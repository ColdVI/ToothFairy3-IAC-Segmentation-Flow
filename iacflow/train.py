from __future__ import annotations

import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import amp_context, atomic_json, fingerprint, file_hash, fm_loss, read_json, sample_path
from .data import PatchDataset, cache_ready, worker_init
from .inference import VolumeEngine, case_seed
from .metrics import case_metrics, summarize, paired_bootstrap


def default_config():
    return dict(
        checkpoint_path="/EDIT/nnUNet_results/.../fold_0/checkpoint_final.pth",
        plans_path="/EDIT/nnUNet_results/.../plans.json",
        dataset_json_path="/EDIT/nnUNet_results/.../dataset.json",
        splits_path="/EDIT/nnUNet_preprocessed/Dataset504_IAC_LR/splits_final.json",
        preprocessed_dir="/EDIT/nnUNet_preprocessed/Dataset504_IAC_LR/nnUNetPlans_3d_fullres",
        preprocessed_data_identifier="nnUNetPlans_3d_fullres",
        cache_dir="/EDIT/local_ssd/iac_flow_cache",run_dir="/EDIT/local_ssd/iac_flow_run",
        backup_dir=None,reference_dir=None,patient_map=None,
        configuration="3d_fullres",fold=0,left_id=1,right_id=2,patch_size=None,
        checkpoint_split_verified=False, # confirm the ORIGINAL checkpoint never trained on this fold's val patients
        delta_mm=3.,band_mm=.6,weight_floor=.05,eta=.01,margin_scale=.1,adapter_hidden=8,
        batch_size=1,accumulation=2,lr=1e-4,encoder_lr_factor=.1,weight_decay=1e-4,
        max_steps=15000,warmup_steps=250,clip_grad=5.,seed=123,
        foreground_probability=.5,intensity_augmentation=False,
        precision="auto",num_workers=2,prefetch_factor=2,cache_workers=1,cpu_threads=2,
        log_every=25,checkpoint_every=500,checkpoint_minutes=30.,max_session_hours=6.,
        first_probe_step=250,probe_every=1000,sentinel_cases=5,
        nfe_values=[1,4],primary_nfe=4,overlap=.5,feature_cache_mb=512,max_inference_ram_gb=16.,
        min_component_mm3=.27,dice_noninferiority_margin=.001,
        stop_on_no_flow_signal=True,collapse_patience=3,collapse_min_step=1000,
        sensitivity_threshold=1e-3,minimum_cldice_gain=1e-4,
        final_hd95=False, # expensive physical surface distances only in explicit final validation
        deterministic=False,
    )


def validate_config(config,info,splits,device):
    from .core import validate_patch
    if not config.get("checkpoint_split_verified"):
        raise ValueError("Set checkpoint_split_verified=True after verifying that the original checkpoint did not see "
                         "the requested validation patients. Fold number alone cannot prove the split.")
    if any(str(config[k]).startswith("/EDIT") for k in ("cache_dir","run_dir","preprocessed_dir")):
        raise ValueError("Edit local paths before running")
    if not 0<config["eta"]<1 or config["delta_mm"]<=0 or config["band_mm"]<=0 or not 0<config["weight_floor"]<=1:
        raise ValueError("Invalid SDF/path/weight scales")
    if config["precision"]=="auto":
        config["precision"]=("bf16" if torch.cuda.is_bf16_supported() else "fp16") if str(device).startswith("cuda") else "fp32"
    if config["precision"] not in ("fp32","fp16","bf16"):
        raise ValueError("precision must be fp32/fp16/bf16/auto")
    if str(device).startswith("cuda") and config["precision"]=="bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support BF16; select fp16 explicitly")
    config["patch_size"]=list(config["patch_size"] or info["patch_size"])
    if config["primary_nfe"] not in config["nfe_values"] or 1 not in config["nfe_values"]:
        raise ValueError("nfe_values must contain 1 and primary_nfe")
    if min(config["batch_size"],config["accumulation"],config["max_steps"],config["log_every"],config["checkpoint_every"],config["probe_every"])<1:
        raise ValueError("Batch/accumulation/step counts must be positive")
    if config["primary_nfe"]<=1:
        raise ValueError("primary_nfe must exceed 1 to measure multistep contribution")
    for path in (config["cache_dir"],config["run_dir"]):
        if Path(path).resolve()==Path(config["preprocessed_dir"]).resolve():
            raise ValueError("Original preprocessed source cannot be used as an output directory")
    if config["backup_dir"] and Path(config["backup_dir"]).resolve()==Path(config["run_dir"]).resolve():
        raise ValueError("Backup and local run directories must differ")
    return config


def seed_all(seed,deterministic=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark=not deterministic


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def make_loader(config,cases,start_sample=0,samples=None):
    total = samples or config["max_steps"]*config["batch_size"]*config["accumulation"]
    ds=PatchDataset(config["cache_dir"],cases,config["patch_size"],total,config["seed"],config["foreground_probability"])
    kwargs=dict(batch_size=config["batch_size"],sampler=range(start_sample,total),num_workers=config["num_workers"],
                pin_memory=torch.cuda.is_available(),drop_last=True,worker_init_fn=worker_init,
                # DataLoader workers must not consume the model/path generator on restart.
                generator=torch.Generator().manual_seed(config["seed"]+991))
    if config["num_workers"]:
        kwargs.update(prefetch_factor=config["prefetch_factor"],persistent_workers=True)
    return DataLoader(ds,**kwargs)


def move_batch(batch,device):
    return {k:v.to(device,non_blocking=True) for k,v in batch.items() if k != "sample_index"}


def objective(model,batch,config):
    image=batch["image"]
    if config["intensity_augmentation"]:
        shape=(len(image),1,1,1,1)
        image=image*(.9+.2*torch.rand(shape,device=image.device))+.05*torch.randn(shape,device=image.device)
    state,t,_=sample_path(batch["target"],config["eta"])
    with amp_context(image.device,config["precision"]):
        clean=model(image,state,t)
    return fm_loss(clean,batch["target"],batch["weight"],batch["valid"])


def profile_training(model,config,splits,repeats=3):
    """Real forward/backward and data wait, no optimizer updates; restore RNG and buffers."""
    device=next(model.parameters()).device
    before=rng_state(); buffers={k:v.detach().clone() for k,v in model.named_buffers()}
    mode=model.training; model.train()
    loader=make_loader(config,splits["train"],samples=config["batch_size"]*(repeats+1))
    iterator=iter(loader); timings=[]
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for i in range(repeats+1):
            t0=time.perf_counter(); batch=move_batch(next(iterator),device); wait=time.perf_counter()-t0
            if device.type=="cuda": torch.cuda.synchronize(device)
            t1=time.perf_counter(); loss=objective(model,batch,config); loss.backward()
            if device.type=="cuda": torch.cuda.synchronize(device)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite smoke loss")
            if i==0:
                frozen_bad=[n for n,p in model.named_parameters() if not p.requires_grad and p.grad is not None]
                deep=[p for p in model.backbone.encoder.stages[-1].parameters() if p.grad is not None]
                adapter=[a.out.weight.grad for a in model.adapters]
                if frozen_bad or not deep or not any(g is not None and torch.any(g!=0) for g in adapter):
                    raise RuntimeError("Gradient contract failed: frozen/deep-stage/state-adapter paths")
            if i: timings.append((wait,time.perf_counter()-t1))
            model.zero_grad(set_to_none=True)
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for k,b in model.named_buffers(): b.copy_(buffers[k])
        model.train(mode); restore_rng(before)
        del iterator,loader
    mean=np.mean(timings,axis=0)
    result=dict(data_wait_seconds=float(mean[0]),forward_backward_seconds=float(mean[1]),
                estimated_training_hours=float(sum(mean)*config["accumulation"]*config["max_steps"]/3600),
                peak_allocated_gb=torch.cuda.max_memory_allocated(device)/2**30 if device.type=="cuda" else None,
                note="Short profile; excludes validation, AdamW, checkpoint/backup and cache preparation. Not a runtime promise.")
    print(json.dumps(result,indent=2,ensure_ascii=False)); return result


def evaluate(model,config,info,cases,nfes=None,step=0,hd95=False,tag="probe"):
    nfes=sorted(nfes or config["nfe_values"])
    root=Path(config["run_dir"]); root.mkdir(parents=True,exist_ok=True)
    allrows=[]; timing=[]; before=rng_state(); mode=model.training; t0=time.perf_counter()
    model.eval()
    try:
        for case in cases:
            cr=Path(config["cache_dir"])/case
            image=np.load(cr/"image.npy",mmap_mode="r",allow_pickle=False)
            label=np.load(cr/"label.npy",mmap_mode="r",allow_pickle=False)
            ref=np.load(cr/"reference.npy",mmap_mode="r",allow_pickle=False)
            cache_key=fingerprint(dict(ref=read_json(cr/"reference.json"),min_component_mm3=config["min_component_mm3"],
                                       hd95=hd95,metric_version=2))[:16]
            metric_path=root/"reference_metrics"/f"{case}_{cache_key}.json"
            if metric_path.exists():
                baseline=read_json(metric_path)
            else:
                baseline=case_metrics(ref,label,info["spacing"],config["left_id"],config["right_id"],config["min_component_mm3"],hd95)
                atomic_json(metric_path,baseline)
            allrows += [dict(m,case=case,method="baseline",nfe=0,step=step) for m in baseline]
            engine=VolumeEngine(model,image,config)
            for nfe in nfes:
                pred,tm=engine.flow(nfe,config["eta"],case_seed(config["seed"],case),probe=nfe==config["primary_nfe"])
                mt=time.perf_counter()
                metrics=case_metrics(pred,label,info["spacing"],config["left_id"],config["right_id"],config["min_component_mm3"],hd95)
                tm.update(case=case,cpu_metrics_seconds=time.perf_counter()-mt,step=step)
                timing.append(tm)
                if any(not r["gt_skeleton_valid"] or not r["pred_skeleton_valid"] for r in metrics):
                    print(f"[{case}, NFE={nfe}] Empty thinning result: affected clDice/gap metrics marked unavailable.",flush=True)
                allrows += [dict(m,case=case,method=f"flow_{nfe}",nfe=nfe,step=step) for m in metrics]
            del engine
            print(f"[{tag} step={step}] {case}: {(time.perf_counter()-t0)/60:.1f}m cumulative",flush=True)
    finally:
        model.train(mode); restore_rng(before)
    groups={name:[r for r in allrows if r["method"]==name] for name in {r["method"] for r in allrows}}
    summary={name:summarize(rows) for name,rows in sorted(groups.items())}
    state_values=[r["state_sensitivity"] for r in timing if r["state_sensitivity"] is not None]
    result=dict(step=step,cases=list(cases),summary=summary,
                state_sensitivity=float(np.mean(state_values)) if state_values else None,
                total_seconds=time.perf_counter()-t0,rows=allrows,timing=timing)
    atomic_json(root/f"{tag}_{step:07d}.json",result)
    with open(root/f"{tag}_{step:07d}.csv","w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    print(json.dumps({"step":step,"summary":summary,"state_sensitivity":result["state_sensitivity"]},ensure_ascii=False))
    return result


def flow_signal(probe,config):
    one=probe["summary"]["flow_1"]; multi=probe["summary"][f"flow_{config['primary_nfe']}"]
    base=probe["summary"]["baseline"]
    delta=lambda a,b:a-b if a is not None and b is not None else None
    return dict(dice_vs_baseline=multi["dice"]-base["dice"],
                cldice_vs_one=delta(multi["cldice"],one["cldice"]),
                betti_improvement_vs_one=one["betti0_error_filtered"]-multi["betti0_error_filtered"],
                gap_improvement_vs_one=delta(one["missing_centerline_fraction"],multi["missing_centerline_fraction"]),
                state_sensitivity=probe["state_sensitivity"])


def should_stop(probes,config):
    n=config["collapse_patience"]
    if len(probes)<n or probes[-1]["step"]<config["collapse_min_step"]:
        return False
    # Budget stop: three small fixed panels, not a hypothesis test or proof that FM is impossible.
    signals=[flow_signal(p,config) for p in probes[-n:]]
    return all(s["state_sensitivity"] is not None and s["state_sensitivity"]<config["sensitivity_threshold"]
               and s["cldice_vs_one"] is not None and s["cldice_vs_one"]<=config["minimum_cldice_gain"]
               and s["betti_improvement_vs_one"]<=0 and s["gap_improvement_vs_one"] is not None
               and s["gap_improvement_vs_one"]<=config["minimum_cldice_gain"]
               for s in signals)


def run_contract(config,info,cache_fingerprint):
    # These change the learning experiment. Paths and session/log/checkpoint limits may change on resume.
    keys=("patch_size","left_id","right_id","delta_mm","band_mm","weight_floor","eta","margin_scale","adapter_hidden",
          "batch_size","accumulation","lr","encoder_lr_factor","weight_decay","max_steps","warmup_steps","clip_grad",
          "seed","foreground_probability","intensity_augmentation","precision","deterministic",
          "overlap","nfe_values","primary_nfe","min_component_mm3","dice_noninferiority_margin",
          "sentinel_cases","sensitivity_threshold","minimum_cldice_gain")
    return dict(config={k:config[k] for k in keys},source_checkpoint=info["checkpoint_sha256"],
                plans=info["plans_sha256"],cache=cache_fingerprint,format=1,
                code={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def save_checkpoint(path,payload,backup_dir=None):
    t0=time.perf_counter();path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".partial")
    torch.save(payload,temp)
    if path.exists():
        os.replace(path,path.with_name("previous.pt"))
    os.replace(temp,path)
    if backup_dir:
        dest=Path(backup_dir);dest.mkdir(parents=True,exist_ok=True)
        # Copy only a completely written checkpoint. No per-batch remote writes.
        shutil.copyfile(path,dest/"latest.pt.partial")
        if (dest/"latest.pt").exists(): os.replace(dest/"latest.pt",dest/"previous.pt")
        os.replace(dest/"latest.pt.partial",dest/"latest.pt")
    return time.perf_counter()-t0


def train(model,config,info,splits,resume_path=None,until_step=None):
    cache_id=cache_ready(config,info,splits);contract=run_contract(config,info,cache_id)
    run=Path(config["run_dir"]);run.mkdir(parents=True,exist_ok=True)
    if (run/"contract.json").exists() and read_json(run/"contract.json")!=contract:
        raise ValueError("Run contract changed. Use the original settings or a new run directory.")
    if (run/"latest.pt").exists() and not resume_path:
        raise ValueError("Existing run has latest.pt: pass resume_path rather than silently restarting")
    atomic_json(run/"contract.json",contract);atomic_json(run/"config.json",config)
    device=next(model.parameters()).device
    optimizer=torch.optim.AdamW(model.parameter_groups(config["lr"],config["encoder_lr_factor"]),weight_decay=config["weight_decay"])
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and config["precision"]=="fp16")
    step=0;sample_count=0;probes=[];skipped_updates=0
    seed_all(config["seed"],config["deterministic"])
    if resume_path:
        ckpt=torch.load(resume_path,map_location="cpu",weights_only=False)
        if ckpt["contract"]!=contract: raise ValueError("Resume contract mismatch")
        model.load_state_dict(ckpt["model"],strict=True);optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"]);step=ckpt["step"];sample_count=ckpt["sample_count"];probes=ckpt["probes"]
        skipped_updates=ckpt.get("skipped_updates",0)
        restore_rng(ckpt["rng"]);del ckpt
        print(f"Resumed optimizer step {step}, next sample {sample_count}")
    model.training_updates=step-skipped_updates
    loader=make_loader(config,splits["train"],sample_count);iterator=iter(loader)
    model.train();session_start=time.perf_counter();last_save=session_start
    loss_sum=torch.zeros((),device=device);logged=0;data_wait=0.;interval_start=session_start
    stop_reason="max_steps";target=min(config["max_steps"],until_step or config["max_steps"])
    def checkpoint():
        return save_checkpoint(run/"latest.pt",dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
               scaler=scaler.state_dict(),step=step,sample_count=sample_count,rng=rng_state(),contract=contract,
               probes=probes,training_updates=step-skipped_updates,skipped_updates=skipped_updates),config.get("backup_dir"))
    boundary_rng=rng_state();boundary_sample=sample_count;updating=False
    try:
        while step<target:
            boundary_rng=rng_state();boundary_sample=sample_count;updating=False
            optimizer.zero_grad(set_to_none=True)
            if step<config["warmup_steps"]:
                factor=(step+1)/max(1,config["warmup_steps"])
            else:
                phase=(step-config["warmup_steps"])/max(1,config["max_steps"]-config["warmup_steps"])
                factor=.05+.95*(1+math.cos(math.pi*phase))/2
            for group in optimizer.param_groups: group["lr"]=group["base_lr"]*factor
            total=torch.zeros((),device=device)
            for _ in range(config["accumulation"]):
                t0=time.perf_counter();batch=move_batch(next(iterator),device);data_wait+=time.perf_counter()-t0
                loss=objective(model,batch,config)/config["accumulation"]
                scaler.scale(loss).backward();total+=loss.detach()
                sample_count+=config["batch_size"]
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config["clip_grad"])
            # One synchronization per optimizer update catches corruption before committing weights.
            if not bool(torch.isfinite(norm)&torch.isfinite(total)):
                if scaler.is_enabled():
                    scaler.update();optimizer.zero_grad(set_to_none=True);skipped_updates+=1
                    print("Nonfinite FP16 gradient: scale reduced; batch consumed, no weight update.",flush=True)
                    # A failed scaled update still consumes its scheduled sample block.
                    if skipped_updates>max(5,int(.05*(step+1))):
                        raise FloatingPointError("Repeated FP16 overflow; resume last valid checkpoint after checking scale/data.")
                else:
                    raise FloatingPointError("Nonfinite loss/gradient; stopped before optimizer update")
            else:
                updating=True;scaler.step(optimizer);scaler.update();updating=False
            step+=1;model.training_updates=step-skipped_updates
            loss_sum+=torch.nan_to_num(total,nan=0.,posinf=0.,neginf=0.);logged+=1
            boundary_rng=rng_state();boundary_sample=sample_count
            if step%config["log_every"]==0:
                elapsed=time.perf_counter()-interval_start
                row=dict(step=step,loss=float(loss_sum/max(logged,1)),seconds_per_step=elapsed/max(logged,1),
                         data_wait_seconds=data_wait,lr=optimizer.param_groups[-1]["lr"],
                         skipped_updates=skipped_updates,session_hours=(time.perf_counter()-session_start)/3600)
                with open(run/"history.jsonl","a",encoding="utf-8") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
                print(f"step={step} loss={row['loss']:.5f} sec/update={row['seconds_per_step']:.2f} "
                      f"data_wait={data_wait:.2f}s",flush=True)
                loss_sum.zero_();logged=0;data_wait=0.;interval_start=time.perf_counter()
            due=step==config["first_probe_step"] or step%config["probe_every"]==0
            if due:
                checkpoint();last_save=time.perf_counter()
                cases=sorted(splits["val"])[:config["sentinel_cases"]]
                probe=evaluate(model,config,info,cases,step=step)
                # Save compact summaries in checkpoints; detailed arrays/metrics are separate tiny JSON/CSV files.
                probes.append({k:probe[k] for k in ("step","cases","summary","state_sensitivity")})
                print("Flow diagnostic:",flow_signal(probe,config),flush=True)
                interval_start=time.perf_counter();loss_sum.zero_();logged=0;data_wait=0.
                if should_stop(probes,config) and config["stop_on_no_flow_signal"]:
                    stop_reason="no_flow_signal_budget_stop";break
            now=time.perf_counter()
            if step%config["checkpoint_every"]==0 or now-last_save>=config["checkpoint_minutes"]*60:
                seconds=checkpoint();last_save=time.perf_counter()
                print(f"Checkpoint + optional backup: {seconds:.1f}s",flush=True)
            if (now-session_start)/3600>=config["max_session_hours"]:
                stop_reason="session_time_budget";break
    except KeyboardInterrupt:
        if updating:
            # Interrupted optimizer writes may be partial: do not overwrite last known-good checkpoint.
            print("Interrupted inside optimizer.step. Resume the previously completed latest.pt.",flush=True)
            raise
        restore_rng(boundary_rng);sample_count=boundary_sample;optimizer.zero_grad(set_to_none=True)
        stop_reason="keyboard_interrupt"
    except Exception:
        # Preserve last completed on-disk checkpoint; never save suspect weights.
        raise
    finally:
        del iterator,loader
    checkpoint()
    status=dict(step=step,successful_updates=step-skipped_updates,skipped_updates=skipped_updates,
                sample_count=sample_count,stop_reason=stop_reason,session_hours=(time.perf_counter()-session_start)/3600)
    atomic_json(run/"status.json",status);print(json.dumps(status,indent=2))
    return status


def final_report(evaluation,config):
    groups={name:[r for r in evaluation["rows"] if r["method"]==name]
            for name in {r["method"] for r in evaluation["rows"]}}
    primary=groups[f"flow_{config['primary_nfe']}"]
    mapping=read_json(config["patient_map"]) if config.get("patient_map") else None
    report={}
    for metric in ("dice","cldice","betti0_error_filtered","missing_centerline_fraction"):
        report[f"{metric}_vs_baseline"]=paired_bootstrap(primary,groups["baseline"],metric,patient_map=mapping)
        report[f"{metric}_vs_NFE1"]=paired_bootstrap(primary,groups["flow_1"],metric,patient_map=mapping)
    ni=report["dice_vs_baseline"]["ci95_low"]>=-config["dice_noninferiority_margin"]
    topology=report["betti0_error_filtered_vs_baseline"]["ci95_high"]<0
    step_gain=report["betti0_error_filtered_vs_NFE1"]["ci95_high"]<0
    # Other topology/centerline metrics remain visible; do not switch the primary metric after seeing outcomes.
    report["decision"]={"dice_noninferiority_pass":ni,"primary_topology_improvement":topology,
                        "primary_topology_multistep_gain":step_gain,
                        "declared_primary_claim_pass":bool(ni and topology and step_gain),
                        "scope":"Exploratory single-fold validation; selected checkpoint/monitoring creates optimism. "
                                "No test-set, clinical, novelty or uncertainty-calibration claim."}
    atomic_json(Path(config["run_dir"])/"final_report.json",report)
    return report
