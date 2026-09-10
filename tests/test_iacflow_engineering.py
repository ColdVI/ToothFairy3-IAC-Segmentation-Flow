from pathlib import Path
import copy
import json

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt
import torch

from iacflow.core import (IACFlow,check_head_initialization,decode_sdf,fm_loss,load_backbone,
                          sample_path,velocity_from_clean)
from iacflow.data import clipped_sdf,prepare_cpu_cache,load_splits,write_reference,reference_complete,cache_ready
from iacflow.demo import create_demo_workspace,tiny_backbone
from iacflow.inference import VolumeEngine
from iacflow.metrics import case_metrics,paired_bootstrap
from iacflow.train import train,validate_config,profile_training

torch.set_num_threads(1)


def test_closed_form_and_weighted_objective():
    torch.manual_seed(11)
    y=torch.randn(2,2,4,5,6);x,t,u=sample_path(y,.01)
    v=velocity_from_clean(y,x,t,.01)
    torch.testing.assert_close(v,u,rtol=2e-5,atol=2e-5)
    estimate=y+.15*torch.randn_like(y)
    w=torch.rand_like(y)+.05;valid=torch.rand(2,1,4,5,6)>.2
    vv=velocity_from_clean(estimate,x,t,.01)
    s=(1-.99*t)[:,None,None,None,None]
    actual=fm_loss(estimate,y,w,valid)
    expected=((w*valid*s.square()*(vv-u).square()).sum((1,2,3,4))/(w*valid).sum((1,2,3,4))).mean()
    torch.testing.assert_close(actual,expected)


@pytest.mark.parametrize("spacing",[(.3,.3,.3),(.2,.4,.7)])
@pytest.mark.parametrize("kind",["tube","border","empty","solid"])
def test_clipped_edt_matches_full_volume(spacing,kind):
    a=np.zeros((29,25,23),dtype=bool)
    if kind=="tube":a[3:27,11:14,10:13]=1
    elif kind=="border":a[:7,:4,:5]=1
    elif kind=="solid":a[:]=1
    actual=clipped_sdf(a,spacing,.6)
    full=np.clip(distance_transform_edt(~a,sampling=spacing)-distance_transform_edt(a,sampling=spacing),-.6,.6)/.6
    if kind=="empty":full=np.ones_like(full)
    np.testing.assert_allclose(actual,full,atol=1e-7)
    np.testing.assert_array_equal(actual<0,a)


@pytest.mark.parametrize("residual",[False,True])
def test_warm_head_and_gradient_contract(residual):
    model=IACFlow(tiny_backbone(residual),left_id=2,right_id=1,adapter_hidden=4)
    image=torch.randn(1,1,16,16,16)
    result=check_head_initialization(model,image)
    assert result["confident_mismatch_voxels"]==0
    model.train();y=torch.randn(1,2,16,16,16);x,t,u=sample_path(y)
    loss=fm_loss(model(image,x,t),y,torch.ones_like(y),torch.ones_like(y[:,0:1]))
    loss.backward()
    assert model.trainable_report()["trainable_encoder_stages"]==[2]
    assert all(p.grad is None for p in model.backbone.encoder.stages[0].parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone.encoder.stages[-1].parameters())
    assert any(a.out.weight.grad.abs().sum()>0 for a in model.adapters)
    # Zero final adapter projection blocks inner gradient only on the very first update, intentionally.
    assert all(a.features[0].weight.grad.abs().sum()==0 for a in model.adapters)
    with torch.no_grad():
        f=model.encode(image)
        torch.testing.assert_close(model.clean(f,x,t),model.clean(f,torch.randn_like(x),1-t),rtol=0,atol=0)


class ConstantClean(torch.nn.Module):
    def __init__(self):
        super().__init__();self.p=torch.nn.Parameter(torch.zeros(()));self.left_id=1;self.right_id=2
    def encode(self,image):return [image]
    def clean(self,features,state,t):
        im=features[0]
        return torch.cat((im-.2,.1-im),dim=1)


def test_shared_volume_euler_constant_clean_exact_and_nfe_counts():
    config=dict(precision="fp32",patch_size=[8,8,8],feature_cache_mb=2,max_inference_ram_gb=2.,overlap=.5)
    rng=np.random.default_rng(12);image=rng.normal(size=(1,12,13,14)).astype(np.float32)
    m=ConstantClean();engine=VolumeEngine(m,image,config)
    before=engine.decoder_calls
    a,_=engine.flow(1,.01,123)
    calls_one=engine.decoder_calls-before
    b,_=engine.flow(8,.01,123)
    assert engine.decoder_calls-before==9*calls_one
    np.testing.assert_array_equal(a,b)
    noise=np.random.default_rng(123).standard_normal((2,12,13,14),dtype=np.float32)
    expected=decode_sdf(np.concatenate((image-.2,.1-image),0)+.01*noise)
    np.testing.assert_array_equal(a,expected)


def setup_cache(tmp_path):
    config=create_demo_workspace(tmp_path)
    backbone,info=load_backbone(config["checkpoint_path"],config["plans_path"],config["dataset_json_path"],"3d_fullres",0)
    splits=load_splits(config["splits_path"],0)
    validate_config(config,info,splits,"cpu")
    prepare_cpu_cache(config,info,splits)
    # Synthetic fixture references; NOT a clinical baseline measurement.
    for c in splits["train"]+splits["val"]:
        label=np.load(Path(config["cache_dir"])/c/"label.npy")
        write_reference(config["cache_dir"],c,label,info,config)
    return config,info,splits,backbone


def test_resume_matches_uninterrupted_optimizer_rng_and_data(tmp_path):
    config,info,splits,backbone=setup_cache(tmp_path)
    config.update(first_probe_step=999,probe_every=999,checkpoint_every=99,max_steps=4)
    torch.manual_seed(4);first=IACFlow(copy.deepcopy(backbone),adapter_hidden=4)
    second=copy.deepcopy(first)
    config["run_dir"]=str(tmp_path/"continuous")
    train(first,config,info,splits)
    config["run_dir"]=str(tmp_path/"resumed")
    train(second,config,info,splits,until_step=2)
    # Simulates a fresh process: create another model, then restore model+optimizer+RNG+sample cursor.
    restored=IACFlow(copy.deepcopy(backbone),adapter_hidden=4)
    train(restored,config,info,splits,resume_path=Path(config["run_dir"])/"latest.pt")
    for k,v in first.state_dict().items():
        torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)


def test_profile_no_update_and_fold_rejection(tmp_path):
    config,info,splits,backbone=setup_cache(tmp_path)
    model=IACFlow(backbone,adapter_hidden=4)
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    profile_training(model,config,splits,repeats=1)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match="fold"):
        load_backbone(config["checkpoint_path"],config["plans_path"],config["dataset_json_path"],"3d_fullres",1)
    wrong=copy.deepcopy(info);wrong["checkpoint_sha256"]="wrong"
    assert not reference_complete(config["cache_dir"],splits["train"][0],wrong,config)


def test_real_gap_vs_speckle_and_patient_bootstrap():
    label=np.zeros((24,24,24),dtype=np.int8);label[2:22,8:11,7:10]=1;label[2:22,15:18,16:19]=2
    p=label.copy();p[10:13,8:11,7:10]=0;p[0,0,0]=1
    rows=case_metrics(p,label,[.3,.3,.3],min_component_mm3=.27)
    left=rows[0]
    assert left["components"]==3 and left["components_filtered"]==2
    assert left["betti0_error_filtered"]==1
    assert left["missing_centerline_fraction"]>0 and left["max_gap_extent_mm"]>0
    assert 0<left["cldice"]<1
    for r in rows:r["case"]="patient_A"
    result=paired_bootstrap(rows,rows,"dice")
    assert result["patients"]==1 and result["mean"]==0


def test_empty_skeleton_is_not_reported_as_a_measured_zero():
    # Lee thinning can erase even-by-even tubes, despite a nonempty segmentation.
    label=np.zeros((20,12,12),dtype=np.int8);label[2:18,4:8,4:8]=1
    row=case_metrics(label,label,[.3,.3,.3])[0]
    assert row["dice"]==1
    if row["gt_skeleton_voxels"]==0:
        assert row["cldice"] is None and row["max_gap_extent_mm"] is None
    else:
        assert row["cldice"]==1
