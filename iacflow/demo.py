"""Tiny synthetic engineering check. No dental data, no segmentation performance claim."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .core import atomic_json
from .train import default_config


def tiny_backbone(residual=False):
    from dynamic_network_architectures.architectures.unet import PlainConvUNet,ResidualEncoderUNet
    cls=ResidualEncoderUNet if residual else PlainConvUNet
    kwargs={"n_blocks_per_stage" if residual else "n_conv_per_stage":[1,1,1]}
    return cls(input_channels=1,n_stages=3,features_per_stage=[4,8,12],conv_op=nn.Conv3d,
               kernel_sizes=[[3,3,3]]*3,strides=[[1,1,1],[2,2,2],[2,2,2]],
               num_classes=3,n_conv_per_stage_decoder=[1,1],conv_bias=True,
               norm_op=nn.InstanceNorm3d,norm_op_kwargs={"eps":1e-5,"affine":True},
               nonlin=nn.LeakyReLU,nonlin_kwargs={"negative_slope":.01,"inplace":True},
               deep_supervision=False,**kwargs)


def create_demo_workspace(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/"run"/"latest.pt").exists():
        raise ValueError("Demo output already exists. Use a fresh demo directory to preserve the previous run.")
    data=root/"preprocessed";data.mkdir(exist_ok=True)
    torch.manual_seed(7);rng=np.random.default_rng(7)
    for i in range(4):
        zz,yy,xx=np.indices((16,20,20))
        mask=np.zeros(zz.shape,dtype=np.int8)
        mask[((yy-9)**2+(xx-5-.05*zz)**2)<5]=1
        mask[((yy-10)**2+(xx-15+.05*zz)**2)<5]=2
        image=(mask>0).astype(np.float32)+rng.normal(0,.2,mask.shape).astype(np.float32)
        np.save(data/f"demo_{i}.npy",image[None]);np.save(data/f"demo_{i}_seg.npy",mask[None])
    plans={"configurations":{"3d_fullres":{"spacing":[.3,.3,.3],"patch_size":[16,16,16],
         "data_identifier":"demo_preprocessed","architecture":{
         "network_class_name":"dynamic_network_architectures.architectures.unet.PlainConvUNet",
         "arch_kwargs":{"n_stages":3,"features_per_stage":[4,8,12],"conv_op":"torch.nn.Conv3d",
         "kernel_sizes":[[3,3,3]]*3,"strides":[[1,1,1],[2,2,2],[2,2,2]],"n_conv_per_stage":[1,1,1],
         "n_conv_per_stage_decoder":[1,1],"conv_bias":True,"norm_op":"torch.nn.InstanceNorm3d",
         "norm_op_kwargs":{"eps":1e-5,"affine":True},"dropout_op":None,"dropout_op_kwargs":None,
         "nonlin":"torch.nn.LeakyReLU","nonlin_kwargs":{"negative_slope":.01,"inplace":True}},
         "_kw_requires_import":["conv_op","norm_op","dropout_op","nonlin"]}}}}
    atomic_json(root/"plans.json",plans)
    atomic_json(root/"dataset.json",{"channel_names":{"0":"CT"},"labels":{"background":0,"IAC_L":1,"IAC_R":2}})
    atomic_json(root/"splits.json",[{"train":["demo_0","demo_1"],"val":["demo_2","demo_3"]}])
    torch.save({"network_weights":tiny_backbone().state_dict(),"init_args":{"fold":0,"configuration":"3d_fullres"},
                "current_epoch":0},root/"original.pth")
    config=default_config()
    config.update(checkpoint_path=str(root/"original.pth"),plans_path=str(root/"plans.json"),
        dataset_json_path=str(root/"dataset.json"),splits_path=str(root/"splits.json"),preprocessed_dir=str(data),
        preprocessed_data_identifier="demo_preprocessed",cache_dir=str(root/"cache"),run_dir=str(root/"run"),
        checkpoint_split_verified=True,patch_size=[16,16,16],num_workers=0,cache_workers=1,cpu_threads=1,
        batch_size=1,accumulation=1,max_steps=4,warmup_steps=1,first_probe_step=4,probe_every=4,
        log_every=1,checkpoint_every=2,sentinel_cases=1,nfe_values=[1,2],primary_nfe=2,
        feature_cache_mb=32,precision="fp32",deterministic=True,adapter_hidden=4,max_session_hours=1.)
    return config
