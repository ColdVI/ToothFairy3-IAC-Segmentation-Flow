from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

CONNECTIVITY = np.ones((3,3,3),dtype=bool)


def binary_dice(a,b):
    na,nb = int(a.sum()),int(b.sum())
    return 1.0 if na+nb == 0 else 2*int(np.count_nonzero(a&b))/(na+nb)


def component_count(mask):
    return int(ndi.label(mask,structure=CONNECTIVITY)[1])


def remove_small(mask,spacing,minimum_mm3):
    labels,n = ndi.label(mask,structure=CONNECTIVITY)
    sizes = np.bincount(labels.ravel())
    keep = sizes*np.prod(spacing) >= minimum_mm3-1e-12
    keep[0] = False
    return keep[labels]


def crop_union(a,b):
    coords = np.where(a|b)
    if not len(coords[0]):
        return a,b
    lo = [max(0,int(c.min())-1) for c in coords]
    hi = [min(n,int(c.max())+2) for c,n in zip(coords,a.shape)]
    sl = tuple(slice(l,h) for l,h in zip(lo,hi))
    return a[sl],b[sl]


def side_metrics(pred,gt,spacing,min_component_mm3=.27,hd95=False):
    pred,gt = crop_union(pred.astype(bool),gt.astype(bool))
    raw_components,gt_components = component_count(pred),component_count(gt)
    filtered = remove_small(pred,spacing,min_component_mm3)
    # Filtering is prediction postprocessing, applied identically to baseline/Flow; GT is the fixed reference.
    filt_components = component_count(filtered)
    skp = skeletonize(pred).astype(bool); skg = skeletonize(gt).astype(bool)
    ps,gs = int(skp.sum()),int(skg.sum())
    gt_skeleton_valid=bool(gs>0 or not gt.any())
    pred_skeleton_valid=bool(ps>0 or not pred.any())
    if not pred.any() and not gt.any():
        cldice=1.; precision=1.; recall=1.
    elif not pred.any() or not gt.any():
        cldice=0.; precision=1. if not pred.any() else 0.; recall=1. if not gt.any() else 0.
    else:
        precision=float(np.count_nonzero(skp&gt))/ps if ps else None
        recall=float(np.count_nonzero(skg&pred))/gs if gs else None
        if precision is None or recall is None:
            cldice=None # Lee thinning can erase some even-width 3D tubes: do not fabricate a score.
        else:
            cldice=2*precision*recall/(precision+recall) if precision+recall else 0.
    missing = skg & ~pred
    gap_labels,n = ndi.label(missing,structure=CONNECTIVITY)
    # Physical bounding-box diagonal: a reproducible gap extent proxy, NOT geodesic gap/branch length.
    max_extent=0.
    for sl in ndi.find_objects(gap_labels):
        if sl:
            ext=np.asarray([s.stop-s.start for s in sl])*spacing
            max_extent=max(max_extent,float(np.linalg.norm(ext)))
    result = {"dice":binary_dice(pred,gt),"dice_filtered":binary_dice(filtered,gt),
              "components":raw_components,"gt_components":gt_components,
              "betti0_error":abs(raw_components-gt_components),
              "components_filtered":filt_components,"betti0_error_filtered":abs(filt_components-gt_components),
              "cldice":cldice,
              "missing_centerline_fraction":(1-recall) if recall is not None and gt_skeleton_valid else None,
              "outside_centerline_fraction":(1-precision) if precision is not None and pred_skeleton_valid else None,
              "gap_components":int(n) if gt_skeleton_valid else None,
              "max_gap_extent_mm":max_extent if gt_skeleton_valid else None,
              "gt_skeleton_valid":gt_skeleton_valid,"pred_skeleton_valid":pred_skeleton_valid,
              "gt_skeleton_voxels":gs,"pred_skeleton_voxels":ps}
    if hd95:
        if pred.any() and gt.any():
            sp=pred & ~ndi.binary_erosion(pred,structure=CONNECTIVITY,border_value=0)
            sg=gt & ~ndi.binary_erosion(gt,structure=CONNECTIVITY,border_value=0)
            a=ndi.distance_transform_edt(~sg,sampling=spacing)[sp]
            b=ndi.distance_transform_edt(~sp,sampling=spacing)[sg]
            result["hd95_mm"]=float(max(np.percentile(a,95),np.percentile(b,95)))
        else:
            result["hd95_mm"]=0. if not pred.any() and not gt.any() else None
    return result


def case_metrics(pred,label,spacing,left_id=1,right_id=2,min_component_mm3=.27,hd95=False):
    valid=label>=0
    rows=[]
    for side,c,opposite in (("L",left_id,right_id),("R",right_id,left_id)):
        p=(pred==c)&valid; g=label==c
        m=side_metrics(p,g,np.asarray(spacing),min_component_mm3,hd95)
        m.update(side=side,swapped_dice=binary_dice((pred==opposite)&valid,g))
        rows.append(m)
    correct=np.mean([m["dice"] for m in rows]); swapped=np.mean([m["swapped_dice"] for m in rows])
    for m in rows:
        m["swap_suspected"]=bool(swapped>correct+.05)
    return rows


def summarize(rows):
    keys=("dice","dice_filtered","cldice","betti0_error","betti0_error_filtered",
          "missing_centerline_fraction","outside_centerline_fraction","max_gap_extent_mm")
    summary={}
    for k in keys:
        available=[r[k] for r in rows if r[k] is not None]
        summary[k]=float(np.mean(available)) if available else None
        if len(available)!=len(rows):summary[k+"_available_sides"]=len(available)
    summary["skeleton_failure_sides"]=sum(not r["gt_skeleton_valid"] or not r["pred_skeleton_valid"] for r in rows)
    return summary


def paired_bootstrap(rows_a,rows_b,key,seed=123,draws=2000,patient_map=None):
    # Cluster by patient, NOT by patch or L/R side. Optional mapping for split L/R files.
    import re
    patient_map=patient_map or {}
    groups_a={}; groups_b={}
    for rows,groups in ((rows_a,groups_a),(rows_b,groups_b)):
        for r in rows:
            case=r["case"]; patient=patient_map.get(case,re.sub(r"_(L|R)$","",case))
            groups.setdefault(patient,[]).append(float(r[key]) if r[key] is not None else None)
    if set(groups_a) != set(groups_b):
        raise ValueError("Paired bootstrap requires exactly the same patients")
    usable=[c for c in sorted(groups_a) if all(v is not None and np.isfinite(v) for v in groups_a[c]+groups_b[c])]
    diffs=np.asarray([np.mean(groups_a[c])-np.mean(groups_b[c]) for c in usable])
    if not len(diffs):
        return {"mean":None,"ci95_low":None,"ci95_high":None,"patients":0,"excluded_patients":len(groups_a)}
    rng=np.random.default_rng(seed)
    boot=np.mean(rng.choice(diffs,(draws,len(diffs)),replace=True),axis=1)
    return {"mean":float(diffs.mean()),"ci95_low":float(np.quantile(boot,.025)),
            "ci95_high":float(np.quantile(boot,.975)),"patients":len(diffs),"excluded_patients":len(groups_a)-len(diffs)}
