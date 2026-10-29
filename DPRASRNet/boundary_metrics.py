"""
Boundary-sensitive segmentation metrics: HD95, ASSD, NSD.
==========================================================

Implements the metrics requested to support the "precise boundary delineation"
claims of the paper:

  * HD95  -- 95th-percentile Hausdorff distance (MONAI-compatible convention:
              95th percentile over the concatenated directional distance sets).
  * ASSD  -- Average Symmetric Surface Distance (mean of the two directional
              mean surface distances).
  * NSD   -- Normalized Surface Dice at tolerance tau:
              fraction of surface points closer than tau, averaged over both
              directions. tau is in PIXELS (all masks are evaluated at the
              network input resolution, e.g. 256x256; tau=2 px is a common
              choice, state it in the paper).

Empty-mask convention (important for BUSI, which contains normal images):
  * pred empty AND gt empty  -> counted as perfect (HD=0, ASSD=0, NSD=1).
  * exactly one empty        -> the sample is INVALID for boundary metrics;
                                it is excluded from the aggregate mean and the
                                valid count is reported alongside.
  * Region metrics (Dice/IoU/...) are still computed for all samples.

Dependencies: numpy, scipy (pip install scipy).
"""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _surface(mask_bool):
    """Return binary surface of a boolean mask (mask minus its erosion)."""
    if mask_bool.sum() == 0:
        return mask_bool.astype(bool)
    eroded = binary_erosion(mask_bool, structure=np.ones((3, 3), dtype=bool))
    return mask_bool ^ eroded


def surface_distances(pred_bool, gt_bool):
    """Directional surface point distances (in pixels).

    Returns (d_pred_to_gt, d_gt_to_pred): for every surface pixel of one mask,
    the Euclidean distance to the nearest surface pixel of the other mask.
    """
    surf_pred = _surface(pred_bool)
    surf_gt = _surface(gt_bool)

    # Distance to the nearest gt surface pixel, evaluated at every pixel.
    if surf_gt.sum() == 0:
        dist_to_gt_surf = None
    else:
        dist_to_gt_surf = distance_transform_edt(~surf_gt)
    if surf_pred.sum() == 0:
        dist_to_pred_surf = None
    else:
        dist_to_pred_surf = distance_transform_edt(~surf_pred)

    d1 = dist_to_gt_surf[surf_pred] if (
        dist_to_gt_surf is not None and surf_pred.sum() > 0) else np.array([])
    d2 = dist_to_pred_surf[surf_gt] if (
        dist_to_pred_surf is not None and surf_gt.sum() > 0) else np.array([])
    return d1, d2


def hd95_assd_nsd(pred_bool, gt_bool, tau=2.0):
    """Compute (HD95, ASSD, NSD, valid) for a single binary mask pair.

    valid=False means one mask is empty and the other is not -> the pair is
    excluded from boundary-metric aggregates.
    """
    pred_empty = pred_bool.sum() == 0
    gt_empty = gt_bool.sum() == 0
    if pred_empty and gt_empty:
        return 0.0, 0.0, 1.0, True
    if pred_empty != gt_empty:
        return float("inf"), float("inf"), 0.0, False

    d1, d2 = surface_distances(pred_bool, gt_bool)  # both non-empty here
    all_d = np.concatenate([d1, d2])
    hd95 = float(np.percentile(all_d, 95))
    assd = float((d1.mean() + d2.mean()) / 2.0)
    nsd = float(
        ((d1 <= tau).sum() / max(len(d1), 1)
         + (d2 <= tau).sum() / max(len(d2), 1)) / 2.0
    )
    return hd95, assd, nsd, True


def boundary_metrics_batch(preds, targets, threshold=0.5, tau=2.0):
    """Aggregate boundary metrics over a batch / eval loop.

    Args:
        preds:   torch.Tensor (B,1,H,W) probabilities (after sigmoid), or
                 numpy array with the same shape.
        targets: same shape, binary ground truth.
        threshold: binarization threshold for preds.
        tau: NSD tolerance in pixels.

    Returns dict:
        hd95 / assd / nsd : means over VALID samples (excludes one-side-empty)
        n_valid / n_total : sample accounting
        per_sample        : lists of per-sample values ('nan' where invalid)
    """
    import torch

    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    preds = np.asarray(preds)
    targets = np.asarray(targets)
    if preds.ndim == 4:
        preds, targets = preds[:, 0], targets[:, 0]

    hd95s, assds, nsds = [], [], []
    per = {"hd95": [], "assd": [], "nsd": []}
    n_valid, n_total = 0, targets.shape[0]

    for p, g in zip(preds, targets):
        p_bin = p > threshold
        g_bin = g > 0.5
        hd, assd, nsd, valid = hd95_assd_nsd(p_bin, g_bin, tau=tau)
        n_total_i = 1
        if valid:
            n_valid += 1
            hd95s.append(hd)
            assds.append(assd)
            nsds.append(nsd)
            per["hd95"].append(hd)
            per["assd"].append(assd)
            per["nsd"].append(nsd)
        else:
            per["hd95"].append(float("nan"))
            per["assd"].append(float("nan"))
            per["nsd"].append(float("nan"))
        del n_total_i

    return {
        "hd95": float(np.mean(hd95s)) if hd95s else float("nan"),
        "assd": float(np.mean(assds)) if assds else float("nan"),
        "nsd": float(np.mean(nsds)) if nsds else float("nan"),
        "n_valid": n_valid,
        "n_total": n_total,
        "per_sample": per,
    }


if __name__ == "__main__":
    # ---------------- self-test ----------------
    import torch

    def circle(h, w, cy, cx, r):
        yy, xx = np.ogrid[:h, :w]
        return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r

    H = W = 256
    gt = circle(H, W, 128, 128, 60)

    # case 1: identical -> hd95=0, assd=0, nsd=1
    hd, assd, nsd, ok = hd95_assd_nsd(gt.copy(), gt)
    assert ok and hd == 0 and assd == 0 and nsd == 1.0, (hd, assd, nsd)
    print(f"identical     : HD95={hd:.3f} ASSD={assd:.3f} NSD={nsd:.3f}")

    # case 2: shifted by 5 px -> HD95 ~= 5 (max displacement);
    # ASSD ~= shift * mean|cos| = 5 * 2/pi ~= 3.18 (normal-projection average)
    pred = circle(H, W, 123, 128, 60)
    hd, assd, nsd, ok = hd95_assd_nsd(pred, gt)
    assert ok and 4.5 < hd < 5.5 and 2.5 < assd < 3.8, (hd, assd)
    print(f"shift 5px     : HD95={hd:.3f} ASSD={assd:.3f} NSD={nsd:.3f}")

    # case 3: both empty -> perfect by convention
    e = np.zeros((H, W), dtype=bool)
    hd, assd, nsd, ok = hd95_assd_nsd(e.copy(), e.copy())
    assert ok and nsd == 1.0
    print(f"both empty    : HD95={hd:.3f} ASSD={assd:.3f} NSD={nsd:.3f} (valid)")

    # case 4: one empty -> invalid
    hd, assd, nsd, ok = hd95_assd_nsd(e.copy(), gt)
    assert not ok
    print(f"one empty     : invalid (excluded from aggregate) -> OK")

    # case 5: batch API with torch tensors
    pb = torch.rand(2, 1, H, W)
    tb = torch.zeros(2, 1, H, W)
    tb[0, 0] = torch.from_numpy(gt.astype(np.float32))
    res = boundary_metrics_batch(pb, tb)
    assert res["n_total"] == 2 and res["n_valid"] == 1
    print(f"batch API     : n_valid={res['n_valid']}/{res['n_total']} "
          f"HD95={res['hd95']:.3f} NSD={res['nsd']:.3f} -> OK")

    print("\nAll boundary-metric self-tests passed.")
