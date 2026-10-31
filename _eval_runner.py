# -*- coding: utf-8 -*-
"""_eval_runner.py - test-only runner used by eval_boundary.py.

Loads a baseline script, forces its ``MODE`` to "test" (so the saved
checkpoint is used and no training happens), transparently wraps the
script's own ``calculate_metrics`` to additionally accumulate HD95 / ASSD /
NSD on exactly the same test batches, then runs the script's ``main()``.

Usage (must be started from the model's own directory):
    python <repo>/_eval_runner.py main.py
"""
import json
import os
import re
import sys
import types

import numpy as np
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    from DPRASRNet.boundary_metrics import boundary_metrics_batch
except Exception:  # pragma: no cover - fallback when the package layout changes
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "_bm", os.path.join(REPO, "DPRASRNet", "boundary_metrics.py")
    )
    _bm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_bm)
    boundary_metrics_batch = _bm.boundary_metrics_batch

MARKER = "@@BOUNDARY@@"


def _run(target):
    with open(target, encoding="utf-8") as f:
        src = f.read()

    # Force test mode: every baseline hard-codes `MODE = "train"`.
    src, n = re.subn(r'MODE\s*=\s*"train"', 'MODE = "test"', src)
    if n == 0:
        print(f"@@EVAL@@ERROR: no `MODE = \"train\"` assignment found in {target}")
        return 2

    mod = types.ModuleType("__eval_target__")
    mod.__file__ = os.path.abspath(target)
    exec(compile(src, target, "exec"), mod.__dict__)

    orig = mod.__dict__.get("calculate_metrics")
    if orig is None:
        print("@@EVAL@@ERROR: calculate_metrics() not found in the target script")
        return 3

    acc = {"hd95": [], "assd": [], "nsd": []}

    def wrapped(pred, target_mask, *args, **kwargs):
        out = orig(pred, target_mask, *args, **kwargs)
        try:
            with torch.no_grad():
                p = pred.detach() if torch.is_tensor(pred) else torch.as_tensor(pred)
                t = (
                    target_mask.detach()
                    if torch.is_tensor(target_mask)
                    else torch.as_tensor(target_mask)
                )
                p, t = p.float(), t.float()
                # All baselines already apply sigmoid; guard for logits anyway.
                if float(p.min()) < 0.0 or float(p.max()) > 1.0:
                    p = torch.sigmoid(p)
                res = boundary_metrics_batch(p, t, threshold=0.5, tau=2.0)
                for k in ("hd95", "assd", "nsd"):
                    acc[k].extend(res["per_sample"][k])
        except Exception as exc:  # never let metric bookkeeping break evaluation
            print(f"@@EVAL@@WARN boundary metrics failed: {exc}")
        return out

    mod.calculate_metrics = wrapped
    mod.main()

    summary = {}
    for k, v in acc.items():
        arr = np.asarray(v, dtype=float)
        arr = arr[~np.isnan(arr)]
        summary[k] = float(arr.mean()) if arr.size else float("nan")
    summary["n"] = int(len(acc["hd95"]))
    print(MARKER + json.dumps(summary))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python _eval_runner.py <script.py>")
        sys.exit(64)
    sys.exit(_run(sys.argv[1]))
