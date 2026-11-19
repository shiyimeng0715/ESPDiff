import os
import sys
import json
import time

import cv2
import numpy as np

from utils.metrics import Smeasure, Fmeasure, WeightedFmeasure, _cal_mae, _prepare_data
from utils.eval import cal_em_with_cumsumhistogram

MASK_PATH = './ORSI_dataset/test_dataset/'

SM = Smeasure()
FM = Fmeasure(beta=0.3)   # F_β^max: β² = 0.3
WFM = WeightedFmeasure(beta=1)  # F_β^ω: β = 1

# (label, pred_path, dataset_name) — pred_root = pred_path/<dataset>
JOBS = [
    # full models
    ('ours · EORSSD', 'results_t5_seed0/EORSSD', 'EORSSD'),
    ('ours · ORSSD', 'results_t5_seed0', 'ORSSD'),  # ORSSD 的 PNG 直接位于 results_t5_seed0/ORSSD/ 下（无嵌套子目录）
    ('ours · ORSI-4199', 'results_t5_seed0/ORS4199', 'ors4199'),
    # ablations (all on EORSSD)
    ('w/o Multi-scale', 'results_t5_seed0/ablate_compose_multiscale', 'EORSSD'),
    ('w/o Channel-split', 'results_t5_seed0/ablate_compose_channelsplit', 'EORSSD'),
    ('w/o Shared-context', 'results_t5_seed0/ablate_compose_sharedcontext', 'EORSSD'),
    ('w/o Boundary-branch', 'results_t5_seed0/ablate_inject_boundary', 'EORSSD'),
    ('w/o Skeleton-branch', 'results_t5_seed0/ablate_inject_skeleton', 'EORSSD'),
    ('仅 L_diff', 'results_t5_seed0/ablate_loss_diffonly', 'EORSSD'),
    ('L_diff + 0.5·L_pvt', 'results_t5_seed0/ablate_loss_diffpvt', 'EORSSD'),
    ('L_diff + L_compose', 'results_t5_seed0/ablate_loss_diffcompose', 'EORSSD'),
    ('w/o Boundary prior', 'results_t5_seed0/ablate_prior_boundary', 'EORSSD'),
    ('w/o Skeleton prior', 'results_t5_seed0/ablate_prior_skeleton', 'EORSSD'),
    ('w/o GPDM', 'results_t5_seed0/ablate_woGPDM', 'EORSSD'),
    ('w/o GPIM', 'results_t5_seed0/ablate_woGPIM', 'EORSSD'),
    ('w/o GPDM & GPIM', 'results_t5_seed0/ablate_woGPDM_and_GPIM', 'EORSSD'),
    # sampling-step ablations (EORSSD)
    ('T=1', 'results_t5_seed0/ablate_T/results_ablate6_T1', 'EORSSD'),
    ('T=5', 'results_t5_seed0/ablate_T/results_ablate6_T5', 'EORSSD'),
    ('T=10', 'results_t5_seed0/ablate_T/results_ablate6_T10', 'EORSSD'),
    ('T=15', 'results_t5_seed0/ablate_T/results_ablate6_T15', 'EORSSD'),
    ('T=20', 'results_t5_seed0/ablate_T/results_ablate6_T20', 'EORSSD'),
]


def measure_one(mask_root, pred_root, mask_name):
    gt = cv2.imread(os.path.join(mask_root, mask_name), cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(os.path.join(pred_root, mask_name), cv2.IMREAD_GRAYSCALE)
    assert gt is not None and pred is not None, f'read fail: {mask_name}'
    pred, gt = _prepare_data(pred=pred, gt=gt)

    sm = SM.cal_sm(pred, gt)

    _, _, fm_curve = FM.cal_pr(pred, gt)  # (256,) changeable F-measure curve

    gt_fg_numel = np.count_nonzero(gt)
    gt_size = gt.shape[0] * gt.shape[1]
    em_curve = cal_em_with_cumsumhistogram(pred, gt, gt_fg_numel, gt_size)

    if np.all(~gt):
        wfm = 0.0
    else:
        wfm = WFM.cal_wfm(pred, gt)

    mae = _cal_mae(pred, gt)
    return sm, fm_curve, wfm, em_curve, mae


def eval_folder(pred_path, dataset_name):
    mask_root = os.path.join(MASK_PATH, dataset_name, 'GT')
    pred_root = os.path.join(pred_path, dataset_name)
    names = sorted(os.listdir(mask_root))

    sms, fm_curves, wfms, em_curves, maes = [], [], [], [], []
    for n in names:
        sm, fcurve, wfm, ecurve, mae = measure_one(mask_root, pred_root, n)
        sms.append(sm)
        fm_curves.append(fcurve)
        wfms.append(wfm)
        em_curves.append(ecurve)
        maes.append(mae)

    fm_curves = np.array(fm_curves, dtype=np.float64)   # (N, 256)
    em_curves = np.array(em_curves, dtype=np.float64)   # (N, 256)

    return {
        'S_alpha': float(np.mean(sms)),
        'F_beta_max': float(fm_curves.mean(axis=0).max()),
        'F_beta_w': float(np.mean(wfms)),
        'E_xi_max': float(em_curves.mean(axis=0).max()),
        'E_xi_mean': float(em_curves.mean(axis=0).mean()),
        'MAE': float(np.mean(maes)),
    }


def main():
    results = {}
    t0 = time.time()
    for i, (label, pred_path, dataset) in enumerate(JOBS, 1):
        print(f"\n[{i}/{len(JOBS)}] {label} (dataset={dataset})", flush=True)
        try:
            r = eval_folder(pred_path, dataset)
            results[label] = r
            print(f"    S_a={r['S_alpha']:.4f} Fb_max={r['F_beta_max']:.4f} Fb_w={r['F_beta_w']:.4f} "
                  f"E_max={r['E_xi_max']:.4f} E_mean={r['E_xi_mean']:.4f} MAE={r['MAE']:.6f}", flush=True)
        except Exception as e:
            print(f"    [FAILED] {type(e).__name__}: {e}", flush=True)
            results[label] = {"error": str(e)}

    print("\n\n========== 结果汇总 ==========")
    print(f"{'模型/变体':<24}{'S_α↑':>9}{'F_β^max↑':>11}{'F_β^ω↑':>11}{'E_ξ^max↑':>11}{'E_ξ^mean↑':>12}{'MAE↓':>12}")
    rows = []
    for label, r in results.items():
        if 'error' in r:
            rows.append(f"{label:<24}    ERROR: {r['error']}")
            print(rows[-1])
            continue
        line = (f"{label:<24}{r['S_alpha']:>9.4f}{r['F_beta_max']:>11.4f}{r['F_beta_w']:>11.4f}"
                f"{r['E_xi_max']:>11.4f}{r['E_xi_mean']:>12.4f}{r['MAE']:>12.6f}")
        rows.append(line)
        print(line)

    print(f"\n总耗时: {time.time() - t0:.1f}s")

    with open('eval_all_t5.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # markdown table (paper format)
    md = ["# results_t5_seed0 指标结果", "",
          "| 模型/变体 | S_α↑ | F_β^max↑ | F_β^ω↑ | E_ξ^max↑ | E_ξ^mean↑ | MAE↓ |",
          "|---|---|---|---|---|---|---|"]
    for label, r in results.items():
        if 'error' in r:
            md.append(f"| {label} | ERROR | - | - | - | - | - |")
            continue
        md.append(f"| {label} | {r['S_alpha']:.4f} | {r['F_beta_max']:.4f} | {r['F_beta_w']:.4f} | "
                  f"{r['E_xi_max']:.4f} | {r['E_xi_mean']:.4f} | {r['MAE']:.6f} |")
    md.append("")
    os.makedirs('paper', exist_ok=True)
    with open('paper/results_t5_seed0_metrics.md', 'w') as f:
        f.write("\n".join(md))
    print("已保存: eval_all_t5.json / paper/results_t5_seed0_metrics.md")


if __name__ == '__main__':
    main()
