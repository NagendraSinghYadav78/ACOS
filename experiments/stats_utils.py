"""
experiments/stats_utils.py

Statistical testing utilities used to rigorously compare paired
forecast-error samples (Holt's method vs. a naive baseline) across
SKUs/stores, rather than reporting only point-estimate means/medians.

Implements:
  - Wilcoxon signed-rank test (paired, non-parametric; appropriate here
    because per-SKU MAPE is not normally distributed and samples are
    paired -- same SKU, both methods, same holdout window)
  - Bootstrap confidence interval for the median of paired differences
  - Matched-pairs rank-biserial correlation as the effect size
    conventionally reported alongside a Wilcoxon test
"""
from __future__ import annotations

import random
from typing import Dict, List

from scipy import stats


def paired_comparison(a: List[float], b: List[float], n_bootstrap: int = 10000,
                       seed: int = 0, alpha: float = 0.05) -> Dict[str, object]:
    """Compare paired samples a vs b (e.g. Holt's MAPE vs naive-baseline
    MAPE, one pair per SKU). Returns Wilcoxon signed-rank statistic and
    p-value, a bootstrap CI for the median of (a - b), and matched-pairs
    rank-biserial correlation as effect size."""
    assert len(a) == len(b), "paired samples must be the same length"
    diffs = [x - y for x, y in zip(a, b)]

    # Wilcoxon signed-rank test (drops exact-zero differences per convention)
    try:
        if all(d == 0 for d in diffs):
            raise ValueError("all paired differences are zero")
        wilcoxon_stat, p_value = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        # all differences are zero, or too few non-zero pairs
        wilcoxon_stat, p_value = float("nan"), 1.0

    # Matched-pairs rank-biserial correlation effect size:
    # r = (W+ - W-) / (W+ + W-), computed from signed ranks of non-zero diffs
    nonzero = [d for d in diffs if d != 0]
    if nonzero:
        abs_ranks = stats.rankdata([abs(d) for d in nonzero])
        w_pos = sum(r for d, r in zip(nonzero, abs_ranks) if d > 0)
        w_neg = sum(r for d, r in zip(nonzero, abs_ranks) if d < 0)
        rank_biserial = (w_pos - w_neg) / (w_pos + w_neg) if (w_pos + w_neg) else 0.0
    else:
        rank_biserial = 0.0

    # Bootstrap CI for the median of the paired differences
    rng = random.Random(seed)
    n = len(diffs)
    boot_medians = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        sample.sort()
        mid = n // 2
        med = sample[mid] if n % 2 else (sample[mid - 1] + sample[mid]) / 2
        boot_medians.append(med)
    boot_medians.sort()
    lo_idx = int((alpha / 2) * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap) - 1

    return {
        "n_pairs": n,
        "median_diff_a_minus_b": round(sorted(diffs)[n // 2] if n % 2 else
                                        (sorted(diffs)[n // 2 - 1] + sorted(diffs)[n // 2]) / 2, 4),
        "bootstrap_ci_95": [round(boot_medians[lo_idx], 4), round(boot_medians[hi_idx], 4)],
        "wilcoxon_statistic": round(float(wilcoxon_stat), 4) if wilcoxon_stat == wilcoxon_stat else None,
        "wilcoxon_p_value": round(float(p_value), 4),
        "significant_at_0.05": bool(p_value < 0.05),
        "matched_pairs_rank_biserial_r": round(rank_biserial, 4),
        "n_a_better": sum(1 for d in diffs if d < 0),
        "n_b_better": sum(1 for d in diffs if d > 0),
        "n_tied": sum(1 for d in diffs if d == 0),
    }


def holm_bonferroni(pvalues: list, alpha: float = 0.05) -> list:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values
    in the ORIGINAL order of the input list."""
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [None] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (n - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def benjamini_hochberg(pvalues: list, alpha: float = 0.05) -> list:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values in
    the ORIGINAL order of the input list."""
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [None] * n
    running_min = 1.0
    for rank in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank]
        bh_val = p * n / (rank + 1)
        running_min = min(running_min, bh_val)
        adjusted[orig_idx] = min(1.0, running_min)
    return adjusted

