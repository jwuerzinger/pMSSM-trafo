#!/usr/bin/env python3
"""Two tests of whether AL's narrower analysis mix costs anything.

Test 1  Is each arm's distribution a SHARPENING of the pool's (same ordering,
        more concentrated) or a REORDERING? Sharpening is consistent with the
        loop following sensitivity: it spends more where the strongest analyses
        already set the limit. Reordering would mean it elevates an analysis the
        pool says is weak, which is a coverage concern.

Test 2  Shares are the wrong denominator. An arm delivering YIELD_MULT times the
        pool's in-band points per attempt gives, per analysis,

            points_AL / points_pool = YIELD_MULT * share_AL / share_pool

        so it delivers MORE points for analysis A whenever share_AL(A) exceeds
        share_pool(A) / YIELD_MULT. If that holds for every analysis, the
        narrower mix starves nothing and is purely a bonus on the leader.

Counts are printed alongside every ratio: the tail analyses have few points and
their ratios are correspondingly soft, which matters because Test 2 turns on the
tail rather than on the leader.
"""
import json
import math
import sys
from pathlib import Path

# Per-attempt in-band yield relative to a random draw, from Table tab:expr-yield.
YIELD_MULT = {
    "deep_gp_expr": 3.30,
    "exact_gp_expr": 2.12,
    "transformer_expr": 1.17,
    "dnn_expr": 1.00,
    "tabpfn_expr": 0.67,
}
DISPLAY = {"deep_gp_expr": "Deep GP", "exact_gp_expr": "Exact GP",
           "transformer_expr": "Transformer", "dnn_expr": "DNN",
           "tabpfn_expr": "TabPFN"}
MIN_N = 10          # below this a share is too soft to judge a ratio on


def load(p):
    d = json.loads(Path(p).read_text())
    return d if "by_analysis" not in d else {Path(p).stem: d}


def shares(rec):
    n = max(rec["n_inband"], 1)
    return {a: c / n for a, c in rec["by_analysis"].items()}, rec["by_analysis"], n


def main(arms_json, pool_json):
    arms = load(arms_json)
    pool_all = load(pool_json)
    pool = next(iter(pool_all.values()))
    ps, pc, pn = shares(pool)
    print(f"pool: {pn} in-band points, {len(ps)} analyses, "
          f"effective {pool['effective_analyses']:.2f}\n")

    order_pool = [a for a, _ in sorted(ps.items(), key=lambda kv: -kv[1])]

    for key, rec in arms.items():
        if key not in YIELD_MULT:
            continue
        s, c, n = shares(rec)
        mult = YIELD_MULT[key]
        print("=" * 78)
        print(f"{DISPLAY[key]}   n={n}   effective {rec['effective_analyses']:.2f}   "
              f"yield x{mult:.2f} vs random")

        # ---- Test 1: ordering ----
        common = [a for a in order_pool if a in s and (pc[a] >= MIN_N or c.get(a, 0) >= MIN_N)]
        arm_rank = {a: i for i, a in enumerate(
            sorted(common, key=lambda x: -s.get(x, 0)))}
        pool_rank = {a: i for i, a in enumerate(common)}
        # Spearman on the analyses that are well populated in either sample
        if len(common) > 2:
            d2 = sum((arm_rank[a] - pool_rank[a]) ** 2 for a in common)
            k = len(common)
            rho = 1 - 6 * d2 / (k * (k * k - 1))
            print(f"  Test 1  rank correlation with the pool over {k} well-populated "
                  f"analyses: rho = {rho:+.3f}")
            top_pool, top_arm = order_pool[0], max(s, key=s.get)
            print(f"          pool's strongest: {top_pool}")
            print(f"          arm's strongest:  {top_arm}"
                  + ("   (same)" if top_arm == top_pool else "   (REORDERED)"))

        # ---- Test 2: absolute points per attempt ----
        print(f"  Test 2  points per attempt vs the pool, per analysis "
              f"(>1 means AL delivers more)")
        rows = []
        for a in sorted(set(s) | set(ps), key=lambda x: -(ps.get(x, 0))):
            sa, sp = s.get(a, 0.0), ps.get(a, 0.0)
            if sp == 0 and sa == 0:
                continue
            ratio = (mult * sa / sp) if sp > 0 else math.inf
            rows.append((a, sp, sa, ratio, pc.get(a, 0), c.get(a, 0)))
        starved = [r for r in rows if r[3] < 1.0 and r[4] >= MIN_N]
        for a, sp, sa, ratio, npool, narm in rows[:12]:
            soft = "" if (npool >= MIN_N and narm >= MIN_N) else "  (soft: few points)"
            flag = "  <-- fewer" if ratio < 1.0 and npool >= MIN_N else ""
            print(f"    {a:32s} pool {100*sp:5.1f}%  arm {100*sa:5.1f}%  "
                  f"x{ratio:5.2f}  (n {npool:4d}/{narm:4d}){flag}{soft}")
        print(f"  -> analyses where AL delivers FEWER points (pool n>={MIN_N}): "
              f"{len(starved)} of {len([r for r in rows if r[4] >= MIN_N])}")
        if starved:
            for a, sp, sa, ratio, npool, narm in starved:
                print(f"       {a:32s} x{ratio:.2f}")
        print()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
