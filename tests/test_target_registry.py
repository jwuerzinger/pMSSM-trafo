"""Target-registry regression tests for the two AL target axes.

Run directly (this repo has no pytest harness):

    ./.pixi/envs/rocm/bin/python tests/test_target_registry.py

The point of these tests is asymmetric. For ``DMRD`` they pin the *historical*
behaviour: the relic-density path fed published numbers, so every registry
lookup added when the second target went in must return exactly what was
hardcoded before. For ``ExpR`` they pin the new contract: the boundary sits at
log(r) = 0, the ``-1.`` sentinel is dropped, and the excluded ``r > 1`` half is
kept rather than cut away.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmssm.config import TARGET_CONFIG, MODELGEN_STEP_DEFS
from pmssm.data import transform_y, inverse_transform_y, target_validity_mask


FAILURES = []


def check(cond, msg):
    if cond:
        print(f"  ok    {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def close(a, b, tol=0.0):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------- registry ---
def test_registry_complete():
    print("registry completeness")
    required = ("true_value", "threshold", "branch", "label", "valid_max",
                "hist_range", "gen_require_neutralino_lsp", "gen_steps",
                "has_mcmc_reference")
    for name, cfg in TARGET_CONFIG.items():
        missing = [k for k in required if k not in cfg]
        check(not missing, f"{name} has every required key (missing: {missing})")
        unknown = [s for s in cfg["gen_steps"] if s not in MODELGEN_STEP_DEFS]
        check(not unknown, f"{name} gen_steps are all known (unknown: {unknown})")
    check("ExpR" in TARGET_CONFIG, "ExpR is registered")
    check(TARGET_CONFIG["DMRD"]["has_mcmc_reference"] is True,
          "DMRD is the only target with an MCMC reference")
    check([n for n, c in TARGET_CONFIG.items() if c["has_mcmc_reference"]] == ["DMRD"],
          "no other target claims an MCMC reference")


def test_dmrd_values_unchanged():
    print("DMRD registry values match the historical hardcodes")
    c = TARGET_CONFIG["DMRD"]
    check(close(c["true_value"], 0.12), "true_value is 0.12")
    check(close(c["threshold"], 0.0), "threshold is 0.0")
    check(c["branch"] == "MO_Omega", "branch is MO_Omega")
    check(close(c["valid_max"], 1.0), "valid_max is 1.0 (the sub-dominant-DM cut)")
    check(tuple(c["hist_range"]) == (0.0, 1.0), "hist_range is (0, 1)")
    check(c["gen_require_neutralino_lsp"] is True,
          "generated data keeps the neutralino veto")
    check(tuple(c["gen_steps"]) == ("prep_input", "SPheno", "micromegas"),
          "gen_steps are the original three (no SModelS)")


def test_expr_values():
    print("ExpR registry values encode the exclusion boundary")
    c = TARGET_CONFIG["ExpR"]
    check(close(c["true_value"], 1.0), "true_value is 1.0 (r = 1 is the boundary)")
    check(close(c["threshold"], 0.0), "threshold is 0.0")
    check(c["branch"] == "SModelS_bestExpR_r_expected", "branch is the SModelS r-value")
    check(c["valid_max"] is None,
          "valid_max is None, so the excluded r > 1 half is NOT cut away")
    check(c["gen_require_neutralino_lsp"] is False,
          "no neutralino veto: a collider limit does not need a DM candidate")
    steps = tuple(c["gen_steps"])
    check("SModelS" in steps, "gen_steps include SModelS")
    check(steps[-1] == "SModelS",
          "SModelS is LAST (it appends xsecs to the SPheno output in place)")
    check("micromegas" in steps,
          "micromegas is kept, so generated points also carry MO_Omega")


# --------------------------------------------------------------- transform ---
def test_transforms():
    print("transform puts each boundary at 0 and round-trips")
    for name, boundary in (("DMRD", 0.12), ("ExpR", 1.0)):
        t = transform_y(torch.tensor([boundary]), target=name)
        check(close(t, 0.0, 1e-12), f"{name}: transform_y({boundary}) == 0")
        back = inverse_transform_y(torch.tensor([0.0]), target=name)
        check(close(back, boundary, 1e-6), f"{name}: inverse of 0 is {boundary}")
        y = torch.tensor([0.031, 0.4, 3.7])
        rt = inverse_transform_y(transform_y(y, target=name), target=name)
        check(torch.allclose(y, rt, rtol=1e-5), f"{name}: round-trips")
    # sign convention: above the boundary is positive, i.e. "excluded" for ExpR
    check(float(transform_y(torch.tensor([2.0]), target="ExpR")) > 0,
          "ExpR: r = 2 (excluded) maps above the threshold")
    check(float(transform_y(torch.tensor([0.5]), target="ExpR")) < 0,
          "ExpR: r = 0.5 (allowed) maps below the threshold")


# -------------------------------------------------------------------- mask ---
def test_masks():
    print("validity mask: DMRD reproduces the old expression exactly")
    # -1 is Run3ModelGen's "branch not filled" sentinel; 578 is a real r-value.
    Y = np.array([-1.0, 0.0, 0.05, 0.12, 0.999, 1.0, 2.5, 578.0, 0.3])
    mh = np.array([125., 125., 125., 125., 125., 125., 125., 125., -1.0])

    old = (Y > 0) & (Y < 1.0) & (mh != -1)
    new, desc = target_validity_mask(Y, mh, target="DMRD")
    check(bool((old == new).all()), "DMRD mask == (Y>0)&(Y<1)&(mh!=-1)")
    check(desc == "MO_Omega > 0 & < 1 & SP_m_h != -1",
          f"DMRD filter text is byte-identical to the archived log line ({desc!r})")

    m, d = target_validity_mask(Y, mh, target="ExpR")
    check(not m[0] and not m[1], "ExpR drops the -1 sentinel and exact 0")
    check(bool(m[6]) and bool(m[7]), "ExpR KEEPS r = 2.5 and r = 578 (excluded region)")
    check(not m[8], "ExpR still drops rows where SPheno failed (SP_m_h == -1)")
    check(d == "SModelS_bestExpR_r_expected > 0 & SP_m_h != -1",
          f"ExpR filter text has no upper cut ({d!r})")


def test_hit_band():
    print("relative hit band around each boundary")
    # the band metric used throughout the analysis scripts
    for name, tv, inside, outside in (
        ("DMRD", 0.12, [0.115, 0.126], [0.10, 0.14]),
        ("ExpR", 1.0, [0.95, 1.05], [0.8, 1.2]),
    ):
        tol = 0.1
        ins = np.array(inside)
        outs = np.array(outside)
        check(bool((np.abs(ins - tv) / tv < tol).all()),
              f"{name}: {inside} are inside the 10% band")
        check(bool((np.abs(outs - tv) / tv >= tol).all()),
              f"{name}: {outside} are outside the 10% band")


# ------------------------------------------------- generated-data loading ---
def _write_ntuple(path, n, target_branch, target_vals, mh_vals, lsp_vals=None,
                  omit_target=False):
    """Minimal 'susy' tree with the branches load_generated_data reads."""
    import uproot
    from pmssm.config import PARAM_ORDER
    data = {b: np.linspace(1.0, 100.0, n) for b in PARAM_ORDER}
    if not omit_target:
        data[target_branch] = np.asarray(target_vals, dtype=np.float64)
    data["SP_m_h"] = np.asarray(mh_vals, dtype=np.float64)
    if lsp_vals is not None:
        data["SP_LSP_type"] = np.asarray(lsp_vals, dtype=np.float64)
    with uproot.recreate(path) as f:
        f["susy"] = data


def test_load_generated_data():
    print("load_generated_data: target-driven branch, mask and veto")
    import logging
    from pmssm.model_generation import load_generated_data
    logging.basicConfig(level=logging.CRITICAL)
    log = logging.getLogger("silent")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # ExpR: sentinel row, an allowed row, two excluded rows, one SPheno fail,
        # and one non-neutralino LSP that must SURVIVE (no veto for this target).
        p = td / "expr.root"
        _write_ntuple(p, 6, "SModelS_bestExpR_r_expected",
                      [-1.0, 0.3, 2.5, 40.0, 0.7, 1.5],
                      [125., 125., 125., 125., -1.0, 125.],
                      [1, 1, 2, 3, 1, 1000014])
        X, Y = load_generated_data(p, log, target="ExpR")
        check(X is not None and len(X) == 4,
              f"ExpR keeps 4 of 6 rows (got {0 if X is None else len(X)})")
        check(Y is not None and int((Y > 1).sum()) == 3,
              "ExpR keeps all three excluded (r > 1) rows")
        check(Y is not None and float(Y.min()) > 0, "no sentinel survived")

        # DMRD on the same shape: veto MUST drop the sneutrino row.
        p2 = td / "dmrd.root"
        _write_ntuple(p2, 5, "MO_Omega",
                      [-1.0, 0.11, 0.5, 0.11, 2.0],
                      [125., 125., 125., 125., 125.],
                      [1, 1, 2, 1000014, 1])
        # 5 rows in, 3 dropped: the -1 sentinel, the Omega > 1 row, and the
        # sneutrino-LSP row that the veto removes. 2 survive.
        Xd, Yd = load_generated_data(p2, log, target="DMRD")
        check(Xd is not None and len(Xd) == 2,
              f"DMRD keeps 2 of 5: drops sentinel, Omega>1, and the sneutrino "
              f"(got {0 if Xd is None else len(Xd)})")
        check(Yd is not None and float(Yd.max()) < 1.0,
              "DMRD still applies the sub-dominant-DM upper cut")

        # A batch where no model produced the target branch at all: the ntupler
        # only creates a branch when >=1 model filled it. Must return empties,
        # not raise.
        p3 = td / "nobranch.root"
        _write_ntuple(p3, 3, "SModelS_bestExpR_r_expected", None,
                      [125., 125., 125.], [1, 1, 1], omit_target=True)
        Xn, Yn = load_generated_data(p3, log, target="ExpR")
        check(Xn is None and Yn is None,
              "absent target branch returns empty rather than raising")

        # SP_m_h absent: SPheno produced nothing usable.
        p4 = td / "nomh.root"
        import uproot
        from pmssm.config import PARAM_ORDER
        with uproot.recreate(p4) as f:
            f["susy"] = {b: np.linspace(1.0, 9.0, 3) for b in PARAM_ORDER}
        Xm, Ym = load_generated_data(p4, log, target="ExpR")
        check(Xm is None and Ym is None, "absent SP_m_h returns empty")


def main():
    for fn in (test_registry_complete, test_dmrd_values_unchanged, test_expr_values,
               test_transforms, test_masks, test_hit_band, test_load_generated_data):
        fn()
        print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
