from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import pandas as pd


@dataclass
class GPBoostDebugConfig:
    # thresholds
    singleton_warn: float = 0.25       # warn if >25% singleton groups
    smallgroup_warn: float = 0.50      # warn if >50% groups have size <=2
    nearconst_within_group_eps: float = 1e-12
    nearperfect_corr: float = 0.999999
    rank_tol: float = 1e-10
    max_pairs: int = 50
    # limit printed pairs for correlations / nesting


def _safe_corrcoef(X: np.ndarray) -> np.ndarray:
    """Correlation for columns; robust to constant columns."""
    X = np.asarray(X, dtype=float)
    # standardize with protection
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (X - mu) / sd
    return (Z.T @ Z) / (Z.shape[0] - 1)


def _matrix_rank(X: np.ndarray, tol: float) -> int:
    if X.size == 0:
        return 0
    s = np.linalg.svd(X, compute_uv=False)
    return int(np.sum(s > tol))


def _functional_dependency(
        df: pd.DataFrame, A: str, B: str
        ) -> tuple[bool, float]:
    """
    Returns (A -> B holds strictly?, proportion of A-levels
    that map to exactly one B).
    """
    nunique = df.groupby(A, dropna=False)[B].nunique(dropna=False)
    prop_one = float((nunique == 1).mean())
    return bool((nunique == 1).all()), prop_one


def debug_gpboost_structure(
    df: pd.DataFrame,
    group_vars: Sequence[str] | None,
    Z: np.ndarray | None,
    pointers: Sequence[int] | None,
    random_effects: Mapping[str, Iterable[str | int]] | None = None,
    config: GPBoostDebugConfig = GPBoostDebugConfig(),
    print_report: bool = True,
) -> dict[str, Any]:
    """
    General diagnostics for GPBoost grouped random effects + random slopes.

    Parameters
    ----------
    df : DataFrame
        Data used for fitting (after dropna / filtering).
    group_vars : list[str] | None
        Grouping variables (columns in df) in the same order as passed
        to group_data.
    Z : ndarray | None
        group_rand_coef_data passed to GPBoost (n x p).
    pointers : list[int] | None
        ind_effect_group_rand_coef (length p). Values are 1-based indices
        referring to group_vars.
    random_effects : mapping | None
        Optional: mapping grouping var -> iterable of covariate names / 0 / 1.
        Used only for nicer labeling; not required.
    config : GPBoostDebugConfig
        thresholds and limits.

    Returns
    -------
    report : dict
        Machine-readable summary of issues and statistics.
    """
    report: dict[str, Any] = {
        "group_vars": list(group_vars) if group_vars is not None else None,
        "n": int(len(df)),
        "group_stats": {},
        "nesting": {"strict": [], "near": []},
        "Z": None,
        "issues": [],
    }

    # -----------------------------
    # Group-level diagnostics
    # -----------------------------
    if group_vars is None or len(group_vars) == 0:
        report["issues"].append("No group_vars provided.")
    else:
        for gv in group_vars:
            vc = df[gv].value_counts(dropna=False)
            stats = {
                "n_levels": int(vc.shape[0]),
                "min": int(vc.min()),
                "median": float(vc.median()),
                "mean": float(vc.mean()),
                "p_singleton": float((vc == 1).mean()),
                "p_le_2": float((vc <= 2).mean()),
            }
            report["group_stats"][gv] = stats

            if stats["n_levels"] < 2:
                report["issues"].append(
                    f"Grouping variable '{gv}' has <2 levels -> "
                    "RE variance not identifiable.")
            if stats["p_singleton"] > config.singleton_warn:
                report["issues"].append(
                    f"High singleton rate for '{gv}' "
                    f"(p={stats['p_singleton']:.3f}). Random slopes at this "
                    "level often unstable."
                )
            if stats["p_le_2"] > config.smallgroup_warn:
                report["issues"].append(
                    f"Many tiny groups for '{gv}' "
                    f"(p(size<=2)={stats['p_le_2']:.3f})."
                )

        # -----------------------------
        # Nesting / functional dependencies between grouping vars
        # -----------------------------
        # Check A -> B for all ordered pairs
        strict, near = [], []
        for i, A in enumerate(group_vars):
            for j, B in enumerate(group_vars):
                if i == j:
                    continue
                holds, prop_one = _functional_dependency(df, A, B)
                entry = {
                    "A": A,
                    "B": B,
                    "A_to_B": holds,
                    "prop_A_levels_map_to_one_B": prop_one}
                if holds:
                    strict.append(entry)
                elif prop_one > 0.95:
                    near.append(entry)
        # sort by strength
        strict = sorted(
            strict, key=lambda e: e["prop_A_levels_map_to_one_B"],
            reverse=True)
        near = sorted(
            near, key=lambda e: e["prop_A_levels_map_to_one_B"],
            reverse=True)
        report["nesting"]["strict"] = strict[: config.max_pairs]
        report["nesting"]["near"] = near[: config.max_pairs]

        if len(strict) > 0:
            report["issues"].append(
                "Strict functional dependencies "
                "found among grouping variables (nesting). "
                "Redundant intercept stacks can push some variances to ~0."
            )

    # -----------------------------
    # Random slope diagnostics (Z + pointers)
    # -----------------------------
    if Z is None:
        report["Z"] = None
    else:
        Z = np.asarray(Z, dtype=float)
        n, p = Z.shape
        ptr: None | np.ndarray
        if pointers is None:
            report["issues"].append("Z provided but pointers is None.")
            ptr = None
        else:
            ptr = np.asarray(list(pointers), dtype=int)
            if ptr.shape[0] != p:
                report["issues"].append(
                    f"Length mismatch: pointers has {ptr.shape[0]} but Z has {p} columns.")
            # pointers are 1-based indices into group_vars
            if (ptr < 1).any():
                report["issues"].append(
                    "pointers contains values <1 (expected 1-based indices).")
            if group_vars is not None and (ptr > len(group_vars)).any():
                report["issues"].append(
                    "pointers contains values > len(group_vars).")

        Z_summary: dict[str, Any] = {
            "shape": (int(n), int(p)),
            "col_std": Z.std(axis=0).tolist(),
            "rank": _matrix_rank(Z, config.rank_tol),
        }

        # if random_effects provided, try to label Z columns
        # but we can't reconstruct exact order reliably; keep it optional
        Z_summary["pointers"] = ptr.tolist() if ptr is not None else None

        # Within-group variation for slopes: for each pointer g and each Z col in that group,
        # compute proportion of groups with within-group std ~ 0.
        within_var = []
        if group_vars is not None and ptr is not None:
            for g_idx in sorted(set(ptr.tolist())):
                gv = group_vars[g_idx - 1]
                cols = np.where(ptr == g_idx)[0]
                if cols.size == 0:
                    continue
                for c in cols:
                    # within-group std across gv
                    s = df.assign(_z=Z[:, c]).groupby(
                        gv, dropna=False)["_z"].std()
                    p0 = float(
                        (s.fillna(
                            0) < config.nearconst_within_group_eps).mean())
                    within_var.append({
                        "group": gv,
                        "z_col": int(c),
                        "p_within_std_near0": p0})
                    if p0 > 0.5:
                        report["issues"].append(
                            f"Random slope column {c} has little within-group variation for '{gv}' "
                            f"(p(within-std≈0)={p0:.3f})."
                        )
        Z_summary["within_group_variation"] = within_var[: config.max_pairs]

        # Correlations / duplicates within each pointer group (most relevant)
        dup_pairs = []
        corr_pairs = []
        if pointers is not None:
            for g_idx in sorted(set(ptr.tolist())):
                cols = np.where(ptr == g_idx)[0]
                if cols.size < 2:
                    continue
                Zg = Z[:, cols]
                C = _safe_corrcoef(Zg)
                # scan upper triangle
                for a in range(len(cols)):
                    for b in range(a + 1, len(cols)):
                        r = float(C[a, b])
                        if np.isfinite(r) and abs(r) >= config.nearperfect_corr:
                            corr_pairs.append({
                                "pointer_group": int(g_idx),
                                "cols": (int(cols[a]), int(cols[b])),
                                "corr": r
                            })
                        # exact duplicate check (tighter)
                        if np.allclose(Z[:, cols[a]], Z[:, cols[b]]):
                            dup_pairs.append({
                                "pointer_group": int(g_idx),
                                "cols": (int(cols[a]), int(cols[b]))
                            })

                # rank per pointer group
                rg = _matrix_rank(Zg, config.rank_tol)
                if rg < Zg.shape[1]:
                    report["issues"].append(
                        f"Z columns for pointer-group {g_idx} are rank-deficient: rank={rg} < {Zg.shape[1]}."
                    )

        Z_summary["nearperfect_corr_pairs_within_pointer"] = corr_pairs[: config.max_pairs]
        Z_summary["exact_duplicate_pairs_within_pointer"] = dup_pairs[: config.max_pairs]

        report["Z"] = Z_summary

    # -----------------------------
    # Pretty print
    # -----------------------------
    if print_report:
        print("=" * 80)
        print("GPBoost structure diagnostics")
        print(f"n = {report['n']}")
        print("group_vars:", report["group_vars"])
        print("-" * 80)

        if report["group_vars"] is not None:
            print("Group size stats:")
            for gv, st in report["group_stats"].items():
                print(
                    f"  {gv:>10s}: levels={st['n_levels']:>5d} "
                    f"min={st['min']:>3d} med={st['median']:.1f} "
                    f"p1={st['p_singleton']:.3f} p<=2={st['p_le_2']:.3f}"
                )
            print("-" * 80)

            if report["nesting"]["strict"]:
                print("Strict nesting / functional dependencies (A -> B):")
                for e in report["nesting"]["strict"][: config.max_pairs]:
                    print(f"  {e['A']} -> {e['B']}  (prop={e['prop_A_levels_map_to_one_B']:.3f})")
                print("-" * 80)
            if report["nesting"]["near"]:
                print("Near-nesting (A -> B holds for most A-levels):")
                for e in report["nesting"]["near"][: config.max_pairs]:
                    print(f"  {e['A']} -> {e['B']}  (prop={e['prop_A_levels_map_to_one_B']:.3f})")
                print("-" * 80)

        if report["Z"] is None:
            print("Z: None (no random slopes).")
        else:
            Zs = report["Z"]
            print(f"Z shape: {Zs['shape']}, rank: {Zs['rank']}")
            # show only first few stds
            stds = np.array(Zs["col_std"], dtype=float)
            print("Z col std (min/median/max):",
                float(np.min(stds)), float(np.median(stds)), float(np.max(stds)))
            if Zs["within_group_variation"]:
                print("Within-group std≈0 proportions (first few):")
                for e in Zs["within_group_variation"][: min(10, len(Zs["within_group_variation"]))]:
                    print(f"  group={e['group']:<10s} col={e['z_col']:>2d} p(std≈0)={e['p_within_std_near0']:.3f}")
            if Zs["exact_duplicate_pairs_within_pointer"]:
                print("Exact duplicate Z column pairs (within same pointer group):")
                for e in Zs["exact_duplicate_pairs_within_pointer"]:
                    print(f"  pointer={e['pointer_group']} cols={e['cols']}")
            if Zs["nearperfect_corr_pairs_within_pointer"]:
                print("Near-perfectly correlated Z column pairs (within same pointer group):")
                for e in Zs["nearperfect_corr_pairs_within_pointer"]:
                    print(f"  pointer={e['pointer_group']} cols={e['cols']} corr={e['corr']:.6f}")

        if report["issues"]:
            print("-" * 80)
            print("Potential issues:")
            for msg in report["issues"]:
                print("  -", msg)
        print("=" * 80)

    return report