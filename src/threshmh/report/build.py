"""Regenerate every table and figure from the raw result CSVs.

Nothing in ``paper/`` is written by hand.  ``make report`` rebuilds all of it,
so a number in the README and the number in the results file cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..optimizers import LABELS  # noqa: E402
from ..stats.compare import cliffs_delta, friedman_ranks, magnitude, posthoc_holm  # noqa: E402

ALGO_ORDER = ["random", "de", "pso", "ga"]
PALETTE = {"random": "#8c8c8c", "de": "#1f77b4", "pso": "#d62728", "ga": "#2ca02c"}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"[report] wrote {path}")


# -- tables ------------------------------------------------------------------


def table_hit_rate(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Fraction of runs that reached the proven optimum, by algorithm and C."""
    piv = (
        df.groupby(["algorithm", "n_classes"])["hit_optimum"]
        .mean()
        .unstack("n_classes")
        .reindex(ALGO_ORDER)
    )
    _write(outdir / "tables" / "hit_rate.md", _md(piv, "Hit rate (fraction of runs reaching the exact optimum)", pct=True))
    return piv


def table_rel_gap(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Median relative gap to the optimum, with IQR, by algorithm and C."""
    g = df.groupby(["algorithm", "n_classes"])["rel_gap"]
    med = g.median().unstack("n_classes").reindex(ALGO_ORDER)
    q1 = g.quantile(0.25).unstack("n_classes").reindex(ALGO_ORDER)
    q3 = g.quantile(0.75).unstack("n_classes").reindex(ALGO_ORDER)

    lines = ["| Algorithm | " + " | ".join(f"C={c}" for c in med.columns) + " |",
             "|---" * (len(med.columns) + 1) + "|"]
    for a in med.index:
        cells = [f"{med.loc[a, c]:.2e} [{q1.loc[a, c]:.0e}, {q3.loc[a, c]:.0e}]" for c in med.columns]
        lines.append(f"| {LABELS[a]} | " + " | ".join(cells) + " |")
    _write(outdir / "tables" / "rel_gap.md",
           "Median relative gap to the proven optimum [IQR], over all images and criteria.\n\n"
           + "\n".join(lines) + "\n")
    return med


def table_statistics(df: pd.DataFrame, outdir: Path) -> str:
    """Friedman ranks + Holm-corrected pairwise Wilcoxon, across images.

    Blocks are (image, criterion, class count, repair); the value per block is
    the median relative gap over the 25 runs.  Lower is better.
    """
    # Block value is the MEAN relative gap over the 25 runs, not the median.
    # DE, PSO and GA reach the exact optimum in more than half their runs, so
    # their median gap is identically zero in most cells and a test built on it
    # would be comparing ties. The mean retains the misses, which is the only
    # thing that distinguishes these algorithms. The descriptive table still
    # reports median + IQR, per the experimental contract.
    blocks = (
        df.groupby(["image", "criterion", "n_classes", "repair", "algorithm"])["rel_gap"]
        .mean()
        .unstack("algorithm")[ALGO_ORDER]
        .dropna()
    )
    stat, p, ranks = friedman_ranks(blocks, lower_is_better=True)
    post = posthoc_holm(blocks)

    out = [
        "## Across-suite comparison",
        "",
        f"Friedman chi-square = {stat:.2f}, p = {p:.3e}, over {len(blocks)} blocks",
        "(block = image x criterion x class count x repair; value = **mean** relative gap over",
        "25 runs -- the median is identically zero for DE/PSO/GA in most cells, because they",
        "reach the exact optimum in over half their runs, so a test on medians would compare ties).",
        "",
        "### Mean ranks (1 = best)",
        "",
        "| Algorithm | Mean rank |",
        "|---|---|",
    ]
    out += [f"| {LABELS[a]} | {r:.3f} |" for a, r in ranks.items()]
    out += ["", "### Pairwise Wilcoxon signed-rank, Holm-corrected", "",
            "| | " + " | ".join(ALGO_ORDER) + " |", "|---" * (len(ALGO_ORDER) + 1) + "|"]
    for a in ALGO_ORDER:
        cells = ["--" if a == b else f"{post.loc[a, b]:.2e}" for b in ALGO_ORDER]
        out.append(f"| **{a}** | " + " | ".join(cells) + " |")

    # Effect sizes come from the RAW RUNS, not the block aggregates. Different
    # question, different data -- see stats/compare.py.
    out += ["", "### Cliff's delta vs. random search (raw runs, pooled)", "",
            "| Algorithm | delta | Magnitude |", "|---|---|---|"]
    rnd = df.loc[df.algorithm == "random", "rel_gap"].to_numpy()
    for a in ALGO_ORDER:
        if a == "random":
            continue
        d = cliffs_delta(df.loc[df.algorithm == a, "rel_gap"].to_numpy(), rnd)
        out.append(f"| {LABELS[a]} | {d:+.3f} | {magnitude(d)} |")
    out.append("")
    out.append("Negative delta means a *smaller* gap than random search, i.e. better.")

    text = "\n".join(out) + "\n"
    _write(outdir / "tables" / "statistics.md", text)
    return text


def table_repair(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Does the continuous-to-discrete rule matter?

    Almost no paper in this literature states which discretisation it used, so
    the honest thing is to measure whether the choice is consequential at all.
    Reported as hit rate and mean gap; the median is zero nearly everywhere and
    discriminates nothing.
    """
    hit = df.groupby(["repair", "n_classes"])["hit_optimum"].mean().unstack()
    gap = df.groupby(["repair", "n_classes"])["rel_gap"].mean().unstack()

    lines = ["Two ways of turning a continuous search vector into sorted integer thresholds:",
             "",
             "- **distinct** -- duplicates pushed apart, so all C classes are always used",
             "- **collapse** -- duplicates permitted, letting the search use fewer than C classes",
             "",
             "### Hit rate (runs reaching the exact optimum)",
             "",
             "| Repair | " + " | ".join(f"C={c}" for c in hit.columns) + " |",
             "|---" * (len(hit.columns) + 1) + "|"]
    for r in hit.index:
        lines.append(f"| {r} | " + " | ".join(f"{hit.loc[r, c]:.1%}" for c in hit.columns) + " |")

    lines += ["", "### Mean relative gap (lower is better)", "",
              "| Repair | " + " | ".join(f"C={c}" for c in gap.columns) + " |",
              "|---" * (len(gap.columns) + 1) + "|"]
    for r in gap.index:
        lines.append(f"| {r} | " + " | ".join(f"{gap.loc[r, c]:.2e}" for c in gap.columns) + " |")

    best = hit.mean(axis=1).idxmax()
    spread = float(abs(hit.loc["distinct"] - hit.loc["collapse"]).max())
    lines += ["",
              f"Overall the **{best}** strategy is marginally better, and the largest difference "
              f"at any class count is {spread:.1%}.",
              "",
              "So the choice is close to immaterial *for these algorithms on this problem* -- "
              "which is itself worth reporting, given that papers routinely leave it unstated "
              "and a reader cannot tell whether it mattered."]
    _write(outdir / "tables" / "repair.md", "\n".join(lines) + "\n")
    return hit


def table_search_space(df: pd.DataFrame, outdir: Path, n_bins: int = 256) -> pd.DataFrame:
    """Size of the search space against the budget spent exploring it.

    The feasible set is every sorted vector of C-1 distinct thresholds drawn
    from L-1 candidate positions, i.e. ``binom(L-1, C-1)``.  At small C that
    number is smaller than the evaluation budget, which means the budget could
    have enumerated the entire space -- and the dynamic program does better
    still, because it never enumerates at all.
    """
    from math import comb

    rows = []
    for C in sorted(df.n_classes.unique()):
        space = comb(n_bins - 1, C - 1)
        budget = int(df.loc[df.n_classes == C, "budget"].iloc[0])
        rows.append({
            "n_classes": C,
            "search_space": space,
            "budget": budget,
            "budget_over_space": budget / space,
            "exhaustive_cheaper": budget >= space,
        })
    tab = pd.DataFrame(rows)

    lines = [
        "Feasible set = binom(255, C-1): every sorted vector of C-1 distinct thresholds.",
        "",
        "| Classes | Search space | Budget (FEs) | Budget / space | Budget could enumerate it all |",
        "|---|---|---|---|---|",
    ]
    for _, r in tab.iterrows():
        lines.append(
            f"| {int(r.n_classes)} | {int(r.search_space):,} | {int(r.budget):,} | "
            f"{r.budget_over_space:.2e} | {'**yes**' if r.exhaustive_cheaper else 'no'} |"
        )
    lines += ["",
              "The dynamic program does not search this space at all: it is "
              "`O(C x 256^2)` regardless of C, which is why its cost is flat."]
    _write(outdir / "tables" / "search_space.md", "\n".join(lines) + "\n")
    return tab


def table_scaling(scaling: pd.DataFrame, outdir: Path) -> None:
    """Cost of the two exact methods.

    CPU time is the primary figure: these measurements were taken on a machine
    that was not exclusively idle, and ``time.process_time`` counts only cycles
    charged to the process. Wall-clock is shown alongside it.
    """
    lines = [
        "Both methods return the **same proven optimum** -- they differ only in how they find it.",
        "",
        "CPU time (`time.process_time`) is primary; the machine was not exclusively idle and CPU",
        "time is largely insensitive to unrelated load. Wall-clock is reported alongside it.",
        "",
        "| Classes | DP CPU (ms) | DP wall (ms) | skimage CPU (ms) | skimage wall (ms) | Speed-up (CPU) | Optima agree |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in scaling.iterrows():
        finite = np.isfinite(r.skimage_ms)
        sk_c = f"{r.skimage_ms:,.1f}" if finite else "not run"
        sk_w = f"{r.skimage_wall_ms:,.1f}" if finite else "not run"
        sp = f"{r.speedup:,.0f}x" if np.isfinite(r.speedup) else "--"
        ag = {True: "yes", False: "**NO**"}.get(r.values_agree, "not run")
        lines.append(
            f"| {int(r.n_classes)} | {r.dp_ms:.2f} | {r.dp_wall_ms:.2f} | {sk_c} | {sk_w} | {sp} | {ag} |"
        )

    dp = scaling.dp_ms
    lines += ["",
              f"The DP is flat in C: {dp.min():.1f}-{dp.max():.1f} ms CPU across "
              f"C={int(scaling.n_classes.min())}..{int(scaling.n_classes.max())}, consistent with "
              "its `O(C x L^2)` bound.",
              "`threshold_multiotsu` searches combinations, with documented complexity",
              "`O(C x h^(C-1) / (C-1)!)`, so its cost rises steeply with C.",
              "",
              "**This is not a defect in scikit-image** -- its complexity is documented, and for the",
              "two- and three-class cases it is used for overwhelmingly, it is perfectly adequate.",
              "It matters here because that combinatorial cost is what makes exact multilevel",
              "thresholding appear expensive, which is the premise the metaheuristic literature rests on.",
              ""]
    _write(outdir / "tables" / "scaling.md", "\n".join(lines) + "\n")


def table_metric_disagreement(metrics: pd.DataFrame, outdir: Path) -> str:
    """Do the conventional metrics pick the same criterion that Dice does?

    Reported **with the Dice margin**, because a disagreement between two
    segmentations that differ by 0.002 Dice is a different claim from a
    disagreement between two that differ by 0.4.  Omitting the margin would
    make an arbitrary tie-break look like a substantive contradiction.
    """
    ph = metrics[metrics.kind == "phantom"]
    rows = []
    for (img, C), grp in ph.groupby(["image", "n_classes"]):
        if grp.criterion.nunique() < 2:
            continue
        dice_sorted = grp.sort_values("dice", ascending=False)
        best_dice = dice_sorted.criterion.iloc[0]
        margin = float(dice_sorted.dice.iloc[0] - dice_sorted.dice.iloc[1])
        rows.append({
            "image": img, "n_classes": int(C), "dice_winner": best_dice,
            "dice_margin": margin,
            "psnr_winner": grp.loc[grp.psnr.idxmax(), "criterion"],
            "ssim_winner": grp.loc[grp.ssim.idxmax(), "criterion"],
            "fsim_winner": grp.loc[grp.fsim.idxmax(), "criterion"],
        })
    tab = pd.DataFrame(rows)
    for m in ("psnr", "ssim", "fsim"):
        tab[f"{m}_disagrees"] = tab[f"{m}_winner"] != tab["dice_winner"]

    lines = [
        "Both criteria are evaluated at their **proven optimal** thresholds, so the",
        "optimizer plays no part in any disagreement -- it is the metric, not the search.",
        "",
        "| Image | C | Dice picks | Dice margin | PSNR picks | SSIM picks | FSIM picks |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in tab.iterrows():
        def mark(metric):
            w = r[f"{metric}_winner"]
            return f"**{w}**" if r[f"{metric}_disagrees"] else w
        lines.append(
            f"| {r.image} | {r.n_classes} | {r.dice_winner} | {r.dice_margin:.4f} | "
            f"{mark('psnr')} | {mark('ssim')} | {mark('fsim')} |"
        )

    n = len(tab)
    lines += ["", "Bold marks a metric selecting a different criterion than Dice.", "",
              "| Metric | Disagrees with Dice |", "|---|---|"]
    for m in ("psnr", "ssim", "fsim"):
        lines.append(f"| {m.upper()} | {int(tab[f'{m}_disagrees'].sum())} of {n} |")

    big = tab[tab.dice_margin > 0.05]
    small = tab[tab.dice_margin <= 0.05]
    lines += ["", "### Reading this honestly", ""]
    if len(big):
        agree_big = int((~big[["psnr_disagrees", "ssim_disagrees", "fsim_disagrees"]].any(axis=1)).sum())
        lines.append(
            f"- Where the two segmentations genuinely differ (Dice margin > 0.05, {len(big)} case"
            f"{'s' if len(big) != 1 else ''}), every metric agrees with Dice in {agree_big} of them."
        )
    if len(small):
        lines.append(
            f"- Where they are near-identical (Dice margin <= 0.05, {len(small)} cases), the "
            "metrics disagree freely -- they are tie-breaking between segmentations that are "
            "practically the same."
        )
    lines += ["",
              "So the conventional metrics are not simply wrong. They track segmentation quality "
              "when the difference is large, and become arbitrary when it is small -- which is "
              "exactly the regime where papers report them to separate competing methods.", ""]

    text = "\n".join(lines) + "\n"
    _write(outdir / "tables" / "metric_disagreement.md", text)
    return text


def _md(piv: pd.DataFrame, caption: str, pct: bool = False) -> str:
    head = "| Algorithm | " + " | ".join(f"C={c}" for c in piv.columns) + " |"
    sep = "|---" * (len(piv.columns) + 1) + "|"
    rows = []
    for a in piv.index:
        fmt = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.3e}")
        label = LABELS.get(a, a)
        rows.append(f"| {label} | " + " | ".join(fmt(piv.loc[a, c]) for c in piv.columns) + " |")
    return f"{caption}\n\n" + "\n".join([head, sep, *rows]) + "\n"


# -- figures -----------------------------------------------------------------


def figure_scaling(scaling: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.semilogy(scaling.n_classes, scaling.dp_ms, "o-", color="#1f77b4", lw=2,
                label="Dynamic programming (this work), $O(CL^2)$")
    sk = scaling.dropna(subset=["skimage_ms"])
    ax.semilogy(sk.n_classes, sk.skimage_ms, "s-", color="#d62728", lw=2,
                label="skimage.threshold_multiotsu, $O(Ch^{C-1}/(C-1)!)$")
    for _, r in sk.iterrows():
        if np.isfinite(r.speedup) and r.speedup > 10:
            ax.annotate(f"{r.speedup:,.0f}x", (r.n_classes, r.skimage_ms),
                        textcoords="offset points", xytext=(-4, 7), fontsize=7.5,
                        color="#d62728", ha="right")
    ax.set_xlabel("Number of classes $C$")
    ax.set_ylabel("CPU time to exact optimum (ms, log scale)")
    ax.set_title("Same proven optimum, very different cost")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "scaling.png", dpi=200)
    plt.close(fig)
    print("[report] wrote scaling.png")


def figure_gap_by_class(df: pd.DataFrame, outdir: Path) -> None:
    counts = sorted(df.n_classes.unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    width = 0.8 / len(ALGO_ORDER)
    for i, a in enumerate(ALGO_ORDER):
        sub = df[df.algorithm == a]
        pos = np.arange(len(counts)) + (i - (len(ALGO_ORDER) - 1) / 2) * width
        data = [sub.loc[sub.n_classes == c, "rel_gap"].to_numpy() for c in counts]
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=1.2))
        for patch in bp["boxes"]:
            patch.set_facecolor(PALETTE[a])
            patch.set_alpha(0.75)
    # Runs that reach the optimum have rel_gap == 0, which a log axis cannot
    # show. symlog keeps them visible and honest instead of silently dropping
    # exactly the runs that succeeded.
    ax.set_yscale("symlog", linthresh=1e-9)
    ax.set_xticks(np.arange(len(counts)), [str(c) for c in counts])
    ax.set_xlabel("Number of classes $C$")
    ax.set_ylabel("Relative gap to proven optimum (symlog; 0 = exact)")
    ax.set_title("Distance from the exact optimum, 25 runs per cell")
    ax.axhline(0, color="k", lw=0)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=PALETTE[a], alpha=0.75) for a in ALGO_ORDER]
    ax.legend(handles, [LABELS[a] for a in ALGO_ORDER], fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "gap_by_class.png", dpi=200)
    plt.close(fig)
    print("[report] wrote gap_by_class.png")


def figure_hit_rate(df: pd.DataFrame, outdir: Path) -> None:
    piv = df.groupby(["algorithm", "n_classes"])["hit_optimum"].mean().unstack("n_classes")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for a in ALGO_ORDER:
        ax.plot(piv.columns, piv.loc[a] * 100, "o-", color=PALETTE[a], lw=2, label=LABELS[a])
    ax.axhline(100, color="k", ls="--", lw=1.2, label="Exact DP (always, in ~4 ms)")
    ax.set_xlabel("Number of classes $C$")
    ax.set_ylabel("Runs reaching the exact optimum (%)")
    ax.set_ylim(-3, 105)
    ax.set_title("Success rate against a solver that never fails")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "hit_rate.png", dpi=200)
    plt.close(fig)
    print("[report] wrote hit_rate.png")


def write_summary(df, scaling, metrics, outdir: Path) -> None:
    """Auto-generate RESULTS.md, including the headline numbers.

    Generated rather than hand-written so the prose cannot drift away from the
    data it describes -- the same reason nothing in paper/ is edited by hand.
    """
    lines = ["# Results", "",
             "Generated by `make report`. Do not edit by hand.", ""]

    n_runs = len(df)
    n_cells = df.groupby(["image", "criterion", "n_classes", "repair", "algorithm"]).ngroups
    lines += [
        "## Scope", "",
        f"- {n_runs:,} runs over {n_cells:,} configuration cells",
        f"- {df.image.nunique()} images, {df.criterion.nunique()} criteria, "
        f"class counts {sorted(int(c) for c in df.n_classes.unique())}",
        f"- {df.seed.nunique()} seeds per cell, budget = 1000 x (C-1) function evaluations",
        f"- Budget never exceeded: {bool((df.evaluations <= df.budget).all())}",
        f"- No run ever beat the proven optimum: "
        f"{bool((df.best_score <= df.optimal_score + 1e-9).all())}",
        "",
    ]

    lines += ["## 1. How often does a metaheuristic reach the proven optimum?", ""]
    hit = df.groupby(["algorithm", "n_classes"])["hit_optimum"].mean().unstack()
    for a in ALGO_ORDER:
        per_c = ", ".join(f"C={c}: {hit.loc[a, c]:.0%}" for c in hit.columns)
        lines.append(f"- **{LABELS[a]}** -- {per_c}")
    overall = df.groupby("algorithm")["hit_optimum"].mean()
    best_algo = overall.idxmax()
    lines += ["",
              f"Best overall: **{LABELS[best_algo]}** at {overall.max():.1%} of runs.",
              "The exact dynamic program reaches it in **100%** of cases, by construction, "
              "in a few milliseconds.", ""]

    lines += ["## 2. How close do they get when they miss?", ""]
    med = df.groupby("algorithm")["rel_gap"].median()
    for a in ALGO_ORDER:
        lines.append(f"- {LABELS[a]}: median relative gap {med[a]:.2e}")
    lines += ["",
              "The gaps are small in absolute terms. That is the point: the methods work, "
              "and they are still solving a problem that was already solved exactly.", ""]

    if scaling is not None:
        lines += ["## 3. Cost of the two exact methods", ""]
        agree = scaling.dropna(subset=["skimage_value"])
        all_agree = bool(agree.values_agree.all()) if len(agree) else None
        lines.append(f"- Optima agree with `skimage.threshold_multiotsu` at every tested C: {all_agree}")
        fin = scaling[np.isfinite(scaling.speedup)]
        if len(fin):
            worst = fin.loc[fin.speedup.idxmax()]
            lines.append(
                f"- Largest measured speed-up: **{worst.speedup:,.0f}x** at C={int(worst.n_classes)} "
                f"({worst.dp_ms:.1f} ms vs {worst.skimage_ms:,.0f} ms)"
            )
        lines.append(f"- DP time across C={int(scaling.n_classes.min())}..{int(scaling.n_classes.max())}: "
                     f"{scaling.dp_ms.min():.1f}-{scaling.dp_ms.max():.1f} ms (flat in C)")
        lines.append("")

    if metrics is not None:
        ph = metrics[metrics.kind == "phantom"]
        if len(ph):
            lines += ["## 4. Do the conventional metrics track segmentation quality?", ""]
            counts = {"psnr": 0, "ssim": 0, "fsim": 0}
            n_tot = 0
            big_agree = big_tot = 0
            for _, grp in ph.groupby(["image", "n_classes"]):
                if grp.criterion.nunique() < 2:
                    continue
                n_tot += 1
                ds = grp.sort_values("dice", ascending=False)
                winner = ds.criterion.iloc[0]
                margin = float(ds.dice.iloc[0] - ds.dice.iloc[1])
                disagreed = False
                for m in counts:
                    if grp.loc[grp[m].idxmax(), "criterion"] != winner:
                        counts[m] += 1
                        disagreed = True
                if margin > 0.05:
                    big_tot += 1
                    big_agree += int(not disagreed)
            for m, c in counts.items():
                lines.append(f"- {m.upper()} selects a different criterion than Dice in "
                             f"**{c} of {n_tot}** cases")
            lines += ["",
                      f"- Where the two segmentations genuinely differ (Dice margin > 0.05): "
                      f"{big_agree} of {big_tot} cases have every metric agreeing with Dice.",
                      "- Where they are near-identical, the metrics disagree freely. They track "
                      "quality when the gap is large and become arbitrary when it is small -- "
                      "which is the regime papers use them in.",
                      "- Both criteria are evaluated at proven optimal thresholds, so the "
                      "optimizer plays no part in this.",
                      ""]

    _write(outdir.parent / "RESULTS.md", "\n".join(lines) + "\n")


def figure_examples(outdir: Path) -> None:
    """Qualitative panel: exact segmentations at increasing class counts."""
    from ..datasets import load_real
    from ..exact import exact_thresholds
    from ..histogram import HistogramStats
    from ..metrics import label_map

    names = ["cell", "human_mitosis", "coins", "camera"]
    imgs = load_real(names)
    counts = [2, 3, 4, 6]
    if not imgs:
        return

    rows = len(imgs)
    fig, axes = plt.subplots(rows, len(counts) + 1, figsize=(2.05 * (len(counts) + 1), 2.05 * rows))
    axes = np.atleast_2d(axes)

    for r, (name, img) in enumerate(imgs.items()):
        stats = HistogramStats(np.bincount(img.ravel(), minlength=256))
        axes[r, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_ylabel(name, fontsize=9)
        if r == 0:
            axes[r, 0].set_title("original", fontsize=9)
        for c, C in enumerate(counts, start=1):
            t, _ = exact_thresholds(stats, C, "otsu")
            axes[r, c].imshow(label_map(img, t), cmap="viridis", interpolation="nearest")
            if r == 0:
                axes[r, c].set_title(f"$C={C}$", fontsize=9)
        for ax in axes[r]:
            ax.set_xticks([]), ax.set_yticks([])

    fig.suptitle("Exact multilevel Otsu segmentations (dynamic programming, ~4 ms each)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outdir / "figures" / "examples.png", dpi=170)
    plt.close(fig)
    print("[report] wrote examples.png")


def figure_criteria(metrics: pd.DataFrame, outdir: Path) -> None:
    """Otsu vs Kapur at their proven optima, against ground truth.

    The point of the panel is that both columns are *optimal* -- any visible
    difference is the criterion, not the optimizer.
    """
    from ..datasets import load_phantoms
    from ..exact import exact_thresholds
    from ..histogram import HistogramStats
    from ..metrics import label_map

    phantoms = load_phantoms([3, 4, 5])
    fig, axes = plt.subplots(len(phantoms), 4, figsize=(8.6, 2.2 * len(phantoms)))
    axes = np.atleast_2d(axes)

    for r, (name, (img, truth)) in enumerate(phantoms.items()):
        C = int(truth.max()) + 1
        stats = HistogramStats(np.bincount(img.ravel(), minlength=256))
        axes[r, 0].imshow(img, cmap="gray"), axes[r, 0].set_ylabel(f"{name}", fontsize=8)
        axes[r, 1].imshow(truth, cmap="viridis", interpolation="nearest")
        if r == 0:
            axes[r, 0].set_title("noisy input", fontsize=9)
            axes[r, 1].set_title("ground truth", fontsize=9)
        for c, criterion in enumerate(("otsu", "kapur"), start=2):
            t, _ = exact_thresholds(stats, C, criterion)
            axes[r, c].imshow(label_map(img, t), cmap="viridis", interpolation="nearest")
            row = metrics[(metrics.image == name) & (metrics.criterion == criterion)]
            if len(row):
                d, ps = float(row.dice.iloc[0]), float(row.psnr.iloc[0])
                axes[r, c].set_xlabel(f"Dice {d:.3f} | PSNR {ps:.1f}", fontsize=7.5)
            if r == 0:
                axes[r, c].set_title(f"exact {criterion}", fontsize=9)
        for ax in axes[r]:
            ax.set_xticks([]), ax.set_yticks([])

    fig.suptitle("Both criteria solved exactly - the difference is the objective, not the search", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "figures" / "criteria.png", dpi=170)
    plt.close(fig)
    print("[report] wrote criteria.png")


def build_all(results: Path, scaling_path: Path, metrics_path: Path, outdir: Path) -> None:
    # Both subdirectories must exist before anything writes: _write creates
    # parents for tables, but fig.savefig does not.
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results)
    table_hit_rate(df, outdir)
    table_rel_gap(df, outdir)
    table_statistics(df, outdir)
    table_repair(df, outdir)
    table_search_space(df, outdir)
    figure_gap_by_class(df, outdir)
    figure_hit_rate(df, outdir)

    if scaling_path.exists():
        _sc = pd.read_csv(scaling_path)
        table_scaling(_sc, outdir)
        figure_scaling(_sc, outdir)
    sc = pd.read_csv(scaling_path) if scaling_path.exists() else None
    mt = None
    figure_examples(outdir)
    if metrics_path.exists():
        mt = pd.read_csv(metrics_path)
        table_metric_disagreement(mt, outdir)
        figure_criteria(mt, outdir)
    write_summary(df, sc, mt, outdir)
    print("[report] done")
