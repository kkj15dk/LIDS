import argparse
import os
import re

import matplotlib as mpl
import seaborn as sns

from eval import LATEX_RC_PARAMS, _parse_map_metrics_file, visualize_map_over_rounds

mpl.rcParams.update(LATEX_RC_PARAMS)


def _first_crossing_rounds(base_dir: str, mean_ci_max: float) -> list[int | None]:
    """
    For every duplicate_* folder in base_dir, return the first round at which map_value crosses (>=) mean_ci_max (the known true maximum), or None if that duplicate never does.
    """
    duplicate_dirs = sorted(
        d for d in os.listdir(base_dir) if d.startswith("duplicate_") and os.path.isdir(os.path.join(base_dir, d))
    )
    first_rounds: list[int | None] = []
    for duplicate_name in duplicate_dirs:
        dup_dir = os.path.join(base_dir, duplicate_name)
        round_dirs = sorted(
            (
                d for d in os.listdir(dup_dir)
                if d.startswith("round_") and d.endswith("_obs") and os.path.isdir(os.path.join(dup_dir, d))
            ),
            key=lambda d: int(d.split("_")[1]),
        )
        found_round: int | None = None
        for round_dir in round_dirs:
            metrics_path = os.path.join(dup_dir, round_dir, "map_metrics.txt")
            if not os.path.exists(metrics_path):
                continue
            _, map_value = _parse_map_metrics_file(metrics_path)
            if map_value >= mean_ci_max:
                found_round = int(round_dir.split("_")[1])
                break
        first_rounds.append(found_round)
    return first_rounds


def _min_median_max_first_round(first_rounds: list[int | None]) -> tuple[int | str, float | str, int | str]:
    """
    min/median/max of the per-duplicate first-crossing round (see _first_crossing_rounds), computed over all duplicates.
    """
    n = len(first_rounds)
    successes_sorted = sorted(r for r in first_rounds if r is not None)
    s = len(successes_sorted)

    def _order_stat(rank: int) -> int | str:
        return successes_sorted[rank - 1] if rank <= s else "N/A"

    min_round = _order_stat(1)
    max_round = _order_stat(n)
    if n % 2 == 1:
        median_round: float | str = _order_stat((n + 1) // 2)
    else:
        lo = _order_stat(n // 2)
        hi = _order_stat(n // 2 + 1)
        if (lo == "N/A" or hi == "N/A"):
            median_round = "N/A"
        else:
            assert isinstance(lo, int) and isinstance(hi, int)
            median_round = (lo + hi) / 2
    return min_round, median_round, max_round


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAP over rounds.")
    parser.add_argument(
        "--average",
        action="store_true",
        help="Plot the average over duplicates for each label.",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help=(
            "Shade the mean's 95%% CI band (delta-method, through the logit transform - see "
            "eval.py's _logit_mean_ci_band). Requires --average. The mean line itself is always "
            "plotted regardless of this flag; --ci only controls whether the band around it is drawn."
        ),
    )
    args = parser.parse_args()
    if args.ci and not args.average:
        parser.error("--ci requires --average")

    sigma_pattern = re.compile(r"\\sigma=([^$]+)")

    def _base_label(label: str) -> str:
        return label.split("(", 1)[0].strip()

    def _sigma_value(label: str) -> str | None:
        match = sigma_pattern.search(label)
        return match.group(1) if match else None

    # color palette
    colors = sns.color_palette("colorblind", n_colors=5)
    approach_colors: dict[str, str | tuple[float, float, float]] = {
        "LIDS": colors[0],
        "Thompson": colors[1],
        "Hamming EI": colors[4],
        "LOCK EI": colors[3],
        "LIDS-argmax": colors[2],
    }
    # q-misspecification (rho) ablation
    rho_colors = sns.color_palette("magma", n_colors=5)
    M_colors = sns.color_palette("magma", n_colors=3)
    approach_colors.update({
        r"$\rho=0$": rho_colors[0],
        r"$\rho=0.05$": rho_colors[1],
        r"$\rho=0.1$": rho_colors[2],
        r"$\rho=0.2$": rho_colors[3],
        r"$\rho=0.4$": rho_colors[4],
    })
    approach_colors.update({
        "M=1000": M_colors[2],
        "M=10000": M_colors[1],
        "M=100000": M_colors[0],
    })
    sigma_linestyles = {
        "10^{-4}": "-",
        "10^{-3}": "-.",
        "10^{-2}": "--",
    }
    legend_bbox_to_anchor = None
    average_duplicates = None

    XLABEL = r"experimental round"
    XLABEL_simple = r"experimental round"
    YLABEL_DNA = (
        r"activity"
    )
    YLABEL_AA = (
        r"top-5 activity"
    )
    YLABEL_AA_simple = (
        r"activity"
    )

    def _plot_dataset(
        folder_names: list[str],
        labels: list[str],
        output_filename: str,
        mean_ci_max: float,
        topk: int,
        title: str | None = None,
        force_solid: bool = False,
        ylabel: str | None = None,
        xlabel: str | None = None,
        max_round_plotted: int | None = None,
        ci_variants: bool = False,
        legend_loc: str = "upper right",
        annotate_label: str | None = None,
        annotate_text: str = "LIDS",
        annotate_offset: tuple[float, float] = (30.0, -30.0),
        annotate_tip_offset: tuple[float, float] = (0.0, 0.0),
        annotate_arrow_scale: float = 1.0,
        label_size: float = 44,
        title_pad: float | None = None,
    ) -> None:
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        missing = [f for f in folder_names if not os.path.isdir(os.path.join("outputs", f))]
        if missing:
            print(f"Skipping {output_filename}: missing {missing}")
            return
        label_styles: dict[str, dict[str, str | tuple[float, float, float]]] = {}
        legend_label_map: dict[str, str] = {}
        for label in labels:
            base = _base_label(label)
            sigma = _sigma_value(label)
            color = approach_colors.get(base, "#333333")
            if force_solid:
                linestyle = "-"
            else:
                linestyle = sigma_linestyles.get(sigma, "-") if sigma is not None else "-"
            label_styles[label] = {"color": color, "linestyle": linestyle}
            legend_label_map[label] = base

        # ci_variants
        if ci_variants:
            name, ext = os.path.splitext(output_filename)
            runs = [(output_filename, False), (f"{name}_CI{ext}", True)]
        else:
            runs = [(output_filename, args.ci)]

        for filename, show_ci in runs:
            path = visualize_map_over_rounds(
                folder_names,
                labels,
                output_filename=filename,
                average=True if ci_variants else args.average,
                average_duplicates=average_duplicates,
                topk=topk,
                max_round=max_round_plotted,
                mean_ci_max=mean_ci_max,
                show_ci=show_ci,
                f_max=mean_ci_max,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                label_styles=label_styles,
                legend_label_map=legend_label_map,
                legend_loc=legend_loc,
                legend_bbox_to_anchor=legend_bbox_to_anchor,
                legend_deduplicate=True,
                annotate_label=annotate_label,
                annotate_text=annotate_text,
                annotate_offset=annotate_offset,
                annotate_tip_offset=annotate_tip_offset,
                annotate_arrow_scale=annotate_arrow_scale,
                label_size=label_size,
                title_pad=title_pad,
            )
            print(path)

        for folder_name, label in zip(folder_names, labels):
            first_rounds = _first_crossing_rounds(os.path.join("outputs", folder_name), mean_ci_max)
            min_round, median_round, max_round = _min_median_max_first_round(first_rounds)
            num_found = sum(1 for r in first_rounds if r is not None)
            print(
                f"  [{label}] round first found max (>= {mean_ci_max}): "
                f"min={min_round} median={median_round} max={max_round} "
                f"({num_found}/{len(first_rounds)} duplicates found it)"
            )

    AA_PETase_TITLE = "PETase"
    AA_RUBISCO_TITLE = "RuBisCO"
    AA_SazCA_TITLE = "SazCA"

    # PETase (N20_L5_E96_AA, sigma=10^-4 only)
    _plot_dataset(
        folder_names=[
            "N20_L5_E96_AA_lifted_categorical_OraclePLM_sigma10-4_PETase",
            "N20_L5_E96_AA_thompson_OraclePLM_sigma10-4_PETase",
        ],
        labels=[
            "LIDS($\\sigma=10^{-4}$)(ours)",
            "Thompson($\\sigma=10^{-4}$)",
        ],
        output_filename="map_over_rounds_N20_L5_E96_AA_PETase_CI_topk5.pdf",
        mean_ci_max=8.82,  # known true maximum activity
        topk=5,
        title=AA_PETase_TITLE,
        ylabel=YLABEL_AA,
        xlabel=XLABEL,
    )
    # PETase (N20_L5_E96_AA, sigma=10^-4 only) ALL
    _plot_dataset(
        folder_names=[
            "N20_L5_E96_AA_lifted_categorical_OraclePLM_sigma10-4_PETase",
            "N20_L5_E96_AA_thompson_OraclePLM_sigma10-4_PETase",
            "L5_E96_AA_hamming_ei_OraclePLM_sigma10-4_PETase",
            "L5_E96_AA_lock_ei_OraclePLM_sigma10-4_PETase",
            "N20_L5_E96_AA_lifted_argmax_OraclePLM_sigma10-4_PETase",
        ],
        labels=[
            "LIDS($\\sigma=10^{-4}$)(ours)",
            "Thompson($\\sigma=10^{-4}$)",
            "Hamming EI($\\sigma=10^{-4}$)",
            "LOCK EI($\\sigma=10^{-4}$)",
            "LIDS-argmax($\\sigma=10^{-4}$)",
        ],
        output_filename="map_over_rounds_N20_L5_E96_AA_PETase_CI_topk5_ALL.pdf",
        mean_ci_max=8.82,  # known true maximum activity
        topk=5,
        title=AA_PETase_TITLE,
        ylabel=YLABEL_AA,
        xlabel=XLABEL,
    )

    # SazCA (N20_L5_E96_AA, sigma=10^-4 only)
    _plot_dataset(
        folder_names=[
            "N20_L5_E96_AA_lifted_categorical_OraclePLM_sigma10-4_SazCA",
            "N20_L5_E96_AA_thompson_OraclePLM_sigma10-4_SazCA",
        ],
        labels=[
            "LIDS($\\sigma=10^{-4}$)(ours)",
            "Thompson($\\sigma=10^{-4}$)",
        ],
        output_filename="map_over_rounds_N20_L5_E96_AA_SazCA_CI_topk5.pdf",
        mean_ci_max=1.87,  # known true maximum activity
        topk=5,
        title=AA_SazCA_TITLE,
        ylabel=YLABEL_AA,
        xlabel=XLABEL,
    )
    # SazCA (N20_L5_E96_AA, sigma=10^-4 only) ALL
    _plot_dataset(
        folder_names=[
            "N20_L5_E96_AA_lifted_categorical_OraclePLM_sigma10-4_SazCA",
            "N20_L5_E96_AA_thompson_OraclePLM_sigma10-4_SazCA",
            "L5_E96_AA_hamming_ei_OraclePLM_sigma10-4_SazCA",
            "L5_E96_AA_lock_ei_OraclePLM_sigma10-4_SazCA",
            "N20_L5_E96_AA_lifted_argmax_OraclePLM_sigma10-4_SazCA",
        ],
        labels=[
            "LIDS($\\sigma=10^{-4}$)(ours)",
            "Thompson($\\sigma=10^{-4}$)",
            "Hamming EI($\\sigma=10^{-4}$)",
            "LOCK EI($\\sigma=10^{-4}$)",
            "LIDS-argmax($\\sigma=10^{-4}$)",
        ],
        output_filename="map_over_rounds_N20_L5_E96_AA_SazCA_CI_topk5_ALL.pdf",
        mean_ci_max=1.87,  # known true maximum activity
        topk=5,
        title=AA_SazCA_TITLE,
        ylabel=YLABEL_AA,
        xlabel=XLABEL,
    )

    # rubisco (N20_L5_E96_AA, sigma=10^-4 only)
    _plot_dataset(
        folder_names=[
            "N20_L5_E96_AA_lifted_categorical_OraclePLM_sigma10-4_rubisco",
            "N20_L5_E96_AA_thompson_OraclePLM_sigma10-4_rubisco",
        ],
        labels=[
            "LIDS($\\sigma=10^{-4}$)(ours)",
            "Thompson($\\sigma=10^{-4}$)",
        ],
        output_filename="map_over_rounds_N20_L5_E96_AA_rubisco_CI_topk5.pdf",
        mean_ci_max=4.55,  # known true maximum activity
        topk=5,
        title=AA_RUBISCO_TITLE,
        ylabel=YLABEL_AA,
        xlabel=XLABEL,
    )

    # rubisco (N20_L5_E96_AA, sigma=10^-4 only) ALL
    _plot_dataset(
        folder_names=[
            "N20_L5_E96_AA_lifted_categorical_OraclePLM_sigma10-4_rubisco",
            "N20_L5_E96_AA_thompson_OraclePLM_sigma10-4_rubisco",
            "L5_E96_AA_hamming_ei_OraclePLM_sigma10-4_rubisco",
            "L5_E96_AA_lock_ei_OraclePLM_sigma10-4_rubisco",
            "N20_L5_E96_AA_lifted_argmax_OraclePLM_sigma10-4_rubisco",
        ],
        labels=[
            "LIDS($\\sigma=10^{-4}$)(ours)",
            "Thompson($\\sigma=10^{-4}$)",
            "Hamming EI($\\sigma=10^{-4}$)",
            "LOCK EI($\\sigma=10^{-4}$)",
            "LIDS-argmax($\\sigma=10^{-4}$)",
        ],
        output_filename="map_over_rounds_N20_L5_E96_AA_rubisco_CI_topk5_ALL.pdf",
        mean_ci_max=4.55,  # known true maximum activity
        topk=5,
        title=AA_RUBISCO_TITLE,
        ylabel=YLABEL_AA,
        xlabel=XLABEL,
        max_round_plotted=100,
    )
    # rubisco (N20_L5_E96_AA, sigma=10^-4 only) simple
    _plot_dataset(
        folder_names=[
            "N20_L5_E96_AA_lifted_categorical_OraclePLM_sigma10-4_rubisco",
            "N20_L5_E96_AA_thompson_OraclePLM_sigma10-4_rubisco",
            "L5_E96_AA_hamming_ei_OraclePLM_sigma10-4_rubisco",
            "L5_E96_AA_lock_ei_OraclePLM_sigma10-4_rubisco",
        ],
        labels=[
            "LIDS($\\sigma=10^{-4}$)(ours)",
            "Thompson($\\sigma=10^{-4}$)",
            "Hamming EI($\\sigma=10^{-4}$)",
            "LOCK EI($\\sigma=10^{-4}$)",
        ],
        output_filename="map_over_rounds_N20_L5_E96_AA_rubisco_CI_topk5_ALL_simple.pdf",
        mean_ci_max=4.55,  # known true maximum activity
        topk=5,
        title=AA_RUBISCO_TITLE,
        ylabel=YLABEL_AA_simple,
        xlabel=XLABEL_simple,
        max_round_plotted=100,
        annotate_label="LIDS",
        annotate_text="",
        annotate_offset=(50.0, -50.0),
        annotate_tip_offset=(6.0, -6.0),
        annotate_arrow_scale=5.0,
    )

    # rubisco (N20_L5_E96_AA, LIDS only) q-misspecification (rho) ablation
    _plot_dataset(
        folder_names=[
            "N20_L5_E96_AA_lifted_categorical_OraclePLM_sigma10-4_rubisco",
            "N20_L5_E96_AA_lifted_categorical_OraclePLM_sigma10-4_rubisco_rho0.05",
            "N20_L5_E96_AA_lifted_categorical_OraclePLM_sigma10-4_rubisco_rho0.1",
            "N20_L5_E96_AA_lifted_categorical_OraclePLM_sigma10-4_rubisco_rho0.2",
            "N20_L5_E96_AA_lifted_categorical_OraclePLM_sigma10-4_rubisco_rho0.4",
        ],
        labels=[r"$\rho=0$", r"$\rho=0.05$", r"$\rho=0.1$", r"$\rho=0.2$", r"$\rho=0.4$"],
        output_filename="map_over_rounds_N20_L5_E96_AA_rubisco_rho_ablation.pdf",
        mean_ci_max=4.55,  # known true maximum activity
        topk=5,
        title=AA_RUBISCO_TITLE,
        force_solid=True,
        ylabel=YLABEL_AA,
        xlabel=XLABEL,
        ci_variants=True,
        legend_loc="lower right",
    )
    # rubisco (N20_L5_E96_AA, LIDS only) posterior samples ablation (M)
    _plot_dataset(
        folder_names=[
            "N20_L5_E96_AA_lifted_categorical_OraclePLM_sigma10-4_rubisco_M1000",
            "N20_L5_E96_AA_lifted_categorical_OraclePLM_sigma10-4_rubisco_M10000",
            "N20_L5_E96_AA_lifted_categorical_OraclePLM_sigma10-4_rubisco",            
        ],
        labels=["M=1000", "M=10000", "M=100000"],
        output_filename="map_over_rounds_N20_L5_E96_AA_rubisco_M_ablation.pdf",
        mean_ci_max=4.55,  # known true maximum activity
        topk=5,
        title=AA_RUBISCO_TITLE,
        force_solid=True,
        ylabel=YLABEL_AA,
        xlabel=XLABEL,
        ci_variants=True,
    )

    # DNA (N8_L8_E12_DNA bump oracle)
    DNA_SIGMA10_2_TITLE = "$\\mathrm{peak}\\textsc{-SNR}=40$dB"
    DNA_SIGMA10_4_TITLE = "$\\mathrm{peak}\\textsc{-SNR}=80$dB"
    DNA_SIGMA10_3_TITLE = "$\\mathrm{peak}\\textsc{-SNR}=60$dB"

    _plot_dataset(
        folder_names=[
            "N8_L8_E12_DNA_lifted_categorical_bump_sigma10-2",
            "N8_L8_E12_DNA_thompson_bump_sigma10-2",
        ],
        labels=[
            "LIDS($\\sigma=10^{-2}$)(ours)",
            "Thompson($\\sigma=10^{-2}$)",
        ],
        output_filename="map_over_rounds_N8_L8_E12_DNA_sigma10-2_CI.pdf",
        mean_ci_max=1.0,
        topk=1,
        title=DNA_SIGMA10_2_TITLE,
        title_pad=14,
        force_solid=True,
        ylabel=YLABEL_DNA,
        xlabel=XLABEL,
    )
    _plot_dataset(
        folder_names=[
            "N8_L8_E12_DNA_lifted_categorical_bump_sigma10-3",
            "N8_L8_E12_DNA_thompson_bump_sigma10-3",
        ],
        labels=[
            "LIDS($\\sigma=10^{-3}$)(ours)",
            "Thompson($\\sigma=10^{-3}$)",
        ],
        output_filename="map_over_rounds_N8_L8_E12_DNA_sigma10-3_CI.pdf",
        mean_ci_max=1.0,
        topk=1,
        title=DNA_SIGMA10_3_TITLE,
        title_pad=14,
        force_solid=True,
        ylabel=YLABEL_DNA,
        xlabel=XLABEL,
    )
    _plot_dataset(
        folder_names=[
            "N8_L8_E12_DNA_lifted_categorical_bump_sigma10-4",
            "N8_L8_E12_DNA_thompson_bump_sigma10-4",
        ],
        labels=[
            "LIDS($\\sigma=10^{-4}$)(ours)",
            "Thompson($\\sigma=10^{-4}$)",
        ],
        output_filename="map_over_rounds_N8_L8_E12_DNA_sigma10-4_CI.pdf",
        mean_ci_max=1.0,
        topk=1,
        title=DNA_SIGMA10_4_TITLE,
        title_pad=14,
        force_solid=True,
        ylabel=YLABEL_DNA,
        xlabel=XLABEL,
    )
    # ALL
    _plot_dataset(
        folder_names=[
            "N8_L8_E12_DNA_lifted_categorical_bump_sigma10-2",
            "N8_L8_E12_DNA_thompson_bump_sigma10-2",
            "N8_L8_E12_DNA_gp_hamming_ei_bump_sigma10-2"
        ],
        labels=[
            "LIDS($\\sigma=10^{-2}$)(ours)",
            "Thompson($\\sigma=10^{-2}$)",
            "Hamming EI($\\sigma=10^{-2}$)"
        ],
        output_filename="map_over_rounds_N8_L8_E12_DNA_sigma10-2_CI_ALL.pdf",
        mean_ci_max=1.0,
        topk=1,
        title=DNA_SIGMA10_2_TITLE,
        title_pad=14,
        force_solid=True,
        ylabel=YLABEL_DNA,
        xlabel=XLABEL,
    )
    _plot_dataset(
        folder_names=[
            "N8_L8_E12_DNA_lifted_categorical_bump_sigma10-3",
            "N8_L8_E12_DNA_thompson_bump_sigma10-3",
            "N8_L8_E12_DNA_gp_hamming_ei_bump_sigma10-3"
        ],
        labels=[
            "LIDS($\\sigma=10^{-3}$)(ours)",
            "Thompson($\\sigma=10^{-3}$)",
            "Hamming EI($\\sigma=10^{-3}$)"
        ],
        output_filename="map_over_rounds_N8_L8_E12_DNA_sigma10-3_CI_ALL.pdf",
        mean_ci_max=1.0,
        topk=1,
        title=DNA_SIGMA10_3_TITLE,
        title_pad=14,
        force_solid=True,
        ylabel=YLABEL_DNA,
        xlabel=XLABEL,
    )
    _plot_dataset(
        folder_names=[
            "N8_L8_E12_DNA_lifted_categorical_bump_sigma10-4",
            "N8_L8_E12_DNA_thompson_bump_sigma10-4",
            "N8_L8_E12_DNA_gp_hamming_ei_bump_sigma10-4"
        ],
        labels=[
            "LIDS($\\sigma=10^{-4}$)(ours)",
            "Thompson($\\sigma=10^{-4}$)",
            "Hamming EI($\\sigma=10^{-4}$)"
        ],
        output_filename="map_over_rounds_N8_L8_E12_DNA_sigma10-4_CI_ALL.pdf",
        mean_ci_max=1.0,
        topk=1,
        title=DNA_SIGMA10_4_TITLE,
        title_pad=14,
        force_solid=True,
        ylabel=YLABEL_DNA,
        xlabel=XLABEL,
    )


if __name__ == "__main__":
    main()