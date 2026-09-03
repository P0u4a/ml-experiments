#!/usr/bin/env python3
"""Generate the report figures for the benign swarm-framing experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from swarm_data import (
    CONDITIONS,
    ROOT,
    agents_for,
    group_runs,
    load_observations,
    summarize,
)


OUT = ROOT / "analysis" / "swarm_report_figures"
OUT.mkdir(parents=True, exist_ok=True)

YES = "#4C78A8"
NO = "#E45756"
GRID = "#D9D9D9"
TEXT = "#222222"
DISCLOSURE = "#6B7280"

def style_axes(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=TEXT)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def yes_no_bars(
    ax: plt.Axes,
    labels: list[str],
    summaries: list[dict],
    *,
    title: str,
) -> None:
    y = np.arange(len(labels))
    yes = np.array([item["yes"] / item["n"] for item in summaries])
    no = np.array([item["no"] / item["n"] for item in summaries])
    yes_bars = ax.barh(y, yes, height=0.58, color=YES, label="YES")
    no_bars = ax.barh(y, no, left=yes, height=0.58, color=NO, label="NO")
    ax.set_yticks(y, [f"{label}  (n={item['n']})" for label, item in zip(labels, summaries)])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of submissions")
    ax.set_title(title, loc="left", weight="bold")
    style_axes(ax)

    for bars, key in ((yes_bars, "yes"), (no_bars, "no")):
        for bar, item in zip(bars, summaries):
            count = item[key]
            if count and bar.get_width() >= 0.08:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(count),
                    ha="center",
                    va="center",
                    color="white",
                    weight="bold",
                    fontsize=10,
                )


def main() -> None:
    observations = load_observations()
    runs = group_runs(observations)
    cohorts = {
        "solo_private": agents_for(runs, CONDITIONS["solo_private"]),
        "swarm_private": agents_for(runs, CONDITIONS["swarm_private"]),
        "swarm_shared": agents_for(runs, CONDITIONS["swarm_shared"]),
    }
    stats = {key: summarize(values) for key, values in cohorts.items()}

    # 1. Headline causal comparison: identical concurrent-agent setup and private
    # boards; only the session-versus-swarm framing differs.
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    yes_no_bars(
        ax,
        ["Solo", "Swarm"],
        [stats["solo_private"], stats["swarm_private"]],
        title="Discloses Tampering",
    )
    ax.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.17))
    fig.tight_layout()
    save(fig, "01_matched_framing_outcomes")

    # 2. The same matched result as a direct disclosure-rate estimate. Resample
    # whole four-agent sessions so the interval respects the experimental unit.
    run_no_counts = {
        key: [sum(agent.said_no for agent in run) for run in runs[condition]]
        for key, condition in (
            ("solo_private", CONDITIONS["solo_private"]),
            ("swarm_private", CONDITIONS["swarm_private"]),
            ("swarm_shared", CONDITIONS["swarm_shared"]),
        )
    }
    nondisclosure_by_session = {
        key: np.asarray(values, dtype=float) / 4
        for key, values in run_no_counts.items()
    }
    bootstrap_rng = np.random.default_rng(29)
    estimates = []
    intervals = []
    for key in ("solo_private", "swarm_private"):
        values = nondisclosure_by_session[key]
        boot = np.array(
            [bootstrap_rng.choice(values, size=len(values), replace=True).mean() for _ in range(20_000)]
        )
        estimates.append(values.mean())
        intervals.append(np.quantile(boot, [0.025, 0.975]))

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(2)
    ax.bar(x, estimates, width=0.56, color=DISCLOSURE)
    lower = np.array(estimates) - np.array([item[0] for item in intervals])
    upper = np.array([item[1] for item in intervals]) - np.array(estimates)
    ax.errorbar(x, estimates, yerr=np.vstack([lower, upper]), fmt="none", ecolor=TEXT, capsize=5)
    ax.set_xticks(x, ["Solo\n(8/40)", "Swarm\n(16/40)"])
    ax.set_ylim(0, 0.65)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Share of agents across 10 rollouts")
    ax.set_title("Nondisclosure", loc="left", weight="bold", pad=34)
    ax.text(
        1.0,
        1.015,
        "95% rollout-bootstrap CI",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#555555",
        fontsize=9,
    )
    for index, value in enumerate(estimates):
        ax.text(
            index - 0.20,
            value / 2,
            f"{value:.0%}",
            ha="left",
            va="center",
            color="white",
            weight="bold",
        )
    style_axes(ax, "y")
    fig.tight_layout()
    save(fig, "02_condition_progression")

    # 3. Show that the swarm-framing effect is already present with isolated
    # boards; enabling peer communication did not further change the point estimate.
    communication_keys = ["solo_private", "swarm_private", "swarm_shared"]
    communication_rates = np.array(
        [np.mean(np.asarray(run_no_counts[key], dtype=float) / 4) for key in communication_keys]
    )
    communication_intervals = []
    communication_rng = np.random.default_rng(41)
    for key in communication_keys:
        values = np.asarray(run_no_counts[key], dtype=float) / 4
        boot = np.array(
            [communication_rng.choice(values, size=len(values), replace=True).mean() for _ in range(20_000)]
        )
        communication_intervals.append(np.quantile(boot, [0.025, 0.975]))

    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    x = np.arange(3)
    ax.bar(x, communication_rates, width=0.58, color=[YES, NO, NO])
    lower = communication_rates - np.array([item[0] for item in communication_intervals])
    upper = np.array([item[1] for item in communication_intervals]) - communication_rates
    ax.errorbar(
        x,
        communication_rates,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor=TEXT,
        capsize=5,
    )
    ax.set_xticks(
        x,
        [
            "Solo\nNo peer communication",
            "Swarm\nNo peer communication",
            "Swarm\nPeer communication enabled",
        ],
    )
    ax.set_ylim(0, 0.72)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Share of agents across 10 rollouts")
    ax.set_title("Nondisclosure with and without peer communication", loc="left", weight="bold", pad=38)
    ax.text(
        1.0,
        1.015,
        "Same initial message · 95% rollout-bootstrap CI",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#555555",
        fontsize=9,
    )
    for index, (rate, key) in enumerate(zip(communication_rates, communication_keys)):
        count = stats[key]["no"]
        ax.text(
            index - 0.22,
            rate / 2,
            f"{rate:.0%}\n({count}/40)",
            ha="left",
            va="center",
            color="white",
            weight="bold",
        )
    style_axes(ax, "y")
    fig.tight_layout()
    save(fig, "03_messageboard_visibility")

    # 4. Show every independent four-agent session and the mean for each arm.
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    jitter_rng = np.random.default_rng(17)
    for position, key, color in (
        (0, "solo_private", YES),
        (1, "swarm_private", NO),
    ):
        values = np.asarray(run_no_counts[key])
        jitter = jitter_rng.uniform(-0.075, 0.075, len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=55,
            color=color,
            alpha=0.9,
        )
        mean = values.mean()
        ax.hlines(mean, position - 0.2, position + 0.2, color=TEXT, linewidth=2.3)
        ax.text(
            position,
            mean - 0.08,
            f"Mean {mean:.1f}",
            ha="center",
            va="top",
            color=TEXT,
            fontsize=9,
        )
    ax.set_xticks(
        [0, 1],
        [
            "Solo\n(2/10 rollouts had ≥2 agents withhold)",
            "Swarm\n(6/10 rollouts had ≥2 agents withhold)",
        ],
    )
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylim(-0.25, 4.25)
    ax.set_ylabel("Agents withholding disclosure (out of 4)")
    ax.set_title("Nondisclosure across rollouts", loc="left", weight="bold", pad=42)
    ax.text(
        1.0,
        1.015,
        "Each dot = one rollout of 4 agents",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#555555",
        fontsize=9,
    )
    style_axes(ax, "y")
    fig.tight_layout()
    save(fig, "04_session_level_outcomes")

    # 5. Full-CoT coding for the matched framing conditions.
    awareness = json.loads(
        (ROOT / "analysis" / "eval_awareness_report.json").read_text(encoding="utf-8")
    )["aggregate"]
    awareness_labels = [
        "Mentions evaluation",
        "Mentions evaluation\nafter board read",
        "Reasons about\nbeing evaluated",
    ]
    session_awareness = np.array(
        [
            awareness["session"]["mentions_eval_term"] / 40,
            awareness["session"]["mentions_eval_term_after_board"] / 40,
            awareness["session"]["strategic_eval_awareness"] / 40,
        ]
    )
    swarm_awareness = np.array(
        [
            awareness["swarm"]["mentions_eval_term"] / 40,
            awareness["swarm"]["mentions_eval_term_after_board"] / 40,
            awareness["swarm"]["strategic_eval_awareness"] / 40,
        ]
    )
    def grouped_horizontal(
        ax: plt.Axes,
        labels: list[str],
        solo_values: np.ndarray,
        swarm_values: np.ndarray,
        title: str,
        xlabel: str,
    ) -> None:
        y = np.arange(len(labels))
        height = 0.34
        solo_bars = ax.barh(y - height / 2, solo_values, height, color=YES, label="Solo")
        swarm_bars = ax.barh(y + height / 2, swarm_values, height, color=NO, label="Swarm")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", weight="bold")
        style_axes(ax, "x")
        for bars in (solo_bars, swarm_bars):
            for bar in bars:
                value = bar.get_width()
                if value >= 0.28:
                    x_text, alignment, color = value - 0.02, "right", "white"
                else:
                    x_text, alignment, color = value + 0.02, "left", TEXT
                ax.text(
                    x_text,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1%}",
                    ha=alignment,
                    va="center",
                    color=color,
                    weight="bold",
                    fontsize=9,
                )

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    grouped_horizontal(
        ax,
        awareness_labels,
        session_awareness,
        swarm_awareness,
        "Evaluation awareness",
        "Share of all agents",
    )
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
    )
    fig.tight_layout()
    save(fig, "05_eval_awareness")

    plotted_data = {
        "cohorts": stats,
        "session_no_counts": run_no_counts,
        "eval_awareness": awareness,
        "notes": {
            "headline_comparison": "Private-board concurrent-agent runs; only solo versus swarm framing differs.",
        },
    }
    (OUT / "figure_data.json").write_text(
        json.dumps(plotted_data, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: {k: v for k, v in item.items() if k != "run_ids"} for key, item in stats.items()}, indent=2))
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
