from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import matplotlib.dates as mdates
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


OUTPUT_DIR = Path("figures")
PNG_PATH = OUTPUT_DIR / "metric_illustration.png"
SVG_PATH = OUTPUT_DIR / "metric_illustration.svg"
SEED = 7
N_POINTS = 156
START_DATE = datetime(2000, 1, 1)
LABELS = ("data", "trend", "seasonal", "remainder")
FIGURE_TITLE: str | None = None


def moving_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    padded = np.pad(values, (len(weights) // 2, len(weights) // 2), mode="edge")
    return np.convolve(padded, weights, mode="valid")


def build_time_axis() -> np.ndarray:
    return np.array(
        [np.datetime64(datetime(START_DATE.year + index // 12, index % 12 + 1, 1)) for index in range(N_POINTS)]
    )


def build_trend(x: np.ndarray) -> np.ndarray:
    anchor_x = np.array([0, 18, 38, 58, 82, 106, 128, 145, 155])
    anchor_y = np.array([80, 86, 84, 111, 90, 99, 114, 80, 90])
    raw_trend = np.interp(x, anchor_x, anchor_y)
    kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    return moving_average(raw_trend, kernel)


def build_seasonal(x: np.ndarray) -> np.ndarray:
    monthly_cycle = 7.5 * np.sin(2 * np.pi * x / 3.0)
    secondary_cycle = 3.5 * np.sin(2 * np.pi * x / 6.0 + 0.7)
    sharp_drop = np.where(x % 10 == 0, -10.0, 0.0)
    return monthly_cycle + secondary_cycle + sharp_drop


def build_remainder(rng: np.random.Generator) -> np.ndarray:
    remainder = rng.normal(0.0, 1.3, N_POINTS)
    spikes = {
        24: 4.8,
        31: -6.2,
        47: 7.6,
        58: -5.8,
        77: 5.3,
        101: 6.1,
        118: -8.5,
        119: -9.4,
        120: -7.1,
        127: 10.2,
        140: 4.4,
        148: -3.5,
    }
    for index, delta in spikes.items():
        remainder[index] += delta
    return remainder


def build_series() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    x = np.arange(N_POINTS)
    trend = build_trend(x)
    seasonal = build_seasonal(x)
    remainder = build_remainder(rng)
    data = trend + seasonal + remainder
    return build_time_axis(), data, trend, seasonal, remainder


def add_side_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.025,
        0.5,
        text,
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        color="#3a3a3a",
        bbox=dict(boxstyle="square,pad=0.55", facecolor="#d9d9d9", edgecolor="#d9d9d9"),
    )


def style_axis(ax: plt.Axes, label: str) -> None:
    ax.set_facecolor("#f4f4f4")
    ax.grid(True, color="white", linewidth=1.1)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    add_side_label(ax, label)


def render_figure(
    timestamps: np.ndarray,
    data: np.ndarray,
    trend: np.ndarray,
    seasonal: np.ndarray,
    remainder: np.ndarray,
) -> None:
    plt.style.use("ggplot")
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

    axes[0].plot(timestamps, data, color="#1f1f1f", linewidth=2.0)
    axes[1].plot(timestamps, trend, color="#3b3b3b", linewidth=2.2)
    axes[2].plot(timestamps, seasonal, color="#454545", linewidth=1.8)
    axes[3].axhline(0.0, color="#242424", linewidth=1.1)
    bar_width = np.timedelta64(18, "D")
    axes[3].bar(timestamps, remainder, width=bar_width, color="#5a5a5a", edgecolor="#5a5a5a")

    for ax, label in zip(axes, LABELS):
        style_axis(ax, label)
        ax.tick_params(colors="#5a5a5a", labelsize=10)
        ax.margins(x=0.02)

    axes[0].set_ylim(60, 130)
    axes[1].set_ylim(78, 115)
    axes[2].set_ylim(-18, 12)
    axes[3].set_ylim(-10, 11)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    if FIGURE_TITLE:
        fig.suptitle(FIGURE_TITLE, fontsize=16, color="#303030")
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamps, data, trend, seasonal, remainder = build_series()
    render_figure(timestamps, data, trend, seasonal, remainder)
    print(f"Saved illustration to {PNG_PATH} and {SVG_PATH}")


if __name__ == "__main__":
    main()
