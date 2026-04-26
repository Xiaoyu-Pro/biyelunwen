from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
from matplotlib import font_manager
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


CJK_FONT_CANDIDATES = (
    "PingFang SC",
    "Songti SC",
    "Heiti TC",
    "STHeiti",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
)
OUTPUT_DIR = Path("figures")
PNG_PATH = OUTPUT_DIR / "metric_illustration.png"
SVG_PATH = OUTPUT_DIR / "metric_illustration.svg"
SEED = 7
N_POINTS = 180
SAMPLE_INTERVAL_SECONDS = 1
LABELS = ("CPU 利用率", "内存占用率", "CPU 温度")
X_AXIS_LABEL = "采样时间（秒）"
Y_AXIS_LABELS = ("CPU 利用率（%）", "内存占用率（%）", "CPU 温度（°C）")
FIGURE_TITLE: str | None = None


def configure_fonts() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in CJK_FONT_CANDIDATES:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def build_time_axis() -> np.ndarray:
    return np.arange(N_POINTS) * SAMPLE_INTERVAL_SECONDS


def build_cpu_utilization(seconds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    baseline = 38 + 8 * np.sin(2 * np.pi * seconds / 45)
    burst = 32 * np.exp(-((seconds - 70) / 12) ** 2) + 24 * np.exp(-((seconds - 132) / 9) ** 2)
    noise = rng.normal(0.0, 2.2, len(seconds))
    return np.clip(baseline + burst + noise, 5, 98)


def build_memory_usage(seconds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    growth = 48 + 0.13 * seconds
    allocation_wave = 4.5 * np.sin(2 * np.pi * seconds / 60 + 0.8)
    release = np.where(seconds >= 125, -10.0, 0.0)
    noise = rng.normal(0.0, 0.9, len(seconds))
    return np.clip(growth + allocation_wave + release + noise, 30, 92)


def build_cpu_temperature(seconds: np.ndarray, cpu_utilization: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    thermal_lag = np.convolve(cpu_utilization, np.ones(12) / 12, mode="same")
    cooling_wave = 1.8 * np.sin(2 * np.pi * seconds / 80 + 1.2)
    noise = rng.normal(0.0, 0.6, len(seconds))
    return np.clip(42 + 0.28 * thermal_lag + cooling_wave + noise, 38, 86)


def build_series() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    seconds = build_time_axis()
    cpu_utilization = build_cpu_utilization(seconds, rng)
    memory_usage = build_memory_usage(seconds, rng)
    cpu_temperature = build_cpu_temperature(seconds, cpu_utilization, rng)
    return seconds, cpu_utilization, memory_usage, cpu_temperature


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
    seconds: np.ndarray,
    cpu_utilization: np.ndarray,
    memory_usage: np.ndarray,
    cpu_temperature: np.ndarray,
) -> None:
    plt.style.use("ggplot")
    configure_fonts()
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, constrained_layout=True)

    axes[0].plot(seconds, cpu_utilization, color="#1f1f1f", linewidth=2.0)
    axes[1].plot(seconds, memory_usage, color="#3b3b3b", linewidth=2.2)
    axes[2].plot(seconds, cpu_temperature, color="#454545", linewidth=1.8)

    for ax, label, y_axis_label in zip(axes, LABELS, Y_AXIS_LABELS):
        style_axis(ax, label)
        ax.set_xlabel(X_AXIS_LABEL, fontsize=11, color="#444444")
        ax.set_ylabel(y_axis_label, fontsize=11, color="#444444")
        ax.tick_params(colors="#5a5a5a", labelsize=10, labelbottom=True)
        ax.margins(x=0.02)

    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(0, 100)
    axes[2].set_ylim(35, 90)
    axes[-1].set_xlim(0, (N_POINTS - 1) * SAMPLE_INTERVAL_SECONDS)

    if FIGURE_TITLE:
        fig.suptitle(FIGURE_TITLE, fontsize=16, color="#303030")
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seconds, cpu_utilization, memory_usage, cpu_temperature = build_series()
    render_figure(seconds, cpu_utilization, memory_usage, cpu_temperature)
    print(f"Saved illustration to {PNG_PATH} and {SVG_PATH}")


if __name__ == "__main__":
    main()
