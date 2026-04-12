# Metric Illustration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable Python script that generates a thesis-ready four-panel metric illustration showing observed data, trend, seasonal variation, and remainder.

**Architecture:** Add one self-contained plotting script under `scripts/` that synthesizes illustrative time-series components with `numpy`, renders a four-row stacked figure with `matplotlib`, and saves the outputs into `figures/`. Keep all parameters near the top of the script so the title, labels, seed, and output paths are easy to adjust later.

**Tech Stack:** Python 3, `numpy`, `matplotlib`

---

## File Structure

- Create: `scripts/plot_metric_illustration.py`
- Create: `figures/`
- Create: `docs/superpowers/plans/2026-04-13-metric-illustration.md`

### Task 1: Create the plotting script

**Files:**
- Create: `scripts/plot_metric_illustration.py`

- [ ] **Step 1: Create `scripts/plot_metric_illustration.py` with a deterministic data generator**

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("figures")
PNG_PATH = OUTPUT_DIR / "metric_illustration.png"
SVG_PATH = OUTPUT_DIR / "metric_illustration.svg"
SEED = 7
N_POINTS = 156


def build_trend(x: np.ndarray) -> np.ndarray:
    anchor_x = np.array([0, 18, 38, 58, 82, 106, 128, 145, 155])
    anchor_y = np.array([80, 86, 84, 111, 90, 99, 114, 80, 90])
    trend = np.interp(x, anchor_x, anchor_y)
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(trend, (2, 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def build_seasonal(x: np.ndarray) -> np.ndarray:
    return (
        7.5 * np.sin(2 * np.pi * x / 3)
        + 3.5 * np.sin(2 * np.pi * x / 6 + 0.7)
        - 5.0 * (x % 10 == 0)
    )
```

- [ ] **Step 2: Add remainder synthesis, total-series assembly, and plotting helpers**

```python
def build_remainder(n_points: int, rng: np.random.Generator) -> np.ndarray:
    remainder = rng.normal(0.0, 1.4, n_points)
    spikes = {
        24: 6.5,
        41: -7.5,
        77: 5.5,
        118: -8.5,
        126: -9.2,
        135: 4.8,
    }
    for index, delta in spikes.items():
        remainder[index] += delta
    return remainder


def build_series() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    x = np.arange(N_POINTS)
    trend = build_trend(x)
    seasonal = build_seasonal(x)
    remainder = build_remainder(N_POINTS, rng)
    data = trend + seasonal + remainder
    return x, data, trend, seasonal, remainder


def style_axis(ax: plt.Axes, label: str) -> None:
    ax.set_facecolor("#f4f4f4")
    ax.grid(True, color="white", linewidth=1.0)
    ax.text(
        -0.08,
        0.5,
        label,
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        color="#444444",
        bbox=dict(boxstyle="square,pad=0.35", facecolor="#d9d9d9", edgecolor="#d9d9d9"),
    )
```

- [ ] **Step 3: Add `main()` to render the four-panel figure and save both formats**

```python
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x, data, trend, seasonal, remainder = build_series()

    plt.style.use("ggplot")
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    axes[0].plot(x, data, color="#1f1f1f", linewidth=2.0)
    axes[1].plot(x, trend, color="#3a3a3a", linewidth=2.2)
    axes[2].plot(x, seasonal, color="#444444", linewidth=1.8)
    axes[3].axhline(0, color="#222222", linewidth=1.2)
    axes[3].bar(x, remainder, color="#4d4d4d", width=0.7)

    for ax, label in zip(axes, ["data", "trend", "seasonal", "remainder"]):
        style_axis(ax, label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("time step")
    fig.suptitle("Illustration of Metric Fluctuation Components", fontsize=15)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Run the script and verify the output files are created**

Run: `python3 scripts/plot_metric_illustration.py`  
Expected: command exits with code `0`, and both `figures/metric_illustration.png` and `figures/metric_illustration.svg` exist.

- [ ] **Step 5: Stage new files**

```bash
git add scripts/plot_metric_illustration.py figures/metric_illustration.png figures/metric_illustration.svg
```

### Task 2: Visual verification and final polish

**Files:**
- Modify: `scripts/plot_metric_illustration.py`
- Update if regenerated: `figures/metric_illustration.png`
- Update if regenerated: `figures/metric_illustration.svg`

- [ ] **Step 1: Inspect the generated figure for thesis suitability**

Check these points:

- The top panel visibly combines slow trend and short-period oscillation
- The second panel is smooth enough to read as a trend
- The third panel shows repetitive seasonal movement with stronger periodic troughs
- The fourth panel stays centered near zero but contains a few obvious spikes

- [ ] **Step 2: If the shape is off, adjust the script in one place only**

Tune these constants first:

```python
SEED = 7
N_POINTS = 156
anchor_y = np.array([80, 86, 84, 111, 90, 99, 114, 80, 90])
7.5 * np.sin(2 * np.pi * x / 3)
3.5 * np.sin(2 * np.pi * x / 6 + 0.7)
spikes = {24: 6.5, 41: -7.5, 77: 5.5, 118: -8.5, 126: -9.2, 135: 4.8}
```

- [ ] **Step 3: Regenerate outputs after any adjustment**

Run: `python3 scripts/plot_metric_illustration.py`  
Expected: files are overwritten successfully and the new image reflects the tuned shape.

- [ ] **Step 4: Stage final outputs**

```bash
git add scripts/plot_metric_illustration.py figures/metric_illustration.png figures/metric_illustration.svg
```

## Self-Review

- Spec coverage: the plan covers one script, two output formats, four stacked panels, thesis-oriented styling, and easy parameter tuning.
- Placeholder scan: no `TODO`, `TBD`, or vague “handle later” text remains.
- Type consistency: all functions and file paths referenced in later steps are defined earlier in the plan.
