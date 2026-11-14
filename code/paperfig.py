"""Shared figure style (colour-blind-safe palette, markers carry series identity)."""
import matplotlib as mpl

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6a66"
MODEL_STYLE = {"qwen2.5:7b": (BLUE, "o"), "llama3.1:8b": (ORANGE, "s"), "gemma2:9b": (AQUA, "^")}
MODEL_NAME = {"qwen2.5:7b": "Qwen2.5-7B", "llama3.1:8b": "Llama3.1-8B", "gemma2:9b": "Gemma2-9B",
              "qwen2.5:14b": "Qwen2.5-14B", "qwen3:8b": "Qwen3-8B, thinking", "command-r:35b": "Command-R-35B",
              "llama3.3:70b": "Llama3.3-70B", "qwen2.5:7b-fp16": "Qwen2.5-7B, FP16"}


def apply_style():
    mpl.rcParams.update({
        "font.size": 7.5, "font.family": "sans-serif", "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.4, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "grid.linewidth": 0.5, "grid.alpha": 0.30, "lines.linewidth": 1.3, "lines.markersize": 3.6,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
