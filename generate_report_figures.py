"""Generate the figures for the report / presentation.

Run from the project root AFTER `python train_model.py`:
    python generate_report_figures.py

Writes PNGs plus INDEX.md (title and caption of each figure) to reports/figures/.
Everything is recomputed from data/sensor_data.csv, the saved model bundle and
reports/model_metrics.json, using the same split (random_state=42) as training.
Figures are rendered for a light slide/report background.
"""
from __future__ import annotations

import json
import textwrap
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from backend.rules import RULES

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "sensor_data.csv"
MODEL = ROOT / "backend" / "models" / "vehicle_failure_model.joblib"
REPORT = ROOT / "reports" / "model_metrics.json"
OUT = ROOT / "reports" / "figures"
SEED = 42
TARGET = "failure_risk"

# Palette (light mode): validated with the dataviz validator. Blue / orange identify
# the two classes everywhere; aqua is only the third line in the threshold chart.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_250, BLUE_400, BLUE_600 = "#86b6ef", "#3987e5", "#184f95"
RED = "#d03b3b"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
    }
)

LABELS = {
    "mileage_km": "Mileage (km)",
    "engine_temp_c": "Engine temperature (°C)",
    "rpm": "Engine RPM",
    "vibration_level": "Vibration level",
    "oil_quality_pct": "Oil quality (%)",
    "brake_pad_mm": "Brake pad (mm)",
    "battery_voltage": "Battery voltage (V)",
    "coolant_level_pct": "Coolant level (%)",
    "vehicle_age_years": "Vehicle age (years)",
}
INDEX: list[tuple[str, str, str]] = []


def compact(v: float) -> str:
    a = abs(v)
    if a >= 10000:
        return f"{v / 1000:.0f}k"
    if a >= 1000:
        return f"{v / 1000:.1f}k"
    return f"{v:.3g}" if a >= 10 else f"{v:.2g}"


def save(fig, name: str, title: str, subtitle: str, tight: bool = True, bottom: float = 0.0) -> None:
    """Add a wrapped title + subtitle header (sized in inches) and save the figure."""
    w, h = fig.get_size_inches()
    t = textwrap.fill(title, width=int(w * 9.0), break_on_hyphens=False)
    sub = textwrap.fill(subtitle, width=int(w * 12), break_on_hyphens=False)
    nt, ns = t.count("\n") + 1, sub.count("\n") + 1
    y_title = 0.12
    y_sub = y_title + nt * 0.25 + 0.10
    header = y_sub + ns * 0.19 + 0.30
    fig.text(0.012, 1 - y_title / h, t, ha="left", va="top", fontsize=13, fontweight="bold", color=INK, linespacing=1.15)
    fig.text(0.012, 1 - y_sub / h, sub, ha="left", va="top", fontsize=9.5, color=INK2, linespacing=1.3)
    top = 1 - header / h
    if tight:
        fig.tight_layout(rect=(0, bottom, 1, top))
    else:
        fig.subplots_adjust(top=top)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    INDEX.append((name, title, subtitle))
    print("saved", name, flush=True)


def clean_axes(ax, grid_axis: str | None = "y") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    features = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    bundle = joblib.load(MODEL)
    pipe = bundle["pipeline"]
    threshold = float(bundle["threshold"])
    ref = bundle["reference"]
    report = json.loads(REPORT.read_text())
    n, pos = len(df), int(y.sum())
    prevalence = pos / n

    # ---------------------------------------------------------------- 01 class balance
    fig, ax = plt.subplots(figsize=(9, 3.8))
    counts = [n - pos, pos]
    ax.barh([1, 0], counts, color=[BLUE, ORANGE], height=0.5, zorder=3)
    for yy, c in zip([1, 0], counts):
        ax.text(c + n * 0.01, yy, f"{c:,}  ·  {c / n * 100:.2f}%", va="center", color=INK, fontsize=10.5)
    ax.set_yticks([1, 0], ["Low risk (0)", "High risk (1)"], color=INK2)
    ax.set_xlim(0, n * 1.16)
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(v):,}")
    ax.set_xlabel("Number of vehicles")
    clean_axes(ax, "x")
    save(
        fig, "fig01_class_balance.png",
        f"Only {prevalence * 100:.2f}% of vehicles carry the high-risk label",
        f"Predicting 'low risk' for every vehicle would score {(1 - prevalence) * 100:.2f}% accuracy and find no failures, "
        "so accuracy alone is not used to judge models.",
    )

    # ---------------------------------------------------------------- 02 distributions
    auc = {f: roc_auc_score(y, X[f]) for f in features}
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, f in zip(axes.ravel(), features):
        lo, hi = X[f].quantile(0.0), X[f].quantile(1.0)
        bins = np.linspace(lo, hi, 31)
        for cls, col in ((0, BLUE), (1, ORANGE)):
            ax.hist(X.loc[y == cls, f], bins=bins, density=True, histtype="stepfilled",
                    color=col, alpha=0.25, linewidth=0, zorder=2)
            ax.hist(X.loc[y == cls, f], bins=bins, density=True, histtype="step",
                    color=col, linewidth=1.6, zorder=3)
        a = auc[f]
        sep = max(a, 1 - a)
        ax.set_title(LABELS[f], loc="left")
        ax.set_title(f"AUC {sep:.2f} {'↑' if a > 0.5 else '↓'}", loc="right", fontsize=9, fontweight="normal", color=INK2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_formatter(lambda v, _: compact(v))
    handles = [plt.Line2D([0], [0], color=BLUE, lw=2), plt.Line2D([0], [0], color=ORANGE, lw=2)]
    fig.legend(handles, ["Low risk (0)", "High risk (1)"], loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    save(
        fig, "fig02_feature_distributions.png",
        "Mileage, age, oil quality and brake wear separate the classes; RPM, vibration and coolant barely do",
        "Density of each feature for low-risk and high-risk vehicles (all 15,000 rows). AUC = how well that one feature separates the "
        "classes (0.5 none, 1.0 perfect); ↑ higher values are riskier, ↓ lower values are riskier.",
        bottom=0.05,
    )

    # ---------------------------------------------------------------- 03 failure rate by quintile
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, f in zip(axes.ravel(), features):
        q = pd.qcut(X[f], 5, duplicates="drop")
        rate = df.groupby(q, observed=True)[TARGET].mean() * 100
        ticks = [f"{compact(i.left)}–\n{compact(i.right)}" for i in rate.index]
        ax.bar(range(len(rate)), rate.values, color=BLUE, width=0.55, zorder=3)
        ax.axhline(prevalence * 100, color=MUTED, linestyle=(0, (4, 3)), linewidth=1, zorder=2)
        for i, v in enumerate(rate.values):
            ax.text(i, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK2, zorder=6,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))
        ax.set_xticks(range(len(rate)), ticks, fontsize=7.5)
        ax.set_title(LABELS[f], loc="left")
        ax.set_ylim(0, max(rate.max() * 1.18, prevalence * 100 * 2))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
        clean_axes(ax, "y")
    axes[0, 0].text(0.02, prevalence * 100 + 0.3, f"dataset average {prevalence * 100:.2f}%", fontsize=7.5,
                    color=INK2, transform=axes[0, 0].get_yaxis_transform())
    save(
        fig, "fig03_failure_rate_by_feature.png",
        "Failure rate rises steadily with wear; temperature and battery matter only at the extremes",
        "High-risk share (%) within each fifth of the feature's range (equal-sized groups). "
        "Dashed line = dataset average. RPM, vibration and coolant show no clear pattern.",
    )

    # ---------------------------------------------------------------- 04 correlation heatmap
    corr = df.corr()
    names = [LABELS[c].split(" (")[0] if c in LABELS else "Failure risk (label)" for c in corr.columns]
    cmap = LinearSegmentedColormap.from_list("div", ["#256abf", "#f0efec", RED])
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v + 0.0:.2f}".replace("-0.00", "0.00"), ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Pearson correlation")
    feat_corr = corr.loc[features, features].abs().values
    max_r = np.max(feat_corr[~np.eye(len(features), dtype=bool)])
    save(
        fig, "fig04_correlation_heatmap.png",
        f"Mileage, age, brake wear and oil quality are heavily intercorrelated (|r| up to {max_r:.2f})",
        "Correlated features share credit in feature-importance and SHAP results, so rankings among them should be read with care.",
    )

    # ---------------------------------------------------------------- 05 model comparison (dumbbell)
    models = ["Logistic Regression", "Random Forest", "XGBoost"]
    base, tuned = report["baseline_5_fold_cv"], report["tuned_5_fold_cv"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (metric, title) in zip(axes, (("roc_auc", "ROC-AUC"), ("average_precision", "Average precision"))):
        for row, m in enumerate(models):
            b, t = base[m][metric], tuned[m]["cv_metrics"][metric]
            ax.plot([b, t], [row, row], color=AXIS, linewidth=2, zorder=2)
            if metric == "average_precision":
                sd = tuned[m]["cv_average_precision_std"]
                ax.errorbar(t, row, xerr=sd, color=BLUE, alpha=0.45, linewidth=1.6, capsize=4, zorder=2)
            ax.scatter([b], [row], s=70, facecolor=SURFACE, edgecolor=INK2, linewidth=1.6, zorder=4)
            ax.scatter([t], [row], s=70, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=5)
            ax.text(t, row - 0.16, f"{t:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
            ax.text(b, row + 0.16, f"{b:.3f}", ha="center", va="top", fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(models)), models, color=INK)
        ax.set_ylim(2.7, -0.7)
        ax.set_title(title, loc="left")
        vals = [base[m][metric] for m in models] + [tuned[m]["cv_metrics"][metric] for m in models]
        pad = 0.07 if metric == "average_precision" else 0.03
        ax.set_xlim(min(vals) - pad, max(vals) + pad)
        clean_axes(ax, "x")
    spread = max(tuned[m]["cv_metrics"]["average_precision"] for m in models) - min(
        tuned[m]["cv_metrics"]["average_precision"] for m in models
    )
    sd_min = min(tuned[m]["cv_average_precision_std"] for m in models)
    save(
        fig, "fig05_model_comparison.png",
        "After tuning, the three models are statistically close; XGBoost scores highest",
        f"5-fold stratified CV on the training split. Tuned average precision spans {spread:.3f}, smaller than the fold-to-fold SD "
        f"(≥ {sd_min:.3f}). ○ default settings, ● tuned, whiskers = ±1 SD across folds (average precision panel).",
    )

    # ---------------------------------------------------------------- 06 threshold trade-off (OOF)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(clone(pipe), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    ts = np.linspace(0.05, 0.95, 181)
    P, R, F = [], [], []
    yt = y_train.to_numpy()
    for t in ts:
        pred = oof >= t
        tp = int((pred & (yt == 1)).sum())
        p = tp / pred.sum() if pred.sum() else np.nan
        r = tp / (yt == 1).sum()
        P.append(p)
        R.append(r)
        F.append(2 * p * r / (p + r) if p and (p + r) else 0)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for series, col, label in ((P, BLUE, "Precision"), (R, ORANGE, "Recall"), (F, AQUA, "F1")):
        ax.plot(ts, series, color=col, linewidth=2.2, zorder=3)
        k = int(np.nanargmin(np.abs(ts - 0.94)))
        ax.text(ts[k] + 0.012, series[k], label, color=INK, fontsize=10, va="center")
    for tv, lab in ((threshold, f"selected {threshold:g}\n(max F1)"), (0.5, "0.5\n(high-recall watch level)")):
        ax.axvline(tv, color=MUTED, linestyle=(0, (4, 3)), linewidth=1, zorder=2)
        ax.text(tv + 0.008, 0.02, lab, fontsize=8.5, color=INK2, va="bottom")
    ax.set_xlim(0.05, 1.03)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Decision threshold on the model score")
    ax.set_ylabel("Score on out-of-fold training predictions")
    clean_axes(ax, "y")
    save(
        fig, "fig06_threshold_tradeoff.png",
        "Raising the threshold trades recall for precision; 0.825 maximises F1",
        "Out-of-fold predictions on the training split only (test labels were not used to choose the threshold).",
    )

    # ---------------------------------------------------------------- 07 confusion matrices
    tm = report["test_metrics"]
    keys = sorted(tm, key=float, reverse=True)  # selected threshold first
    seq = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", BLUE_250, BLUE_400, BLUE_600])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9))
    for ax, k in zip(axes, keys):
        cm = np.array(tm[k]["confusion_matrix"])
        rowpct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rowpct, cmap=seq, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                dark = rowpct[i, j] > 0.55
                ax.text(j, i, f"{cm[i, j]:,}\n{rowpct[i, j] * 100:.1f}% of row", ha="center", va="center",
                        fontsize=11, color="white" if dark else INK)
        ax.set_xticks([0, 1], ["Predicted low", "Predicted high"])
        ax.set_yticks([0, 1], ["Actual low", "Actual high"])
        ax.tick_params(which="both", length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_xticks([0.5], minor=True)
        ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=3)
        ax.set_title(f"Threshold {float(k):g}   precision {tm[k]['precision']:.2f} · recall {tm[k]['recall']:.2f}", loc="left")
    sel, alt = tm[keys[0]]["confusion_matrix"], tm[keys[1]]["confusion_matrix"]
    save(
        fig, "fig07_confusion_matrices.png",
        f"Held-out test: {sel[1][1]} of {sum(sel[1])} high-risk vehicles caught at {float(keys[0]):g}; "
        f"{alt[1][1]} at {float(keys[1]):g} with {alt[0][1]} false alarms",
        "3,000 untouched test vehicles. Colour = share of each actual class; counts are numbers of vehicles.",
    )

    # ---------------------------------------------------------------- 08 ROC and PR (test)
    score = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, score)
    prec, rec, _ = precision_recall_curve(y_test, score)
    roc_auc = roc_auc_score(y_test, score)
    ap = tm[keys[0]]["average_precision"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.4, zorder=3)
    ax2 = axes[1]
    ax2.axhline(y_test.mean(), color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
    ax2.plot(rec, prec, color=BLUE, linewidth=2.4, zorder=3)
    for k in keys:
        t = float(k)
        pred = score >= t
        tp = ((pred) & (y_test.to_numpy() == 1)).sum()
        fp = ((pred) & (y_test.to_numpy() == 0)).sum()
        tpr_t, fpr_t = tp / (y_test == 1).sum(), fp / (y_test == 0).sum()
        prec_t = tp / max(pred.sum(), 1)
        ax.scatter([fpr_t], [tpr_t], s=60, color=INK, zorder=5, edgecolor=SURFACE, linewidth=1.2)
        ax.annotate(f"{t:g}", (fpr_t, tpr_t), textcoords="offset points", xytext=(8, -4), fontsize=9, color=INK)
        ax2.scatter([tpr_t], [prec_t], s=60, color=INK, zorder=5, edgecolor=SURFACE, linewidth=1.2)
        ax2.annotate(f"{t:g}", (tpr_t, prec_t), textcoords="offset points", xytext=(8, 4), fontsize=9, color=INK)
    ax.text(0.5, 0.56, "no-skill diagonal", color=INK2, fontsize=8.5, rotation=38, rotation_mode="anchor", ha="center", va="bottom")
    ax2.text(0.02, y_test.mean() + 0.025, f"no-skill = prevalence {y_test.mean():.3f}", color=INK2, fontsize=8.5)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate (recall)")
    ax.set_title("ROC curve", loc="left")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision–recall curve", loc="left")
    for a_ in (ax, ax2):
        a_.set_xlim(0, 1)
        a_.set_ylim(0, 1.02)
        clean_axes(a_, "both")
    save(
        fig, "fig08_roc_pr_curves.png",
        f"Held-out test: ROC-AUC {roc_auc:.3f}, average precision {ap:.3f} (no-skill {y_test.mean():.3f})",
        "Black dots mark the two operating thresholds. Scores come from the final tuned XGBoost model.",
    )

    # ---------------------------------------------------------------- 09/10 SHAP (test set)
    model, imputer = pipe.named_steps["model"], pipe.named_steps["imputer"]
    Xt = imputer.transform(X_test)
    sv = np.asarray(shap.TreeExplainer(model).shap_values(Xt))
    if sv.ndim == 3:
        sv = sv[..., -1]
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(range(len(mean_abs)), mean_abs.values, color=BLUE, height=0.55, zorder=3)
    for i, v in enumerate(mean_abs.values):
        ax.text(v + mean_abs.max() * 0.012, i, f"{v:.2f}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(mean_abs)), [LABELS[f] for f in mean_abs.index], color=INK)
    ax.set_xlabel("Mean |SHAP value| (log-odds contribution per vehicle)")
    clean_axes(ax, "x")
    top3 = [LABELS[f].split(" (")[0] for f in mean_abs.index[::-1][:3]]
    save(
        fig, "fig09_shap_importance.png",
        f"{top3[0]}, {top3[1].lower()} and {top3[2].lower()} carry most of the model's weight",
        "Average size of each feature's SHAP contribution over the 3,000 test vehicles. Correlated wear features share credit.",
    )

    plt.figure(figsize=(9.5, 5.6))
    shap.summary_plot(sv, pd.DataFrame(Xt, columns=features).rename(columns=LABELS), show=False, plot_size=None)
    fig = plt.gcf()
    fig.axes[0].set_xlabel("SHAP value (impact on model output, log-odds)")
    save(
        fig, "fig10_shap_beeswarm.png",
        "How each feature value pushes a vehicle toward higher or lower risk",
        "Each dot is one test vehicle; colour = feature value (red high, blue low). Right of zero = pushes toward the high-risk label.",
        tight=False,
    )

    # ---------------------------------------------------------------- 11 risk bands
    be = ref["band_evidence"]
    bands = ["LOW", "MODERATE", "HIGH"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    rates = [be[b]["dataset_failure_rate"] * 100 for b in bands]
    ax.bar(range(3), rates, color=[BLUE_250, BLUE_400, BLUE_600], width=0.5, zorder=3)
    ax.axhline(prevalence * 100, color=MUTED, linestyle=(0, (4, 3)), linewidth=1, zorder=2)
    ax.text(2.32, prevalence * 100, f"dataset average\n{prevalence * 100:.2f}%", ha="left", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(-0.5, 3.0)
    for i, b in enumerate(bands):
        ax.text(i, rates[i] + 1.5, f"{rates[i]:.1f}%", ha="center", va="bottom", fontsize=11, color=INK, zorder=6,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.set_xticks(range(3), [f"{b} risk\n{be[b]['n_vehicles']:,} vehicles" for b in bands], color=INK)
    ax.set_ylim(0, max(rates) * 1.25)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
    ax.set_ylabel("Share carrying the high-risk label")
    clean_axes(ax, "y")
    save(
        fig, "fig11_risk_bands.png",
        f"The model's HIGH band carried the high-risk label {rates[2]:.1f}% of the time, versus {rates[0]:.1f}% in LOW",
        "Risk bands shown in the app (LOW / MODERATE / HIGH), checked against the dataset label using out-of-fold training predictions.",
    )

    # ---------------------------------------------------------------- 12 maintenance rule evidence
    ev = ref["rule_evidence"]
    rows = sorted(RULES, key=lambda r: ev[r["id"]]["lift"] or 0)
    RULE_LABELS = {
        "engine_overheat": "Engine temp ≥ 110 °C",
        "oil_quality_low": "Oil quality < 60 %",
        "brake_pad_low": "Brake pad ≤ 3.0 mm",
        "coolant_low": "Coolant level < 60 %",
        "battery_voltage_low": "Battery voltage < 12.0 V",
        "age_or_mileage_high": "Age ≥ 10 yr or mileage ≥ 150k km",
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for i, r in enumerate(rows):
        e = ev[r["id"]]
        ax.barh(i, e["lift"], color=BLUE, height=0.5, zorder=3)
        ax.text(e["lift"] + 0.15, i,
                f"{e['lift']:.1f}×  ·  {e['failure_rate_when_triggered'] * 100:.0f}% high-risk  ·  {e['n_triggered']:,} vehicles  ·  {e['strength']}",
                va="center", fontsize=9, color=INK)
    ax.axvline(1, color=MUTED, linestyle=(0, (4, 3)), linewidth=1, zorder=2)
    ax.text(1.05, len(rows) - 0.45, "1× = dataset average", fontsize=8.5, color=INK2, va="bottom")
    ax.set_yticks(range(len(rows)), [RULE_LABELS.get(r["id"], r["condition"]) for r in rows], color=INK, fontsize=9.5)
    ax.set_xlim(0, max(ev[r["id"]]["lift"] for r in rows) * 1.75)
    ax.set_xlabel("Lift: high-risk rate when the rule fires ÷ overall high-risk rate")
    clean_axes(ax, "x")
    strong = [r["condition"].split(" ")[0] for r in rows if ev[r["id"]]["strength"] == "strong"]
    save(
        fig, "fig12_rule_evidence.png",
        "Most maintenance rules are well supported by the data; coolant and battery rules are weak",
        "Association of each rule-based advisory with the dataset label (training split). Weak rules reflect general maintenance practice, not this dataset.",
    )

    # ---------------------------------------------------------------- index
    lines = ["# Report figures", "", "Generated by `generate_report_figures.py` (light background, 200 dpi).", ""]
    for name, title, sub in INDEX:
        lines += [f"## {name}", f"**{title}**  ", sub, "", f"![{name}]({name})", ""]
    (OUT / "INDEX.md").write_text("\n".join(lines))
    print(f"Wrote {len(INDEX)} figures to {OUT}")


if __name__ == "__main__":
    main()
