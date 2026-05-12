"""OutoLLM — Advanced EDA Module (seaborn + matplotlib + scipy)"""
import math
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from logger import log, sidebar_log

warnings.filterwarnings("ignore")

# ── Sabitler ─────────────────────────────────────────────────────────────────
HIGH_CORR_THRESHOLD = 0.75
MAX_HISTOGRAM_COLS  = 12
MAX_BAR_CHART_COLS  = 8
TOP_VALUE_COUNTS    = 15
MAX_SCATTER_FEATURES = 6
TOP_GROUP_MEANS     = 15

# ── Renk Paleti ───────────────────────────────────────────────────────────────
P = {
    "numeric":     "#0d9488",
    "categorical": "#f97316",
    "binary":      "#8b5cf6",
    "success":     "#10b981",
    "danger":      "#ef4444",
    "warn":        "#f59e0b",
    "bg":          "#0f172a",
    "surface":     "#1e293b",
    "border":      "#334155",
    "text":        "#e2e8f0",
    "muted":       "#94a3b8",
}

# ── Tema Kurulum ──────────────────────────────────────────────────────────────
def _setup():
    sns.set_theme(style="dark", palette="muted", font_scale=1.1)
    plt.rcParams.update({
        "figure.facecolor": P["surface"], "axes.facecolor": P["bg"],
        "axes.edgecolor": P["border"], "text.color": P["text"],
        "axes.labelcolor": P["muted"], "xtick.color": P["muted"],
        "ytick.color": P["muted"], "grid.color": P["border"],
        "grid.alpha": 0.4, "axes.titlesize": 12, "axes.labelsize": 10,
        "figure.dpi": 120,
    })

def _show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS_DONE = False
def _inject_tufte_css():
    global _CSS_DONE
    if _CSS_DONE: return
    st.markdown("""<style>
    .eda-label{font-size:.78rem;font-weight:500;color:#9ca3af;letter-spacing:.04em;margin-bottom:2px}
    .eda-stats{font-size:.75rem;color:#6b7280;margin-bottom:4px}
    .insight-box{background:#1e293b;border-left:3px solid #0d9488;padding:10px 14px;
        border-radius:8px;margin:8px 0;font-size:.85rem;color:#cbd5e1;line-height:1.5}
    .insight-warn{background:#1e293b;border-left:3px solid #f59e0b;padding:10px 14px;
        border-radius:8px;margin:8px 0;font-size:.85rem;color:#fde68a;line-height:1.5}
    .insight-danger{background:#1e293b;border-left:3px solid #ef4444;padding:10px 14px;
        border-radius:8px;margin:8px 0;font-size:.85rem;color:#fca5a5;line-height:1.5}
    .corr-badge{display:inline-block;background:#1f2937;border-left:3px solid #6366f1;
        padding:3px 10px;border-radius:4px;font-size:.82rem;color:#d1d5db;margin:3px 2px}
    .section-header{font-size:1.1rem;font-weight:700;color:#e2e8f0;
        border-bottom:1px solid #334155;padding-bottom:6px;margin:16px 0 10px 0}
    .zero-group-box{background:#2d1b4e;border-left:3px solid #8b5cf6;padding:10px 14px;
        border-radius:8px;margin:8px 0;font-size:.85rem;color:#ddd6fe}
    </style>""", unsafe_allow_html=True)
    _CSS_DONE = True

# ══════════════════════════════════════════════════════════════════════════════
# MODÜLER YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════

def detect_outliers(series: pd.Series) -> pd.Series:
    """IQR tabanlı outlier maskesi döndürür."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)

def generate_insight(series: pd.Series, col: str) -> dict:
    """İstatistiklere dayalı otomatik içgörü: metin + severity döndürür."""
    s = series.dropna()
    skew = s.skew()
    kurt = s.kurtosis()
    cv = s.std() / s.mean() if s.mean() != 0 else 0
    n_out = int(detect_outliers(s).sum())
    pct_max = (s == s.max()).mean()
    pct_min = (s == s.min()).mean()
    pct_zero = (s == 0).mean()
    severity = "info"
    parts = []

    # Skewness
    if skew > 2:   parts.append(f"Strongly right-skewed (skew={skew:.2f}) — long right tail, consider log transform")
    elif skew > 1: parts.append(f"Moderately right-skewed (skew={skew:.2f})")
    elif skew < -2: parts.append(f"Strongly left-skewed (skew={skew:.2f}) — values cluster at high end")
    elif skew < -1: parts.append(f"Moderately left-skewed (skew={skew:.2f})")
    # Kurtosis
    if kurt > 5:   parts.append(f"Leptokurtic (kurt={kurt:.2f}) — extreme values more frequent than normal")
    elif kurt < -1: parts.append(f"Platykurtic (kurt={kurt:.2f}) — flat distribution, low peakedness")
    # Variability
    if cv > 1.5:   parts.append(f"Very high variability (CV={cv:.2f}) — heterogeneous population likely")
    elif cv > 0.8: parts.append(f"High variability (CV={cv:.2f})")
    # Outliers
    if n_out > 0:
        pct_o = n_out / len(s) * 100
        if pct_o > 5: severity = "warn"; parts.append(f"⚠ {n_out} outliers ({pct_o:.1f}%) — investigate before modeling")
        else:          parts.append(f"{n_out} outliers ({pct_o:.1f}%)")
    # Ceiling effect
    if pct_max > 0.20:
        severity = "danger"
        parts.append(f"🚨 Ceiling effect: {pct_max:.0%} of values at maximum ({s.max():.4g}) — truncated distribution")
    elif pct_max > 0.10:
        severity = "warn"
        parts.append(f"⚠ Potential ceiling effect: {pct_max:.0%} at max ({s.max():.4g})")
    # Floor effect
    if pct_min > 0.20 and s.min() == 0:
        severity = max(severity, "warn", key=lambda x: {"info":0,"warn":1,"danger":2}[x])
        parts.append(f"⚠ Floor effect: {pct_min:.0%} at zero — consider zero-inflation model")
    # Zero group
    if 0 < pct_zero <= 0.30 and s.min() < 0 or (pct_zero > 0 and s.min() == 0 and pct_min < 0.20):
        parts.append(f"Zero values: {pct_zero:.0%} of observations — may indicate a special subgroup")

    text = " · ".join(parts) if parts else "✓ Distribution appears approximately normal — no major anomalies detected."
    return {"text": text, "severity": severity}

def _show_insight(insight: dict):
    """Severity'e göre doğru CSS sınıfıyla insight box render eder."""
    cls = {"info": "insight-box", "warn": "insight-warn", "danger": "insight-danger"}.get(insight["severity"], "insight-box")
    st.markdown(f"<div class='{cls}'>💡 {insight['text']}</div>", unsafe_allow_html=True)


def plot_distribution(series: pd.Series, col: str, color: str = None) -> plt.Figure:
    """Histogram + KDE + BoxViolin + QQ — 3 panel."""
    _setup()
    color = color or P["numeric"]
    s = series.dropna()
    out_mask = detect_outliers(s)
    mu, sigma = s.mean(), s.std()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=P["surface"])
    fig.suptitle(col, color=P["text"], fontsize=14, fontweight="bold")

    # Panel 1: Histogram + KDE + annotations
    ax = axes[0]
    ceil_mask = (s == s.max())
    pct_max = ceil_mask.mean()
    normal_s = s[~ceil_mask & ~out_mask]
    out_s = s[out_mask]
    ceil_s = s[ceil_mask]

    # Normal bar
    if len(normal_s): ax.hist(normal_s, bins="auto", color=color, alpha=0.7, label="Normal", density=True)
    # Outlier bar
    if len(out_s): ax.hist(out_s, bins="auto", color=P["danger"], alpha=0.8, label="Outlier", density=True)
    # Ceiling group bar (at max value)
    if pct_max > 0.05 and len(ceil_s):
        ax.hist(ceil_s, bins=1, color=P["warn"], alpha=0.9, label=f"At max ({pct_max:.0%})", density=True)

    x = np.linspace(s.min(), s.max(), 300)
    try:
        kde = stats.gaussian_kde(s)
        ax.plot(x, kde(x), color=P["text"], lw=2, label="KDE")
    except Exception:
        pass
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color=P["warn"], lw=1.5, ls="--", alpha=0.7, label="Normal fit")

    ylim_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1
    for val, lbl, clr, ls in [(mu, "μ", P["warn"], "--"), (s.median(), "Med", "#38bdf8", ":")]:
        ax.axvline(val, color=clr, ls=ls, lw=1.4, alpha=0.9)
        ax.text(val, ylim_top * 0.88, lbl, color=clr, fontsize=8, ha="center", fontweight="bold")
    for val, lbl in [(s.quantile(0.25), "Q1"), (s.quantile(0.75), "Q3")]:
        ax.axvline(val, color=P["binary"], ls="-.", lw=1, alpha=0.8)
        ax.text(val, ylim_top * 0.72, lbl, color=P["binary"], fontsize=7, ha="center")

    # Ceiling effect annotation with arrow
    if pct_max > 0.10:
        ax.axvline(s.max(), color=P["danger"], lw=2.5, alpha=0.9)
        ax.annotate(
            f"Ceiling Effect\n{pct_max:.0%} at {s.max():.4g}",
            xy=(s.max(), ylim_top * 0.55),
            xytext=(s.max() - sigma * 1.2, ylim_top * 0.78),
            arrowprops=dict(arrowstyle="->", color=P["danger"], lw=1.5),
            color=P["danger"], fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=P["surface"], edgecolor=P["danger"], alpha=0.8),
        )
    ax.legend(fontsize=7, labelcolor=P["muted"], loc="upper left")
    stats_txt = f"μ={mu:.3g}  σ={sigma:.3g}  skew={s.skew():.2f}  kurt={s.kurtosis():.2f}"
    ax.set_title(f"Distribution  ·  {stats_txt}", color=P["muted"], fontsize=8.5)
    ax.set_xlabel(col, color=P["muted"])

    # Panel 2: Violin + Box
    ax = axes[1]
    parts = ax.violinplot(s, positions=[0], showmedians=False, showextrema=False)
    for pc in parts.get("bodies", []):
        pc.set_facecolor(color); pc.set_alpha(0.5)
    ax.boxplot(s, positions=[0], widths=0.15, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.3, linewidth=1.5),
               medianprops=dict(color=P["warn"], lw=2.5),
               whiskerprops=dict(color=P["muted"]),
               capprops=dict(color=P["muted"]),
               flierprops=dict(marker="x", color=P["danger"], ms=6))
    ax.set_xticks([0]); ax.set_xticklabels([col])
    ax.set_title("Box + Violin", color=P["muted"])

    # Panel 3: Q-Q plot
    ax = axes[2]
    try:
        (osm, osr), (slope, intercept, r) = stats.probplot(s, dist="norm")
        ax.scatter(osm, osr, color=color, alpha=0.4, s=12)
        ax.plot(osm, np.array(osm)*slope+intercept, color=P["warn"], lw=2)
        ax.set_title(f"Q-Q Plot  (r={r:.3f})", color=P["muted"])
        ax.set_xlabel("Theoretical Quantiles", color=P["muted"])
        ax.set_ylabel("Sample Quantiles", color=P["muted"])
    except Exception:
        ax.text(0.5, 0.5, "Q-Q unavailable", transform=ax.transAxes,
                ha="center", color=P["muted"])
    fig.tight_layout()
    return fig

def plot_relationship(df: pd.DataFrame, x: str, y: str) -> plt.Figure:
    """Scatter + regresyon çizgisi + outlier vurgu."""
    _setup()
    clean = df[[x, y]].dropna()
    out = detect_outliers(clean[x]) | detect_outliers(clean[y])
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=P["surface"])
    ax.scatter(clean.loc[~out, x], clean.loc[~out, y],
               color=P["numeric"], alpha=0.45, s=18, label="Normal")
    if out.any():
        ax.scatter(clean.loc[out, x], clean.loc[out, y],
                   color=P["danger"], alpha=0.85, s=30, marker="x", label="Outlier")
    try:
        sl, ic, r, p, _ = stats.linregress(clean[x], clean[y])
        xr = np.linspace(clean[x].min(), clean[x].max(), 200)
        ax.plot(xr, sl*xr+ic, color=P["warn"], lw=2, label=f"r={r:.3f}")
        p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
        ax.set_title(f"{x} vs {y}  |  r={r:.3f}, p={p_str}", color=P["text"], fontsize=11)
    except Exception:
        ax.set_title(f"{x} vs {y}", color=P["text"])
    ax.set_xlabel(x, color=P["muted"]); ax.set_ylabel(y, color=P["muted"])
    ax.legend(fontsize=8, labelcolor=P["muted"])
    fig.tight_layout()
    return fig

def plot_correlation_matrix(df: pd.DataFrame) -> plt.Figure:
    """Masked lower-triangle Pearson heatmap."""
    _setup()
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    n = len(corr)
    sz = max(7, n * 0.85)
    fig, ax = plt.subplots(figsize=(sz, sz*0.85), facecolor=P["surface"])
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.5, linecolor=P["bg"],
                annot_kws={"size": 9},
                cbar_kws={"shrink": 0.75, "label": "Pearson r"})
    for i in range(n):
        for j in range(i):
            if abs(corr.iloc[i, j]) >= 0.8:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                             edgecolor=P["warn"], lw=2.5))
    ax.set_title("Pearson Correlation Matrix (Lower Triangle)", color=P["text"],
                 fontsize=13, pad=12)
    ax.tick_params(axis="x", rotation=45, colors=P["muted"])
    ax.tick_params(axis="y", rotation=0, colors=P["muted"])
    fig.tight_layout()
    return fig

def plot_binary_analysis(series: pd.Series, col: str) -> plt.Figure:
    """Donut + yatay yüzde bar."""
    _setup()
    vc = series.value_counts()
    total = len(series.dropna())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=P["surface"])
    palette = [P["success"], P["danger"]] if len(vc) == 2 else sns.color_palette("Set2", len(vc))

    # Donut
    ax = axes[0]
    wedges, _, autotexts = ax.pie(vc.values, autopct="%1.1f%%", colors=palette,
                                   startangle=90,
                                   wedgeprops={"width": 0.55, "edgecolor": P["surface"], "lw": 2},
                                   pctdistance=0.75)
    for at in autotexts:
        at.set_color(P["bg"]); at.set_fontsize(11); at.set_fontweight("bold")
    ax.legend(wedges, [f"{k} (n={v})" for k, v in vc.items()],
              loc="center", fontsize=9, labelcolor=P["text"])
    ax.set_title(f"{col} — Donut", color=P["text"], fontsize=12)

    # Horizontal pct bar
    ax = axes[1]
    pcts = (vc / total * 100).sort_values()
    cols_b = palette[:len(pcts)]
    bars = ax.barh(pcts.index.astype(str), pcts.values,
                   color=cols_b, alpha=0.85, edgecolor=P["surface"])
    for bar, k in zip(bars, pcts.index):
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height()/2,
                f"{w:.1f}%  (n={vc[k]})", va="center", color=P["text"], fontsize=10)
    ax.set_xlim(0, 120)
    ax.set_title("Percentage Breakdown", color=P["text"], fontsize=12)
    ax.set_xlabel("Percentage (%)", color=P["muted"])
    fig.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT RENDER FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════════

def render_eda_overview(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame): raise TypeError("df must be DataFrame")
    log.info("EDA overview render.")
    _inject_tufte_css()
    total = df.shape[0] * df.shape[1]
    miss  = df.isnull().sum().sum() / total * 100 if total else 0
    dup   = int(df.duplicated().sum())
    nc = df.select_dtypes(include="number").shape[1]
    cc = df.select_dtypes(exclude="number").shape[1]
    cols = st.columns(5)
    cols[0].metric("Satır", f"{df.shape[0]:,}")
    cols[1].metric("Sütun", f"{df.shape[1]}")
    cols[2].metric("Sayısal", nc)
    cols[3].metric("Kategorik", cc)
    cols[4].metric("Eksik", f"{miss:.1f}%")
    if dup > 0:
        st.warning(f"**{dup:,}** yinelenen satır tespit edildi.")
    else:
        st.success("Yinelenen satır yok.")

def render_correlation_heatmap(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame): raise TypeError("df must be DataFrame")
    _inject_tufte_css()
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        st.info("Korelasyon için en az 2 sayısal sütun gerekir."); return
    log.info(f"Korelasyon: {num_df.shape[1]} sütun")
    fig = plot_correlation_matrix(df)
    _show(fig)
    corr = num_df.corr(numeric_only=True)
    high = []
    cls = corr.columns.tolist()
    for i, c1 in enumerate(cls):
        for c2 in cls[i+1:]:
            v = corr.loc[c1, c2]
            if abs(v) >= HIGH_CORR_THRESHOLD:
                high.append((c1, c2, v))
    if high:
        st.markdown(f"<div class='insight-warn'>⚠️ <b>{len(high)} highly correlated pair(s)</b> found (|r| ≥ {HIGH_CORR_THRESHOLD}). "
                    f"These may cause multicollinearity — consider dropping one from each pair before modeling.</div>",
                    unsafe_allow_html=True)
        for c1, c2, v in sorted(high, key=lambda x: abs(x[2]), reverse=True):
            direction = "positive" if v > 0 else "negative"
            d = "+" if v > 0 else "−"
            interp = "strong" if abs(v) >= 0.9 else "moderate"
            st.markdown(
                f"<span class='corr-badge'><b>{c1}</b> ↔ <b>{c2}</b> &nbsp; r = {d}{abs(v):.2f} "
                f"({interp} {direction})</span>",
                unsafe_allow_html=True
            )

def render_distribution_plots(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame): raise TypeError("df must be DataFrame")
    _inject_tufte_css()
    num_df = df.select_dtypes(include="number")
    if num_df.empty:
        st.info("Sayısal sütun bulunamadı."); return
    cols_to_plot = num_df.columns.tolist()
    if len(cols_to_plot) > MAX_HISTOGRAM_COLS:
        st.caption(f"İlk {MAX_HISTOGRAM_COLS}/{len(cols_to_plot)} sütun gösteriliyor.")
        cols_to_plot = cols_to_plot[:MAX_HISTOGRAM_COLS]
    log.info(f"Dağılım grafikleri: {len(cols_to_plot)} sütun")
    binary_cols  = [c for c in cols_to_plot if num_df[c].nunique() == 2]
    numeric_cols = [c for c in cols_to_plot if c not in binary_cols]
    if numeric_cols:
        st.markdown("#### 📈 Sayısal Dağılımlar")
        for col in numeric_cols:
            s = df[col].dropna()
            n_zero = int((s == 0).sum())
            zero_tag = f" · **{n_zero} zero-value**" if n_zero > 0 else ""
            with st.expander(f"📊 **{col}** — {len(s):,} değer{zero_tag}", expanded=False):
                fig = plot_distribution(s, col)
                _show(fig)
                insight = generate_insight(s, col)
                _show_insight(insight)
                # Zero-group special analysis
                if n_zero > 0 and s.min() == 0:
                    zero_mask = df[col] == 0
                    nonzero_mask = df[col] > 0
                    n_nz = int(nonzero_mask.sum())
                    pct_z = n_zero / len(s) * 100
                    st.markdown(
                        f"<div class='zero-group-box'>"
                        f"🔬 <b>Zero-Value Group Analysis</b> — "
                        f"Students/observations with {col} = 0: <b>{n_zero}</b> ({pct_z:.1f}%)<br>"
                        f"Non-zero group (n={n_nz}): μ = {s[s>0].mean():.3g}, median = {s[s>0].median():.3g}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
    if binary_cols:
        st.markdown("#### 🔵 Binary Değişkenler")
        for col in binary_cols:
            with st.expander(f"🔵 **{col}**", expanded=False):
                fig = plot_binary_analysis(df[col], col)
                _show(fig)

def render_categorical_plots(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame): raise TypeError("df must be DataFrame")
    _inject_tufte_css()
    cat_df = df.select_dtypes(exclude="number")
    if cat_df.empty:
        st.info("Kategorik sütun bulunamadı."); return
    cols_to_plot = cat_df.columns.tolist()[:MAX_BAR_CHART_COLS]
    log.info(f"Kategorik grafikler: {len(cols_to_plot)} sütun")
    n_cols = 2
    n_rows = math.ceil(len(cols_to_plot) / n_cols)
    for ri in range(n_rows):
        grid = st.columns(n_cols)
        for ci in range(n_cols):
            idx = ri * n_cols + ci
            if idx >= len(cols_to_plot): break
            col_name = cols_to_plot[idx]
            vc = df[col_name].value_counts().head(TOP_VALUE_COUNTS)
            n_unique = df[col_name].nunique()
            with grid[ci]:
                if n_unique == 2:
                    fig = plot_binary_analysis(df[col_name], col_name)
                    _show(fig)
                else:
                    _setup()
                    fig, ax = plt.subplots(figsize=(6, max(3, len(vc)*0.35)), facecolor=P["surface"])
                    colors = sns.color_palette("YlOrRd_r", len(vc))
                    bars = ax.barh(vc.index.astype(str)[::-1], vc.values[::-1],
                                   color=colors, alpha=0.85, edgecolor=P["surface"])
                    for bar in bars:
                        w = bar.get_width()
                        ax.text(w + 0.3, bar.get_y() + bar.get_height()/2,
                                f"{int(w)}", va="center", color=P["text"], fontsize=9)
                    ax.set_title(f"{col_name}  ({n_unique} unique)", color=P["text"], fontsize=11)
                    ax.set_xlabel("Count", color=P["muted"])
                    fig.tight_layout()
                    _show(fig)

def render_missing_heatmap(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame): raise TypeError("df must be DataFrame")
    _inject_tufte_css()
    missing = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        st.success("Hiçbir sütunda eksik veri yok."); return
    _setup()
    fig, ax = plt.subplots(figsize=(10, max(3, len(missing)*0.45)), facecolor=P["surface"])
    colors = [P["success"] if v < 20 else P["warn"] if v < 50 else P["danger"] for v in missing.values]
    bars = ax.barh(missing.index[::-1], missing.values[::-1], color=colors[::-1], alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2,
                f"{w:.1f}%", va="center", color=P["text"], fontsize=10)
    ax.set_xlim(0, 115)
    ax.axvline(20, color=P["warn"],   ls="--", lw=1, alpha=0.7, label="20% threshold")
    ax.axvline(50, color=P["danger"], ls="--", lw=1, alpha=0.7, label="50% threshold")
    ax.set_title("Missing Data by Column", color=P["text"], fontsize=13)
    ax.set_xlabel("Missing (%)", color=P["muted"])
    ax.legend(fontsize=9, labelcolor=P["muted"])
    fig.tight_layout()
    _show(fig)
    green = mpatches.Patch(color=P["success"], label="<20% — tolerable")
    orange = mpatches.Patch(color=P["warn"],  label="20-50% — impute")
    red = mpatches.Patch(color=P["danger"],   label=">50% — consider drop")
    st.caption("🟢 <20%: tolerable &nbsp; 🟠 20–50%: consider imputation &nbsp; 🔴 >50%: consider dropping")

def render_target_scatter(df: pd.DataFrame, target_col: str) -> None:
    if not isinstance(df, pd.DataFrame): raise TypeError("df must be DataFrame")
    if target_col not in df.columns:
        st.warning(f"'{target_col}' bulunamadı."); return
    _inject_tufte_css()
    num_feats = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    if not num_feats:
        st.info("Scatter için sayısal özellik gerekir."); return
    log.info(f"Scatter — target: {target_col}, {len(num_feats)} özellik")

    # Correlation ranking banner
    if pd.api.types.is_numeric_dtype(df[target_col]) and len(num_feats) > 0:
        corr_with_target = (
            df[num_feats + [target_col]].corr()[target_col]
            .drop(target_col).abs().sort_values(ascending=False)
        )
        top3 = corr_with_target.head(3)
        ranking_parts = [f"**{c}** (r={corr_with_target[c]:+.2f})" for c in top3.index]
        st.markdown(
            f"<div class='insight-box'>📊 <b>Correlation with {target_col}:</b> "
            + " &nbsp;›&nbsp; ".join(ranking_parts)
            + "</div>",
            unsafe_allow_html=True
        )

    max_show = min(len(num_feats), MAX_SCATTER_FEATURES)
    selected = st.multiselect(
        f"Görselleştirilecek özellikler (maks {MAX_SCATTER_FEATURES}):",
        options=num_feats, default=num_feats[:max_show],
        key="scatter_feature_select",
    )
    if not selected:
        st.info("En az bir özellik seçin."); return
    is_num_target = pd.api.types.is_numeric_dtype(df[target_col])
    n_cols = 2
    n_rows = math.ceil(len(selected) / n_cols)
    for ri in range(n_rows):
        grid = st.columns(n_cols)
        for ci in range(n_cols):
            idx = ri * n_cols + ci
            if idx >= len(selected): break
            feat = selected[idx]
            with grid[ci]:
                if is_num_target:
                    fig = plot_relationship(df, feat, target_col)
                    _show(fig)
                    gm = (df.groupby(target_col)[feat].mean()
                          .sort_values(ascending=False).head(TOP_GROUP_MEANS))
                    _setup()
                    fig, ax = plt.subplots(figsize=(6, 4), facecolor=P["surface"])
                    colors = sns.color_palette("Blues_r", len(gm))
                    ax.bar(gm.index.astype(str), gm.values, color=colors, alpha=0.85)
                    ax.set_title(f"{feat} mean by {target_col}", color=P["text"], fontsize=11)
                    ax.set_xlabel(target_col, color=P["muted"])
                    ax.tick_params(axis="x", rotation=30, colors=P["muted"])
                    fig.tight_layout()
                    _show(fig)

# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED STORY-DRIVEN PLOT FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════════

def plot_ceiling_effect_drama(series: pd.Series, col: str) -> plt.Figure:
    """Ceiling effect için dramatik 4-panel özel görselleştirme."""
    _setup()
    s = series.dropna()
    max_val = s.max()
    ceil_mask = s == max_val
    pct_ceil  = ceil_mask.mean()
    ceil_n    = int(ceil_mask.sum())
    mu, sigma = s.mean(), s.std()

    fig = plt.figure(figsize=(17, 5), facecolor=P["surface"])
    gs  = fig.add_gridspec(1, 4, wspace=0.38)

    # ── Panel 1: Annotated stacked histogram ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    bins_arr = np.linspace(s.min(), max_val, 30)
    normal_s = s[~ceil_mask]
    ax1.hist(normal_s, bins=bins_arr, color=P["numeric"], alpha=0.75, label="Normal scores")
    # Dramatic ceiling bar
    bar_w = (bins_arr[1] - bins_arr[0]) * 0.55
    ax1.bar(max_val, ceil_n, width=bar_w, color=P["danger"], alpha=0.95,
            label=f"Score = {max_val:.0f}  ({pct_ceil:.1%})")
    # Arrow annotation
    ax1.annotate(
        f"🚨 CEILING EFFECT\n{ceil_n:,} students\n({pct_ceil:.1%} of total)",
        xy=(max_val, ceil_n * 0.95),
        xytext=(max_val - sigma * 1.8, ceil_n * 0.75),
        arrowprops=dict(arrowstyle="-|>", color=P["danger"], lw=2),
        color=P["danger"], fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=P["bg"], edgecolor=P["danger"], lw=1.5),
    )
    ax1.set_title("Score Distribution", color=P["text"], fontsize=11)
    ax1.set_xlabel(col, color=P["muted"]); ax1.set_ylabel("Count", color=P["muted"])
    ax1.legend(fontsize=8, labelcolor=P["muted"])

    # ── Panel 2: KDE + Normal overlay showing spike ───────────────────────────
    ax2 = fig.add_subplot(gs[1])
    x = np.linspace(s.min(), max_val * 1.005, 500)
    try:
        kde = stats.gaussian_kde(s, bw_method=0.12)
        ax2.fill_between(x, kde(x), alpha=0.35, color=P["numeric"])
        ax2.plot(x, kde(x), color=P["numeric"], lw=2.2, label="KDE")
    except Exception:
        pass
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), color=P["warn"], lw=1.8,
             ls="--", alpha=0.8, label="Expected (normal)")
    ax2.axvline(max_val, color=P["danger"], lw=2.5, alpha=0.9, label=f"Max={max_val:.0f}")
    ax2.axvline(mu, color="#38bdf8", lw=1.5, ls=":", label=f"Mean={mu:.1f}")
    ax2.set_title("KDE vs Normal Distribution", color=P["text"], fontsize=11)
    ax2.set_xlabel(col, color=P["muted"]); ax2.legend(fontsize=7, labelcolor=P["muted"])

    # ── Panel 3: Empirical vs Normal CDF ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    sorted_s = np.sort(s)
    ecdf     = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    ax3.step(sorted_s, ecdf, color=P["numeric"], lw=2, label="Empirical CDF")
    ax3.plot(sorted_s, stats.norm.cdf(sorted_s, mu, sigma),
             color=P["warn"], lw=1.8, ls="--", label="Normal CDF")
    ax3.axvline(max_val, color=P["danger"], lw=2, alpha=0.8)
    ax3.axhline(1 - pct_ceil, color=P["binary"], lw=1.2, ls=":", alpha=0.9)
    ax3.text(s.min(), 1 - pct_ceil + 0.02, f"{(1-pct_ceil)*100:.1f}%ile",
             color=P["binary"], fontsize=8)
    ax3.set_title("Cumulative Distribution (ECDF)", color=P["text"], fontsize=11)
    ax3.set_xlabel(col, color=P["muted"]); ax3.set_ylabel("Cumulative Prob.", color=P["muted"])
    ax3.legend(fontsize=7, labelcolor=P["muted"])

    # ── Panel 4: Statistics table ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3]); ax4.axis("off")
    rows = [
        ("N total",      f"{len(s):,}"),
        ("Mean",         f"{mu:.2f}"),
        ("Median",       f"{s.median():.2f}"),
        ("Std Dev",      f"{sigma:.2f}"),
        ("Skewness",     f"{s.skew():.3f}"),
        ("Kurtosis",     f"{s.kurtosis():.3f}"),
        ("Q1 (25%)",     f"{s.quantile(0.25):.2f}"),
        ("Q3 (75%)",     f"{s.quantile(0.75):.2f}"),
        ("",             ""),
        ("At Max",       f"{ceil_n:,}  ({pct_ceil:.1%})"),
        ("Ceiling?",     "⚠ YES" if pct_ceil > 0.1 else "✓ No"),
    ]
    ax4.text(0.05, 0.97, f"Descriptive Stats", color=P["text"], fontsize=10.5,
             fontweight="bold", transform=ax4.transAxes, va="top")
    y = 0.88
    for k, v in rows:
        if k:
            ax4.text(0.05, y, k, color=P["muted"], fontsize=8.5, transform=ax4.transAxes, va="top")
            c = P["danger"] if k == "Ceiling?" and "YES" in v else P["text"]
            ax4.text(0.62, y, v, color=c, fontsize=8.5, fontweight="bold",
                     transform=ax4.transAxes, va="top")
        y -= 0.083

    fig.suptitle(f"📊  {col}  —  Ceiling Effect Analysis",
                 color=P["text"], fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_zero_segment(df: pd.DataFrame, col: str, target: str = None) -> plt.Figure:
    """assignments_completed=0 tipi sıfır grubu vs rest karşılaştırması."""
    _setup()
    s = df[col].dropna()
    zmask = df.index.isin(df[df[col] == 0].index)
    zero_s = s[s == 0]; nz_s = s[s > 0]

    has_target = (target and target in df.columns
                  and pd.api.types.is_numeric_dtype(df[target]))
    n_panels = 3 if has_target else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5), facecolor=P["surface"])

    # Panel 1: Overlaid density comparison
    ax = axes[0]
    if len(nz_s) > 1:
        ax.hist(nz_s, bins="auto", color=P["numeric"], alpha=0.7, density=True,
                label=f"{col} > 0  (n={len(nz_s)})")
        try:
            kde = stats.gaussian_kde(nz_s)
            xr  = np.linspace(nz_s.min(), nz_s.max(), 300)
            ax.plot(xr, kde(xr), color=P["numeric"], lw=2)
        except Exception:
            pass
    if len(zero_s) > 0:
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.1
        ax.bar(0, ymax * 0.85, width=nz_s.std() * 0.08 if len(nz_s) > 0 else 1,
               color=P["binary"], alpha=0.85, label=f"{col} = 0  (n={len(zero_s)}, {len(zero_s)/len(s):.0%})")
        ax.annotate(
            f"Zero Group\nn = {len(zero_s)}\n{len(zero_s)/len(s):.1%} of data",
            xy=(0, ymax * 0.6), xytext=(nz_s.mean() * 0.25 if len(nz_s) else 2, ymax * 0.75),
            arrowprops=dict(arrowstyle="->", color=P["binary"], lw=1.5),
            color=P["binary"], fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=P["bg"], edgecolor=P["binary"], lw=1.5),
        )
    ax.set_title(f"{col}: Zero vs Non-Zero", color=P["text"], fontsize=11)
    ax.set_xlabel(col, color=P["muted"]); ax.legend(fontsize=8, labelcolor=P["muted"])

    # Panel 2: Box plot by segment
    ax = axes[1]
    groups = [zero_s.values] if len(zero_s) > 0 else []
    groups.append(nz_s.values)
    labels = ([f"= 0\n(n={len(zero_s)})"] if len(zero_s) > 0 else []) + [f"> 0\n(n={len(nz_s)})"]
    bp = ax.boxplot(groups, labels=labels, patch_artist=True,
                    medianprops=dict(color=P["warn"], lw=2.5),
                    whiskerprops=dict(color=P["muted"]),
                    capprops=dict(color=P["muted"]),
                    flierprops=dict(marker="x", color=P["danger"], ms=5))
    clrs = [P["binary"], P["numeric"]] if len(zero_s) > 0 else [P["numeric"]]
    for patch, c in zip(bp["boxes"], clrs):
        patch.set_facecolor(c); patch.set_alpha(0.4)
    ax.set_title(f"Segment Distribution: {col}", color=P["text"], fontsize=11)
    ax.set_ylabel(col, color=P["muted"])

    # Panel 3: Target outcome by segment (violin + t-test)
    if has_target:
        ax = axes[2]
        z_tgt  = df.loc[df[col] == 0,  target].dropna()
        nz_tgt = df.loc[df[col] > 0,   target].dropna()
        v_data, v_pos = [], []
        if len(z_tgt) > 1:  v_data.append(z_tgt.values);  v_pos.append(0)
        if len(nz_tgt) > 1: v_data.append(nz_tgt.values); v_pos.append(1)
        if v_data:
            parts = ax.violinplot(v_data, positions=v_pos, showmedians=True)
            clrs2 = [P["binary"], P["numeric"]] if len(v_data) == 2 else [P["numeric"]]
            for pc, c in zip(parts.get("bodies", []), clrs2):
                pc.set_facecolor(c); pc.set_alpha(0.5)
        xlbls = []
        if len(z_tgt) > 0:  xlbls.append(f"{col}=0\nμ={z_tgt.mean():.1f}")
        if len(nz_tgt) > 0: xlbls.append(f"{col}>0\nμ={nz_tgt.mean():.1f}")
        ax.set_xticks(list(range(len(xlbls)))); ax.set_xticklabels(xlbls, color=P["muted"], fontsize=9)
        title = f"{target} by {col} group"
        if len(z_tgt) > 1 and len(nz_tgt) > 1:
            try:
                t, p = stats.ttest_ind(z_tgt, nz_tgt)
                sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                title += f"\n(t={t:.2f}, p={p:.3f} {sig})"
            except Exception:
                pass
        ax.set_title(title, color=P["text"], fontsize=10)
        ax.set_ylabel(target, color=P["muted"])

    fig.suptitle(f"🔬  Zero-Group Segment Analysis: {col}",
                 color=P["text"], fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_scatter_matrix(df: pd.DataFrame, cols: list, target: str = None) -> plt.Figure:
    """Pairplot-style scatter matrix with per-cell regression and r annotation."""
    _setup()
    valid = [c for c in cols if c in df.columns
             and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]
    if len(valid) < 2:
        return None
    n   = len(valid)
    sz  = max(2.5, 12 / n)
    fig, axes = plt.subplots(n, n, figsize=(sz * n, sz * n), facecolor=P["surface"])
    fig.suptitle("Pairplot — Scatter Matrix with Regression Lines",
                 color=P["text"], fontsize=13, fontweight="bold", y=1.01)

    for i, rc in enumerate(valid):
        for j, cc in enumerate(valid):
            ax = axes[i][j]
            if i == j:
                # Diagonal: KDE
                s = df[rc].dropna()
                try:
                    kde = stats.gaussian_kde(s)
                    xv  = np.linspace(s.min(), s.max(), 200)
                    ax.fill_between(xv, kde(xv), color=P["numeric"], alpha=0.4)
                    ax.plot(xv, kde(xv), color=P["numeric"], lw=1.8)
                except Exception:
                    ax.hist(s, bins=15, color=P["numeric"], alpha=0.6)
                ax.set_title(rc, color=P["text"], fontsize=8, pad=2)
            else:
                clean = df[[cc, rc]].dropna()
                ax.scatter(clean[cc], clean[rc], color=P["numeric"], alpha=0.22, s=7, rasterized=True)
                try:
                    sl, ic, r, p, _ = stats.linregress(clean[cc], clean[rc])
                    xr = np.linspace(clean[cc].min(), clean[cc].max(), 100)
                    lc = P["warn"] if abs(r) > 0.5 else P["muted"]
                    ax.plot(xr, sl * xr + ic, color=lc, lw=1.8)
                    ax.text(0.05, 0.90, f"r={r:+.2f}", transform=ax.transAxes,
                            color=lc, fontsize=7.5, fontweight="bold", va="top")
                except Exception:
                    pass
            if j == 0: ax.set_ylabel(rc, color=P["muted"], fontsize=7.5)
            else: ax.set_yticklabels([])
            if i == n - 1: ax.set_xlabel(cc, color=P["muted"], fontsize=7.5)
            else: ax.set_xticklabels([])
            ax.tick_params(labelsize=6, colors=P["muted"])

    fig.tight_layout()
    return fig


def render_storytelling_dashboard(df: pd.DataFrame,
                                   target_col: str = None,
                                   ceiling_cols: list = None,
                                   zero_seg_cols: list = None,
                                   pairplot_cols: list = None) -> None:
    """
    Seçili sütunlar için story-driven EDA dashboard render eder.
    Streamlit EDA sekmesinin herhangi bir yerinden çağrılabilir.
    """
    _inject_tufte_css()

    # ── Ceiling Effect Panels ─────────────────────────────────────────────────
    if ceiling_cols:
        for col in ceiling_cols:
            if col not in df.columns: continue
            s = df[col].dropna()
            pct_max = (s == s.max()).mean()
            if pct_max > 0.05:
                st.markdown(f"<div class='insight-danger'>🚨 <b>Ceiling Effect Detected in <code>{col}</code></b> "
                            f"— {pct_max:.1%} of observations sit at the maximum value ({s.max():.4g}). "
                            f"This indicates a truncated distribution; scores above this threshold cannot be captured.</div>",
                            unsafe_allow_html=True)
                fig = plot_ceiling_effect_drama(s, col)
                _show(fig)
            else:
                st.info(f"ℹ️ No significant ceiling effect in `{col}` ({pct_max:.1%} at max).")

    # ── Zero-Group Segment Panels ─────────────────────────────────────────────
    if zero_seg_cols:
        for col in zero_seg_cols:
            if col not in df.columns: continue
            n_zero = int((df[col] == 0).sum())
            if n_zero == 0:
                st.info(f"ℹ️ No zero values in `{col}`."); continue
            pct_z = n_zero / len(df[col].dropna()) * 100
            st.markdown(f"<div class='zero-group-box'>🔬 <b>Zero-Group Found in <code>{col}</code></b> "
                        f"— {n_zero} observations ({pct_z:.1f}%) have value = 0. "
                        f"Analyzing as a separate segment below.</div>",
                        unsafe_allow_html=True)
            fig = plot_zero_segment(df, col, target=target_col)
            _show(fig)

    # ── Pairplot Matrix ───────────────────────────────────────────────────────
    if pairplot_cols and len(pairplot_cols) >= 2:
        valid = [c for c in pairplot_cols if c in df.columns]
        if len(valid) >= 2:
            st.markdown("<div class='section-header'>📐 Pairplot — Variable Relationships</div>",
                        unsafe_allow_html=True)
            fig = plot_scatter_matrix(df, valid, target=target_col)
            if fig: _show(fig)
