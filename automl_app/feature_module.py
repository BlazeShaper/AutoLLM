"""
OutoLLM — Keşifsel Veri Analizi (EDA) Modülü
Korelasyon ısı haritası, dağılım grafikleri, kategorik çubuk grafikleri.
Yalnızca standart kütüphane (pandas, streamlit) kullanılır — ek kurulum gerekmez.
"""
import streamlit as st
import pandas as pd
import math

from logger import log, sidebar_log


# ─── Genel EDA Özet Paneli ───────────────────────────────────────────────────

def render_eda_overview(df: pd.DataFrame):
    """Satır/sütun/eksik/duplikat metriklerini kartlarla gösterir."""
    log.info("EDA genel özet paneli render ediliyor.")
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (df.isnull().sum().sum() / total_cells * 100) if total_cells else 0
    dup_count   = df.duplicated().sum()
    num_cols    = df.select_dtypes(include="number").shape[1]
    cat_cols    = df.select_dtypes(exclude="number").shape[1]

    cols = st.columns(5)
    cols[0].metric("📋 Satır", f"{df.shape[0]:,}")
    cols[1].metric("📊 Sütun", f"{df.shape[1]}")
    cols[2].metric("🔢 Sayısal", f"{num_cols}")
    cols[3].metric("🔤 Kategorik", f"{cat_cols}")
    cols[4].metric("❓ Eksik", f"{missing_pct:.1f}%")

    if dup_count > 0:
        st.warning(f"⚠️ **{dup_count:,}** yinelenen satır tespit edildi.")
    else:
        st.success("✅ Yinelenen satır yok.")


# ─── Korelasyon Isı Haritası ─────────────────────────────────────────────────

def render_correlation_heatmap(df: pd.DataFrame):
    """
    Sayısal sütunlar arasındaki Pearson korelasyonunu
    Streamlit'in built-in veri tablosu + renklendirme ile gösterir.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        st.info("ℹ️ Korelasyon için en az 2 sayısal sütun gerekir.")
        return

    log.info(f"Korelasyon hesaplanıyor: {numeric_df.shape[1]} sayısal sütun")
    corr = numeric_df.corr(numeric_only=True).round(2)

    # Streamlit gradient renklendirme
    st.markdown("#### 🔥 Pearson Korelasyon Matrisi")
    st.dataframe(
        corr.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
        use_container_width=True,
    )

    # Yüksek korelasyonlu çiftleri bul (|r| > 0.85, çapraz hariç)
    high_pairs = []
    cols_list = corr.columns.tolist()
    for i, c1 in enumerate(cols_list):
        for c2 in cols_list[i + 1:]:
            val = corr.loc[c1, c2]
            if abs(val) >= 0.85:
                high_pairs.append((c1, c2, val))

    if high_pairs:
        st.markdown("##### ⚠️ Yüksek Korelasyonlu Çiftler (|r| ≥ 0.85)")
        pair_df = pd.DataFrame(high_pairs, columns=["Sütun A", "Sütun B", "Korelasyon"])
        pair_df = pair_df.sort_values("Korelasyon", key=abs, ascending=False)
        st.dataframe(pair_df, use_container_width=True, hide_index=True)
        st.caption("Bu sütunlardan birini çıkarmak multikolinearite sorununu azaltabilir.")


# ─── Sayısal Sütun Histogramları ─────────────────────────────────────────────

def render_distribution_plots(df: pd.DataFrame):
    """
    Sayısal sütunlar için temel istatistik + streamlit bar_chart ile histogram.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        st.info("ℹ️ Sayısal sütun bulunamadı.")
        return

    log.info(f"Dağılım grafikleri: {numeric_df.shape[1]} sütun")
    st.markdown("#### 📈 Sayısal Sütun Dağılımları")

    cols_to_plot = numeric_df.columns.tolist()
    # max 12 sütun göster
    if len(cols_to_plot) > 12:
        st.caption(f"İlk 12 sayısal sütun gösteriliyor ({len(cols_to_plot)} toplamda).")
        cols_to_plot = cols_to_plot[:12]

    n_cols = 3
    n_rows = math.ceil(len(cols_to_plot) / n_cols)

    for row_i in range(n_rows):
        grid = st.columns(n_cols)
        for col_i in range(n_cols):
            idx = row_i * n_cols + col_i
            if idx >= len(cols_to_plot):
                break
            col_name = cols_to_plot[idx]
            series = df[col_name].dropna()

            with grid[col_i]:
                st.markdown(f"**`{col_name}`**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Ort.", f"{series.mean():.3g}")
                c2.metric("Std", f"{series.std():.3g}")
                c3.metric("Çarpıklık", f"{series.skew():.2f}")

                # Pandas cut ile bin hesapla → bar_chart
                try:
                    counts, bin_edges = pd.cut(series, bins=20, retbins=True)
                    hist_df = pd.DataFrame({
                        "aralık": [f"{b:.2g}" for b in bin_edges[:-1]],
                        "adet": counts.value_counts(sort=False).values,
                    }).set_index("aralık")
                    st.bar_chart(hist_df, height=140)
                except Exception:
                    st.caption("Grafik oluşturulamadı.")


# ─── Kategorik Sütun Grafikleri ──────────────────────────────────────────────

def render_categorical_plots(df: pd.DataFrame):
    """
    Kategorik sütunlar için değer sayıları (bar chart).
    """
    cat_df = df.select_dtypes(exclude="number")
    if cat_df.empty:
        st.info("ℹ️ Kategorik sütun bulunamadı.")
        return

    log.info(f"Kategorik grafikler: {cat_df.shape[1]} sütun")
    st.markdown("#### 🏷️ Kategorik Sütun Dağılımları")

    cols_to_plot = cat_df.columns.tolist()
    if len(cols_to_plot) > 8:
        st.caption(f"İlk 8 kategorik sütun gösteriliyor ({len(cols_to_plot)} toplamda).")
        cols_to_plot = cols_to_plot[:8]

    n_cols = 2
    n_rows = math.ceil(len(cols_to_plot) / n_cols)

    for row_i in range(n_rows):
        grid = st.columns(n_cols)
        for col_i in range(n_cols):
            idx = row_i * n_cols + col_i
            if idx >= len(cols_to_plot):
                break
            col_name = cols_to_plot[idx]
            vc = df[col_name].value_counts().head(15)

            with grid[col_i]:
                n_unique = df[col_name].nunique()
                st.markdown(f"**`{col_name}`** — {n_unique} benzersiz değer")
                bar_df = pd.DataFrame({"adet": vc}).rename_axis("değer")
                st.bar_chart(bar_df, height=160)


# ─── Eksik Veri Isı Haritası ─────────────────────────────────────────────────

def render_missing_heatmap(df: pd.DataFrame):
    """
    Sütun bazında eksik veri yüzdesini gradient bar chart ile gösterir.
    """
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        st.success("✅ Hiçbir sütunda eksik veri yok.")
        return

    st.markdown("#### 🟥 Eksik Veri Yoğunluk Haritası")
    miss_df = pd.DataFrame({
        "Eksik (%)": missing.values
    }, index=missing.index)

    st.dataframe(
        miss_df.style.bar(
            subset=["Eksik (%)"],
            color=["#fee2e2", "#ef4444"],
            vmin=0, vmax=100,
        ),
        use_container_width=True,
    )


# ─── Hedef vs Özellik Scatter Plot ───────────────────────────────────────────

def render_target_scatter(df: pd.DataFrame, target_col: str):
    """
    Seçilen hedef sütun ile sayısal özellikler arasındaki ilişkiyi gösterir.
    Sayısal hedef → st.scatter_chart; Kategorik hedef → gruplu bar chart.
    """
    if target_col not in df.columns:
        st.warning(f"⚠️ Hedef sütun '{target_col}' bulunamadı.")
        return

    numeric_features = [
        c for c in df.select_dtypes(include="number").columns
        if c != target_col
    ]

    if not numeric_features:
        st.info("ℹ️ Scatter için en az 1 sayısal özellik sütunu gerekir.")
        return

    log.info(f"Scatter plot — hedef: {target_col}, özellik sayısı: {len(numeric_features)}")
    st.markdown(f"#### 🎯 Hedef: `{target_col}` vs Özellikler")

    max_show = min(len(numeric_features), 6)
    selected_features = st.multiselect(
        "Görselleştirilecek özellikleri seçin (maks 6):",
        options=numeric_features,
        default=numeric_features[:max_show],
        key="scatter_feature_select",
    )

    if not selected_features:
        st.info("ℹ️ En az bir özellik seçin.")
        return

    is_numeric_target = pd.api.types.is_numeric_dtype(df[target_col])
    n_cols = 2
    n_rows = math.ceil(len(selected_features) / n_cols)

    for row_i in range(n_rows):
        grid = st.columns(n_cols)
        for col_i in range(n_cols):
            idx = row_i * n_cols + col_i
            if idx >= len(selected_features):
                break
            feat = selected_features[idx]

            with grid[col_i]:
                st.markdown(f"**`{feat}`** ↔ **`{target_col}`**")
                plot_df = df[[feat, target_col]].dropna()

                if is_numeric_target:
                    try:
                        st.scatter_chart(plot_df, x=feat, y=target_col, height=220)
                    except Exception:
                        st.caption("Scatter grafiği oluşturulamadı.")
                else:
                    try:
                        group_means = (
                            plot_df.groupby(target_col)[feat]
                            .mean()
                            .sort_values(ascending=False)
                            .head(15)
                        )
                        st.bar_chart(
                            pd.DataFrame({"Ortalama": group_means}),
                            height=220,
                        )
                        st.caption(f"Sınıf bazlı `{feat}` ortalaması")
                    except Exception:
                        st.caption("Grafik oluşturulamadı.")

