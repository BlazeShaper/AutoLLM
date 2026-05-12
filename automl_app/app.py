"""
OutoLLM — Uçtan Uca AutoML Platform
Streamlit tabanlı, modüler yapı.
"""
import streamlit as st
import pandas as pd
import json
import os

from logger import log, sidebar_log
from config import (
    ALL_CLASSIFICATION_MODELS,
    ALL_REGRESSION_MODELS,
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    NUMERIC_IMPUTATION_METHODS,
    CATEGORICAL_IMPUTATION_METHODS,
)
from ui_components import (
    render_header,
    render_mode_toggle,
    render_step_indicator,
    render_dataframe_preview,
    render_warning_banner,
)
from data_module import (
    load_data,
    clean_columns,
    generate_profile,
    detect_cleaning_issues,
    apply_cleaning,
    render_missing_summary,
    remove_duplicates,
    detect_outliers_iqr,
    clip_outliers,
    fill_missing_values,
)
from train_module import (
    detect_task_type,
    check_imbalance,
    run_setup,
    compare_models,
    tune_selected_model,
    save_model_with_card,
    render_model_plots,
    make_pycaret_train_fn,
)
from inference_module import (
    load_model_safe,
    validate_compatibility,
    run_prediction,
    format_results,
    render_prediction_summary,
)
from feature_module import (
    render_eda_overview,
    render_correlation_heatmap,
    render_distribution_plots,
    render_categorical_plots,
    render_missing_heatmap,
    render_target_scatter,
    render_storytelling_dashboard,
)
from optimization_module import render_smart_hyperparams

# ─── Sayfa Yapılandırması ────────────────────────────────────────────────────

st.set_page_config(
    page_title="OutoLLM — AutoML Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session State Başlatma ──────────────────────────────────────────────────

_defaults = {
    "raw_df": None,
    "cleaned_df": None,
    "task_type": None,
    "setup_obj": None,
    "pc_module": None,
    "leaderboard": None,
    "best_model": None,
    "model_card": None,
    "target_col": None,
    "mode": "standard",
    "smart_params": None,   # Optuna'dan gelen en iyi parametreler
    "opt_report": None,     # Tam optimizasyon raporua
    "manual_params": None,  # Kullanıcının manuel hiper-parametreleri
    "manual_model_selection": None,  # Kullanıcının seçtiği model listesi
    "profile_html": None,            # Oluşturulan profil raporu HTML içeriği
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── Sidebar + Header ────────────────────────────────────────────────────────

render_mode_toggle()
render_header()

# ─── Sekmeler ────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📂 1. Yükleme",
    "🧹 2. Temizlik",
    "🧠 3. Eğitim",
    "📊 4. Değerlendirme",
    "🔮 5. Tahmin",
    "🔍 6. EDA Analizi",
])

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 1 — Veri Yükleme
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    render_step_indicator(0)
    st.subheader("Veri Yükleme")

    uploaded_file = st.file_uploader(
        "CSV veya Excel dosyası seçin",
        type=["csv", "xlsx", "xls"],
        help="Maksimum dosya boyutu: 50 MB",
    )

    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("current_file_key") != file_key:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state["raw_df"] = df
                st.session_state["cleaned_df"] = df.copy()
                st.session_state["current_file_key"] = file_key
                # Yeni dosya yüklendiğinde eski eğitim verilerini temizle
                for k in ["setup_obj", "pc_module", "task_type", "leaderboard", "best_model", "target_col"]:
                    st.session_state[k] = None

        if st.session_state.get("raw_df") is not None:
            render_dataframe_preview(st.session_state["raw_df"])

            st.markdown("---")
            if st.button("📊 Hızlı Profil Raporu Oluştur", key="btn_profile"):
                html_content = generate_profile(st.session_state["raw_df"])
                if html_content:
                    st.session_state["profile_html"] = html_content

            # Profil daha önce oluşturulduysa indirme butonunu göster
            if st.session_state.get("profile_html"):
                st.download_button(
                    label="⬇️ Profil Raporunu İndir (.html)",
                    data=st.session_state["profile_html"].encode("utf-8"),
                    file_name="veri_profili.html",
                    mime="text/html",
                    key="btn_download_profile",
                    use_container_width=True,
                )

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 2 — Veri Temizliği
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    render_step_indicator(1)
    st.subheader("Veri Temizliği")

    if st.session_state["raw_df"] is None:
        st.info("ℹ️ Lütfen önce **1. Yükleme** sekmesinden veri yükleyin.")
    else:
        df = st.session_state["cleaned_df"]
        raw_df = st.session_state["raw_df"]

        # ── Ham vs Temizlenmiş Karşılaştırma ─────────────────────────────
        st.markdown("### 📊 Veri Durumu")
        _cmp1, _cmp2, _cmp3, _cmp4, _cmp5 = st.columns(5)
        _cmp1.metric("🔢 Satır (Ham)",      f"{raw_df.shape[0]:,}")
        _cmp2.metric("🔢 Satır (Güncel)",   f"{df.shape[0]:,}",
                     delta=f"{df.shape[0] - raw_df.shape[0]:+,}" if df.shape[0] != raw_df.shape[0] else None)
        _cmp3.metric("📋 Sütun (Ham)",      f"{raw_df.shape[1]}")
        _cmp4.metric("📋 Sütun (Güncel)",   f"{df.shape[1]}",
                     delta=f"{df.shape[1] - raw_df.shape[1]:+}" if df.shape[1] != raw_df.shape[1] else None)
        _miss_pct = df.isnull().sum().sum() / max(df.shape[0] * df.shape[1], 1) * 100
        _cmp5.metric("❓ Eksik Veri", f"{_miss_pct:.1f}%")
        st.markdown("---")

        # ── Eksik veri özeti + Doldurma
        render_missing_summary(df)

        _has_missing = df.isnull().sum().sum() > 0
        if _has_missing:
            with st.expander("🧰 Eksik Veri Doldurma Seçenekleri", expanded=False):
                ic1, ic2, ic3 = st.columns([2, 2, 1])
                with ic1:
                    num_method = st.selectbox(
                        "Sayısal doldurma:",
                        ["mean", "median", "zero", "sabit"],
                        key="imp_num",
                    )
                with ic2:
                    cat_method = st.selectbox(
                        "Kategorik doldurma:",
                        ["mode", "sabit"],
                        key="imp_cat",
                    )
                with ic3:
                    fill_val = st.text_input("Sabit değer:", value="0", key="imp_val")

                if st.button("🧰 Eksik Verileri Doldur", key="btn_impute"):
                    filled_df = fill_missing_values(
                        df,
                        numeric_method=num_method,
                        categorical_method=cat_method,
                        fill_value=fill_val,
                    )
                    st.session_state["cleaned_df"] = filled_df.copy()
                    for k in ["setup_obj", "pc_module", "task_type", "leaderboard", "best_model"]:
                        st.session_state[k] = None
                    df = filled_df
                    remaining = filled_df.isnull().sum().sum()
                    st.success(
                        f"✅ Eksik değerler dolduruldu. "
                        f"Kalan eksik hücre: **{remaining}**"
                    )
                    st.rerun()
        st.markdown("---")

        # ── Duplikat Kaldırma
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            st.markdown("### 🗑️ Yinelenen Satırlar")
            st.warning(f"⚠️ **{dup_count:,}** satır birebir yineleniyor.")
            if st.button(f"🗑️ {dup_count:,} Yinelenen Satırı Kaldır", key="btn_drop_dup"):
                cleaned_dup, removed = remove_duplicates(df)
                st.session_state["cleaned_df"] = cleaned_dup.copy()
                for k in ["setup_obj", "pc_module", "task_type", "leaderboard", "best_model"]:
                    st.session_state[k] = None
                st.success(f"✅ {removed:,} yinelenen satır kaldırıldı. Kalan: **{len(cleaned_dup):,}** satır.")
                st.rerun()
        else:
            st.success("✅ Yinelenen satır yok.")

        st.markdown("---")

        # ── Aykırı Değer Tespiti (IQR)
        st.markdown("### ⚠️ Aykırı Değer Tespiti (IQR Yöntemi)")
        outlier_report = detect_outliers_iqr(df)

        if not outlier_report:
            st.success("✅ Aykırı değer tespit edilmedi.")
        else:
            st.markdown(
                f"⚠️ **{len(outlier_report)} sütun**da aykırı değer tespit edildi. "
                "Kırpma uygulamak istediğiniz sütunları seçin:"
            )
            cols_to_clip = []
            for col, info in outlier_report.items():
                oc1, oc2 = st.columns([0.5, 5])
                with oc1:
                    if st.checkbox("Kırp", key=f"clip_{col}"):
                        cols_to_clip.append(col)
                with oc2:
                    st.markdown(
                        f"<code>{col}</code>: "
                        f"**{info['outlier_count']}** aykırı değer "
                        f"(%{info['outlier_pct']}) — "
                        f"Alt sınır: `{info['lower_bound']}`, "
                        f"Üst sınır: `{info['upper_bound']}`",
                        unsafe_allow_html=True,
                    )

            _btn_c1, _btn_c2 = st.columns(2)
            with _btn_c1:
                if cols_to_clip:
                    if st.button("✂️ Seçili Sütunlarda Outlier Kırp", key="btn_clip"):
                        clipped_df = clip_outliers(df, cols_to_clip)
                        st.session_state["cleaned_df"] = clipped_df.copy()
                        for k in ["setup_obj", "pc_module", "task_type", "leaderboard", "best_model"]:
                            st.session_state[k] = None
                        st.success(f"✅ {len(cols_to_clip)} sütunda aykırı değerler IQR sınırlarına kırpıldı.")
                        st.rerun()
            with _btn_c2:
                if st.button(
                    f"⚡ Tümünü Otomatik Kırp ({len(outlier_report)} sütun)",
                    key="btn_clip_all",
                    type="primary",
                    help="Tüm sütunlardaki aykırı değerleri IQR yöntemiyle tek seferde kırpar.",
                ):
                    all_cols = list(outlier_report.keys())
                    clipped_df = clip_outliers(df, all_cols)
                    st.session_state["cleaned_df"] = clipped_df.copy()
                    for k in ["setup_obj", "pc_module", "task_type", "leaderboard", "best_model"]:
                        st.session_state[k] = None
                    st.success(f"✅ {len(all_cols)} sütunun tamamında aykırı değerler kırpıldı!")
                    st.rerun()

        st.markdown("---")

        # ── 🚑 R² Kurtarma Merkezi ────────────────────────────────────────────
        st.markdown("### 🚑 R² Kurtarma Merkezi")
        st.caption("Model doğruluğu düşükse (R² < 0.3) bu araçlarla veriyi iyileştirin.")

        with st.expander("🔬 Hedef Sütunu Dağılım Analizi (Skewness)", expanded=True):
            import numpy as np
            _target_saved = st.session_state.get("target_col")
            if _target_saved and _target_saved in df.columns and pd.api.types.is_numeric_dtype(df[_target_saved]):
                _y = df[_target_saved].dropna()
                _skew = float(_y.skew())
                _kurt = float(_y.kurtosis())

                sk_c1, sk_c2, sk_c3 = st.columns(3)
                sk_c1.metric("📐 Çarpıklık (Skewness)", f"{_skew:.3f}",
                             help="0'a yakın = normal. |skew| > 1 = çarpık dağılım.")
                sk_c2.metric("📈 Basıklık (Kurtosis)",  f"{_kurt:.3f}")
                sk_c3.metric("📊 Min / Max",
                             f"{_y.min():.2f} / {_y.max():.2f}")

                if abs(_skew) > 1.0:
                    st.warning(
                        f"⚠️ Hedef sütun **{_target_saved}** çarpık dağılım gösteriyor "
                        f"(skewness={_skew:.2f}). "
                        "Log dönüşümü R²'yi önemli ölçüde artırabilir."
                    )
                    if _y.min() > 0:
                        if st.button(
                            f"📉 '{_target_saved}' sütununa log1p() uygula",
                            key="btn_log1p_target",
                            type="primary",
                        ):
                            log1p_df = df.copy()
                            log1p_df[_target_saved] = np.log1p(log1p_df[_target_saved])
                            st.session_state["cleaned_df"] = log1p_df
                            for k in ["setup_obj", "pc_module", "task_type", "leaderboard", "best_model"]:
                                st.session_state[k] = None
                            st.success(
                                f"✅ `np.log1p({_target_saved})` uygulandı! "
                                "Yeni skewness: "
                                f"**{float(log1p_df[_target_saved].skew()):.3f}**"
                            )
                            st.rerun()
                    else:
                        st.info(
                            "ℹ️ Hedef sütun sıfır veya negatif değerler içeriyor. "
                            "Eğitim sekmesindeki **'📉 Hedef Log Dönüşümü'** toggle'ını kullanın."
                        )
                else:
                    st.success(f"✅ Hedef dağılımı normale yakın (skewness={_skew:.2f}). Log dönüşümü gerekmez.")
            else:
                st.info(
                    "ℹ️ Hedef sütunu analiz için **3. Eğitim** sekmesinden hedefi seçin, "
                    "ardından bu sekmeye geri dönün."
                )

        with st.expander("⚙️ Eğitim Sekmesi Hızlı Öneri Listesi", expanded=False):
            st.markdown("""
**R² = 0.1 için kanıtlanmış çözümler (Eğitim → Ön İşleme panelinden uygulayın):**

| # | Öneri | Açıklama |
|---|-------|----------|
| 1 | ✅ **Normalize Et** | Sayısal özellikleri aynı ölçeğe getirir — lineer modeller için kritik |
| 2 | ✅ **Aykırı Değerleri Temizle** | PyCaret'in kendi outlier temizleyicisi ek koruma sağlar |
| 3 | ✅ **Özellik Seçimi (%80)** | Gürültülü sütunları eler, sinyal/gürültü oranını artırır |
| 4 | ✅ **Çoklu Doğrusallığı Gider** | Korelasyonlu özellikleri temizler, lineer modelleri iyileştirir |
| 5 | ✅ **Hedef Log Dönüşümü** | Çarpık hedef dağılımları için — yukarıda otomatik uygulayın |

> 💡 **Tavsiye:** Tüm 5 seçeneği açıp "Eğitimi Başlat"a basın. Çoğu durumda R² 0.1 → 0.6+ olur.
""")

        st.markdown("---")

        # ── Sorun tespiti (sütun bazlı)
        st.markdown("### 🔎 Sütun Sorunları")
        issues = detect_cleaning_issues(df)

        if not issues:
            st.success("✅ Veri setinizde belirgin bir sütun sorunu tespit edilmedi.")
        else:
            st.markdown(
                f"⚠️ **{len(issues)} sütun** için potansiyel sorun tespit edildi. "
                "Çıkarmak istediklerinizi işaretleyin:"
            )
            cols_to_drop = []

            for col, info in issues.items():
                severity = info.get("severity", "medium")
                badge_color = "#ef4444" if severity == "high" else "#f59e0b"
                badge_label = "Yüksek Risk" if severity == "high" else "Orta Risk"

                c1, c2, c3 = st.columns([0.5, 1.5, 3])
                with c1:
                    if st.checkbox("Kaldır", key=f"drop_{col}"):
                        cols_to_drop.append(col)
                with c2:
                    st.markdown(
                        f"<b style='color:{badge_color};'>[{badge_label}]</b> <code>{col}</code>",
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(f"{info['reason']} → *{info['action']}*")

            st.markdown("---")
            if st.button("🧹 Seçili Sütunları Kaldır", key="btn_clean"):
                cleaned = apply_cleaning(df, cols_to_drop)
                st.session_state["cleaned_df"] = cleaned.copy()
                # Silinen sütunlar varsa hedef sütun referansını da sıfırla
                for k in ["setup_obj", "pc_module", "task_type", "leaderboard", "best_model", "target_col"]:
                    st.session_state[k] = None
                st.success(
                    f"✅ Temizlik uygulandı. "
                    f"Kalan sütun sayısı: **{cleaned.shape[1]}** "
                    f"(çıkarılan: **{len(cols_to_drop)}**)"
                )
                st.rerun()

        # ── Manuel Sütun Silme
        st.markdown("### 🗑️ Manuel Sütun Silme")
        st.caption("Veri setinden çıkarmak istediğiniz sütunları seçin.")
        
        cols_to_manually_drop = st.multiselect(
            "Kaldırılacak sütunlar:",
            options=list(df.columns),
            key="manual_col_drop_select"
        )
        if cols_to_manually_drop:
            if st.button(f"🗑️ Seçili {len(cols_to_manually_drop)} Sütunu Kaldır", key="btn_manual_drop"):
                cleaned = df.drop(columns=cols_to_manually_drop)
                st.session_state["cleaned_df"] = cleaned.copy()
                for k in ["setup_obj", "pc_module", "task_type", "leaderboard", "best_model"]:
                    st.session_state[k] = None
                
                if st.session_state.get("target_col") in cols_to_manually_drop:
                    st.session_state["target_col"] = None
                    
                st.success(f"✅ Seçilen {len(cols_to_manually_drop)} sütun başarıyla kaldırıldı.")
                st.rerun()

        # ── Güncel Veri Önizlemesi (her zaman en altta göster) ───────────────
        st.markdown("---")
        st.markdown("### 📋 Güncel Veri Önizlemesi")
        st.caption(
            f"Temizlik uygulandıktan sonraki veri — "
            f"**{df.shape[0]:,} satır × {df.shape[1]} sütun**"
        )

        _prev_c1, _prev_c2 = st.columns([1, 3])
        with _prev_c1:
            _n_preview = st.number_input(
                "Gösterilecek satır:",
                min_value=5, max_value=min(500, df.shape[0]),
                value=min(20, df.shape[0]),
                step=5,
                key="preview_n_rows",
            )
        with _prev_c2:
            _num_cols = df.select_dtypes(include="number").columns.tolist()
            _search_col = st.selectbox(
                "Sütun istatistiği gör:",
                ["— Seçin —"] + _num_cols,
                key="preview_col_stat",
            )

        st.dataframe(df.head(int(_n_preview)), use_container_width=True)

        if _search_col and _search_col != "— Seçin —":
            _s = df[_search_col].dropna()
            _sc1, _sc2, _sc3, _sc4, _sc5 = st.columns(5)
            _sc1.metric("Min",    f"{_s.min():.4f}")
            _sc2.metric("Max",    f"{_s.max():.4f}")
            _sc3.metric("Ortalama", f"{_s.mean():.4f}")
            _sc4.metric("Medyan", f"{_s.median():.4f}")
            _sc5.metric("Std",    f"{_s.std():.4f}")

        with st.expander("📥 Temizlenmiş Veriyi İndir", expanded=False):
            import io as _io
            _csv_buf = _io.StringIO()
            df.to_csv(_csv_buf, index=False)
            st.download_button(
                label="⬇️ CSV olarak indir",
                data=_csv_buf.getvalue().encode("utf-8"),
                file_name="cleaned_data.csv",
                mime="text/csv",
                key="btn_download_cleaned",
                use_container_width=True,
            )

# SEKME 3 — Model Eğitimi
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    render_step_indicator(2)
    st.subheader("Model Eğitimi")

    if st.session_state["cleaned_df"] is None:
        st.info("ℹ️ Lütfen önce veri yükleyin.")
    else:
        df = st.session_state["cleaned_df"]

        # Mevcut cleaned_df sütunlarını al — temizlik sonrası güncel liste
        available_cols = list(st.session_state["cleaned_df"].columns)

        # Kaydedilmiş hedef sütun artık mevcut değilse sıfırla
        if st.session_state.get("target_col") not in available_cols:
            st.session_state["target_col"] = None

        # Dropdown index'ini güvenli hesapla — silinmiş sütun seçili kalmasın
        _prev_target = st.session_state.get("target_col") or ""
        _opts        = [""] + available_cols
        _idx         = _opts.index(_prev_target) if _prev_target in _opts else 0

        target_col = st.selectbox(
            "🎯 Hedef (Target) Sütunu Seçin",
            _opts,
            index=_idx,
            key="target_col_select",
            help="Tahmin etmek istediğiniz sütunu seçin. Kümeleme için boş bırakın.",
        )

        if target_col:
            # session_state'e kaydet
            st.session_state["target_col"] = target_col

            try:
                task_type, msg = detect_task_type(df, target_col)
                st.info(f"🤖 **Otomatik Analiz:** {msg}")
                st.session_state["task_type"] = task_type

                check_imbalance(df, target_col, task_type)

                st.markdown("### ⚙️ Eğitim Ayarları")
                imputation_dict = {}

                # ── Veri Ön İşleme Seçenekleri (her modda görünür) ──────────
                with st.expander("🔧 Veri Ön İşleme Seçenekleri", expanded=True):
                    st.markdown(
                        "<small style='color:#9ca3af;'>Aşağıdaki seçenekler PyCaret setup() fonksiyonuna "
                        "doğrudan iletilir ve model başarımını önemli ölçüde artırabilir.</small>",
                        unsafe_allow_html=True,
                    )
                    pp_col1, pp_col2 = st.columns(2)
                    with pp_col1:
                        do_normalize = st.toggle(
                            "📐 Veriyi Normalize Et",
                            value=False,
                            key="pp_normalize",
                            help="Sayısal özellikleri 0-1 aralığına ölçekler (normalize=True). "
                                 "Lineer modeller (Ridge, Lasso, SVM) için özellikle önemlidir.",
                        )
                        do_feature_selection = st.toggle(
                            "🎯 Otomatik Özellik Seçimi",
                            value=False,
                            key="pp_feature_selection",
                            help="En bilgilendirici %80 özelliği tutar, geri kalanı eler "
                                 "(feature_selection=True, n_features_to_select=0.8). "
                                 "Gürültülü/ilgisiz sütunları modelden uzaklaştırır.",
                        )
                    with pp_col2:
                        do_remove_outliers = st.toggle(
                            "🚫 Aykırı Değerleri Temizle",
                            value=False,
                            key="pp_remove_outliers",
                            help="Aykırı değerleri eğitim verisinden otomatik çıkarır "
                                 "(remove_outliers=True). Regresyon R² skorunu artırır.",
                        )
                        do_pca = st.toggle(
                            "📉 PCA Uygula",
                            value=False,
                            key="pp_pca",
                            help="Boyut indirgeme için PCA uygular (pca=True). "
                                 "Çok sayıda özellik olduğunda eğitimi hızlandırır.",
                        )

                    st.markdown("---")
                    pp_col3, pp_col4 = st.columns(2)
                    with pp_col3:
                        do_remove_multicollinearity = st.toggle(
                            "🔗 Çoklu Doğrusallığı Gider",
                            value=False,
                            key="pp_multicollinearity",
                            help="Birbirleriyle yüksek korelasyonlu (çok doğrusal) özelliklerden "
                                 "birini otomatik çıkarır (remove_multicollinearity=True). "
                                 "Lineer modellerin tahmin gücünü önemli ölçüde artırır.",
                        )
                    with pp_col4:
                        st.caption(
                            "💡 **İpucu:** Çoklu doğrusallık + Özellik Seçimi birlikte "
                            "kullanıldığında gürültü en aza iner."
                        )

                    # Regresyon için Log Dönüşümü satırı
                    if task_type == "regression":
                        st.markdown("---")
                        pp_col5, pp_col6 = st.columns(2)
                        with pp_col5:
                            do_log_transform = st.toggle(
                                "📉 Hedef Log Dönüşümü (log1p)",
                                value=False,
                                key="pp_log_transform",
                                help="Eğitimden önce hedef sütuna numpy.log1p() uygular. "
                                     "Power-law / log-normal dağılımlı (sağa çarpık) hedeflerde "
                                     "R² skorunu ciddi ölçüde artırabilir. "
                                     "Hedef sütun tamamen pozitif değerler içermelidir.",
                            )
                        with pp_col6:
                            st.caption(
                                "💡 EDA → ‘🎯 Hedef vs Özellik’ sekmesinde "
                                "hedef histogramı sağa çarpıksa bu seçeneği aç."
                            )
                    else:
                        do_log_transform = False

                    # Aktif seçenekleri bilgi olarak göster
                    active_opts = []
                    if do_normalize:                  active_opts.append("`normalize`")
                    if do_remove_outliers:            active_opts.append("`remove_outliers`")
                    if do_feature_selection:          active_opts.append("`feature_selection (n=0.8)`")
                    if do_pca:                        active_opts.append("`pca`")
                    if do_remove_multicollinearity:   active_opts.append("`remove_multicollinearity`")
                    if do_log_transform:              active_opts.append("`log1p(target)`")
                    if active_opts:
                        st.success(f"✅ Aktif ön işleme: {', '.join(active_opts)}")
                    else:
                        st.info("ℹ️ Ön işleme seçilmedi — ham veri kullanılacak.")

                # Seçimleri dict'e topla
                preprocessing_opts = {
                    "normalize":                  do_normalize,
                    "remove_outliers":            do_remove_outliers,
                    "feature_selection":          do_feature_selection,
                    "pca":                        do_pca,
                    "remove_multicollinearity":   do_remove_multicollinearity,
                    "log_transform_target":       do_log_transform,
                }

                # ── Manuel Model Seçimi + Hiper-Parametre Paneli ─────────────────
                with st.expander("🎛️ Manuel Model Seçimi & Hiper-Parametreler", expanded=True):
                    st.markdown(
                        "<small style='color:#9ca3af;'>Eğitilecek modelleri seçin ve hiperparametreleri kendiniz belirleyin.</small>",
                        unsafe_allow_html=True,
                    )

                    # Model listesini görev tipine göre belirle
                    _all_mdl = (
                        ALL_CLASSIFICATION_MODELS
                        if task_type == "classification"
                        else ALL_REGRESSION_MODELS
                    )
                    _def_mdl = (
                        ["lr", "dt", "rf", "knn", "nb"]
                        if task_type == "classification"
                        else ["lr", "dt", "rf", "knn", "en"]
                    )

                    _saved_sel = st.session_state.get("manual_model_selection") or []
                    _saved_sel = [m for m in _saved_sel if m in _all_mdl]  # Geçersizleri temizle

                    selected_manual_models = st.multiselect(
                        "🤖 Eğitilecek Modeller",
                        options=_all_mdl,
                        default=_saved_sel if _saved_sel else _def_mdl,
                        key="manual_model_select",
                        help="Birden fazla model seçebilirsiniz. En iyi model otomatik belirlenir.",
                    )
                    if not selected_manual_models:
                        selected_manual_models = _def_mdl
                    st.session_state["manual_model_selection"] = selected_manual_models

                    st.markdown("---")
                    st.markdown(
                        "<b style='color:#c7d2fe;'>⚙️ Hiper-Parametreler</b> "
                        "<small style='color:#9ca3af;'>(MLP · GBM · XGBoost · LightGBM · CatBoost · Ridge/Lasso/ElasticNet için uygulanır)</small>",
                        unsafe_allow_html=True,
                    )

                    _hp1, _hp2, _hp3, _hp4 = st.columns(4)
                    with _hp1:
                        _saved_mp = st.session_state.get("manual_params") or {}
                        manual_epochs = st.number_input(
                            "🔁 Epoch / max_iter",
                            min_value=10, max_value=2000,
                            value=int(_saved_mp.get("epochs", 200)),
                            step=50, key="hp_epochs",
                            help="Toplam eğitim adımı sayısı (MLP, Boosting vb.)",
                        )
                    with _hp2:
                        manual_batch = st.number_input(
                            "📦 Batch Size",
                            min_value=4, max_value=1024,
                            value=int(_saved_mp.get("batch_size", 32)),
                            step=8, key="hp_batch",
                            help="Her adımda işlenen örnek sayısı (MLP)",
                        )
                    with _hp3:
                        manual_lr = st.number_input(
                            "📈 Learning Rate",
                            min_value=0.0001, max_value=1.0,
                            value=float(_saved_mp.get("learning_rate", 0.01)),
                            step=0.001, format="%.4f", key="hp_lr",
                            help="Öğrenme hızı (Boosting, MLP, Lojistik Reg.)",
                        )
                    with _hp4:
                        manual_reg = st.number_input(
                            "🔒 Regularization",
                            min_value=0.0, max_value=100.0,
                            value=float(_saved_mp.get("regularization", 0.1)),
                            step=0.01, format="%.4f", key="hp_reg",
                            help="Düzenleştirme katsayısı (alpha, l2 vb.)",
                        )

                    manual_params = {
                        "epochs":        int(manual_epochs),
                        "batch_size":    int(manual_batch),
                        "learning_rate": float(manual_lr),
                        "regularization": float(manual_reg),
                    }
                    st.session_state["manual_params"] = manual_params

                    # Özet kutusu
                    st.info(
                        f"🎛️ **Seçili modeller ({len(selected_manual_models)}):** "
                        f"{', '.join(f'`{m}`' for m in selected_manual_models)}  \n"
                        f"⚙️ Epoch: **{manual_epochs}** | Batch: **{manual_batch}** | "
                        f"LR: **{manual_lr}** | Reg: **{manual_reg}**"
                    )

                if st.session_state["mode"] == "expert":
                    c1, c2 = st.columns(2)
                    with c1:
                        num_imp = st.selectbox(
                            "Sayısal Eksik Veri Doldurma:", NUMERIC_IMPUTATION_METHODS
                        )
                        imputation_dict["numeric"] = num_imp
                    with c2:
                        cat_imp = st.selectbox(
                            "Kategorik Eksik Veri Doldurma:", CATEGORICAL_IMPUTATION_METHODS
                        )
                        imputation_dict["categorical"] = cat_imp

                    # ── Akıllı Hiperparametre Motoru — Gerçek PyCaret Entegrasyonu ──
                    # PyCaret setup tamamlandıysa gerçek train_fn üret; yoksa güvenli fallback kullan.
                    _pc_module_live  = st.session_state.get("pc_module")
                    _task_type_live  = task_type  # yukarıda detect_task_type'tan gelir

                    # Hangi metriği optimize edeceğiz?
                    from config import CLASSIFICATION_METRICS, REGRESSION_METRICS
                    _default_metric = (
                        "Accuracy" if _task_type_live == "classification" else "R2"
                    )
                    # Eğer Tab4'te seçilmiş bir metrik varsa onu kullan
                    _opt_metric = st.session_state.get("_optuna_metric", _default_metric)

                    if _pc_module_live is not None:
                        # Setup tamamlandı → gerçek PyCaret train_fn
                        from config import (
                            ALL_CLASSIFICATION_MODELS,
                            ALL_REGRESSION_MODELS,
                            DEFAULT_CLASSIFICATION_MODELS,
                            DEFAULT_REGRESSION_MODELS,
                        )
                        _include = (
                            DEFAULT_CLASSIFICATION_MODELS
                            if _task_type_live == "classification"
                            else DEFAULT_REGRESSION_MODELS
                        )
                        _pycaret_train_fn = make_pycaret_train_fn(
                            _pc_module_live, _include, _opt_metric
                        )
                        st.caption(
                            f"🔗 Optuna → PyCaret entegrasyonu **aktif** "
                            f"(metrik: `{_opt_metric}`, {len(_include)} model)"
                        )

                        # Optuna metriği için ayrı selectbox (Tab3 içinde)
                        _metrics_list = (
                            CLASSIFICATION_METRICS
                            if _task_type_live == "classification"
                            else REGRESSION_METRICS
                        )
                        _chosen_metric = st.selectbox(
                            "🎯 Optuna Optimizasyon Metriği",
                            _metrics_list,
                            index=(_metrics_list.index(_opt_metric)
                                   if _opt_metric in _metrics_list else 0),
                            key="hp_optuna_metric_tab3",
                            help="Optuna bu metriği maksimize edecek; eğer setup tamamsa "
                                 "PyCaret leaderboard skoru ile bire bir aynı metrik kullanılır.",
                        )
                        # Seçimi session_state'e kaydet
                        st.session_state["_optuna_metric"] = _chosen_metric

                        # train_fn'i seçilen metriğe göre yeniden üret (metrik değişmişse)
                        if _chosen_metric != _opt_metric:
                            _pycaret_train_fn = make_pycaret_train_fn(
                                _pc_module_live, _include, _chosen_metric
                            )

                        opt_results = render_smart_hyperparams(df, target_col, _pycaret_train_fn)

                        # Optuna bitti ve optimized parametreler varsa session_state'e kaydet
                        if opt_results and opt_results.get("optimized"):
                            st.session_state["opt_report"]   = opt_results
                            st.session_state["smart_params"] = opt_results["optimized"]
                    else:
                        # Hata 2 Çözümü: Setup henüz yapılmadı → fallback kullanma, sadece uyarı göster ve paneli gizle.
                        st.warning("⚠️ Optuna için önce 'Eğitimi Başlat' butonuna basın.")

                st.markdown("---")
                if st.button("🚀 Eğitimi Başlat", key="btn_train"):
                    with st.status("Pipeline çalışıyor…", expanded=True) as status:
                        st.write("⏳ PyCaret setup başlatılıyor…")
                        try:
                            # 1. Standardize features before training
                            df = clean_columns(df)
                            import re
                            target_col = re.sub(r"[^\w]", "", target_col.strip().lower().replace(" ", "_"))
                            
                            setup_obj, pc_module = run_setup(
                                df, target_col, task_type, imputation_dict,
                                st.session_state["mode"],
                                preprocessing_opts=preprocessing_opts,
                            )
                            st.session_state["setup_obj"] = setup_obj
                            st.session_state["pc_module"] = pc_module
                            st.write("✅ Setup tamamlandı.")

                            # Manuel seçim yoksa mode'a göre default kullan
                            _sel_models = st.session_state.get("manual_model_selection") or None

                            st.write(f"⏳ {len(_sel_models or [])} model karşılaştırılıyor…")
                            best_model, leaderboard = compare_models(
                                task_type, _sel_models,
                                st.session_state["mode"], pc_module,
                                smart_params=st.session_state.get("smart_params"),
                                manual_params=st.session_state.get("manual_params"),
                            )
                            st.session_state["best_model"] = best_model
                            st.session_state["leaderboard"] = leaderboard

                            # ── Optuna (smart_params) tune adımı — SADECE expert modda
                            _smart = st.session_state.get("smart_params")
                            if (
                                best_model is not None
                                and st.session_state["mode"] == "expert"
                                and task_type in ("classification", "regression")
                                and _smart  # manuel tune zaten compare_models içinde yapıldı
                            ):
                                _tune_metric = st.session_state.get(
                                    "_optuna_metric",
                                    "Accuracy" if task_type == "classification" else "R2",
                                )
                                st.write(
                                    f"⚙️ Optuna parametreleri tune_model'e aktarılıyor "
                                    f"(metrik: `{_tune_metric}`)…"
                                )
                                try:
                                    tuned_model, tune_lb = tune_selected_model(
                                        best_model, task_type, n_iter=20,
                                        optimize_metric=_tune_metric,
                                        pc_module=pc_module,
                                        smart_params=_smart,
                                    )
                                    if tuned_model is not None:
                                        st.session_state["best_model"] = tuned_model
                                        best_model = tuned_model
                                        if tune_lb is not None and not tune_lb.empty:
                                            st.session_state["leaderboard"] = tune_lb
                                            leaderboard = tune_lb
                                        st.write("✅ Optuna tune tamamlandı.")
                                except Exception as _te:
                                    log.warning(f"Optuna tune atlandı: {_te}")
                                    st.warning(f"⚠️ Optuna tune atlandı: {_te}")

                            status.update(
                                label="✅ Eğitim tamamlandı!", state="complete"
                            )
                        except Exception as e:
                            status.update(label="❌ Hata oluştu", state="error")
                            log.exception(f"Eğitim pipeline hatası: {e}")
                            sidebar_log(f"❌ Pipeline hatası: {type(e).__name__}: {e}", "error")
                            st.error(str(e))

            except ValueError as e:
                log.warning(f"Değer hatası (hedef seçimi): {e}")
                st.error(str(e))
            except Exception as e:
                log.exception(f"Tab3 beklenmeyen hata: {e}")
                sidebar_log(f"❌ Beklenmeyen hata: {e}", "error")
                st.error(f"❌ Beklenmeyen hata oluştu: {e}")
        else:
            st.info("ℹ️ Kümeleme (Clustering) için hedef sütun seçmeden devam edebilirsiniz. "
                    "(Bu özellik geliştirilmeye devam etmektedir.)")

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 4 — Değerlendirme & Dışa Aktarma
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    render_step_indicator(3)
    st.subheader("Model Sonuçları ve Dışa Aktarma")

    if st.session_state["leaderboard"] is None or st.session_state["best_model"] is None:
        st.info("ℹ️ Lütfen önce **3. Eğitim** sekmesinden modeli eğitin.")
    else:
        task_type   = st.session_state["task_type"]

        if task_type == "clustering":
            st.info(
                "🔹 Kümeleme (Clustering) sonucu — model ve liderlik tablosu aşağıda gösterilmektedir. "
                "Model grafikleri için aşağıdaki **Model Grafikleri** bölümünü açın."
            )

        st.markdown("### 🏆 Liderlik Tablosu")
        _lb = st.session_state["leaderboard"]
        st.dataframe(_lb, use_container_width=True)

        # ── Model Karşılaştırma Çubuk Grafiği
        if task_type in ("classification", "regression") and _lb is not None and not _lb.empty:
            st.markdown("### 📊 Model Karşılaştırma Grafiği")
            with st.container(border=True):
                _possible = ["Accuracy", "AUC", "F1", "R2", "RMSE", "MAE"]
                _avail = [c for c in _possible if c in _lb.columns]

                if _avail:
                    _col_sel, _col_thr = st.columns([3, 2])
                    with _col_sel:
                        _metric = st.selectbox(
                            "Karşılaştırma metriği:",
                            _avail,
                            key="lb_metric_select",
                        )
                    _use_filter = (_metric == "R2" and task_type == "regression")
                    with _col_thr:
                        if _use_filter:
                            _r2_threshold = st.number_input(
                                "R² eşiği (altı gizlenir):",
                                min_value=-1.0, max_value=1.0,
                                value=0.10, step=0.05,
                                key="r2_threshold",
                            )
                        else:
                            _r2_threshold = None

                    # Ham veriyi hazırla
                    if "Model" in _lb.columns:
                        _chart_df = _lb[["Model", _metric]].dropna().set_index("Model")
                    else:
                        _chart_df = _lb[[_metric]].dropna()
                    _chart_df = _chart_df.sort_values(_metric, ascending=False)

                    # R² filtresi
                    if _use_filter and _r2_threshold is not None:
                        _total    = len(_chart_df)
                        _filtered = _chart_df[_chart_df[_metric] > _r2_threshold]
                        _hidden_n = _total - len(_filtered)
                        if _hidden_n > 0:
                            st.warning(
                                f"⚠️ **{_hidden_n} model** R² ≤ {_r2_threshold:.2f} olduğu için "
                                f"grafikten gizlendi. Toplam {_total} modelden "
                                f"**{len(_filtered)}** tanesi gösteriliyor."
                            )
                            with st.expander(f"🔽 Gizlenen {_hidden_n} model (R² ≤ {_r2_threshold:.2f})"):
                                _hidden_df = _chart_df[_chart_df[_metric] <= _r2_threshold].reset_index()
                                st.dataframe(_hidden_df, use_container_width=True, hide_index=True)
                        _chart_df = _filtered

                    if _chart_df.empty:
                        st.info("ℹ️ Seçili eşiğin üzerinde model bulunamadı. Eşiği düşürün.")
                    else:
                        st.bar_chart(_chart_df, height=300)

                    # ── Negatif R² tanı paneli — regresyon + R² seçiliyse
                    if task_type == "regression" and _metric == "R2" and "R2" in _lb.columns:
                        _all_r2   = _lb["R2"].dropna()
                        _neg_ratio = float((_all_r2 <= 0).sum()) / max(len(_all_r2), 1)
                        if _neg_ratio > 0.3:
                            with st.expander(
                                f"🔬 Tanı: Modellerin %{_neg_ratio*100:.0f}'i negatif R² veriyor — Kök neden analizi",
                                expanded=True
                            ):
                                st.markdown("""
**Negatif R², modelin sabit ortalamayı tahmin etmekten bile kötü performans sergilediği anlamına gelir.**
Bu genellikle şu nedenlerden kaynaklanır:

| # | Kök Neden | Belirti | Önerilen Düzeltme |
|---|-----------|---------|-------------------|
| 1 | **Ölçeklenmemiş özellikler** | Lineer modeller (lr, ridge, lasso) en çok etkilenir | Eğitim sekmesinde Uzman Mod → `normalize=True` ekleyin |
| 2 | **Aykırı değerler (Outliers)** | Tek bir uç değer tüm CV fold'unu bozabilir | Temizlik sekmesinde IQR kırpma uygulayın |
| 3 | **Düşük sinyal / gürültü oranı** | Hiçbir özellik hedefle korelasyon taşımıyor | EDA → “Hedef vs Özellik” sekmesini inceleyin |
| 4 | **Az veri + çok özellik** | CV fold başına yetersiz örneklem (overfitting) | Fold sayısını 2'ye indirin veya feature selection yapın |
| 5 | **Hedef dağılımı çok çarpık** | Log-normal / power-law dağılımlı hedef | Hedef sütununa `np.log1p()` dönüşümü uygulayın |

**⚡ Hızlı kontrol listesi:**
- Temizlik sekmesinden outlier kırpma uygulandı mı?
- Sayısal özellikler çok farklı ölçeklerde mi? (birisi 0–1, diğeri 0–1.000.000)
- EDA'da hedef sütunun histogramı normal mi görünüyor?
- Eğitim sekmesinde Uzman Mod → `normalize=True` seçili mi?
                                """)
                else:
                    st.caption("Grafik için desteklenen metrik bulunamadı.")


        best_model  = st.session_state["best_model"]
        pc_module   = st.session_state["pc_module"]
        task_type   = st.session_state["task_type"]

        # ── Uzman Mod: Hiperparametre Optimizasyonu ──
        if st.session_state["mode"] == "expert" and task_type in ("classification", "regression"):
            with st.expander("⚙️ Hiperparametre Optimizasyonu (Uzman Mod)", expanded=False):

                # Önceki Optuna raporunu burada da göster
                _saved_rep = st.session_state.get("opt_report")
                if _saved_rep and _saved_rep.get("optimized"):
                    st.info("💡 Akıllı Hiperparametre Motoru sonuçları mevcut — Tune işlemi bu değerleri kullanacak.")
                    _risks_tab4 = _saved_rep.get("risks", [])
                    if _risks_tab4:
                        for _rv in _risks_tab4:
                            st.warning(f"⚠️ {_rv}")
                    else:
                        st.success("✅ Risk analizi temiz — optimizasyon sonucu güvenli.")

                c1, c2 = st.columns(2)
                with c1:
                    n_iter = st.number_input(
                        "İterasyon Sayısı", min_value=10, max_value=200, value=20, step=10
                    )
                with c2:
                    metrics = CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
                    opt_metric = st.selectbox("Optimizasyon Metriği", metrics)

                if st.button("🔧 En İyi Modeli Optimize Et (Tune)", key="btn_tune"):
                    smart_params = st.session_state.get("smart_params")
                    tuned_model, tune_results = tune_selected_model(
                        best_model, task_type, n_iter, opt_metric, pc_module, smart_params
                    )
                    st.session_state["best_model"] = tuned_model
                    best_model = tuned_model
                    st.success("✅ Model başarıyla optimize edildi!")
                    st.dataframe(tune_results, use_container_width=True)

        # ── Model Grafikleri ──
        with st.expander("📈 Model Grafikleri", expanded=False):
            if best_model is not None and pc_module is not None:
                render_model_plots(best_model, task_type, pc_module)

        st.markdown("---")

        # ── Model Kaydetme ──
        st.markdown("### 💾 Modeli Kaydet ve İndir")
        model_name = st.text_input("Model Adı", value="my_automl_model", key="model_name_input")

        if st.button("💾 Modeli Kaydet", key="btn_save"):
            df = st.session_state["cleaned_df"]
            target = st.session_state.get("target_col") or "Target"
            features = [c for c in df.columns if c != target]

            try:
                pkl_path, json_path, m_card = save_model_with_card(
                    best_model, model_name, task_type, target, features, df, pc_module
                )
                st.session_state["model_card"] = m_card
                st.success(f"✅ Model kaydedildi:\n- `{pkl_path}`\n- `{json_path}`")

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    with open(pkl_path, "rb") as fh:
                        st.download_button(
                            "⬇️ Modeli İndir (.pkl)",
                            data=fh,
                            file_name=os.path.basename(pkl_path),
                            key="dl_pkl",
                        )
                with col_d2:
                    with open(json_path, "rb") as fh:
                        st.download_button(
                            "⬇️ Kimlik Kartını İndir (.json)",
                            data=fh,
                            file_name=os.path.basename(json_path),
                            mime="application/json",
                            key="dl_json",
                        )
            except Exception as e:
                st.error(f"❌ Kaydetme hatası: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 5 — Inference / Tahmin
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    render_step_indicator(4)
    st.subheader("Yeni Veri Üzerinde Tahmin (Inference)")

    st.markdown(
        "Daha önce eğittiğiniz bir model (`.pkl`) ve kimlik kartını (`.json`) "
        "yükleyerek yeni veriler üzerinde tahmin yapabilirsiniz."
    )

    c1, c2 = st.columns(2)
    with c1:
        model_file = st.file_uploader("Model Dosyası (.pkl)", type=["pkl"], key="inf_model")
    with c2:
        card_file = st.file_uploader("Kimlik Kartı (.json)", type=["json"], key="inf_card")

    inf_data_file = st.file_uploader(
        "Tahmin Edilecek Yeni Veri (CSV/Excel)",
        type=["csv", "xlsx", "xls"],
        key="inf_data",
    )

    if model_file and card_file and inf_data_file:
        try:
            card_data = json.load(card_file)
            task_type = card_data.get("task_type", "classification")

            # Modeli geçici dosyaya yaz
            temp_pkl = "temp_inference_model.pkl"
            with open(temp_pkl, "wb") as fh:
                fh.write(model_file.getbuffer())

            # Veri yükle
            inf_df = load_data(inf_data_file)

            if inf_df is not None:
                render_dataframe_preview(inf_df, title="📋 Tahmin Verisi Önizleme")
                st.markdown("---")

                is_valid = validate_compatibility(card_data, inf_df)

                if is_valid:
                    if st.button("🔮 Tahminleri Üret", key="btn_predict"):
                        loaded_model = load_model_safe(temp_pkl, task_type)
                        if loaded_model:
                            predictions = run_prediction(loaded_model, inf_df, task_type)

                            st.success("✅ Tahminler başarıyla üretildi!")
                            render_prediction_summary(predictions, task_type)

                            st.markdown("#### Tahmin Sonuçları")
                            formatted = format_results(predictions, task_type)
                            st.dataframe(formatted, use_container_width=True)

                            csv_bytes = predictions.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "⬇️ Sonuçları CSV Olarak İndir",
                                data=csv_bytes,
                                file_name="predictions.csv",
                                mime="text/csv",
                                key="dl_csv",
                            )

                            # Geçici dosyayı temizle
                            try:
                                os.remove(temp_pkl)
                            except OSError:
                                pass

        except json.JSONDecodeError:
            st.error("❌ Kimlik kartı (.json) dosyası okunamadı. Geçerli bir JSON dosyası yükleyin.")
        except Exception as e:
            log.exception(f"Inference Tab hatası: {e}")
            st.error(f"❌ Hata: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# SEKME 6 — EDA Analizi
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🔍 Keşifsel Veri Analizi (EDA)")

    if "cleaned_df" in st.session_state and st.session_state["cleaned_df"] is not None:
        _eda_df = st.session_state["cleaned_df"]
    elif "raw_df" in st.session_state and st.session_state["raw_df"] is not None:
        _eda_df = st.session_state["raw_df"]
    else:
        _eda_df = None

    if _eda_df is None:
        st.warning("Lütfen önce Yükleme sekmesinden bir veri seti yükleyin.")
        st.stop()

    if _eda_df is None:
        st.info("ℹ️ Lütfen önce **1. Yükleme** sekmesinden bir veri seti yükleyin.")
    else:
        _src = "Temizlenmiş" if st.session_state.get("cleaned_df") is not None else "Ham"
        st.caption(f"Kaynak: **{_src} Veri** — {_eda_df.shape[0]:,} satır × {_eda_df.shape[1]} sütun")

        # ── Genel Özet
        st.markdown("### 📋 Genel Bakış")
        render_eda_overview(_eda_df)

        st.markdown("---")

        # ── Sekmeli EDA bölümleri (6 sekme)
        eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5, eda_tab6 = st.tabs([
            "🔥 Korelasyon",
            "📈 Sayısal Dağılımlar",
            "🏷️ Kategorik Dağılımlar",
            "❓ Eksik Veri Haritası",
            "🎯 Hedef vs Özellik",
            "🚀 Story Dashboard",
        ])

        with eda_tab1:
            render_correlation_heatmap(_eda_df)

        with eda_tab2:
            render_distribution_plots(_eda_df)

        with eda_tab3:
            render_categorical_plots(_eda_df)

        with eda_tab4:
            render_missing_heatmap(_eda_df)

        with eda_tab5:
            # Hedef sütun seçimi — eğitimde seçildiyse onu varsayılan yap
            _default_target = st.session_state.get("target_col") or ""
            _all_cols = _eda_df.columns.tolist()
            _scatter_target = st.selectbox(
                "🎯 Hedef (Target) Sütununu Seçin:",
                ["— Seçin —"] + _all_cols,
                index=([0] + [i + 1 for i, c in enumerate(_all_cols) if c == _default_target] or [0])[0] if _default_target else 0,
                key="eda_scatter_target",
            )
            if _scatter_target != "— Seçin —":
                render_target_scatter(_eda_df, _scatter_target)
            else:
                st.info("ℹ️ Scatter grafikleri için bir hedef sütun seçin.")

        with eda_tab6:
            st.markdown("### 🚀 Story-Driven EDA Dashboard")
            st.caption("Otomatik ceiling effect, zero-segment ve pairplot analizi.")

            _num_cols_eda  = _eda_df.select_dtypes(include="number").columns.tolist()
            _cat_cols_eda  = _eda_df.select_dtypes(exclude="number").columns.tolist()

            _sd_c1, _sd_c2 = st.columns(2)
            with _sd_c1:
                _story_target = st.selectbox(
                    "Hedef sütun (opsiyonel):",
                    ["— Yok —"] + _num_cols_eda,
                    key="story_target_col",
                )
                _story_target = None if _story_target == "— Yok —" else _story_target

                _ceil_candidates = [
                    c for c in _num_cols_eda
                    if (_eda_df[c] == _eda_df[c].max()).mean() > 0.05
                ]
                _ceiling_sel = st.multiselect(
                    "🚨 Ceiling effect analizi (otomatik tespit edildi):",
                    options=_num_cols_eda,
                    default=_ceil_candidates[:3],
                    key="story_ceiling_cols",
                )

            with _sd_c2:
                _zero_candidates = [
                    c for c in _num_cols_eda
                    if (_eda_df[c] == 0).sum() > 0 and c != _story_target
                ]
                _zero_sel = st.multiselect(
                    "🔬 Zero-segment analizi:",
                    options=_num_cols_eda,
                    default=_zero_candidates[:2],
                    key="story_zero_cols",
                )
                _pair_default = _num_cols_eda[:min(4, len(_num_cols_eda))]
                _pair_sel = st.multiselect(
                    "📐 Pairplot değişkenleri:",
                    options=_num_cols_eda,
                    default=_pair_default,
                    key="story_pair_cols",
                )

            if st.button("▶ Dashboard'u Oluştur", key="btn_story_dashboard", use_container_width=True):
                render_storytelling_dashboard(
                    _eda_df,
                    target_col=_story_target,
                    ceiling_cols=_ceiling_sel or None,
                    zero_seg_cols=_zero_sel or None,
                    pairplot_cols=_pair_sel or None,
                )
            else:
                st.info("ℹ️ Ayarları yapıp '▶ Dashboard'u Oluştur' butonuna basın.")

        log.info("EDA sekmesi render tamamlandı.")
        sidebar_log("🔍 EDA analizi görüntülendi", "info")

