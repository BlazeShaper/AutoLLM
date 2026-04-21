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
)

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
                    st.session_state.pop(k, None)

        if st.session_state.get("raw_df") is not None:
            render_dataframe_preview(st.session_state["raw_df"])

            st.markdown("---")
            if st.button("📊 Hızlı Profil Raporu Oluştur", key="btn_profile"):
                generate_profile(st.session_state["raw_df"])

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
                        st.session_state.pop(k, None)
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
                    st.session_state.pop(k, None)
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

            if cols_to_clip:
                if st.button("✂️ Seçili Sütunlarda Outlier Kırp", key="btn_clip"):
                    clipped_df = clip_outliers(df, cols_to_clip)
                    st.session_state["cleaned_df"] = clipped_df.copy()
                    for k in ["setup_obj", "pc_module", "task_type", "leaderboard", "best_model"]:
                        st.session_state.pop(k, None)
                    st.success(f"✅ {len(cols_to_clip)} sütunda aykırı değerler IQR sınırlarına kırpıldı.")
                    st.rerun()

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
                for k in ["setup_obj", "pc_module", "task_type", "leaderboard", "best_model"]:
                    st.session_state.pop(k, None)
                st.success(
                    f"✅ Temizlik uygulandı. "
                    f"Kalan sütun sayısı: **{cleaned.shape[1]}** "
                    f"(çıkarılan: **{len(cols_to_drop)}**)"
                )
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SEKME 3 — Model Eğitimi
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    render_step_indicator(2)
    st.subheader("Model Eğitimi")

    if st.session_state["cleaned_df"] is None:
        st.info("ℹ️ Lütfen önce veri yükleyin.")
    else:
        df = st.session_state["cleaned_df"]

        target_col = st.selectbox(
            "🎯 Hedef (Target) Sütunu Seçin",
            [""] + list(st.session_state["cleaned_df"].columns),
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

                st.markdown("---")
                if st.button("🚀 Eğitimi Başlat", key="btn_train"):
                    with st.status("Pipeline çalışıyor…", expanded=True) as status:
                        st.write("⏳ PyCaret setup başlatılıyor…")
                        try:
                            setup_obj, pc_module = run_setup(
                                df, target_col, task_type, imputation_dict,
                                st.session_state["mode"]
                            )
                            st.session_state["setup_obj"] = setup_obj
                            st.session_state["pc_module"] = pc_module
                            st.write("✅ Setup tamamlandı.")

                            selected_models = None
                            if st.session_state["mode"] == "expert":
                                selected_models = (
                                    ALL_CLASSIFICATION_MODELS
                                    if task_type == "classification"
                                    else ALL_REGRESSION_MODELS
                                )

                            st.write("⏳ Modeller karşılaştırılıyor…")
                            best_model, leaderboard = compare_models(
                                task_type, selected_models,
                                st.session_state["mode"], pc_module
                            )
                            st.session_state["best_model"] = best_model
                            st.session_state["leaderboard"] = leaderboard

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
            with st.expander("📊 Model Karşılaştırma Grafiği", expanded=False):
                # Hangi metrik sütunları mevcut?
                _possible = ["Accuracy", "AUC", "F1", "R2", "RMSE", "MAE"]
                _avail = [c for c in _possible if c in _lb.columns]

                if _avail:
                    _metric = st.selectbox(
                        "Karşılaştırma metriği:",
                        _avail,
                        key="lb_metric_select",
                    )
                    _chart_df = (
                        _lb[["Model", _metric]]
                        .dropna()
                        .set_index("Model")
                        .sort_values(_metric, ascending=False)
                    ) if "Model" in _lb.columns else (
                        _lb[[_metric]].dropna().sort_values(_metric, ascending=False)
                    )
                    st.bar_chart(_chart_df, height=280)
                else:
                    st.caption("Grafik için desteklenen metrik bulunamadı.")


        best_model  = st.session_state["best_model"]
        pc_module   = st.session_state["pc_module"]
        task_type   = st.session_state["task_type"]

        # ── Uzman Mod: Hiperparametre Optimizasyonu ──
        if st.session_state["mode"] == "expert" and task_type in ("classification", "regression"):
            with st.expander("⚙️ Hiperparametre Optimizasyonu (Uzman Mod)", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    n_iter = st.number_input(
                        "İterasyon Sayısı", min_value=10, max_value=200, value=20, step=10
                    )
                with c2:
                    metrics = CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
                    opt_metric = st.selectbox("Optimizasyon Metriği", metrics)

                if st.button("🔧 En İyi Modeli Optimize Et (Tune)", key="btn_tune"):
                    tuned_model, tune_results = tune_selected_model(
                        best_model, task_type, n_iter, opt_metric, pc_module
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

        # ── Sekmeli EDA bölümleri (5 sekme)
        eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
            "🔥 Korelasyon",
            "📈 Sayısal Dağılımlar",
            "🏷️ Kategorik Dağılımlar",
            "❓ Eksik Veri Haritası",
            "🎯 Hedef vs Özellik",
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

        log.info("EDA sekmesi render tamamlandı.")
        sidebar_log("🔍 EDA analizi görüntülendi", "info")
