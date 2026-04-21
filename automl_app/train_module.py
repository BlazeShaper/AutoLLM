import streamlit as st
import pandas as pd
import json
import os
import datetime

import pycaret.classification as pc_class
import pycaret.regression as pc_reg
import pycaret.clustering as pc_clust
import pycaret

from config import DEFAULT_CLASSIFICATION_MODELS, DEFAULT_REGRESSION_MODELS
from logger import log, sidebar_log


# ─── Görev Tipi Tespiti ──────────────────────────────────────────────────────

def detect_task_type(df: pd.DataFrame, target_col: str) -> tuple[str, str]:
    """
    Hedef sütuna göre sınıflandırma / regresyon / kümeleme kararı verir.
    Returns: (task_type, açıklama_mesajı)
    """
    if not target_col:
        return "clustering", "Hedef sütun seçilmedi → **Kümeleme (Clustering)** yapılacak."

    target_series = df[target_col]
    num_unique = target_series.nunique()
    total_rows = len(target_series.dropna())

    if num_unique <= 1:
        raise ValueError(
            f"Hedef sütun **'{target_col}'** sadece {num_unique} benzersiz değer içeriyor. "
            "Model öğrenemez. Farklı bir sütun seçin."
        )

    dtype = target_series.dtype

    if pd.api.types.is_numeric_dtype(dtype):
        if num_unique < 20 and (total_rows == 0 or num_unique / total_rows < 0.05):
            return (
                "classification",
                f"Hedef **'{target_col}'**, az sayıda değer içeriyor "
                f"({num_unique} farklı). → **Sınıflandırma** öneriliyor.",
            )
        else:
            return (
                "regression",
                f"Hedef **'{target_col}'** sürekli sayısal değerler içeriyor. "
                "→ **Regresyon** öneriliyor.",
            )
    else:
        return (
            "classification",
            f"Hedef **'{target_col}'** kategorik/metin değerler içeriyor. "
            "→ **Sınıflandırma** öneriliyor.",
        )


# ─── Dengesizlik Kontrolü ────────────────────────────────────────────────────

def check_imbalance(df: pd.DataFrame, target_col: str, task_type: str):
    if task_type != "classification":
        return
    counts = df[target_col].value_counts()
    min_ratio = counts.min() / counts.sum()
    if min_ratio < 0.1:
        st.warning(
            f"⚠️ Veri seti **dengesiz** — en az temsil edilen sınıf oranı "
            f"**%{min_ratio*100:.1f}**. "
            "Accuracy yerine **F1 / Kappa / AUC** metriğini kullanmanızı öneririz."
        )
    with st.expander("📊 Sınıf Dağılımı"):
        dist_df = pd.DataFrame({
            "Sınıf": counts.index.astype(str),
            "Adet": counts.values,
            "Oran (%)": (counts.values / counts.sum() * 100).round(1),
        })
        st.dataframe(dist_df, use_container_width=True, hide_index=True)


# ─── Setup ───────────────────────────────────────────────────────────────────

def run_setup(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    imputation_dict: dict,
    mode: str,
):
    """PyCaret setup() çalıştırır; (setup_obj, pc_module) döndürür."""
    num_imp = imputation_dict.get("numeric", "mean")
    cat_imp = imputation_dict.get("categorical", "mode")

    common_kwargs = dict(session_id=42, verbose=False)

    log.info(f"PyCaret setup başlıyor — görev: {task_type}, hedef: {target}, mod: {mode}")
    sidebar_log(f"⚙️ PyCaret setup: {task_type} — hedef: '{target}'", "info")

    if task_type == "classification":
        setup_obj = pc_class.setup(
            data=df,
            target=target,
            numeric_imputation=num_imp,
            categorical_imputation=cat_imp,
            **common_kwargs,
        )
        log.info("Setup tamamlandı: classification")
        sidebar_log("✅ Setup tamamlandı (sınıflandırma)", "success")
        return setup_obj, pc_class

    elif task_type == "regression":
        setup_obj = pc_reg.setup(
            data=df,
            target=target,
            numeric_imputation=num_imp,
            categorical_imputation=cat_imp,
            **common_kwargs,
        )
        log.info("Setup tamamlandı: regression")
        sidebar_log("✅ Setup tamamlandı (regresyon)", "success")
        return setup_obj, pc_reg

    elif task_type == "clustering":
        setup_obj = pc_clust.setup(data=df, **common_kwargs)
        log.info("Setup tamamlandı: clustering")
        sidebar_log("✅ Setup tamamlandı (kümeleme)", "success")
        return setup_obj, pc_clust

    raise ValueError(f"Geçersiz task_type: {task_type}")


# ─── Model Karşılaştırma ─────────────────────────────────────────────────────

def compare_models(task_type: str, selected_models, mode: str, pc_module):
    """Modelleri karşılaştırır; (best_model, leaderboard) döndürür."""
    if task_type in ("classification", "regression"):
        if mode == "standard":
            include = (
                DEFAULT_CLASSIFICATION_MODELS
                if task_type == "classification"
                else DEFAULT_REGRESSION_MODELS
            )
            label = "Standart Mod"
        else:
            include = selected_models
            label = "Uzman Modu"

        log.info(f"Model karşılaştırması başlıyor — {task_type} | {label} | {len(include)} model")
        sidebar_log(f"🔄 {len(include)} model karşılaştırılıyor ({label})…", "info")

        with st.spinner(f"🔄 Modeller eğitiliyor ve karşılaştırılıyor… ({label})"):
            best_model = pc_module.compare_models(include=include)
            leaderboard = pc_module.pull()

        best_name = type(best_model).__name__
        log.info(f"En iyi model: {best_name}")
        sidebar_log(f"🏆 En iyi model: {best_name}", "success")
        return best_model, leaderboard

    # Clustering
    num_clusters = 4
    log.info(f"Kümeleme modeli oluşturuluyor: K-Means, k={num_clusters}")
    sidebar_log(f"🔄 K-Means ({num_clusters} küme) eğitiliyor…", "info")
    with st.spinner(f"🔄 K-Means modeli {num_clusters} küme ile eğitiliyor…"):
        best_model = pc_module.create_model("kmeans", num_clusters=num_clusters)
        leaderboard = pc_module.pull()
    sidebar_log("✅ Kümeleme modeli hazır", "success")
    return best_model, leaderboard


# ─── Hiperparametre Optimizasyonu ────────────────────────────────────────────

def tune_selected_model(model, task_type: str, n_iter: int, optimize_metric: str, pc_module):
    log.info(f"Hiperparametre optimizasyonu — metrik: {optimize_metric}, iterasyon: {n_iter}")
    sidebar_log(f"⚙️ Model optimize ediliyor ({optimize_metric}, {n_iter} iterasyon)…", "info")
    with st.spinner(
        f"⚙️ Model optimize ediliyor — metrik: **{optimize_metric}**, "
        f"iterasyon: **{n_iter}** …"
    ):
        tuned = pc_module.tune_model(model, n_iter=n_iter, optimize=optimize_metric)
        results = pc_module.pull()
    log.info("Hiperparametre optimizasyonu tamamlandı.")
    sidebar_log("✅ Optimizasyon tamamlandı", "success")
    return tuned, results


# ─── Model Kaydetme ──────────────────────────────────────────────────────────

def save_model_with_card(
    model,
    model_name: str,
    task_type: str,
    target: str,
    features: list,
    df: pd.DataFrame,
    pc_module,
) -> tuple[str, str, dict]:
    """
    Modeli .pkl olarak kaydeder; JSON kimlik kartı oluşturur.
    Returns: (pkl_path, json_path, model_card_dict)
    """
    # Kayıt dizini
    save_dir = os.path.join(os.getcwd(), "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_prefix = os.path.join(save_dir, f"{model_name}_{timestamp}")

    log.info(f"Model kaydediliyor: {file_prefix}")
    sidebar_log(f"💾 Model kaydediliyor: {model_name}", "info")
    pc_module.save_model(model, file_prefix)
    pkl_path = f"{file_prefix}.pkl"

    model_card = {
        "model_name": model_name,
        "timestamp": timestamp,
        "task_type": task_type,
        "target_col": target,
        "features": features,
        "pycaret_version": pycaret.__version__,
        "num_rows_trained": int(len(df)),
        "num_features": int(len(features)),
    }

    json_path = f"{file_prefix}_card.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=4, ensure_ascii=False)

    log.info(f"Model kaydı tamamlandı: {pkl_path}")
    sidebar_log(f"✅ Model kaydedildi: {os.path.basename(pkl_path)}", "success")
    return pkl_path, json_path, model_card


# ─── Model Grafikleri ────────────────────────────────────────────────────────

def render_model_plots(model, task_type: str, pc_module):
    """Mevcut PyCaret görselleştirmelerini gösterir (hata yutulur)."""
    st.markdown("#### 📈 Model Grafikleri")

    # ── Kümeleme grafikleri ──────────────────────────────────────────────────
    if task_type == "clustering":
        col1, col2 = st.columns(2)
        with col1:
            try:
                st.markdown("**Elbow Plot**")
                pc_module.plot_model(model, plot="elbow", display_format="streamlit")
            except Exception:
                st.caption("Elbow grafiği oluşturulamadı.")
        with col2:
            try:
                st.markdown("**Silhouette Plot**")
                pc_module.plot_model(model, plot="silhouette", display_format="streamlit")
            except Exception:
                st.caption("Silhouette grafiği oluşturulamadı.")
        return

    # ── Sınıflandırma / Regresyon grafikleri ────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        if task_type == "classification":
            try:
                st.markdown("**Confusion Matrix**")
                pc_module.plot_model(model, plot="confusion_matrix", display_format="streamlit")
            except Exception:
                st.caption("Grafik oluşturulamadı.")
        else:
            try:
                st.markdown("**Residuals**")
                pc_module.plot_model(model, plot="residuals", display_format="streamlit")
            except Exception:
                st.caption("Grafik oluşturulamadı.")

    with col2:
        try:
            st.markdown("**Feature Importance**")
            pc_module.plot_model(model, plot="feature", display_format="streamlit")
        except Exception:
            st.caption("Bu model için özellik önemi grafiği mevcut değil.")
