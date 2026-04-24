import streamlit as st
import pandas as pd
import json
import os
import datetime
import time
import sys
import io
import threading

import pycaret.classification as pc_class
import pycaret.regression as pc_reg
import pycaret.clustering as pc_clust
import pycaret

from config import DEFAULT_CLASSIFICATION_MODELS, DEFAULT_REGRESSION_MODELS
from logger import log, sidebar_log


# ─── Görev Tipi Tespiti ──────────────────────────────────────────────────────

def detect_task_type(df: pd.DataFrame, target_col: str) -> tuple[str, str]:
    """Hedef sütuna göre sınıflandırma / regresyon / kümeleme kararı verir."""
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

def _auto_fold(n_rows: int) -> int:
    """
    Veri boyutuna göre güvenli CV fold sayısı döndürür.
    Az veriyle 10-fold yapmak her fold'da çok az örnek bırakır.
    """
    if n_rows < 500:
        return 3
    elif n_rows < 1000:
        return 5
    else:
        return 10


def run_setup(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    imputation_dict: dict,
    mode: str,
    preprocessing_opts: dict = None,
):
    """
    PyCaret setup() çalıştırır; (setup_obj, pc_module, working_df) döndürür.

    preprocessing_opts anahtarları (hepsi bool, aksi belirtilmedikçe):
        normalize                 – özellikleri [0,1] aralığına ölçekler
        remove_outliers           – aykırı değerleri eğitimden çıkarır
        feature_selection         – en iyi %80 özelliği tutar (n_features_to_select=0.8)
        pca                       – boyut indirgeme için PCA uygular
        remove_multicollinearity  – yüksek korelasyonlu özelliklerden birini çıkarır
        log_transform_target      – çarpık hedef için log1p dönüşümü uygular
    """
    import numpy as np

    num_imp = imputation_dict.get("numeric", "mean")
    cat_imp = imputation_dict.get("categorical", "mode")

    # Preprocessing seçeneklerini güvenli şekilde al
    pp = preprocessing_opts or {}
    do_normalize                = bool(pp.get("normalize",                False))
    do_remove_outliers          = bool(pp.get("remove_outliers",          False))
    do_feature_selection        = bool(pp.get("feature_selection",        False))
    do_pca                      = bool(pp.get("pca",                      False))
    do_remove_multicollinearity = bool(pp.get("remove_multicollinearity", False))
    do_log_transform            = bool(pp.get("log_transform_target",     False))

    # ── A) Dinamik CV Fold Sayısı ────────────────────────────────────
    fold = _auto_fold(len(df))
    log.info(f"Dinamik fold sayısı: {fold} (veri boyutu: {len(df)} satır)")
    sidebar_log(f"🔢 CV fold sayısı: {fold} (veri: {len(df):,} satır)", "info")

    # ── B) Hedef Log Dönüşümü ──────────────────────────────────────
    working_df = df.copy()
    if do_log_transform and task_type == "regression":
        target_series = working_df[target]
        if (target_series > 0).all():
            working_df[target] = np.log1p(target_series)
            log.info(f"Hedef sütun '{target}' log1p dönüşümü uygulandı.")
            sidebar_log(f"📉 log1p dönüşümü: '{target}'", "info")
        else:
            log.warning("log_transform_target seçili ama hedef sıfır/negatif değer içeriyor — atlandı.")
            sidebar_log("⚠️ Log dönüşümü atlandı (hedef ≤ 0 içeriyor)", "warning")

    # ── C) n_jobs=-1 Paralel Eğitim ─────────────────────────────────
    common_kwargs = dict(session_id=42, verbose=False, fold=fold, n_jobs=-1)

    # Aktif seçenekleri logla
    active_pp = [
        k for k, v in {
            "normalize":                do_normalize,
            "remove_outliers":          do_remove_outliers,
            "feature_selection":        do_feature_selection,
            "pca":                      do_pca,
            "remove_multicollinearity": do_remove_multicollinearity,
            "log_transform_target":     do_log_transform,
        }.items() if v
    ]
    log.info(
        f"PyCaret setup başlıyor — görev: {task_type}, hedef: {target}, mod: {mode}, "
        f"ön işleme: {active_pp or 'yok'}"
    )
    sidebar_log(f"⚙️ PyCaret setup: {task_type} — hedef: '{target}'", "info")
    if active_pp:
        sidebar_log(f"🔧 Aktif ön işleme: {', '.join(active_pp)}", "info")

    if task_type == "classification":
        setup_obj = pc_class.setup(
            data=working_df,
            target=target,
            numeric_imputation=num_imp,
            categorical_imputation=cat_imp,
            normalize=do_normalize,
            remove_outliers=do_remove_outliers,
            feature_selection=do_feature_selection,
            n_features_to_select=0.8 if do_feature_selection else 1.0,
            pca=do_pca,
            remove_multicollinearity=do_remove_multicollinearity,
            **common_kwargs,
        )
        log.info("Setup tamamlandı: classification")
        sidebar_log("✅ Setup tamamlandı (sınıflandırma)", "success")
        return setup_obj, pc_class

    elif task_type == "regression":
        setup_obj = pc_reg.setup(
            data=working_df,
            target=target,
            numeric_imputation=num_imp,
            categorical_imputation=cat_imp,
            normalize=do_normalize,
            remove_outliers=do_remove_outliers,
            feature_selection=do_feature_selection,
            n_features_to_select=0.8 if do_feature_selection else 1.0,
            pca=do_pca,
            remove_multicollinearity=do_remove_multicollinearity,
            **common_kwargs,
        )
        log.info("Setup tamamlandı: regression")
        sidebar_log("✅ Setup tamamlandı (regresyon)", "success")
        return setup_obj, pc_reg

    elif task_type == "clustering":
        # Clustering setup normalize/PCA destekler ama remove_outliers/feature_selection desteklemez
        setup_obj = pc_clust.setup(
            data=working_df,
            normalize=do_normalize,
            pca=do_pca,
            **common_kwargs,
        )
        log.info("Setup tamamlandı: clustering")
        sidebar_log("✅ Setup tamamlandı (kümeleme)", "success")
        return setup_obj, pc_clust

    raise ValueError(f"Geçersiz task_type: {task_type}")


# ─── Canlı Eğitim Logu Yakalama ──────────────────────────────────────────────

class _StreamCapture(io.StringIO):
    """Stdout/stderr'i yakalayıp canlı olarak bir liste'ye yazar."""
    def __init__(self, lines: list):
        super().__init__()
        self._lines = lines
        self._orig = None

    def write(self, text: str):
        super().write(text)
        stripped = text.strip()
        if stripped:
            self._lines.append(stripped)

    def flush(self):
        super().flush()


def _build_live_log_lines(model_list: list, task_type: str, mode: str) -> list:
    """Eğitim başlamadan önce ne yapılacağını listeleyen log satırları."""
    lines = [
        f"🚀 **Eğitim Başlatıldı** — görev: `{task_type}`, mod: `{mode}`",
        f"📦 **Eğitilecek model sayısı:** {len(model_list)}",
        "─" * 50,
    ]
    for m in model_list:
        lines.append(f"  ▸ `{m}`")
    lines.append("─" * 50)
    return lines


def render_training_live_log(
    model_list: list,
    task_type: str,
    mode: str,
    smart_params: dict = None,
) -> st.empty:
    """
    Eğitim öncesinde terminale benzer bir canlı log paneli oluşturur.
    Döndürülen placeholder eğitim sırasında güncellenecektir.
    """
    placeholder = st.empty()
    lines = _build_live_log_lines(model_list, task_type, mode)
    if smart_params:
        lines.append("⚙️ **Aktif Hiperparametreler (Manuel/Optuna):**")
        for k, v in smart_params.items():
            lines.append(f"  • `{k}` = **{v}**")
        lines.append("─" * 50)
    _update_log_panel(placeholder, lines)
    return placeholder, lines


def _update_log_panel(placeholder, lines: list):
    """Log satırlarını terminale benzer koyu tema panelde gösterir."""
    display = "\n".join(lines[-40:])   # son 40 satır
    placeholder.markdown(
        f"""
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;
padding:14px 16px;font-family:'Courier New',monospace;font-size:0.82em;
color:#94a3b8;max-height:380px;overflow-y:auto;line-height:1.6;">
<pre style="margin:0;white-space:pre-wrap;">{display}</pre>
</div>
""",
        unsafe_allow_html=True,
    )


# ─── Model Karşılaştırma ─────────────────────────────────────────────────────

def compare_models(task_type: str, selected_models, mode: str, pc_module, smart_params: dict = None):
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

        # ── Canlı log panelini oluştur
        st.markdown("##### 📡 Canlı Eğitim Terminali")
        log_placeholder, log_lines = render_training_live_log(include, task_type, label, smart_params)

        start_t = time.time()
        try:
            # stdout'u yakala (PyCaret çıktısı için)
            captured_lines: list = []
            capture = _StreamCapture(captured_lines)
            old_stdout = sys.stdout
            sys.stdout = capture

            def _poll_stdout():
                """Arka planda stdout'u izler ve log panelini günceller."""
                while getattr(_poll_stdout, "running", True):
                    new_items = captured_lines[:]
                    if new_items:
                        for item in new_items:
                            log_lines.append(f"  {item}")
                        captured_lines.clear()
                        _update_log_panel(log_placeholder, log_lines)
                    time.sleep(0.5)

            _poll_stdout.running = True
            poll_thread = threading.Thread(target=_poll_stdout, daemon=True)
            poll_thread.start()

            # Model karşılaştırma
            for i, model_id in enumerate(include, 1):
                elapsed = time.time() - start_t
                log_lines.append(
                    f"⏳ [{i:02d}/{len(include):02d}] `{model_id}` eğitiliyor… "
                    f"(toplam geçen: {elapsed:.1f}s)"
                )
                _update_log_panel(log_placeholder, log_lines)

            best_model = pc_module.compare_models(include=include)

            _poll_stdout.running = False
            sys.stdout = old_stdout

            if best_model is None:
                raise RuntimeError("PyCaret geçerli bir model eğitemedi. Veri setini kontrol edin.")

        except Exception as e:
            sys.stdout = old_stdout
            log.exception("Model karşılaştırma başarısız.")
            st.error(f"❌ Model eğitimi sırasında hata: {e}")
            return None, None

        dur = time.time() - start_t
        leaderboard = pc_module.pull()
        best_name = type(best_model).__name__

        # ── Eğitim özeti logu
        log_lines.append("─" * 50)
        log_lines.append(f"✅ **Eğitim tamamlandı** — Süre: {dur:.2f} saniye")
        log_lines.append(f"🏆 **En iyi model:** `{best_name}`")
        if leaderboard is not None and not leaderboard.empty:
            metric_cols = [c for c in ["Accuracy", "AUC", "F1", "R2", "RMSE", "MAE"] if c in leaderboard.columns]
            for mc in metric_cols[:3]:
                best_val = leaderboard.iloc[0][mc]
                log_lines.append(f"  📊 {mc}: **{best_val:.4f}**")
        _update_log_panel(log_placeholder, log_lines)

        log.info(f"Eğitim tamamlandı. Süre: {dur:.2f}s — En iyi model: {best_name}")
        sidebar_log(f"🏆 En iyi model: {best_name}", "success")
        return best_model, leaderboard

    # ── Clustering ────────────────────────────────────────────────────────────
    num_clusters = 4
    log.info(f"Kümeleme modeli oluşturuluyor: K-Means, k={num_clusters}")
    sidebar_log(f"🔄 K-Means ({num_clusters} küme) eğitiliyor…", "info")

    with st.spinner(f"🔄 K-Means modeli {num_clusters} küme ile eğitiliyor…"):
        start_t = time.time()
        try:
            best_model = pc_module.create_model("kmeans", num_clusters=num_clusters)
            if best_model is None:
                raise RuntimeError("Kümeleme modeli eğitilemedi.")
        except Exception as e:
            log.exception("Kümeleme modeli başarısız.")
            st.error(f"❌ Kümeleme eğitimi hatası: {e}")
            return None, None
        dur = time.time() - start_t
        log.info(f"Eğitim tamamlandı. Süre: {dur:.2f}s")
        leaderboard = pc_module.pull()
    sidebar_log("✅ Kümeleme modeli hazır", "success")
    return best_model, leaderboard


# ─── Hiperparametre Optimizasyonu ────────────────────────────────────────────

def tune_selected_model(
    model,
    task_type: str,
    n_iter: int,
    optimize_metric: str,
    pc_module,
    smart_params: dict = None,
):
    log.info(f"Hiperparametre optimizasyonu — metrik: {optimize_metric}, iterasyon: {n_iter}")
    sidebar_log(f"⚙️ Model optimize ediliyor ({optimize_metric}, {n_iter} iterasyon)…", "info")

    custom_grid = None
    if smart_params:
        model_name = type(model).__name__.lower()
        if "logisticregression" in model_name or "svc" in model_name:
            custom_grid = {"C": [1.0 / smart_params.get("regularization", 1.0)]}
        elif "ridge" in model_name or "lasso" in model_name or "elasticnet" in model_name:
            custom_grid = {"alpha": [smart_params.get("regularization", 1.0)]}
        elif "gbm" in model_name or "xgboost" in model_name or "catboost" in model_name or "lightgbm" in model_name:
            custom_grid = {
                "learning_rate": [smart_params.get("learning_rate", 0.01)],
                "reg_alpha": [smart_params.get("regularization", 0.0)],
            }
        elif "mlp" in model_name:
            custom_grid = {
                "learning_rate_init": [smart_params.get("learning_rate", 0.001)],
                "alpha": [smart_params.get("regularization", 0.0001)],
                "batch_size": [smart_params.get("batch_size", 32)],
                "max_iter": [smart_params.get("epochs", 200)],
            }

        if custom_grid:
            log.info(f"Akıllı hiperparametreler custom_grid olarak ayarlandı: {custom_grid}")

    # Tune log paneli
    st.markdown("##### 📡 Tune Terminali")
    tune_placeholder = st.empty()
    tune_lines = [
        f"🔧 **Tune başlatıldı** — Metrik: `{optimize_metric}`, İterasyon: `{n_iter}`",
        f"🤖 **Model:** `{type(model).__name__}`",
        "─" * 50,
    ]
    if custom_grid:
        tune_lines.append("⚙️ **Kullanılan Custom Grid (Akıllı Parametreler):**")
        for k, v in custom_grid.items():
            tune_lines.append(f"  • `{k}` = {v}")
    _update_log_panel(tune_placeholder, tune_lines)

    start_t = time.time()
    with st.spinner(
        f"⚙️ Model optimize ediliyor — metrik: **{optimize_metric}**, "
        f"iterasyon: **{n_iter}** …"
    ):
        if custom_grid:
            tuned = pc_module.tune_model(model, custom_grid=custom_grid, n_iter=n_iter, optimize=optimize_metric)
        else:
            tuned = pc_module.tune_model(model, n_iter=n_iter, optimize=optimize_metric)
        results = pc_module.pull()

    dur = time.time() - start_t
    tune_lines.append("─" * 50)
    tune_lines.append(f"✅ **Tune tamamlandı** — Süre: {dur:.2f}s")
    if results is not None and not results.empty:
        metric_cols = [c for c in ["Accuracy", "AUC", "F1", "R2", "RMSE", "MAE"] if c in results.columns]
        for mc in metric_cols[:3]:
            tune_lines.append(f"  📊 {mc}: **{results.iloc[0][mc]:.4f}**")
    _update_log_panel(tune_placeholder, tune_lines)

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
    """Modeli .pkl olarak kaydeder; JSON kimlik kartı oluşturur."""
    save_dir = os.path.join(os.getcwd(), "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_prefix = os.path.join(save_dir, f"{model_name}_{timestamp}")

    log.info(f"Model kaydediliyor: {file_prefix}")
    sidebar_log(f"💾 Model kaydediliyor: {model_name}", "info")
    pc_module.save_model(model, file_prefix)
    pkl_path = f"{file_prefix}.pkl"

    model_card = {
        "model_name":       model_name,
        "timestamp":        timestamp,
        "task_type":        task_type,
        "target_col":       target,
        "features":         features,
        "pycaret_version":  pycaret.__version__,
        "num_rows_trained": int(len(df)),
        "num_features":     int(len(features)),
    }

    json_path = f"{file_prefix}_card.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=4, ensure_ascii=False)

    log.info(f"Model kaydı tamamlandı: {pkl_path}")
    sidebar_log(f"✅ Model kaydedildi: {os.path.basename(pkl_path)}", "success")
    return pkl_path, json_path, model_card


# ─── Model Grafikleri ────────────────────────────────────────────────────────

def render_model_plots(model, task_type: str, pc_module):
    """Mevcut PyCaret görselleştirmelerini gösterir."""
    st.markdown("#### 📈 Model Grafikleri")

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
