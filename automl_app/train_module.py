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


def _build_live_log_lines(model_list: list, task_type: str, mode: str, manual_params: dict = None) -> list:
    """Eğitim başlamadan önce ne yapılacağını listeleyen log satırları."""
    lines = [
        f"🚀 **Eğitim Başlatıldı** — görev: `{task_type}`, mod: `{mode}`",
        f"📦 **Eğitilecek model sayısı:** {len(model_list)}",
        "─" * 50,
    ]
    for m in model_list:
        lines.append(f"  ▸ `{m}`")
    lines.append("─" * 50)
    if manual_params:
        lines.append("🎛️ **Manuel Hiperparametreler:**")
        lines.append(f"  • Epoch / max_iter  : {manual_params.get('epochs', '—')}")
        lines.append(f"  • Batch Size        : {manual_params.get('batch_size', '—')}")
        lines.append(f"  • Learning Rate     : {manual_params.get('learning_rate', '—')}")
        lines.append(f"  • Regularization    : {manual_params.get('regularization', '—')}")
        lines.append("  ℹ️  MLP · GBM · XGBoost · LightGBM · CatBoost için uygulanır")
        lines.append("─" * 50)
    return lines


def render_training_live_log(
    model_list: list,
    task_type: str,
    mode: str,
    smart_params: dict = None,
    manual_params: dict = None,
) -> st.empty:
    """
    Eğitim öncesinde terminale benzer bir canlı log paneli oluşturur.
    Döndürülen placeholder eğitim sırasında güncellenecektir.
    """
    placeholder = st.empty()
    lines = _build_live_log_lines(model_list, task_type, mode, manual_params)
    if smart_params:
        lines.append("⚙️ **Aktif Hiperparametreler (Optuna):**")
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

def _build_manual_custom_grid(model, manual_params: dict) -> dict | None:
    """Model tipine göre manuel parametrelerden PyCaret custom_grid oluşturur."""
    if not manual_params:
        return None
    name = type(model).__name__.lower()
    epochs  = int(manual_params.get("epochs", 200))
    batch   = int(manual_params.get("batch_size", 32))
    lr      = float(manual_params.get("learning_rate", 0.01))
    reg     = float(manual_params.get("regularization", 0.1))

    if "mlp" in name or "multilayer" in name:
        return {"learning_rate_init": [lr], "alpha": [reg], "batch_size": [batch], "max_iter": [epochs]}
    elif "xgb" in name or "xgboost" in name:
        return {"learning_rate": [lr], "reg_alpha": [reg], "n_estimators": [epochs]}
    elif "lgbm" in name or "lightgbm" in name:
        return {"learning_rate": [lr], "reg_alpha": [reg], "n_estimators": [epochs]}
    elif "catboost" in name:
        return {"learning_rate": [lr], "l2_leaf_reg": [reg], "iterations": [epochs]}
    elif "gradientboosting" in name or "gbr" in name or "gbc" in name:
        return {"learning_rate": [lr], "n_estimators": [epochs]}
    elif "adaboost" in name or "ada" in name:
        return {"learning_rate": [lr], "n_estimators": [epochs]}
    elif "elasticnet" in name or "ridge" in name or "lasso" in name:
        return {"alpha": [reg], "max_iter": [epochs]}
    elif "logisticregression" in name:
        return {"C": [round(1.0 / max(reg, 1e-9), 6)], "max_iter": [epochs]}
    elif "randomforest" in name or "extratrees" in name:
        return {"n_estimators": [epochs]}
    return None


def compare_models(task_type: str, selected_models, mode: str, pc_module, smart_params: dict = None, manual_params: dict = None):
    """Modelleri karşılaştırır; (best_model, leaderboard) döndürür."""
    if task_type in ("classification", "regression"):
        if selected_models:
            include = selected_models
            label   = "Manuel Seçim"
        elif mode == "standard":
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
        progress_bar = st.progress(0)
        log_placeholder, log_lines = render_training_live_log(
            include, task_type, label, smart_params, manual_params
        )

        start_t = time.time()
        try:
            # stdout'u yakala (PyCaret çıktısı için)
            captured_lines: list = []
            capture = _StreamCapture(captured_lines)
            old_stdout = sys.stdout
            sys.stdout = capture

            def _poll_stdout():
                """Arka planda stdout'u izler ve log panelini günceller."""
                from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
                try:
                    ctx = get_script_run_ctx()
                    if ctx:
                        add_script_run_ctx(threading.current_thread(), ctx)
                except Exception:
                    pass

                curr_prog = 0.0
                while getattr(_poll_stdout, "running", True):
                    curr_prog += 0.02
                    if curr_prog > 0.90:
                        curr_prog = 0.90
                    try:
                        progress_bar.progress(curr_prog)
                    except Exception:
                        pass

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

            log_lines.append(f"⏳ {len(include)} model için eşzamanlı eğitim (compare_models) başlatıldı...")
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
        try:
            progress_bar.progress(1.0)
        except Exception:
            pass
            
        leaderboard = pc_module.pull()
        best_name = type(best_model).__name__
        
        num_models = len(include)
        avg_time = dur / num_models if num_models > 0 else dur

        # ── Eğitim özeti logu
        log_lines.append("─" * 50)
        log_lines.append(f"✅ **Eğitim tamamlandı**")
        log_lines.append(f"⏱️ **Ortalama model süresi:** {avg_time:.2f} sn")
        
        if leaderboard is not None and not leaderboard.empty:
            log_lines.append("─" * 50)
            log_lines.append("📊 **Model Performansları (Cross-Validation Ortalaması):**")
            metric_cols = [c for c in ["Accuracy", "AUC", "F1", "R2", "RMSE", "MAE"] if c in leaderboard.columns]
            
            for idx, row in leaderboard.iterrows():
                metrics_str = " | ".join([f"{mc}: {row[mc]:.4f}" for mc in metric_cols[:3]])
                log_lines.append(f"  ▸ `{idx}`")
                log_lines.append(f"     {metrics_str}")
                
            log_lines.append("─" * 50)
            log_lines.append("🏆 **EN İYİ MODEL**")
            log_lines.append(f"Model: {best_name}")
            if len(metric_cols) > 0:
                best_metric = metric_cols[0]
                best_val = leaderboard.iloc[0][best_metric]
                log_lines.append(f"{best_metric}: {best_val:.4f}")
            log_lines.append(f"Toplam Süre: {dur:.2f} sn")

        # ── Manuel parametrelerle otomatik tune ────────────────────────
        if manual_params:
            custom_grid = _build_manual_custom_grid(best_model, manual_params)
            if custom_grid:
                log_lines.append("─" * 50)
                log_lines.append(f"🔧 **Manuel Parametrelerle Tune Başlatıldı:** `{best_name}`")
                for k, v in custom_grid.items():
                    log_lines.append(f"  • `{k}` = {v}")
                _update_log_panel(log_placeholder, log_lines)
                try:
                    best_model = pc_module.tune_model(
                        best_model,
                        custom_grid=custom_grid,
                        n_iter=1,
                        optimize="R2" if task_type == "regression" else "Accuracy",
                        verbose=False,
                    )
                    leaderboard = pc_module.pull()
                    log_lines.append("✅ **Manuel tune tamamlandı.**")
                except Exception as _te:
                    log_lines.append(f"⚠️ Manuel tune atlandı: {_te}")
                    log.warning(f"Manuel tune hatası: {_te}")
                _update_log_panel(log_placeholder, log_lines)
            else:
                log_lines.append(f"ℹ️  `{best_name}` için manuel parametre grid'i oluşturulamadı — tune atlandı.")
                _update_log_panel(log_placeholder, log_lines)

        _update_log_panel(log_placeholder, log_lines)
        log.info(f"Eğitim tamamlandı. Süre: {dur:.2f}s — En iyi model: {best_name}")
        sidebar_log(f"🏆 En iyi model: {best_name}", "success")
        
        if leaderboard is not None and not leaderboard.empty:
            st.markdown("#### 🏆 Leaderboard Tablosu")
            st.dataframe(leaderboard, use_container_width=True)
            
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


# ─── PyCaret Optuna Entegrasyon Köprüsü ─────────────────────────────────────

def make_pycaret_train_fn(pc_module, include_models: list, optimize_metric: str):
    """
    Optuna'ya geçirilecek gerçek PyCaret train_fn closure'ı döndürür.

    Döndürülen fonksiyon:
      - params dict'inden learning_rate / regularization değerlerini alır,
      - PyCaret'in mevcut (setup tamamlanmış) session'ında compare_models() çalıştırır,
      - pull() ile leaderboard'u çeker ve optimize_metric değerini float olarak döndürür.

    Bu sayede Optuna'nın gördüğü skor ile PyCaret leaderboard'undaki skor
    birebir aynı metrik üzerinden hesaplanır (birim farkı sıfır).

    Args:
        pc_module:        pycaret.regression veya pycaret.classification modülü.
        include_models:   compare_models(include=...) listesi.
        optimize_metric:  PyCaret metrik adı — "R2", "Accuracy", "AUC", "F1" vb.

    Returns:
        Callable[[dict], float]
    """
    # Metrik sütun adını PyCaret leaderboard sütunuyla eşleştir
    _METRIC_COL_MAP = {
        "R2":       "R2",
        "MAE":      "MAE",
        "MSE":      "MSE",
        "RMSE":     "RMSE",
        "RMSLE":    "RMSLE",
        "MAPE":     "MAPE",
        "Accuracy": "Accuracy",
        "AUC":      "AUC",
        "F1":       "F1",
        "Recall":   "Recall",
        "Precision":"Precision",
        "Kappa":    "Kappa",
        "MCC":      "MCC",
    }
    # Büyük-küçük harf duyarsız arama
    metric_col = _METRIC_COL_MAP.get(optimize_metric, optimize_metric)

    # MAE/RMSE gibi hata metrikleri minimize → negatif alarak Optuna'ya maximize ettir
    _LOWER_IS_BETTER = {"MAE", "MSE", "RMSE", "RMSLE", "MAPE"}
    negate = optimize_metric in _LOWER_IS_BETTER

    def _train_fn(params: dict) -> float:
        """
        Optuna trial parametrelerini içeren params dict'ini alır;
        PyCaret compare_models() → tune_model() → pull() → metrik değeri döndürür.
        """
        try:
            # Adım 1: Mevcut PyCaret session'ında en iyi modeli bul
            best = pc_module.compare_models(
                include=include_models,
                sort=optimize_metric,
                verbose=False,
            )
            if best is None:
                log.warning("make_pycaret_train_fn: compare_models None döndürdü.")
                return 0.0

            # Adım 2: Model sınıf adına göre params'tan uygun custom_grid oluştur
            model_name = type(best).__name__.lower()

            if any(k in model_name for k in ("elasticnet", "ridge", "lasso")):
                # Lineer düzenlemeli modeller → alpha, regularization'dan gelir
                custom_grid = {"alpha": [params["regularization"]]}

            elif any(k in model_name for k in ("gbm", "xgb", "lgbm", "catboost", "gradientboosting")):
                # Gradient boosting ailesi → learning_rate + L1 cezası
                custom_grid = {
                    "learning_rate": [params["learning_rate"]],
                    "reg_alpha":     [params["regularization"]],
                }

            elif "mlp" in model_name:
                # Çok katmanlı algılayıcı → tüm hiperparametreler iletilir
                custom_grid = {
                    "learning_rate_init": [params["learning_rate"]],
                    "alpha":              [params["regularization"]],
                    "batch_size":         [int(params["batch_size"])],
                    "max_iter":           [int(params["epochs"])],
                }

            else:
                # Desteklenmeyen model tipi → tune atlanır, compare skoru kullanılır
                custom_grid = None

            # Adım 3 & 4: custom_grid varsa tune_model çağır, yoksa compare sonucunu kullan
            if custom_grid is not None:
                log.info(f"_train_fn: {model_name} tune_model çağrılıyor, grid={custom_grid}")
                pc_module.tune_model(
                    best,
                    custom_grid=custom_grid,
                    n_iter=3,                  # Trial hızını korumak için düşük iterasyon
                    optimize=optimize_metric,
                    verbose=False,
                )
                lb = pc_module.pull()          # tune_model sonrası leaderboard'u çek
            else:
                log.info(f"_train_fn: {model_name} için custom_grid yok, compare skoru kullanılıyor.")
                lb = pc_module.pull()          # compare_models sonrası leaderboard'u çek

            # Adım 5: Leaderboard'dan metrik değerini oku
            if lb is None or lb.empty:
                log.warning("make_pycaret_train_fn: pull() boş döndürdü.")
                return 0.0

            if metric_col not in lb.columns:
                # İstenen sütun yoksa ilk sayısal sütunu fallback olarak kullan
                num_cols = lb.select_dtypes("number").columns.tolist()
                val = float(lb.iloc[0][num_cols[0]]) if num_cols else 0.0
            else:
                val = float(lb.iloc[0][metric_col])

            # negate=True ise (MAE/RMSE gibi hata metrikleri) Optuna maximize için negatife çevir
            return -val if negate else val

        except Exception as exc:
            log.warning(f"make_pycaret_train_fn._train_fn hatası: {exc}")
            return 0.0

    return _train_fn


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
