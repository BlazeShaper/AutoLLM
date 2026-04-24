"""
OutoLLM — Dinamik Hiperparametre ve Optimizasyon Modülü
Veri setinin istatistiksel analizine dayalı hiperparametre önerileri,
manuel kontrol paneli ve Optuna tabanlı Bayesyen Optimizasyon motoru içerir.
"""

import optuna
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from typing import Dict, Any, Callable, List, Tuple
import streamlit as st

from logger import log, sidebar_log


# ─── VERİ ANALİZİ (STEP 1) ───────────────────────────────────────────────────

def analyze_dataset(df: pd.DataFrame, target: str) -> Dict[str, float]:
    """Veri setinin istatistiksel yapısını analiz eder."""
    log.info(f"Veri seti analizi başlatıldı. Hedef: {target}")

    if target not in df.columns:
        raise ValueError(f"Hedef sütun '{target}' veri setinde bulunamadı.")

    numeric_df = df.select_dtypes(include="number")

    if target in numeric_df.columns:
        X = numeric_df.drop(columns=[target])
        y_is_numeric = True
        y = numeric_df[target]
    else:
        X = numeric_df
        y_is_numeric = False
        y = df[target]

    n, p = X.shape
    if p == 0:
        log.warning("Veri setinde sayısal özellik bulunamadı, analiz metrikleri varsayılan olacak.")
        return {
            "n_rows": n, "n_features": 0, "skewness_mean": 0.0,
            "kurtosis_mean": 0.0, "sparsity": 0.0, "corr_mean": 0.0, "corr_max": 0.0
        }

    skew_vals = skew(X, nan_policy="omit")
    kurt_vals = kurtosis(X, nan_policy="omit")
    sparsity = float((X == 0).sum().sum() / (n * p)) if (n * p) > 0 else 0.0

    if y_is_numeric and p > 0:
        corr = X.corrwith(y).abs()
        corr_mean = float(corr.mean()) if not corr.isna().all() else 0.0
        corr_max = float(corr.max()) if not corr.isna().all() else 0.0
    else:
        corr_mean = 0.0
        corr_max = 0.0

    stats = {
        "n_rows": n,
        "n_features": p,
        "skewness_mean": float(np.nanmean(skew_vals)) if not np.isnan(np.nanmean(skew_vals)) else 0.0,
        "kurtosis_mean": float(np.nanmean(kurt_vals)) if not np.isnan(np.nanmean(kurt_vals)) else 0.0,
        "sparsity": sparsity,
        "corr_mean": corr_mean,
        "corr_max": corr_max,
    }

    log.info(f"Veri analizi tamamlandı: {stats}")
    return stats


# ─── SEZGİSEL (HEURISTIC) MOTOR (STEP 2) ─────────────────────────────────────

def compute_batch_size(n: int, p: int, sparsity: float, gpu_memory_gb: float = 8.0) -> int:
    base = int(np.sqrt(n))
    if sparsity > 0.7:
        base = int(base * 0.5)
    mem_limit = (gpu_memory_gb * 1024) / max(p * 4, 1)
    batch = int(min(base, mem_limit))
    return int(np.clip(batch, 16, 512))


def compute_learning_rate(skewness: float, kurtosis_val: float) -> float:
    lr = 0.01
    if abs(skewness) > 1:
        lr *= 0.5
    if kurtosis_val > 3:
        lr *= 0.7
    return float(np.clip(lr, 0.0005, 0.05))


def compute_regularization(corr_mean: float, corr_max: float) -> float:
    if corr_max > 0.7:
        return 0.01
    elif corr_mean < 0.1:
        return 10.0
    return 1.0


def compute_patch_size(p: int, n: int) -> int:
    if p > 100:
        return 32
    elif n < 2000:
        return 8
    return 16


def compute_epochs(n: int, p: int) -> int:
    """Veri boyutuna göre önerilen epoch sayısını hesaplar."""
    if n < 500:
        return 200
    elif n < 5000:
        return 100
    elif n < 50000:
        return 50
    return 20


def compute_bias(df: pd.DataFrame, target: str, task_type_hint: str = "regression") -> float:
    """Hedef sütunun ortalamasını (bias başlangıcı) hesaplar."""
    if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        return float(df[target].mean())
    return 0.0


def heuristic_params(stats: Dict[str, float], df: pd.DataFrame = None, target: str = None, gpu_memory_gb: float = 8.0) -> Dict[str, Any]:
    """Tüm sezgisel hesaplamaları birleştirip bir hyperparam dict'i döner."""
    n = int(stats["n_rows"])
    p = int(stats["n_features"])
    result = {
        "batch_size": compute_batch_size(n, p, stats["sparsity"], gpu_memory_gb),
        "patch_size": compute_patch_size(p, n),
        "learning_rate": compute_learning_rate(stats["skewness_mean"], stats["kurtosis_mean"]),
        "regularization": compute_regularization(stats["corr_mean"], stats["corr_max"]),
        "epochs": compute_epochs(n, p),
        "bias": compute_bias(df, target) if (df is not None and target) else 0.0,
    }
    return result


# ─── OPTUNA OPTİMİZASYONU (STEP 3) ───────────────────────────────────────────

def get_search_space(base: Dict[str, Any]) -> Dict[str, Tuple]:
    return {
        "lr":    (base["learning_rate"] * 0.5,             base["learning_rate"] * 2),
        "batch": (max(16, int(base["batch_size"] * 0.5)),  min(512, int(base["batch_size"] * 1.5))),
        "reg":   (base["regularization"] * 0.1,            base["regularization"] * 10),
        "epochs":(max(5, int(base["epochs"] * 0.5)),       min(500, int(base["epochs"] * 2))),
    }


def objective(trial: optuna.Trial, base_params: Dict[str, Any], train_fn: Callable) -> float:
    space = get_search_space(base_params)
    params = {
        "learning_rate":  trial.suggest_float("lr",    space["lr"][0],    space["lr"][1],    log=True),
        "batch_size":     trial.suggest_int(  "batch", int(space["batch"][0]), int(space["batch"][1])),
        "regularization": trial.suggest_float("reg",   space["reg"][0],   space["reg"][1],   log=True),
        "epochs":         trial.suggest_int(  "epochs",int(space["epochs"][0]), int(space["epochs"][1])),
        "patch_size":     base_params.get("patch_size", 16),
        "bias":           base_params.get("bias", 0.0),
    }
    try:
        score = train_fn(params)
        return float(score)
    except Exception as e:
        log.warning(f"Optuna trial hatası: {e}")
        raise optuna.TrialPruned()


def run_optimization(
    base_params: Dict[str, Any],
    train_fn: Callable,
    n_trials: int = 20,
    progress_placeholder=None,
) -> Dict[str, Any]:
    """Bayesyen optimizasyonu başlatır; ilerlemeyi isteğe bağlı bir Streamlit placeholder'a yazar."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")

    log.info(f"Optuna optimizasyonu başladı ({n_trials} trial)...")

    trial_logs: list = []

    def _callback(study, trial):
        msg = (
            f"  Trial #{trial.number + 1:02d} — "
            f"Skor: **{trial.value:.4f}** | "
            f"lr={trial.params.get('lr', '?'):.5f}, "
            f"batch={trial.params.get('batch', '?')}, "
            f"epochs={trial.params.get('epochs', '?')}, "
            f"reg={trial.params.get('reg', '?'):.4f}"
        )
        trial_logs.append(msg)
        if progress_placeholder is not None:
            progress_placeholder.markdown("\n\n".join(trial_logs[-15:]))  # son 15 trial göster

    try:
        study.optimize(
            lambda trial: objective(trial, base_params, train_fn),
            n_trials=n_trials,
            n_jobs=1,
            callbacks=[_callback],
        )
    except Exception as e:
        log.error(f"Optuna çalışırken hata oluştu: {e}")
        return base_params

    best = study.best_params
    optimized = {
        "learning_rate":  best.get("lr",     base_params["learning_rate"]),
        "batch_size":     best.get("batch",  base_params["batch_size"]),
        "regularization": best.get("reg",    base_params["regularization"]),
        "epochs":         best.get("epochs", base_params["epochs"]),
        "patch_size":     base_params["patch_size"],
        "bias":           base_params["bias"],
    }
    log.info(f"Optuna tamamlandı. En iyi skor: {study.best_value:.4f}. Parametreler: {optimized}")
    return optimized


# ─── RİSK ANALİZİ (STEP 4) ───────────────────────────────────────────────────

def analyze_risk(params: Dict[str, Any], stats: Dict[str, float]) -> List[str]:
    risks = []
    if params.get("batch_size", 32) < 32 and stats["n_rows"] < 1000:
        risks.append("overfitting — Düşük batch size & küçük veri seti: Aşırı öğrenme riski yüksek.")
    if params.get("learning_rate", 0.01) < 0.001:
        risks.append("vanishing_gradient — Çok düşük öğrenme oranı: Kaybolan gradyan riski.")
    if params.get("learning_rate", 0.01) > 0.05:
        risks.append("unstable_training — Yüksek öğrenme oranı: Eğitim kararsızlığı riski.")
    if params.get("epochs", 50) > 300 and stats["n_rows"] < 2000:
        risks.append("overfitting — Çok fazla epoch & küçük veri seti: Aşırı öğrenme riski.")
    return risks


# ─── PARAMETRE ÖNERİ AÇIKLAMA KARTLARI ───────────────────────────────────────

def _render_param_badge(label: str, recommended: Any, unit: str, reason: str, icon: str = "💡"):
    """Tek bir parametre için önerir ve açıklar."""
    st.markdown(
        f"""
<div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.3);
border-radius:10px;padding:10px 14px;margin-bottom:6px;">
  <span style="font-size:1.1em;">{icon}</span>
  <b style="color:#818cf8;">{label}</b>
  <span style="float:right;background:#312e81;color:#c7d2fe;border-radius:6px;
  padding:2px 8px;font-size:0.9em;">{recommended} {unit}</span>
  <br/><small style="color:#94a3b8;">{reason}</small>
</div>
""",
        unsafe_allow_html=True,
    )


# ─── STREAMLIT ENTEGRASYONU (STEP 5) ─────────────────────────────────────────

def render_smart_hyperparams(df: pd.DataFrame, target: str, train_fn: Callable) -> Dict[str, Any]:
    """
    Uzman Mod'da çağrılan tam hiperparametre paneli.
    Öneri kartları + manuel sliderlar + Optuna optimizasyonu + canlı log.
    """
    st.markdown("---")
    st.markdown("### 🎛️ Hiperparametre Kontrol Paneli")
    st.caption(
        "Aşağıda veri setinizin istatistiksel yapısına göre **önerilen değerler** gösterilmektedir. "
        "İstediğiniz parametreyi manuel olarak değiştirebilirsiniz."
    )

    # ── 1. Veri Analizi ──────────────────────────────────────────────────────
    with st.spinner("📊 Veri istatistikleri çıkarılıyor…"):
        try:
            stats = analyze_dataset(df, target)
            base = heuristic_params(stats, df=df, target=target)
        except Exception as e:
            st.error(f"❌ Analiz hatası: {e}")
            log.exception(f"analyze_dataset başarısız: {e}")
            return {}

    result_payload = {
        "stats":     stats,
        "heuristic": base,
        "optimized": None,
        "risks":     [],
    }

    # ── 2. Öneri Kartları ────────────────────────────────────────────────────
    with st.expander("📋 Akıllı Parametre Önerileri (Neden bu değerler?)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            _render_param_badge(
                "Learning Rate", f"{base['learning_rate']:.5f}", "",
                f"Skewness={stats['skewness_mean']:.2f} → "
                + ("düşük LR (çarpık dağılım)" if abs(stats["skewness_mean"]) > 1 else "standart LR"),
                "📈",
            )
            _render_param_badge(
                "Batch Size", base["batch_size"], "",
                f"√(n={stats['n_rows']:.0f}) formülü + seyreklik={stats['sparsity']:.2f}",
                "📦",
            )
            _render_param_badge(
                "Epochs", base["epochs"], "",
                f"Veri boyutu ({stats['n_rows']:.0f} satır) bazlı öneri",
                "🔄",
            )
        with c2:
            _render_param_badge(
                "Patch Size", base["patch_size"], "",
                f"Özellik sayısı={stats['n_features']} bazlı (transformer/patch modeller için)",
                "🧩",
            )
            _render_param_badge(
                "Regularization", f"{base['regularization']:.4f}", "",
                f"corr_max={stats['corr_max']:.2f} → "
                + ("zayıf regularizasyon (yüksek korelasyon)" if stats["corr_max"] > 0.7 else "güçlü regularizasyon"),
                "⚖️",
            )
            _render_param_badge(
                "Bias (Intercept)", f"{base['bias']:.4f}", "",
                "Hedef sütunun ortalaması — modelin başlangıç tahmini",
                "🎯",
            )

    # ── 3. Manuel Kontrol Paneli ─────────────────────────────────────────────
    st.markdown("#### ✋ Manuel Parametre Ayarı")
    st.info("💡 Önerilen değerler sliderlara varsayılan olarak yüklenmiştir. Dilediğiniz gibi değiştirebilirsiniz.")

    col1, col2, col3 = st.columns(3)

    with col1:
        manual_lr = st.select_slider(
            "📈 Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
            value=_nearest(base["learning_rate"], [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]),
            key="hp_lr",
            help="Öğrenme adım büyüklüğü. Küçük = yavaş ama kararlı, Büyük = hızlı ama kararsız.",
        )
        manual_bias = st.number_input(
            "🎯 Bias / Intercept",
            value=round(base["bias"], 4),
            step=0.0001,
            format="%.4f",
            key="hp_bias",
            help="Modelin başlangıç tahmini (sabit terim). Genellikle hedef ortalaması.",
        )

    with col2:
        manual_batch = st.select_slider(
            "📦 Batch Size",
            options=[8, 16, 32, 64, 128, 256, 512],
            value=_nearest(base["batch_size"], [8, 16, 32, 64, 128, 256, 512]),
            key="hp_batch",
            help="Her adımda işlenen örnek sayısı. Küçük = daha fazla güncelleme, Büyük = daha hızlı.",
        )
        manual_patch = st.select_slider(
            "🧩 Patch Size",
            options=[4, 8, 16, 32, 64, 128],
            value=_nearest(base["patch_size"], [4, 8, 16, 32, 64, 128]),
            key="hp_patch",
            help="Transformer/patch tabanlı modeller için girdi parça büyüklüğü.",
        )

    with col3:
        manual_epochs = st.slider(
            "🔄 Epochs",
            min_value=5,
            max_value=500,
            value=int(base["epochs"]),
            step=5,
            key="hp_epochs",
            help="Veri setinin kaç kez tam dolaşılacağı. Fazla epoch = overfitting riski.",
        )
        manual_reg = st.select_slider(
            "⚖️ Regularization (L2)",
            options=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            value=_nearest(base["regularization"], [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
            key="hp_reg",
            help="Overfitting önleme gücü. Büyük = daha fazla düzenleme.",
        )

    # Manuel parametreleri topla
    manual_params = {
        "learning_rate":  manual_lr,
        "batch_size":     manual_batch,
        "patch_size":     manual_patch,
        "epochs":         manual_epochs,
        "regularization": manual_reg,
        "bias":           manual_bias,
    }

    # Canlı risk göstergesi (manuel değerler değişince anında güncellenir)
    live_risks = analyze_risk(manual_params, stats)
    if live_risks:
        st.markdown("##### ⚠️ Anlık Risk Analizi")
        for r in live_risks:
            st.warning(f"⚠️ {r}")
    else:
        st.success("✅ Anlık risk analizi: Seçilen parametreler güvenli görünüyor.")

    # Parametreleri session_state'e yaz (eğitim sırasında kullanılacak)
    st.session_state["smart_params"] = manual_params

    # ── 4. Optuna Otomatik Optimizasyon ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🚀 Optuna ile Otomatik Optimizasyon")
    st.caption(
        "Optuna, manuel ayarladığınız değerleri başlangıç noktası olarak alarak "
        "Bayesyen arama ile en iyi kombinasyonu bulur."
    )

    opt_col1, opt_col2 = st.columns([1, 2])
    with opt_col1:
        n_trials = st.number_input(
            "Trial Sayısı",
            min_value=5, max_value=100, value=15, step=5,
            key="hp_n_trials",
            help="Daha fazla trial = daha iyi optimizasyon ama daha uzun süre.",
        )
    with opt_col2:
        st.markdown("<br/>", unsafe_allow_html=True)
        run_optuna = st.button("🔍 Optuna'yı Başlat", key="btn_run_optuna", use_container_width=True)

    if run_optuna:
        sidebar_log("🚀 Optuna optimizasyonu başlatıldı...", "info")
        st.markdown("##### 📡 Canlı Optimizasyon Logu")
        log_placeholder = st.empty()

        with st.spinner("🔄 Optuna Bayesyen Optimizasyonu çalışıyor…"):
            best = run_optimization(
                base_params=manual_params,
                train_fn=train_fn,
                n_trials=int(n_trials),
                progress_placeholder=log_placeholder,
            )

        result_payload["optimized"] = best
        st.session_state["smart_params"] = best

        risks = analyze_risk(best, stats)
        result_payload["risks"] = risks

        st.success("✅ Optimizasyon tamamlandı! En iyi parametreler aşağıda:")

        # Öneri vs Optuna karşılaştırma tablosu
        _rows = []
        for k in ["learning_rate", "batch_size", "patch_size", "epochs", "regularization", "bias"]:
            _rows.append({
                "Parametre":           k,
                "Önerilen (Heuristic)": base.get(k, "—"),
                "Manuel":              manual_params.get(k, "—"),
                "Optuna (En İyi)":     best.get(k, "—"),
            })
        import pandas as _pd
        st.dataframe(_pd.DataFrame(_rows), use_container_width=True, hide_index=True)

        if risks:
            for r in risks:
                st.warning(f"⚠️ **Risk:** {r}")
        else:
            st.success("✅ Optimizasyon sonrası risk analizi: Temiz.")

        sidebar_log("✅ Optuna tamamlandı", "success")

    return result_payload


# ─── YARDIMCI FONKSİYON ───────────────────────────────────────────────────────

def _nearest(value: float, options: list):
    """Verilen değere en yakın seçeneği döner (slider için)."""
    return min(options, key=lambda x: abs(x - value))
