"""
OutoLLM — Dinamik Hiperparametre ve Optimizasyon Modülü
Veri setinin istatistiksel analizine dayalı hiperparametre önerileri
ve Optuna tabanlı Bayesyen Optimizasyon motoru içerir.
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
    """Veri setinin istatistiksel yapısını analiz eder.
    
    Args:
        df (pd.DataFrame): Analiz edilecek tam veri seti.
        target (str): Hedef değişkenin sütun adı.
        
    Returns:
        Dict[str, float]: n_rows, n_features, skewness, kurtosis, sparsity, ve korelasyon metrikleri.
    """
    log.info(f"Veri seti analizi başlatıldı. Hedef: {target}")
    
    if target not in df.columns:
        raise ValueError(f"Hedef sütun '{target}' veri setinde bulunamadı.")

    # Sayısal sütunları seç (hedef hariç)
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

    # Skewness ve Kurtosis
    skew_vals = skew(X, nan_policy="omit")
    kurt_vals = kurtosis(X, nan_policy="omit")

    # Sparsity (Sıfır oranı)
    sparsity = float((X == 0).sum().sum() / (n * p)) if (n * p) > 0 else 0.0

    # Korelasyon (Sadece hedef sayısal ise hesaplanabilir, aksi halde dummy)
    if y_is_numeric and p > 0:
        corr = X.corrwith(y).abs()
        corr_mean = float(corr.mean()) if not corr.isna().all() else 0.0
        corr_max = float(corr.max()) if not corr.isna().all() else 0.0
    else:
        # Sınıflandırma durumunda veya sayısal değilse korelasyonu şimdilik 0 varsay
        corr_mean = 0.0
        corr_max = 0.0

    stats = {
        "n_rows": n,
        "n_features": p,
        "skewness_mean": float(np.nanmean(skew_vals)) if not np.isnan(np.nanmean(skew_vals)) else 0.0,
        "kurtosis_mean": float(np.nanmean(kurt_vals)) if not np.isnan(np.nanmean(kurt_vals)) else 0.0,
        "sparsity": sparsity,
        "corr_mean": corr_mean,
        "corr_max": corr_max
    }
    
    log.info(f"Veri analizi tamamlandı: {stats}")
    return stats


# ─── SEZGİSEL (HEURISTIC) MOTOR (STEP 2) ─────────────────────────────────────

def compute_batch_size(n: int, p: int, sparsity: float, gpu_memory_gb: float = 8.0) -> int:
    """Veri boyutu ve seyrekliğine göre ideal batch size hesaplar."""
    base = int(np.sqrt(n))
    if sparsity > 0.7:
        base = int(base * 0.5)

    mem_limit = (gpu_memory_gb * 1024) / max(p * 4, 1)  # Sıfıra bölme koruması
    batch = int(min(base, mem_limit))
    return int(np.clip(batch, 16, 512))


def compute_learning_rate(skewness: float, kurtosis: float) -> float:
    """Veri dağılımı çarpıklığına göre öğrenme oranını ayarlar."""
    lr = 0.01
    if abs(skewness) > 1:
        lr *= 0.5
    if kurtosis > 3:
        lr *= 0.7
    return float(np.clip(lr, 0.0005, 0.05))


def compute_regularization(corr_mean: float, corr_max: float) -> float:
    """Korelasyon metriklerine göre L2 (Ridge) regulasyon çarpanı belirler."""
    if corr_max > 0.7:
        return 0.01
    elif corr_mean < 0.1:
        return 10.0
    return 1.0


def compute_patch_size(p: int, n: int) -> int:
    """Boyutlara göre varsayımsal patch büyüklüğü (örn. transformer tabanlı modeller için)."""
    if p > 100:
        return 32
    elif n < 2000:
        return 8
    return 16


def heuristic_params(stats: Dict[str, float], gpu_memory_gb: float = 8.0) -> Dict[str, float]:
    """Tüm sezgisel hesaplamaları birleştirip bir hyperparam dict'i döner."""
    return {
        "batch_size": compute_batch_size(
            stats["n_rows"],
            stats["n_features"],
            stats["sparsity"],
            gpu_memory_gb
        ),
        "learning_rate": compute_learning_rate(
            stats["skewness_mean"],
            stats["kurtosis_mean"]
        ),
        "regularization": compute_regularization(
            stats["corr_mean"],
            stats["corr_max"]
        ),
        "patch_size": compute_patch_size(
            stats["n_features"],
            stats["n_rows"]
        )
    }


# ─── OPTUNA OPTİMİZASYONU (STEP 3) ───────────────────────────────────────────

def get_search_space(base: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """Heuristic temel değerleri baz alarak Optuna arama uzayını (search space) üretir."""
    return {
        "lr": (base["learning_rate"] * 0.5, base["learning_rate"] * 2),
        "batch": (max(16, int(base["batch_size"] * 0.5)), min(512, int(base["batch_size"] * 1.5))),
        "reg": (base["regularization"] * 0.1, base["regularization"] * 10)
    }


def objective(trial: optuna.Trial, base_params: Dict[str, float], train_fn: Callable) -> float:
    """Optuna tarafından minimize/maksimize edilecek objektif fonksiyon."""
    space = get_search_space(base_params)

    params = {
        "learning_rate": trial.suggest_float("lr", space["lr"][0], space["lr"][1], log=True),
        "batch_size": trial.suggest_int("batch", int(space["batch"][0]), int(space["batch"][1])),
        "regularization": trial.suggest_float("reg", space["reg"][0], space["reg"][1], log=True)
    }
    
    # patch_size optimizasyona dahil değilse sabit bırak
    params["patch_size"] = base_params.get("patch_size", 16)

    # Kullanıcı tanımlı eğitim fonksiyonunu çağır (Optuna hata yakalamayı destekler)
    try:
        score = train_fn(params)
        return float(score)
    except Exception as e:
        log.warning(f"Optuna trial hatası: {e}")
        raise optuna.TrialPruned()


def run_optimization(base_params: Dict[str, float], train_fn: Callable, n_trials: int = 20) -> Dict[str, float]:
    """Bayesyen optimizasyonu başlatır ve en iyi parametre setini döner."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")

    log.info(f"Optuna optimizasyonu başladı ({n_trials} trial)...")
    try:
        study.optimize(
            lambda trial: objective(trial, base_params, train_fn),
            n_trials=n_trials,
            n_jobs=1  # Streamlit ile güvenli çalışması için 1 (thread-safe)
        )
    except Exception as e:
        log.error(f"Optuna çalışırken hata oluştu: {e}")
        return base_params  # Hata durumunda base paramlara geri dön

    best = study.best_params
    
    # Optuna'nın kullandığı kısa isimleri asıl isimlere mapliyoruz
    optimized = {
        "learning_rate": best.get("lr", base_params["learning_rate"]),
        "batch_size": best.get("batch", base_params["batch_size"]),
        "regularization": best.get("reg", base_params["regularization"]),
        "patch_size": base_params["patch_size"]
    }
    log.info(f"Optuna tamamlandı. En iyi skor: {study.best_value}. Parametreler: {optimized}")
    return optimized


# ─── RİSK ANALİZİ (STEP 4) ───────────────────────────────────────────────────

def analyze_risk(params: Dict[str, float], stats: Dict[str, float]) -> List[str]:
    """Optimum parametrelerin veri seti üzerinde yaratabileceği riskleri analiz eder."""
    risks = []

    if params["batch_size"] < 32 and stats["n_rows"] < 1000:
        risks.append("overfitting (Aşırı Öğrenme Riski - Düşük batch size & küçük veri seti)")

    if params["learning_rate"] < 0.001:
        risks.append("vanishing_gradient (Kaybolan Gradyan Riski - Çok düşük öğrenme oranı)")

    if params["learning_rate"] > 0.05:
        risks.append("unstable_training (Kararsız Eğitim Riski - Yüksek öğrenme oranı)")

    return risks


# ─── STREAMLIT ENTEGRASYONU (STEP 5) ─────────────────────────────────────────

def render_smart_hyperparams(df: pd.DataFrame, target: str, train_fn: Callable) -> Dict[str, Any]:
    """Arayüzde akıllı hiperparametre analizini gösterir ve tüm sonuçları içeren dict döner."""
    
    st.markdown("---")
    st.markdown("### 💡 Akıllı Hiperparametre Motoru (Optuna)")
    st.caption("Veri setinizin istatistiksel özelliklerine dayalı dinamik hiperparametre önerileri.")

    # 1. Analiz ve Sezgisel Parametreler
    with st.spinner("📊 Veri istatistikleri çıkarılıyor..."):
        try:
            stats = analyze_dataset(df, target)
            base = heuristic_params(stats)
        except Exception as e:
            st.error(f"❌ Analiz hatası: {e}")
            log.exception(f"analyze_dataset başarısız: {e}")
            return {}

    # Çıktı Dictionary Başlangıcı
    result_payload = {
        "stats": stats,
        "heuristic": base,
        "optimized": None,
        "risks": []
    }

    st.write("**Sezgisel (Heuristic) Başlangıç Değerleri:**")
    st.json(base)

    if "smart_params" in st.session_state and st.session_state["smart_params"] is not None:
        st.info("✅ Optimizasyon tamamlanmış ve hafızaya alınmış.")
        st.json(st.session_state["smart_params"])

    if st.button("🚀 Optuna ile Optimize Et", key="btn_run_optuna"):
        sidebar_log("🚀 Optuna optimizasyonu başlatıldı...", "info")
        with st.spinner("🔄 Optuna Bayesyen Optimizasyonu çalışıyor (Bu işlem biraz sürebilir)..."):
            # 2. Optimizasyon
            best = run_optimization(base, train_fn, n_trials=10) # Hız için 10 trial
            result_payload["optimized"] = best
            st.session_state["smart_params"] = best
            
            st.success("✅ Optimizasyon Tamamlandı! En İyi Parametreler:")
            st.json(best)
            sidebar_log("✅ Optuna tamamlandı", "success")

            # 3. Risk Analizi
            risks = analyze_risk(best, stats)
            result_payload["risks"] = risks

            if risks:
                for risk in risks:
                    st.warning(f"⚠️ **Risk Tespit Edildi:** {risk}")
            else:
                st.info("✅ Belirgin bir risk tespit edilmedi.")

    return result_payload
