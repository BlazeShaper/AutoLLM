import os
import streamlit as st
import pandas as pd
import pycaret
import pycaret.classification as pc_class
import pycaret.regression as pc_reg
import pycaret.clustering as pc_clust

from logger import log, sidebar_log


# ─── Model Yükleme ───────────────────────────────────────────────────────────

def load_model_safe(pkl_path: str, task_type: str):
    """
    .pkl uzantısından model adını çıkarır ve ilgili PyCaret modülüyle yükler.
    Başarısız olursa None döner.
    """
    try:
        model_name = pkl_path.removesuffix(".pkl") if pkl_path.endswith(".pkl") else pkl_path
        log.info(f"Model yükleniyor: {model_name} | görev: {task_type}")
        sidebar_log(f"🔄 Model yükleniyor: {os.path.basename(pkl_path)}", "info")

        if task_type == "classification":
            model = pc_class.load_model(model_name)
        elif task_type == "regression":
            model = pc_reg.load_model(model_name)
        elif task_type == "clustering":
            model = pc_clust.load_model(model_name)
        else:
            st.error(f"❌ Geçersiz görev tipi: **{task_type}**")
            log.error(f"Geçersiz task_type: {task_type}")
            return None

        log.info(f"Model başarıyla yüklendi: {type(model).__name__}")
        sidebar_log(f"✅ Model yüklendi: {type(model).__name__}", "success")
        return model

    except Exception as e:
        log.exception(f"Model yüklenirken hata: {e}")
        sidebar_log(f"❌ Model yükleme hatası: {e}", "error")
        st.error(f"❌ Model yüklenirken hata: {e}")
        return None


# ─── Uyumluluk Doğrulaması ──────────────────────────────────────────────────

def validate_compatibility(model_card: dict, df: pd.DataFrame) -> bool:
    """
    Modelin kimlik kartı ile yüklenen veri setini karşılaştırır.
    Eksik sütun yoksa True döner.
    """
    warnings_list = []
    errors_list = []

    # Sürüm kontrolü
    current_ver = pycaret.__version__
    card_ver = model_card.get("pycaret_version", "?")
    if card_ver != current_ver:
        warnings_list.append(
            f"PyCaret sürüm uyumsuzluğu — model: **{card_ver}**, mevcut: **{current_ver}**."
        )
        log.warning(f"PyCaret sürüm uyumsuzluğu: model={card_ver}, mevcut={current_ver}")

    # Sütun kontrolü
    expected = set(model_card.get("features", []))
    target = model_card.get("target_col")
    if target and target in expected:
        expected.discard(target)

    provided = set(df.columns)
    missing = expected - provided
    extra = provided - expected

    if missing:
        errors_list.append(f"Eksik sütunlar: **{', '.join(sorted(missing))}**")
        log.error(f"Uyumsuz veri: eksik sütunlar = {sorted(missing)}")
        sidebar_log(f"❌ Eksik sütunlar: {', '.join(sorted(missing))}", "error")
    if extra:
        warnings_list.append(f"Fazladan sütunlar (görmezden gelinecek): **{', '.join(sorted(extra))}**")
        log.warning(f"Fazla sütunlar (yok sayılacak): {sorted(extra)}")

    for w in warnings_list:
        st.warning(f"⚠️ {w}")
    for e in errors_list:
        st.error(f"❌ {e}")

    if not missing and not errors_list:
        log.info("Uyumluluk doğrulandı: veri model ile tam uyumlu.")
        sidebar_log("✅ Veri uyumlu", "success")
        st.success("✅ Veri, model ile **tam uyumlu**!")

    return len(missing) == 0


# ─── Tahmin ──────────────────────────────────────────────────────────────────

def run_prediction(model, df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    """
    Modeli kullanarak tahmin üretir ve ham DataFrame döndürür.
    """
    log.info(f"Tahmin başlıyor: {task_type}, {len(df)} satır")
    sidebar_log(f"🔮 Tahmin üretiliyor… ({len(df)} satır)", "info")
    with st.spinner("🔮 Tahminler üretiliyor…"):
        if task_type == "classification":
            result = pc_class.predict_model(model, data=df)
        elif task_type == "regression":
            result = pc_reg.predict_model(model, data=df)
        elif task_type == "clustering":
            result = pc_clust.predict_model(model, data=df)
        else:
            result = df
    log.info(f"Tahmin tamamlandı: {len(result)} satır sonucu")
    sidebar_log(f"✅ {len(result)} tahmin üretildi", "success")
    return result


# ─── Sonuç Formatlama ────────────────────────────────────────────────────────

def format_results(predictions_df: pd.DataFrame, task_type: str):
    """
    Sınıflandırma için düşük güvenli satırları kırmızıyla vurgular.
    Diğer görevler için ham DataFrame döndürür.
    """
    if task_type != "classification":
        return predictions_df

    # PyCaret 3.x sütun isimleri
    score_col = next(
        (c for c in ["prediction_score", "Score"] if c in predictions_df.columns),
        None,
    )

    if score_col is None:
        return predictions_df

    def _highlight(val):
        try:
            return "background-color: #fecaca" if float(val) < 0.60 else ""
        except Exception:
            return ""

    return predictions_df.style.map(_highlight, subset=[score_col])


# ─── Sonuç Özeti ─────────────────────────────────────────────────────────────

def render_prediction_summary(predictions_df: pd.DataFrame, task_type: str):
    """Tahmin sonuçlarının kısa istatistiksel özetini gösterir."""
    pred_col = next(
        (c for c in ["prediction_label", "Label", "Cluster"] if c in predictions_df.columns),
        None,
    )
    if pred_col is None:
        return

    st.markdown("#### 📊 Tahmin Özeti")
    if task_type == "classification":
        counts = predictions_df[pred_col].value_counts()
        summary_df = pd.DataFrame({
            "Sınıf": counts.index.astype(str),
            "Adet": counts.values,
            "Oran (%)": (counts.values / len(predictions_df) * 100).round(1),
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    elif task_type == "regression":
        score_col = next(
            (c for c in ["prediction_label", "Label"] if c in predictions_df.columns),
            None,
        )
        if score_col:
            vals = predictions_df[score_col]
            cols = st.columns(4)
            cols[0].metric("Ortalama", f"{vals.mean():.3f}")
            cols[1].metric("Medyan",   f"{vals.median():.3f}")
            cols[2].metric("Min",      f"{vals.min():.3f}")
            cols[3].metric("Max",      f"{vals.max():.3f}")
    elif task_type == "clustering":
        if "Cluster" in predictions_df.columns:
            cluster_counts = predictions_df["Cluster"].value_counts().reset_index()
            cluster_counts.columns = ["Küme", "Adet"]
            st.dataframe(
                cluster_counts,
                use_container_width=True,
                hide_index=True,
            )
