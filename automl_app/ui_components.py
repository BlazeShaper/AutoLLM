import os
import streamlit as st
import pandas as pd

from logger import render_sidebar_logs

# ─── Sabit Metinler (Constants) ──────────────────────────────────────────────
MODE_STANDARD_LBL = "🚀 Standart Mod"
MODE_EXPERT_LBL = "🔬 Uzman Modu"
MODE_TOGGLE_HELP = "Standart mod hızlıdır. Uzman mod tüm kontrolleri açar."
REGISTRY_DEFAULT_SELECT = "— Seçin —"


# ─── CSS Enjeksiyonu ─────────────────────────────────────────────────────────

def inject_css():
    """Uygulama geneli için özel CSS kurallarını enjekte eder."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Kart stili — tek ton, gradyan yok */
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #2d2d3f;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 {
        color: #9ca3af;
        font-size: 0.72rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.2rem;
    }
    .metric-card p {
        color: #e5e7eb;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0;
    }

    /* Step indicator */
    .step-done   { color: #4ade80; font-weight: 600; }
    .step-active { color: #818cf8; font-weight: 700; font-size: 1.05rem; }
    .step-wait   { color: #6b7280; }

    /* Section başlığı — sadece sol çizgi, renk nötr */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #d1d5db;
        border-left: 3px solid #6366f1;
        padding-left: 0.55rem;
        margin: 0.75rem 0 0.4rem;
    }

    /* Buton — gölge kaldırıldı, yalnızca hafif yükselme */
    div.stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 500;
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
    }
    </style>
    """, unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────

def render_header():
    """Ana sayfa başlığını ve alt metnini render eder."""
    inject_css()
    st.markdown(
        """
        <div style='padding: 1.2rem 0 0.4rem;'>
          <h1 style='font-size:2rem; font-weight:700; color:#e5e7eb; margin:0;'>
            OutoLLM &mdash; AutoML Platform
          </h1>
          <p style='color:#6b7280; font-size:0.9rem; margin-top:0.25rem;'>
            Verilerinizden yapay zeka modelleri saniyeler içinde oluşturun.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_mode_toggle():
    """Kullanıcı modunu (Standart/Uzman) seçmek için sidebar toggle render eder."""
    st.sidebar.markdown("## ⚙️ Uygulama Ayarları")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "Kullanım Modu:",
        (MODE_STANDARD_LBL, MODE_EXPERT_LBL),
        help=MODE_TOGGLE_HELP
    )
    st.session_state["mode"] = "standard" if mode == MODE_STANDARD_LBL else "expert"

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Pipeline Durumu")

    stages = ["Veri Yükleme", "Temizleme", "Eğitim", "Değerlendirme", "Tahmin", "EDA"]
    flags = [
        st.session_state.get("raw_df") is not None,
        st.session_state.get("cleaned_df") is not None,
        st.session_state.get("best_model") is not None,
        st.session_state.get("leaderboard") is not None,
        st.session_state.get("model_card") is not None,
        st.session_state.get("raw_df") is not None,   # EDA veri yüklendi mi
    ]
    for stage, done in zip(stages, flags):
        icon = "✅" if done else "⏳"
        st.sidebar.markdown(f"{icon} {stage}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small style='color:#6b7280;'>v1.0.0 · PyCaret 3.x · Streamlit</small>",
        unsafe_allow_html=True
    )

    # ── Model Kaydı ──────────────────────────────────────────────────
    render_model_registry()

    # ── Uygulama Logları ──────────────────────────────────────────
    render_sidebar_logs()


# ─── Model Registry (Sidebar) ────────────────────────────────────────

def render_model_registry():
    """saved_models/ altındaki .pkl dosyalarını listeler; seçilen model yüklenir."""
    save_dir = os.path.join(os.getcwd(), "saved_models")
    if not os.path.isdir(save_dir):
        return

    pkl_files = [
        f for f in os.listdir(save_dir)
        if f.endswith(".pkl")
    ]
    if not pkl_files:
        return

    pkl_files_sorted = sorted(pkl_files, reverse=True)   # en yeni üste

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗄️ Kaydedilmiş Modeller")

    selected = st.sidebar.selectbox(
        "Model seçin:",
        [REGISTRY_DEFAULT_SELECT] + pkl_files_sorted,
        key="registry_select",
    )

    if selected and selected != REGISTRY_DEFAULT_SELECT:
        full_path = os.path.join(save_dir, selected)
        card_path = full_path.replace(".pkl", "_card.json")

        # JSON kartını sidebar'da göster
        if os.path.isfile(card_path):
            import json
            with open(card_path, encoding="utf-8") as f:
                card = json.load(f)
            with st.sidebar.expander("📄 Model Detayı", expanded=False):
                st.sidebar.write(f"🎯 Görev: `{card.get('task_type', '?')}`")
                st.sidebar.write(f"📅 Tarih: `{card.get('timestamp', '?')}`")
                st.sidebar.write(f"📊 Satır: `{card.get('num_rows_trained', '?'):,}`")
                st.sidebar.write(f"🔢 Özellik: `{card.get('num_features', '?')}`")
                st.sidebar.write(f"🎯 Hedef: `{card.get('target_col', '?')}`")

        # Modeli oturuma yükle butonu
        if st.sidebar.button("🔄 Bu Modeli Yükle", key="btn_registry_load"):
            import pycaret.classification as pc_c
            import pycaret.regression   as pc_r
            import pycaret.clustering   as pc_cl
            task = card.get("task_type", "classification") if os.path.isfile(card_path) else "classification"
            _pc = {"classification": pc_c, "regression": pc_r, "clustering": pc_cl}.get(task, pc_c)
            model_name = full_path.removesuffix(".pkl")
            try:
                loaded = _pc.load_model(model_name)
                st.session_state["best_model"] = loaded
                st.session_state["pc_module"]   = _pc
                if os.path.isfile(card_path):
                    st.session_state["model_card"]  = card
                    st.session_state["task_type"]   = task
                st.sidebar.success(f"✅ `{selected}` oturuma yüklendi!")
            except Exception as e:
                st.sidebar.error(f"❌ Yükleme hatası: {e}")


# ─── Step Indicator ──────────────────────────────────────────────────────────

def render_step_indicator(current_step: int):
    """Uygulama akışını gösteren yatay adım belirtecini çizer."""
    steps = ["📂 Yükleme", "🧹 Temizlik", "🧠 Eğitim", "📊 Değerlendirme", "🔮 Tahmin"]
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i < current_step:
                st.markdown(f"<p class='step-done'>✅ {step}</p>", unsafe_allow_html=True)
            elif i == current_step:
                st.markdown(f"<p class='step-active'>▶ {step}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='step-wait'>○ {step}</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(99,102,241,0.3);'>", unsafe_allow_html=True)


# ─── Veri Önizleme ───────────────────────────────────────────────────────────

def render_dataframe_preview(df: pd.DataFrame, title: str = "📋 Veri Önizleme"):
    """Yüklenen veya temizlenen verinin özet istatistiklerini ve ilk satırlarını gösterir."""
    st.markdown(f"<p class='section-title'>{title}</p>", unsafe_allow_html=True)

    # Özet metrik kartları
    cols = st.columns(4)
    with cols[0]:
        st.metric("Satır Sayısı", f"{df.shape[0]:,}")
    with cols[1]:
        st.metric("Sütun Sayısı", f"{df.shape[1]}")
    with cols[2]:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        st.metric("Eksik Veri", f"{missing_pct:.1f}%")
    with cols[3]:
        dup_count = df.duplicated().sum()
        st.metric("Yinelenen Satır", f"{dup_count:,}")

    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("🔍 Veri Türleri & İstatistikler"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Sütun Bilgileri**")
            info_df = pd.DataFrame({
                "Sütun": df.columns,
                "Tür": df.dtypes.values.astype(str),
                "Benzersiz": [df[c].nunique() for c in df.columns],
                "Eksik": [df[c].isnull().sum() for c in df.columns],
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("**Sayısal Sütun İstatistikleri**")
            numeric_df = df.select_dtypes(include="number")
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe().T.round(3), use_container_width=True)
            else:
                st.info("Sayısal sütun bulunamadı.")


# ─── Uyarı Banner ────────────────────────────────────────────────────────────

def render_warning_banner(msg: str):
    """Ekranda uyarı banner'ı çıkarır."""
    st.warning(msg, icon="⚠️")
