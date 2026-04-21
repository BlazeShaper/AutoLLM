"""
OutoLLM — Merkezi Logging Modülü
Her pipeline adımı hem logs.log dosyasına hem de (isteğe bağlı)
Streamlit sidebar'a yazılır.
"""
import logging
import os
from datetime import datetime

# ─── Log Dosyası Yolu ────────────────────────────────────────────────────────

LOG_DIR  = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "logs.log")

# ─── Logger Oluşturma ────────────────────────────────────────────────────────

def _build_logger() -> logging.Logger:
    """Singleton logger; tekrar çağrılırsa aynı örneği döner."""
    logger = logging.getLogger("outollm")

    if logger.handlers:          # zaten yapılandırıldıysa yeniden ekleme
        return logger

    logger.setLevel(logging.DEBUG)

    # — Dosya handler —
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] [%(module)-20s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # — Konsol handler (terminal çıkışı) —
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "[%(levelname)s] %(module)s › %(message)s"
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─── Dışa Açık Erişim Noktası ────────────────────────────────────────────────

log = _build_logger()

# ─── Yardımcı: Streamlit Sidebar Logu ────────────────────────────────────────

def sidebar_log(msg: str, level: str = "info"):
    """
    Streamlit sidebar'a renkli log satırı ekler.
    Streamlit context dışında çağrılırsa sessizce geçer.
    """
    try:
        import streamlit as st

        _colors = {
            "info":    "#60a5fa",
            "success": "#4ade80",
            "warning": "#fbbf24",
            "error":   "#f87171",
        }
        color   = _colors.get(level, "#a0a0c0")
        ts      = datetime.now().strftime("%H:%M:%S")
        line    = (
            f"<span style='color:#6b7280;font-size:0.72rem;'>{ts}</span> "
            f"<span style='color:{color};font-size:0.78rem;'>{msg}</span>"
        )

        if "sidebar_logs" not in st.session_state:
            st.session_state["sidebar_logs"] = []

        # Son 50 satırı tut
        st.session_state["sidebar_logs"].append(line)
        st.session_state["sidebar_logs"] = st.session_state["sidebar_logs"][-50:]

    except Exception:
        pass   # Streamlit yoksa (birim test, CLI) hiçbir şey yapma


def render_sidebar_logs():
    """
    ui_components.py tarafından sidebar'ın altında çağrılır.
    Son log satırlarını gösterir.
    """
    try:
        import streamlit as st

        logs = st.session_state.get("sidebar_logs", [])
        if not logs:
            return

        with st.sidebar.expander("📋 Uygulama Logları", expanded=False):
            html = "<br>".join(reversed(logs))
            st.markdown(
                f"<div style='font-family:monospace;line-height:1.6;'>{html}</div>",
                unsafe_allow_html=True,
            )
            if st.button("🗑️ Logları Temizle", key="btn_clear_logs"):
                st.session_state["sidebar_logs"] = []
                st.rerun()

    except Exception:
        pass
