import streamlit as st
import pandas as pd
import io

from logger import log, sidebar_log

try:
    import ydata_profiling
    import streamlit.components.v1 as components
    HAS_PROFILING = True
except ImportError:
    HAS_PROFILING = False

from config import MAX_FILE_MB


# ─── Veri Yükleme ────────────────────────────────────────────────────────────

def load_data(uploaded_file) -> pd.DataFrame | None:
    """CSV veya Excel dosyasını yükler; boyut ve format kontrolü yapar."""
    if uploaded_file is None:
        return None

    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_MB:
        st.error(
            f"❌ Dosya boyutu çok büyük: **{file_size_mb:.2f} MB**. "
            f"Maksimum izin verilen: **{MAX_FILE_MB} MB**."
        )
        return None

    try:
        name = uploaded_file.name.lower()
        log.info(f"Dosya yükleme başladı: {name} ({uploaded_file.size / 1024:.1f} KB)")
        sidebar_log(f"📂 Dosya okunuyor: {name}", "info")

        if name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                log.warning(f"{name} UTF-8 okunamadı, latin-1 deneniyor.")
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin-1")
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Desteklenmeyen format. Lütfen **CSV** veya **Excel** yükleyin.")
            log.error(f"Desteklenmeyen dosya formatı: {name}")
            return None

        if df.empty:
            st.error("❌ Yüklenen dosya boş görünüyor.")
            log.error(f"{name} boş DataFrame döndürdü.")
            return None

        est_min = max(1, int(df.shape[0] / 10000))
        est_max = max(3, int(df.shape[0] / 5000))
        log.info(f"Veri yüklendi: {df.shape[0]} satır × {df.shape[1]} sütun")
        sidebar_log(f"✅ {df.shape[0]:,} satır × {df.shape[1]} sütun yüklendi", "success")
        st.success(
            f"✅ Veri yüklendi! **{df.shape[0]:,}** satır, **{df.shape[1]}** sütun. "
            f"Tahmini eğitim süresi: **{est_min}–{est_max} dk**."
        )
        return df

    except Exception as e:
        log.exception(f"Dosya okunurken beklenmeyen hata: {e}")
        sidebar_log(f"❌ Yükleme hatası: {e}", "error")
        st.error(f"❌ Dosya okunurken hata: {e}")
        return None


# ─── Profil Raporu ───────────────────────────────────────────────────────────

def generate_profile(df: pd.DataFrame):
    """ydata_profiling ile minimal HTML rapor oluşturur."""
    if not HAS_PROFILING:
        log.warning("ydata_profiling kurulu değil, profil raporu atlandı.")
        st.warning(
            "⚠️ `ydata_profiling` yüklü değil. "
            "Yüklemek için: `pip install ydata-profiling`"
        )
        return

    log.info("Profil raporu oluşturuluyor (minimal mod)...")
    sidebar_log("📊 Profil raporu oluşturuluyor…", "info")
    with st.spinner("📊 Profil raporu oluşturuluyor (minimal mod)..."):
        profile = ydata_profiling.ProfileReport(df, minimal=True, title="Veri Profili")
        profile_html = profile.to_html()
        components.html(profile_html, height=650, scrolling=True)
    log.info("Profil raporu tamamlandı.")
    sidebar_log("✅ Profil raporu hazır", "success")


# ─── Temizlik Sorunları Tespiti ──────────────────────────────────────────────

def detect_cleaning_issues(df: pd.DataFrame) -> dict:
    """
    Sütun bazlı potansiyel sorunları döndürür.
    Anahtar: sütun adı, Değer: {reason, action, severity}
    """
    issues = {}
    num_rows = len(df)
    if num_rows == 0:
        return issues

    for col in df.columns:
        num_unique = df[col].nunique(dropna=True)
        num_missing = df[col].isnull().sum()
        missing_pct = num_missing / num_rows

        if num_unique == 1:
            issues[col] = {
                "reason": "Tüm satırlarda aynı değer var — modele katkısı sıfır.",
                "action": "Sütunu çıkar",
                "severity": "high",
            }
        elif num_unique == 0:
            issues[col] = {
                "reason": "Sütun tamamen boş.",
                "action": "Sütunu çıkar",
                "severity": "high",
            }
        elif missing_pct > 0.6:
            issues[col] = {
                "reason": f"**%{missing_pct*100:.0f}** eksik veri içeriyor.",
                "action": "Sütunu çıkarmanızı öneririz",
                "severity": "high",
            }
        elif missing_pct > 0.4:
            issues[col] = {
                "reason": f"**%{missing_pct*100:.0f}** eksik veri içeriyor.",
                "action": "Eksik değerleri doldurun ya da sütunu çıkarın",
                "severity": "medium",
            }
        elif num_unique / num_rows > 0.9 and df[col].dtype == "object":
            issues[col] = {
                "reason": "Çok yüksek benzersiz değer oranı (%90+). Muhtemelen ID/metin sütunu.",
                "action": "Sütunu çıkar",
                "severity": "medium",
            }

    return issues


# ─── Temizlik Uygulama ───────────────────────────────────────────────────────

def apply_cleaning(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """Seçilen sütunları düşürür ve temizlenmiş kopyayı döndürür."""
    cleaned = df.copy()
    if cols_to_drop:
        log.info(f"Temizlik uygulanıyor — çıkarılan sütunlar: {cols_to_drop}")
        sidebar_log(f"🧹 {len(cols_to_drop)} sütun kaldırıldı: {', '.join(cols_to_drop)}", "info")
        cleaned = cleaned.drop(columns=cols_to_drop, errors="ignore")
        log.info(f"Temizlik sonrası: {cleaned.shape[0]} satır × {cleaned.shape[1]} sütun")
    else:
        log.info("Temizlik: kaldırılacak sütun seçilmedi, DataFrame değişmedi.")
    return cleaned


# ─── Duplikat Kaldırma ───────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Birebir yinelenen satırları kaldırır.
    Returns: (temizlenmiş_df, kaldırılan_satır_sayısı)
    """
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)
    log.info(f"Duplikat kaldırma: {removed} satır kaldırıldı ({before} → {len(cleaned)})")
    if removed > 0:
        sidebar_log(f"🗑️ {removed} yinelenen satır kaldırıldı", "info")
    return cleaned, removed


# ─── Aykırı Değer Tespiti (IQR) ─────────────────────────────────────────────

def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    """
    IQR yöntemine göre sayısal sütunlardaki aykırı değer yüzdesini hesaplar.
    Returns: {sütun_adı: {"outlier_count": int, "outlier_pct": float, "q1": float, "q3": float}}
    """
    report = {}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        pct = outliers / len(series) * 100
        if outliers > 0:
            report[col] = {
                "outlier_count": int(outliers),
                "outlier_pct": round(pct, 1),
                "q1": round(float(q1), 4),
                "q3": round(float(q3), 4),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
            }
    return report


def clip_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Seçilen sütunlardaki aykırı değerleri IQR sınırlarına kırpar (clip).
    """
    clipped = df.copy()
    for col in cols:
        if col not in clipped.select_dtypes(include="number").columns:
            continue
        q1 = clipped[col].quantile(0.25)
        q3 = clipped[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        clipped[col] = clipped[col].clip(lower=lower, upper=upper)
    log.info(f"Aykırı değer kırpma uygulandı: {cols}")
    sidebar_log(f"✂️ {len(cols)} sütunda outlier kırpıldı", "info")
    return clipped


# ─── Eksik Veri Özeti ────────────────────────────────────────────────────────

def render_missing_summary(df: pd.DataFrame):
    """Eksik veri barları ile özet tablo gösterir."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.success("✅ Eksik veri bulunamadı.")
        return

    st.markdown("#### Eksik Veri Dağılımı")
    miss_df = pd.DataFrame({
        "Sütun": missing.index,
        "Eksik Sayısı": missing.values,
        "Oran (%)": (missing.values / len(df) * 100).round(1),
    })
    st.dataframe(miss_df, use_container_width=True, hide_index=True)


# ─── Eksik Veri Doldurma ─────────────────────────────────────────────────────

def fill_missing_values(
    df: pd.DataFrame,
    numeric_method: str = "mean",
    categorical_method: str = "mode",
    fill_value: str = "0",
) -> pd.DataFrame:
    """
    Sayısal ve kategorik sütunlardaki eksik değerleri doldurur.

    numeric_method  : "mean" | "median" | "zero" | "sabit"
    categorical_method: "mode" | "sabit"
    fill_value      : "sabit" seçildiğinde kullanılacak değer (str olarak gelir,
                       sayısal sütunlar için float dönüşümü denenir)
    Returns: doldurulmuş DataFrame kopyası
    """
    filled = df.copy()
    num_cols  = filled.select_dtypes(include="number").columns.tolist()
    cat_cols  = filled.select_dtypes(exclude="number").columns.tolist()

    # ── Sayısal sütunlar
    for col in num_cols:
        if filled[col].isnull().sum() == 0:
            continue
        if numeric_method == "mean":
            filled[col] = filled[col].fillna(filled[col].mean())
        elif numeric_method == "median":
            filled[col] = filled[col].fillna(filled[col].median())
        elif numeric_method == "zero":
            filled[col] = filled[col].fillna(0)
        elif numeric_method == "sabit":
            try:
                filled[col] = filled[col].fillna(float(fill_value))
            except ValueError:
                filled[col] = filled[col].fillna(0)

    # ── Kategorik sütunlar
    for col in cat_cols:
        if filled[col].isnull().sum() == 0:
            continue
        if categorical_method == "mode":
            mode_val = filled[col].mode()
            if not mode_val.empty:
                filled[col] = filled[col].fillna(mode_val[0])
        elif categorical_method == "sabit":
            filled[col] = filled[col].fillna(str(fill_value))

    filled_count = df.isnull().sum().sum() - filled.isnull().sum().sum()
    log.info(
        f"Eksik veri doldurma tamamlandı: {filled_count} hücre dolduruldu "
        f"(sayısal={numeric_method}, kategorik={categorical_method})"
    )
    sidebar_log(f"✅ {filled_count} eksik hücre dolduruldu", "success")
    return filled
