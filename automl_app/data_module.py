"""
data_module.py — OutoLLM Veri İşleme Modülü (v3.0)
──────────────────────────────────────────────────
Yenilikler (v3.0):
  • Pandera ile schema doğrulama (dtype, range, null kısıtları)
  • @st.cache_data ile yükleme ve temizlik önbelleği
  • Stratified + TimeSeriesSplit train/test split
  • sklearn Pipeline + ColumnTransformer (data leakage önlendi)
  • LabelEncoder / OneHotEncoder / OrdinalEncoder pipeline entegrasyonu
  • Plotly ile eksik veri ısı haritası + korelasyon matrisi
  • mutual_info + chi2 ile feature seçim ve sıralama
  • Zaman serisi: lag feature, rolling feature, TimeSeriesSplit
  • joblib ile pipeline serialize/deserialize
  • Tüm v2.0 özellikleri korundu
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import streamlit as st

from logger import log, sidebar_log
from config import MAX_FILE_MB

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Opsiyonel bağımlılıklar ──────────────────────────────────────────────────

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

try:
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
    HAS_PANDERA = True
except ImportError:
    HAS_PANDERA = False

try:
    import ydata_profiling
    import streamlit.components.v1 as components
    HAS_PROFILING = True
except ImportError:
    HAS_PROFILING = False

try:
    from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import (
        LabelEncoder, OneHotEncoder, OrdinalEncoder,
        StandardScaler, MinMaxScaler, RobustScaler,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ═══════════════════════════════════════════════════════════════════════════════
# İşlem Geçmişi (Audit Trail)
# ═══════════════════════════════════════════════════════════════════════════════

class AuditTrail:
    """Tüm veri dönüşümlerini kaydeden hafif bir audit trail."""

    def __init__(self):
        self._records: list[dict] = []

    def log(self, operation: str, details: str, rows_before: int = 0, rows_after: int = 0):
        entry = {
            "adım": len(self._records) + 1,
            "işlem": operation,
            "detay": details,
            "satır_önce": rows_before,
            "satır_sonra": rows_after,
        }
        self._records.append(entry)
        log.info(f"[Audit] {operation}: {details}")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def render(self):
        if not self._records:
            st.info("Henüz işlem yapılmadı.")
            return
        st.markdown("#### 🗂 İşlem Geçmişi")
        st.dataframe(self.to_dataframe(), use_container_width=True, hide_index=True)


def get_audit() -> AuditTrail:
    if "audit_trail" not in st.session_state:
        st.session_state["audit_trail"] = AuditTrail()
    return st.session_state["audit_trail"]


# ═══════════════════════════════════════════════════════════════════════════════
# Yardımcı: Sütun Adı Temizleme
# ═══════════════════════════════════════════════════════════════════════════════

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sütun adlarını standardize eder (küçük harf, boşluk → _, özel karakter kaldır)."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# YENİ: Pandera Schema Doğrulama
# ═══════════════════════════════════════════════════════════════════════════════

def build_schema_from_df(df: pd.DataFrame) -> "pa.DataFrameSchema | None":
    """
    DataFrame'den otomatik pandera schema üretir.
    Her sayısal sütun için min/max sınırı, her object için nullable kontrolü.
    
    Returns:
        pandera.DataFrameSchema veya None (pandera yüklü değilse).
    """
    if not HAS_PANDERA:
        return None

    columns = {}
    for col in df.columns:
        series = df[col].dropna()
        nullable = bool(df[col].isnull().any())

        if pd.api.types.is_numeric_dtype(df[col]) and len(series) > 0:
            col_min = float(series.min())
            col_max = float(series.max())
            # Aşırı geniş aralık için ±%50 tolerans ekle
            lower = col_min - abs(col_min) * 0.5
            upper = col_max + abs(col_max) * 0.5
            columns[col] = pa.Column(
                dtype=None,
                checks=[
                    pa.Check.greater_than_or_equal_to(lower),
                    pa.Check.less_than_or_equal_to(upper),
                ],
                nullable=nullable,
                coerce=True,
            )
        else:
            columns[col] = pa.Column(
                dtype=None,
                nullable=nullable,
                coerce=True,
            )

    return pa.DataFrameSchema(columns, coerce=True)


def validate_schema(
    df: pd.DataFrame,
    schema: "pa.DataFrameSchema | None" = None,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """
    Pandera ile DataFrame'i doğrular.

    Args:
        df     : Doğrulanacak DataFrame.
        schema : Önceden oluşturulmuş schema; None ise df'den otomatik üretilir.
        strict : True → hata bulunursa exception fırlat; False → uyarı ver.

    Returns:
        (geçerli_mi: bool, hatalar: list[str])
    """
    if not HAS_PANDERA:
        st.warning("⚠️ Pandera yüklü değil. `pip install pandera`")
        return True, []

    if schema is None:
        schema = build_schema_from_df(df)

    errors: list[str] = []
    try:
        schema.validate(df, lazy=True)
        log.info("Schema doğrulama: başarılı")
        return True, []
    except pa.errors.SchemaErrors as exc:
        for _, row in exc.failure_cases.iterrows():
            errors.append(f"{row.get('column', '?')}: {row.get('check', '?')} — {row.get('failure_case', '?')}")
        log.warning(f"Schema doğrulama: {len(errors)} hata")
        if strict:
            raise
        return False, errors


def render_validation_report(errors: list[str]):
    """Doğrulama hatalarını Streamlit'te gösterir."""
    if not errors:
        st.success("✅ Schema doğrulama geçti.")
        return
    st.error(f"❌ {len(errors)} schema hatası:")
    for e in errors[:20]:
        st.markdown(f"- `{e}`")
    if len(errors) > 20:
        st.caption(f"... ve {len(errors) - 20} hata daha.")


# ═══════════════════════════════════════════════════════════════════════════════
# Veri Yükleme — Cache + Encoding + Chunked
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_encoding(raw_bytes: bytes) -> str:
    if HAS_CHARDET:
        result = chardet.detect(raw_bytes[:50_000])
        enc = result.get("encoding") or "utf-8"
        conf = result.get("confidence", 0)
        log.info(f"Encoding tespit: {enc} (güven: {conf:.0%})")
        return enc
    return "utf-8"


def _read_csv_chunked(uploaded_file: Any, encoding: str, chunk_size: int = 100_000) -> pd.DataFrame:
    chunks = []
    uploaded_file.seek(0)
    reader = pd.read_csv(
        uploaded_file,
        encoding=encoding,
        sep=None,
        engine="python",
        chunksize=chunk_size,
        on_bad_lines="warn",
    )
    for chunk in reader:
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


@st.cache_data(show_spinner=False)
def _load_cached(file_bytes: bytes, file_name: str, chunked_threshold_mb: float) -> pd.DataFrame | None:
    """Gerçek yükleme — cache'lenebilir (bytes tabanlı)."""
    name = file_name.lower()
    file_size_mb = len(file_bytes) / (1024 ** 2)

    try:
        if name.endswith(".csv"):
            encoding = _detect_encoding(file_bytes)
            buf = io.BytesIO(file_bytes)

            if file_size_mb > chunked_threshold_mb:
                log.info(f"Büyük dosya — chunked okuma ({chunked_threshold_mb} MB eşiği).")
                df = _read_csv_chunked(buf, encoding)
            else:
                for enc in [encoding, "utf-8", "latin-1", "cp1252"]:
                    try:
                        buf.seek(0)
                        df = pd.read_csv(buf, encoding=enc, sep=None, engine="python", on_bad_lines="warn")
                        break
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
                else:
                    return None

        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None

        if df.empty:
            return None

        df = clean_columns(df)
        df = _downcast_memory(df)
        return df

    except (ValueError, OSError, pd.errors.ParserError) as e:
        log.exception(f"Cache yükleme hatası: {e}")
        return None


def load_data(uploaded_file: Any, chunked_threshold_mb: float = 20.0) -> pd.DataFrame | None:
    """
    CSV veya Excel dosyasını yükler.
    Önbellekleme için dosya bytes olarak okunur, _load_cached'e iletilir.
    """
    if uploaded_file is None:
        return None

    file_size_mb = uploaded_file.size / (1024 ** 2)
    if file_size_mb > MAX_FILE_MB:
        st.error(
            f"❌ Dosya boyutu çok büyük: **{file_size_mb:.2f} MB**. "
            f"Maksimum: **{MAX_FILE_MB} MB**."
        )
        return None

    name = uploaded_file.name.lower()
    log.info(f"Yükleme başladı: {name} ({file_size_mb:.2f} MB)")
    sidebar_log(f"📂 Dosya okunuyor: {name}", "info")

    # Desteklenmeyen format
    if not name.endswith((".csv", ".xls", ".xlsx")):
        st.error("❌ Desteklenmeyen format. Lütfen **CSV** veya **Excel** yükleyin.")
        return None

    file_bytes = uploaded_file.read()
    df = _load_cached(file_bytes, uploaded_file.name, chunked_threshold_mb)

    if df is None:
        st.error("❌ Dosya okunamadı veya boş.")
        sidebar_log("❌ Yükleme başarısız", "error")
        return None

    rows, cols = df.shape
    initial_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    est_min = max(1, rows // 10_000)
    est_max = max(3, rows // 5_000)

    log.info(f"Yüklendi: {rows:,} × {cols} | Bellek: {initial_mem:.2f} MB")
    sidebar_log(f"✅ {rows:,} satır × {cols} sütun yüklendi", "success")
    st.success(
        f"✅ Veri yüklendi! **{rows:,}** satır, **{cols}** sütun. "
        f"Bellek: **{initial_mem:.1f} MB** | "
        f"Tahmini eğitim süresi: **{est_min}–{est_max} dk**."
    )
    get_audit().log("Veri Yükleme", f"{name} → {rows:,}×{cols}", rows_before=0, rows_after=rows)
    return df


def _downcast_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Otomatik Tip Tespiti & Dönüşümü
# ═══════════════════════════════════════════════════════════════════════════════

_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
    "%Y/%m/%d", "%d-%m-%Y", "%Y.%m.%d",
    "%d.%m.%Y", "%Y-%m-%dT%H:%M:%S",
]

_BOOL_MAP = {
    "true": True, "false": False,
    "yes": True, "no": False,
    "1": True, "0": False,
    "evet": True, "hayır": False,
}


def _try_parse_date(series: pd.Series) -> pd.Series | None:
    sample = series.dropna().astype(str).head(50)
    for fmt in _DATE_FORMATS:
        try:
            parsed = pd.to_datetime(sample, format=fmt, errors="raise")
            if parsed.notna().sum() >= len(sample) * 0.8:
                return pd.to_datetime(series, format=fmt, errors="coerce")
        except (ValueError, TypeError):
            continue
    try:
        parsed = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        if parsed.notna().sum() >= len(series.dropna()) * 0.8:
            return parsed
    except Exception:
        pass
    return None


def _try_parse_bool(series: pd.Series) -> pd.Series | None:
    lower = series.dropna().astype(str).str.lower()
    if lower.isin(_BOOL_MAP.keys()).mean() >= 0.9:
        return series.astype(str).str.lower().map(_BOOL_MAP)
    return None


def auto_detect_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    converted = df.copy()
    report: dict[str, str] = {}
    n = len(df)

    for col in df.columns:
        original_dtype = str(df[col].dtype)

        if df[col].dtype == object:
            parsed = _try_parse_date(df[col])
            if parsed is not None:
                converted[col] = parsed
                report[col] = f"object → datetime64  ({original_dtype})"
                continue

        if df[col].dtype == object:
            parsed = _try_parse_bool(df[col])
            if parsed is not None:
                converted[col] = parsed
                report[col] = f"object → bool  ({original_dtype})"
                continue

        if df[col].dtype == object:
            nunique = df[col].nunique(dropna=True)
            if nunique <= 50 or (n > 0 and nunique / n <= 0.20):
                converted[col] = df[col].astype("category")
                report[col] = f"object → category  ({nunique} benzersiz)"
                continue

        if df[col].dtype in ["float32", "float64"]:
            non_null = df[col].dropna()
            if len(non_null) > 0 and (non_null % 1 == 0).all():
                try:
                    converted[col] = df[col].astype("Int64")
                    report[col] = f"float → Int64  (ondalık değer yok)"
                except (ValueError, TypeError):
                    pass

    if report:
        detail = ", ".join(f"{c}: {v}" for c, v in report.items())
        get_audit().log("Tip Dönüşümü", f"{len(report)} sütun dönüştürüldü: {detail}",
                        rows_before=n, rows_after=n)
    else:
        log.info("Tip tespiti: dönüştürülecek sütun bulunamadı.")

    return converted, report


# ═══════════════════════════════════════════════════════════════════════════════
# Temizlik Sorunları Tespiti
# ═══════════════════════════════════════════════════════════════════════════════

def detect_cleaning_issues(df: pd.DataFrame) -> dict:
    issues = {}
    n = len(df)
    if n == 0:
        return issues

    _neg_keywords = ("age", "price", "count", "amount", "quantity", "weight",
                     "yaş", "fiyat", "adet", "miktar", "ağırlık")

    for col in df.columns:
        series = df[col]
        num_unique = series.nunique(dropna=True)
        num_missing = series.isnull().sum()
        missing_pct = num_missing / n

        if num_unique == 0:
            issues[col] = {"reason": "Sütun tamamen boş.", "action": "Sütunu çıkar", "severity": "high"}
            continue

        if num_unique == 1:
            issues[col] = {"reason": "Tüm satırlarda aynı değer — varyans sıfır.",
                           "action": "Sütunu çıkar", "severity": "high"}
            continue

        if pd.api.types.is_numeric_dtype(series):
            if series.dropna().std() == 0:
                issues[col] = {"reason": "Sayısal sütun — standart sapma sıfır.",
                               "action": "Sütunu çıkar", "severity": "high"}
                continue

        if missing_pct > 0.6:
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
        elif series.dtype == object and n > 0 and num_unique / n > 0.9:
            issues[col] = {
                "reason": "Çok yüksek benzersiz değer oranı (%90+). Muhtemelen ID/metin.",
                "action": "Sütunu çıkar",
                "severity": "medium",
            }

        if pd.api.types.is_numeric_dtype(series):
            col_lower = col.lower()
            if any(kw in col_lower for kw in _neg_keywords):
                if series.min() < 0:
                    existing = issues.get(col, {})
                    issues[col] = {
                        **existing,
                        "reason": (existing.get("reason", "") +
                                   f" ⚠️ '{col}' sütununda negatif değer bulundu (min={series.min():.2f})."),
                        "action": existing.get("action", "Negatif değerleri kontrol edin"),
                        "severity": existing.get("severity", "low"),
                    }

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Temizlik + Duplikat
# ═══════════════════════════════════════════════════════════════════════════════

def apply_cleaning(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    cleaned = df.copy()
    if cols_to_drop:
        cleaned = cleaned.drop(columns=cols_to_drop, errors="ignore")
        msg = f"{len(cols_to_drop)} sütun kaldırıldı: {', '.join(cols_to_drop)}"
        log.info(f"Temizlik: {msg}")
        sidebar_log(f"🧹 {msg}", "info")
        get_audit().log("Sütun Çıkarma", msg, rows_before=len(df), rows_after=len(cleaned))
    return cleaned


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)
    if removed > 0:
        sidebar_log(f"🗑️ {removed} yinelenen satır kaldırıldı", "info")
        get_audit().log("Duplikat Kaldırma", f"{removed} satır silindi",
                        rows_before=before, rows_after=len(cleaned))
    log.info(f"Duplikat: {removed} satır kaldırıldı ({before} → {len(cleaned)})")
    return cleaned, removed


# ═══════════════════════════════════════════════════════════════════════════════
# YENİ: Train/Test Split (Stratified + TimeSeries)
# ═══════════════════════════════════════════════════════════════════════════════

SplitMode = Literal["stratified", "random", "timeseries"]


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    mode: SplitMode = "stratified",
    random_state: int = 42,
    n_splits: int = 5,
) -> dict:
    """
    DataFrame'i train/test olarak böler.

    Args:
        df          : Bölünecek DataFrame.
        target_col  : Hedef sütun adı.
        test_size   : Test oranı (0-1 arası, yalnızca stratified/random için).
        mode        : "stratified" | "random" | "timeseries"
        random_state: Tekrarlanabilirlik tohumu.
        n_splits    : TimeSeriesSplit için kat sayısı.

    Returns:
        {
          "X_train", "X_test", "y_train", "y_test"  — stratified/random için
          "splits": [(X_tr, X_val, y_tr, y_val), ...]  — timeseries için
          "mode": str
        }
    """
    if not HAS_SKLEARN:
        st.error("❌ scikit-learn gerekli. `pip install scikit-learn`")
        return {}

    if target_col not in df.columns:
        st.error(f"❌ '{target_col}' sütunu bulunamadı.")
        return {}

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if mode == "timeseries":
        tss = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        for train_idx, val_idx in tss.split(X):
            splits.append((
                X.iloc[train_idx], X.iloc[val_idx],
                y.iloc[train_idx], y.iloc[val_idx],
            ))
        msg = f"TimeSeriesSplit: {n_splits} kat"
        get_audit().log("Train/Test Split", msg, rows_before=len(df), rows_after=len(df))
        log.info(msg)
        return {"splits": splits, "mode": "timeseries"}

    stratify = y if mode == "stratified" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError:
        log.warning("Stratified split başarısız, random'a düşülüyor.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    msg = (f"{mode} split: train={len(X_train):,}, test={len(X_test):,}, "
           f"oran={test_size:.0%}")
    get_audit().log("Train/Test Split", msg, rows_before=len(df), rows_after=len(df))
    sidebar_log(f"✂️ {msg}", "info")
    log.info(msg)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "mode": mode,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# YENİ: sklearn Pipeline + ColumnTransformer (Data Leakage Önlenir)
# ═══════════════════════════════════════════════════════════════════════════════

ImputeMethod = Literal["mean", "median", "zero", "sabit", "knn", "iterative"]
ScalerType   = Literal["standard", "minmax", "robust", "none"]
EncoderType  = Literal["onehot", "ordinal", "label", "none"]


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    numeric_impute: ImputeMethod = "median",
    scaler: ScalerType = "standard",
    cat_encoder: EncoderType = "onehot",
    knn_neighbors: int = 5,
) -> "Pipeline | None":
    """
    Sayısal ve kategorik sütunlar için ayrı kollu ColumnTransformer Pipeline kurar.
    Sadece train seti üzerinde fit edilmeli; test/inference'a transform uygulanır.

    Args:
        df             : Kaynak DataFrame (tip bilgisi için kullanılır, fit için değil).
        numeric_impute : Sayısal eksik doldurma yöntemi.
        scaler         : Ölçekleme yöntemi.
        cat_encoder    : Kategorik kodlama yöntemi.
        knn_neighbors  : KNN için komşu sayısı.

    Returns:
        sklearn Pipeline veya None.
    """
    if not HAS_SKLEARN:
        st.error("❌ scikit-learn gerekli.")
        return None

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── Sayısal Pipeline ────────────────────────────────────────────────────
    num_steps = []

    if numeric_impute == "knn":
        num_steps.append(("imputer", KNNImputer(n_neighbors=knn_neighbors, weights="distance")))
    elif numeric_impute == "iterative":
        num_steps.append(("imputer", IterativeImputer(max_iter=10, random_state=42)))
    elif numeric_impute == "mean":
        num_steps.append(("imputer", SimpleImputer(strategy="mean")))
    elif numeric_impute == "median":
        num_steps.append(("imputer", SimpleImputer(strategy="median")))
    elif numeric_impute in ("zero", "sabit"):
        num_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0)))

    if scaler == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scaler == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))
    elif scaler == "robust":
        num_steps.append(("scaler", RobustScaler()))

    # ── Kategorik Pipeline ──────────────────────────────────────────────────
    cat_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]

    if cat_encoder == "onehot":
        cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    elif cat_encoder == "ordinal":
        cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
    # "label" ve "none" → encode adımı eklenmez

    # ── ColumnTransformer ────────────────────────────────────────────────────
    transformers = []
    if num_cols and num_steps:
        transformers.append(("numeric", Pipeline(num_steps), num_cols))
    if cat_cols and cat_steps and cat_encoder != "none":
        transformers.append(("categorical", Pipeline(cat_steps), cat_cols))

    if not transformers:
        st.warning("⚠️ Pipeline için uygun sütun bulunamadı.")
        return None

    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer(transformers=transformers, remainder="passthrough"))
    ])

    msg = (f"Pipeline: sayısal={numeric_impute}+{scaler} ({len(num_cols)} sütun), "
           f"kategorik={cat_encoder} ({len(cat_cols)} sütun)")
    log.info(f"Pipeline kuruldu: {msg}")
    sidebar_log(f"⚙️ {msg}", "info")
    get_audit().log("Pipeline Kurulumu", msg, rows_before=0, rows_after=0)
    return pipeline


def fit_transform_pipeline(
    pipeline: "Pipeline",
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
) -> tuple["np.ndarray", "np.ndarray | None"]:
    """
    Pipeline'ı sadece X_train'e fit eder, ikisine de transform uygular.
    Data leakage'ı önler.

    Returns:
        (X_train_transformed, X_test_transformed veya None)
    """
    X_train_t = pipeline.fit_transform(X_train)
    X_test_t  = pipeline.transform(X_test) if X_test is not None else None
    log.info(f"Pipeline fit+transform: train={X_train_t.shape}")
    return X_train_t, X_test_t


def save_pipeline(pipeline: "Pipeline", path: str | Path = "pipeline.joblib") -> Path:
    """Pipeline'ı diske kaydeder."""
    path = Path(path)
    joblib.dump(pipeline, path)
    log.info(f"Pipeline kaydedildi: {path}")
    sidebar_log(f"💾 Pipeline kaydedildi: {path.name}", "success")
    return path


def load_pipeline(path: str | Path = "pipeline.joblib") -> "Pipeline | None":
    """Kaydedilmiş pipeline'ı yükler."""
    path = Path(path)
    if not path.exists():
        log.error(f"Pipeline dosyası bulunamadı: {path}")
        return None
    pipeline = joblib.load(path)
    log.info(f"Pipeline yüklendi: {path}")
    return pipeline


# ═══════════════════════════════════════════════════════════════════════════════
# Eski: Bağımsız Eksik Veri Doldurma (Geriye Dönük Uyum)
# ═══════════════════════════════════════════════════════════════════════════════

def fill_missing_values(
    df: pd.DataFrame,
    numeric_method: ImputeMethod = "mean",
    categorical_method: Literal["mode", "sabit"] = "mode",
    fill_value: str = "0",
    knn_neighbors: int = 5,
) -> pd.DataFrame:
    """
    [Geriye Dönük Uyum] Bağımsız eksik veri doldurma.
    
    ⚠️ Uyarı: Bu fonksiyon tüm seti görür — data leakage riski taşır.
    Yeni projelerde build_preprocessing_pipeline() kullanın.
    """
    if not HAS_SKLEARN and numeric_method in ("knn", "iterative"):
        st.warning(f"⚠️ `{numeric_method}` için scikit-learn gerekli. Median kullanılıyor.")
        numeric_method = "median"

    filled = df.copy()
    n_before = df.isnull().sum().sum()

    num_cols = filled.select_dtypes(include="number").columns.tolist()
    cat_cols = filled.select_dtypes(exclude="number").columns.tolist()

    if numeric_method in ("knn", "iterative") and num_cols:
        num_df = filled[num_cols].copy()
        imputer = (KNNImputer(n_neighbors=knn_neighbors, weights="distance")
                   if numeric_method == "knn"
                   else IterativeImputer(max_iter=10, random_state=42))
        imputed_vals = imputer.fit_transform(num_df)
        filled[num_cols] = pd.DataFrame(imputed_vals, columns=num_cols, index=filled.index)
    else:
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

    for col in cat_cols:
        if filled[col].isnull().sum() == 0:
            continue
        if categorical_method == "mode":
            mode_val = filled[col].mode()
            if not mode_val.empty:
                filled[col] = filled[col].fillna(mode_val[0])
        elif categorical_method == "sabit":
            filled[col] = filled[col].fillna(str(fill_value))

    n_after = filled.isnull().sum().sum()
    filled_count = n_before - n_after
    msg = (f"{filled_count} hücre dolduruldu "
           f"(sayısal={numeric_method}, kategorik={categorical_method})")
    log.info(msg)
    sidebar_log(f"✅ {filled_count} eksik hücre dolduruldu", "success")
    get_audit().log("Eksik Veri Doldurma", msg, rows_before=len(df), rows_after=len(filled))
    return filled


# ═══════════════════════════════════════════════════════════════════════════════
# Outlier Tespiti ve Yönetimi
# ═══════════════════════════════════════════════════════════════════════════════

OutlierMethod = Literal["iqr", "zscore", "isolation_forest", "all"]


def detect_outliers(
    df: pd.DataFrame,
    method: OutlierMethod = "iqr",
    zscore_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    contamination: float = 0.05,
) -> dict:
    if not HAS_SKLEARN and method in ("isolation_forest", "all"):
        st.warning("⚠️ Isolation Forest için scikit-learn gerekli. IQR kullanılıyor.")
        method = "iqr" if method == "isolation_forest" else "zscore"

    report: dict[str, dict] = {}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        col_report: dict[str, dict] = {}

        if method in ("iqr", "all"):
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
                mask = (series < lower) | (series > upper)
                cnt = int(mask.sum())
                if cnt > 0:
                    col_report["iqr"] = {
                        "outlier_count": cnt,
                        "outlier_pct": round(cnt / len(series) * 100, 1),
                        "lower_bound": round(float(lower), 4),
                        "upper_bound": round(float(upper), 4),
                        "q1": round(float(q1), 4),
                        "q3": round(float(q3), 4),
                    }

        if method in ("zscore", "all"):
            mu, sigma = series.mean(), series.std()
            if sigma > 0:
                zscores = ((series - mu) / sigma).abs()
                mask = zscores > zscore_threshold
                cnt = int(mask.sum())
                if cnt > 0:
                    col_report["zscore"] = {
                        "outlier_count": cnt,
                        "outlier_pct": round(cnt / len(series) * 100, 1),
                        "threshold": zscore_threshold,
                        "mean": round(float(mu), 4),
                        "std": round(float(sigma), 4),
                    }

        if method in ("isolation_forest", "all") and HAS_SKLEARN:
            X = series.values.reshape(-1, 1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            preds = clf.fit_predict(X)
            cnt = int((preds == -1).sum())
            if cnt > 0:
                col_report["isolation_forest"] = {
                    "outlier_count": cnt,
                    "outlier_pct": round(cnt / len(series) * 100, 1),
                    "contamination": contamination,
                }

        if col_report:
            report[col] = col_report

    return report


def detect_outliers_iqr(df: pd.DataFrame) -> dict:
    """[Geriye Dönük Uyum] Sadece IQR ile outlier tespit eder."""
    raw = detect_outliers(df, method="iqr")
    return {col: data["iqr"] for col, data in raw.items() if "iqr" in data}


def clip_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    clipped = df.copy()
    for col in cols:
        if col not in clipped.select_dtypes(include="number").columns:
            continue
        q1, q3 = clipped[col].quantile(0.25), clipped[col].quantile(0.75)
        iqr = q3 - q1
        clipped[col] = clipped[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

    msg = f"{len(cols)} sütunda outlier kırpıldı: {cols}"
    log.info(msg)
    sidebar_log(f"✂️ {msg}", "info")
    get_audit().log("Outlier Kırpma", msg, rows_before=len(df), rows_after=len(clipped))
    return clipped


def drop_outliers(df: pd.DataFrame, cols: list, method: OutlierMethod = "iqr") -> tuple[pd.DataFrame, int]:
    mask = pd.Series(False, index=df.index)
    numeric_df = df[cols].select_dtypes(include="number")

    for col in numeric_df.columns:
        series = df[col].dropna()
        if method == "zscore":
            mu, sigma = series.mean(), series.std()
            if sigma > 0:
                zscores = ((df[col] - mu) / sigma).abs()
                mask |= zscores > 3.0
        else:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask |= (df[col] < lower) | (df[col] > upper)

    cleaned = df[~mask].reset_index(drop=True)
    removed = int(mask.sum())
    msg = f"{removed} aykırı satır silindi ({method.upper()})"
    log.info(msg)
    sidebar_log(f"🗑️ {msg}", "info")
    get_audit().log("Outlier Silme", msg, rows_before=len(df), rows_after=len(cleaned))
    return cleaned, removed


# ═══════════════════════════════════════════════════════════════════════════════
# YENİ: Feature Seçimi (mutual_info + chi2)
# ═══════════════════════════════════════════════════════════════════════════════

FeatureScoreMethod = Literal["mutual_info_classif", "mutual_info_regression", "chi2"]


def score_features(
    df: pd.DataFrame,
    target_col: str,
    method: FeatureScoreMethod = "mutual_info_classif",
    top_k: int | None = None,
) -> pd.DataFrame:
    """
    Sayısal feature'ları hedef sütuna göre puanlar ve sıralar.

    Args:
        df         : Kaynak DataFrame.
        target_col : Hedef sütun adı.
        method     : Puanlama yöntemi.
        top_k      : Döndürülecek maksimum feature sayısı; None → tümü.

    Returns:
        ["feature", "score", "rank"] sütunlarından oluşan DataFrame.
    """
    if not HAS_SKLEARN:
        st.error("❌ scikit-learn gerekli.")
        return pd.DataFrame()

    if target_col not in df.columns:
        st.error(f"❌ '{target_col}' sütunu bulunamadı.")
        return pd.DataFrame()

    num_df = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore")
    X = num_df.fillna(num_df.median())
    y = df[target_col]

    if method == "chi2":
        # chi2 negatif değer kabul etmez
        X_pos = X - X.min()
        scores, _ = chi2(X_pos, y)
    elif method == "mutual_info_regression":
        from sklearn.feature_selection import mutual_info_regression
        scores = mutual_info_regression(X, y, random_state=42)
    else:
        scores = mutual_info_classif(X, y, random_state=42)

    result = pd.DataFrame({"feature": X.columns, "score": scores})
    result = result.sort_values("score", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1

    if top_k:
        result = result.head(top_k)

    msg = f"Feature scoring ({method}): {len(result)} feature"
    log.info(msg)
    get_audit().log("Feature Scoring", msg, rows_before=0, rows_after=0)
    return result


def render_feature_importance(score_df: pd.DataFrame):
    """Feature önem sıralamasını bar chart ile gösterir."""
    if score_df.empty:
        st.info("Puanlanacak feature bulunamadı.")
        return

    if HAS_PLOTLY:
        fig = px.bar(
            score_df,
            x="score", y="feature",
            orientation="h",
            title="Feature Önem Sıralaması",
            labels={"score": "Puan", "feature": "Sütun"},
            color="score",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(score_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# YENİ: Zaman Serisi Feature Mühendisliği
# ═══════════════════════════════════════════════════════════════════════════════

def add_lag_features(
    df: pd.DataFrame,
    col: str,
    lags: list[int],
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Belirtilen sütun için lag (gecikme) feature'ları ekler.

    Args:
        col     : Lag uygulanacak sütun.
        lags    : Gecikme adımları listesi, örn. [1, 3, 7].
        drop_na : Lag kaynaklı NaN satırlarını düşür.
    """
    result = df.copy()
    for lag in lags:
        result[f"{col}_lag{lag}"] = result[col].shift(lag)

    if drop_na:
        result = result.dropna().reset_index(drop=True)

    msg = f"Lag feature: {col} → {lags}"
    log.info(msg)
    get_audit().log("Lag Feature", msg, rows_before=len(df), rows_after=len(result))
    return result


def add_rolling_features(
    df: pd.DataFrame,
    col: str,
    windows: list[int],
    funcs: list[Literal["mean", "std", "min", "max"]] | None = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Belirtilen sütun için rolling (hareketli) istatistik feature'ları ekler.

    Args:
        col     : Rolling uygulanacak sütun.
        windows : Pencere boyutları, örn. [7, 14, 30].
        funcs   : İstatistikler listesi; None → ["mean", "std"].
        drop_na : Rolling kaynaklı NaN satırlarını düşür.
    """
    result = df.copy()
    if funcs is None:
        funcs = ["mean", "std"]

    for w in windows:
        roll = result[col].rolling(window=w)
        for fn in funcs:
            result[f"{col}_roll{w}_{fn}"] = getattr(roll, fn)()

    if drop_na:
        result = result.dropna().reset_index(drop=True)

    msg = f"Rolling feature: {col} windows={windows} funcs={funcs}"
    log.info(msg)
    get_audit().log("Rolling Feature", msg, rows_before=len(df), rows_after=len(result))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# YENİ: Görsel Analiz (Plotly)
# ═══════════════════════════════════════════════════════════════════════════════

def render_missing_heatmap(df: pd.DataFrame):
    """Eksik veri ısı haritasını Plotly ile gösterir."""
    if not HAS_PLOTLY:
        render_missing_summary(df)
        return

    missing_cols = [c for c in df.columns if df[c].isnull().any()]
    if not missing_cols:
        st.success("✅ Eksik veri yok.")
        return

    sample = df[missing_cols].head(200).isnull().astype(int)
    fig = px.imshow(
        sample.T,
        color_continuous_scale=["white", "#ef4444"],
        title="Eksik Veri Haritası (ilk 200 satır)",
        labels={"color": "Eksik"},
        aspect="auto",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def render_correlation_matrix(df: pd.DataFrame, method: Literal["pearson", "spearman"] = "pearson"):
    """Korelasyon matrisini Plotly heatmap ile gösterir."""
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        st.info("Korelasyon için en az 2 sayısal sütun gerekli.")
        return

    corr = num_df.corr(method=method).round(2)

    if HAS_PLOTLY:
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title=f"Korelasyon Matrisi ({method.title()})",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(corr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Eksik Veri Özeti (eski — korundu)
# ═══════════════════════════════════════════════════════════════════════════════

def render_missing_summary(df: pd.DataFrame):
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


# ═══════════════════════════════════════════════════════════════════════════════
# Profil Raporu (ydata_profiling)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_profile(df: pd.DataFrame) -> str | None:
    """Profil raporu oluşturur, HTML olarak ekrana gösterir ve HTML stringini döndürür."""
    if not HAS_PROFILING:
        st.warning("⚠️ `ydata_profiling` yüklü değil. `pip install ydata-profiling`")
        return None

    sidebar_log("📊 Profil raporu oluşturuluyor…", "info")
    with st.spinner("📊 Profil raporu hazırlanıyor (minimal mod)..."):
        profile = ydata_profiling.ProfileReport(df, minimal=True, title="Veri Profili")
        profile_html = profile.to_html()
        components.html(profile_html, height=650, scrolling=True)
    log.info("Profil raporu tamamlandı.")
    sidebar_log("✅ Profil raporu hazır", "success")
    return profile_html