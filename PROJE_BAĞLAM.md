# 🧠 OutoLLM — Proje Bağlam ve Hafıza Dosyası
> **⚠️ BU DOSYA KRİTİKTİR.**
> Her yapay zeka oturumunda **ilk önce bu dosyayı oku**.
> Buraya yazılmayan hiçbir şey varsayılmaz, uydurulmaz.

---

## 📌 Projenin Tek Cümle Tanımı

**OutoLLM**, CSV/Excel dosyası yükleyip 6 sekmeli arayüzde (Yükleme → Temizlik → Eğitim → Değerlendirme → Tahmin → EDA) otomatik makine öğrenmesi modeli oluşturan, **Streamlit + PyCaret** tabanlı Türkçe bir web uygulamasıdır.

---

## 📁 Dizin Yapısı (GÜNCEL)

```
C:\Users\ugrys\Desktop\OutoLLM\
├── PROJE_BAĞLAM.md          ← Bu dosya (her zaman ilk oku)
└── automl_app\
    ├── app.py               ← Ana uygulama, 6 sekme (~540 satır) ✅
    ├── config.py            ← Sabitler ve model listeleri ✅
    ├── data_module.py       ← Veri IO, temizlik, duplikat, outlier ✅
    ├── train_module.py      ← Eğitim pipeline + grafik + kayıt ✅
    ├── inference_module.py  ← Model yükleme + tahmin + format ✅
    ├── ui_components.py     ← CSS + header + sidebar + registry ✅
    ├── feature_module.py    ← EDA grafikleri (NEW) ✅
    ├── logger.py            ← Merkezi logging (NEW) ✅
    ├── requirements.txt     ← Bağımlılıklar ✅
    └── logs.log             ← Uygulama logları ✅
```

---

## 🏗️ Modül Sorumlulukları (ASLA KARIŞTIRILMAZ)

| Dosya | Tek Sorumluluğu | İçe Aktarır |
|---|---|---|
| `config.py` | Sabitler (model listesi, metrikler, limit) | — |
| `logger.py` | Dosya + konsol + sidebar logging | `logging`, `streamlit` |
| `data_module.py` | Veri IO, temizlik, duplikat, outlier (IQR) | `config`, `logger`, `streamlit`, `pandas` |
| `train_module.py` | PyCaret setup/compare/tune/save/plot | `config`, `logger`, `pycaret.*`, `streamlit` |
| `inference_module.py` | Model yükleme, uyumluluk, tahmin, format | `logger`, `pycaret.*`, `streamlit`, `os` |
| `feature_module.py` | EDA: korelasyon, histogram, kategorik, eksik harita | `logger`, `streamlit`, `pandas`, `math` |
| `ui_components.py` | CSS, header, sidebar, step-indicator, model registry | `logger`, `streamlit`, `pandas`, `os` |
| `app.py` | Orchestration – 6 sekme arası bağlayıcı | Diğer tüm modüller |

---

## 🔄 6-Sekme Pipeline (session_state üzerinden veri akar)

```
[Tab 1] Yükleme
   └─ load_data() → st.session_state["raw_df"] & ["cleaned_df"]

[Tab 2] Temizlik (GENİŞLETİLDİ)
   ├─ render_missing_summary()
   ├─ remove_duplicates()         → st.session_state["cleaned_df"]
   ├─ detect_outliers_iqr()
   ├─ clip_outliers()             → st.session_state["cleaned_df"]
   └─ detect_cleaning_issues() + apply_cleaning()

[Tab 3] Eğitim
   ├─ detect_task_type() → st.session_state["task_type"]
   ├─ run_setup()        → st.session_state["setup_obj", "pc_module"]
   └─ compare_models()  → st.session_state["best_model", "leaderboard"]

[Tab 4] Değerlendirme
   ├─ (opsiyonel) tune_selected_model()
   ├─ render_model_plots() [classification/regression/clustering]
   └─ save_model_with_card() → .pkl + .json → download

[Tab 5] Tahmin (Inference)
   ├─ load_model_safe()
   ├─ validate_compatibility()
   └─ run_prediction() → format_results() → download CSV

[Tab 6] EDA Analizi (YENİ)
   ├─ render_eda_overview()
   ├─ render_correlation_heatmap()
   ├─ render_distribution_plots()
   ├─ render_categorical_plots()
   └─ render_missing_heatmap()
```

---

## 🔑 Session State Anahtarları (TÜMÜ)

| Anahtar | Tür | Ne Zaman Dolar |
|---|---|---|
| `raw_df` | `pd.DataFrame` | Tab 1 - upload |
| `cleaned_df` | `pd.DataFrame` | Tab 1 (kopya), Tab 2 (güncellenir) |
| `task_type` | `str` → `"classification"/"regression"/"clustering"` | Tab 3 |
| `setup_obj` | PyCaret setup nesnesi | Tab 3 |
| `pc_module` | `pycaret.classification / regression / clustering` | Tab 3 |
| `leaderboard` | `pd.DataFrame` | Tab 3 |
| `best_model` | PyCaret model nesnesi | Tab 3 / Tab 4 tune / Sidebar registry |
| `model_card` | `dict` | Tab 4 save / Sidebar registry |
| `target_col` | `str` | Tab 3 selectbox |
| `mode` | `"standard" / "expert"` | Sidebar radio |
| `sidebar_logs` | `list[str]` | Her pipeline adımında (logger.py) |

---

## 📦 Bağımlılıklar (requirements.txt)

```
streamlit>=1.32.0
pandas>=2.0.0
pycaret>=3.3.0
openpyxl
ydata-profiling  (opsiyonel)
```

---

## ✅ Tamamlanan Özellikler

### Çekirdek Pipeline
- [x] 6 sekmeli Streamlit arayüzü
- [x] CSV/Excel yükleme + boyut kontrolü
- [x] Latin-1 encoding fallback
- [x] PyCaret setup (classification / regression / clustering)
- [x] Hızlı (standart) ve tam (uzman) model karşılaştırma
- [x] Sınıf dengesizliği uyarısı
- [x] Hiperparametre optimizasyonu (Uzman mod)
- [x] Model grafikleri (confusion matrix / residuals / feature importance / elbow / silhouette)
- [x] .pkl + .json model card kaydetme + download

### Veri Temizliği (Tab 2)
- [x] Eksik veri özeti tablosu
- [x] Duplikat satır tespiti ve kaldırma
- [x] IQR aykırı değer tespiti ve kırpma
- [x] Sütun bazlı risk tespiti ve kaldırma

### EDA (Tab 6)
- [x] Genel bakış metrikleri
- [x] Pearson korelasyon ısı haritası
- [x] Sayısal sütun histogramları + istatistikler
- [x] Kategorik sütun çubuk grafikleri
- [x] Eksik veri yoğunluk haritası

### Altyapı
- [x] Merkezi logging (dosya + konsol + sidebar)
- [x] Sidebar model registry (saved_models/ tarama + yükleme)
- [x] Sidebar uygulama logları paneli
- [x] Global hata yakalama (Tab 3, 5)
- [x] Clustering grafikleri (Elbow + Silhouette)
- [x] Türkçe arayüz (tamamı)

---

## 🚧 YAPILACAKLAR (Kalan)

### Öncelik 1
- [x] ~~Tab 2'de eksik veri doldurma seçeneği (mean/median/mode) direkt uygulanabilir olsun~~ ✅
- [x] ~~EDA Tab'ında hedef sütun vs. özellik scatter plot~~ ✅
- [x] ~~Model karşılaştırma çubuk grafikleri (Tab 4)~~ ✅

### Öncelik 2 — İleride Eklenebilir
- [ ] Batch inference: çoklu model formatı desteği (farklı PyCaret sürümleri arası)
- [ ] Kümeleme için k sayısı kullanıcıdan alınabilir (Tab 3)
- [ ] Model export: ONNX/PMML formatı desteği

---

## ⚙️ Çalıştırma Komutu

```powershell
cd C:\Users\ugrys\Desktop\OutoLLM\automl_app
streamlit run app.py
```

Uygulama: http://localhost:8501

---

## 🚫 HALLÜSINASYON ENGELLEYİCİ KURALLAR

1. **Bu projede FastAPI/Flask yoktur.** Tüm backend Streamlit içindedir.
2. **`models/` dizini yoktur.** Modeller `automl_app/saved_models/` altına kaydedilir.
3. **`utils.py` dosyası yoktur.** Yardımcı fonksiyonlar ilgili modüllerdedir.
4. **Celery / Redis / kuyruk sistemi yoktur.** Pipeline senkron çalışır.
5. **Docker/container yoktur.** Doğrudan Python virtualenv ile çalışır.
6. **PyCaret 3.x kullanılır.** 2.x API'si farklıdır, kullanılmaz.
7. **`pc_module` bir modül referansıdır** (nesne değil). `pc_class` / `pc_reg` / `pc_clust` import'larından biri.
8. **Tüm UI dili Türkçedir.** İngilizce etiket eklenmez.
9. **`st.session_state` merkezi depodur.** Global değişken kullanılmaz.
10. **EDA grafikleri için harici kütüphane YOK.** Sadece `pandas` + `st.bar_chart` + `st.dataframe.style`.
11. **`feature_module.py` sadece görüntüleme içerir.** Veri mutasyonu yapmaz.
12. **Yeni dosya eklemeden önce bu listeyi güncelle.**
13. **Pandas DataFrame logical checks:** Never use short-circuit operators (like df1 or df2) when assigning DataFrames from st.session_state to avoid 'ValueError: truth value ambiguous'. Always use explicit 'key in st.session_state' and 'is not None' checks. Use guard clauses with st.stop() for missing data scenarios.
14. Tab 2 data mutations MUST explicitly reassign st.session_state["cleaned_df"]. Never use inplace. Clear PyCaret cache keys after cleaning.

---

## 📝 Değişiklik Günlüğü

| Tarih | Ne Değişti | Dosya |
|---|---|---|
| 2026-04-17 | İlk sürüm oluşturuldu | Tüm modüller |
| 2026-04-17 | Bağlam dosyası oluşturuldu | PROJE_BAĞLAM.md |
| 2026-04-17 | Logging sistemi eklendi | logger.py |
| 2026-04-17 | Clustering grafikleri eklendi | train_module.py |
| 2026-04-17 | Model Registry sidebar | ui_components.py |
| 2026-04-17 | Duplikat + Outlier (IQR) temizlik | data_module.py |
| 2026-04-17 | EDA modülü oluşturuldu | feature_module.py |
| 2026-04-17 | Tab 2 genişletildi, Tab 6 eklendi | app.py |
| 2026-04-17 | **pandas 2.x uyumu:** `fillna(inplace=True)` → assignment | data_module.py |
| 2026-04-17 | **pandas 2.x uyumu:** `applymap` → `map`; clustering rename fix | inference_module.py |
| 2026-04-17 | **UI fix:** Tüm veri mutasyon butonlarına `st.rerun()` eklendi | app.py |
| 2026-04-17 | **UI fix:** Sidebar pipeline durumu EDA + Tahmin adımları güncellendi | ui_components.py |
| 2026-04-17 | **Guard fix:** Tab 4'te `best_model is None` kontrolü eklendi | app.py |
| 2026-04-20 | Fixed Tab 6 EDA DataFrame truth value error, added strict session_state check and st.stop() guard. Updated architecture rules. | app.py, PROJE_BAĞLAM.md |
| 2026-04-20 | Fixed Ghost Data bug (Tab 2 to Tab 3). Enforced explicit state assignment and cleared cache. | app.py, data_module.py, PROJE_BAĞLAM.md |

---

## ⚠️ Bilinen Kısıtlamalar

- `ydata-profiling` opsiyoneldir; kurulmadan da uygulama çalışır (Tab 1 rapor butonu uyarı verir).
- PyCaret clustering `pull()` her zaman anlamlı bir leaderboard döndürmeyebilir — bu normal.
- `st.rerun()` çağrısından önce `st.success()` mesajı kısa süre görünür, sonra sayfa yenilenir.
- Windows'ta log dosyası (`logs.log`) varsayılan olarak `automl_app/` dizinine yazılır.

---

*Son güncelleme: 2026-04-17 18:37 · Bu dosyayı her değişiklikten sonra güncelle.*
