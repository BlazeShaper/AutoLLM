<div align="center">
  <h1>🚀 AutoLLM</h1>
  <p><b>Uçtan Uca, Kullanıcı Dostu ve Güçlü Otomatik Makine Öğrenmesi (AutoML) Platformu</b></p>
</div>

---

**AutoLLM** (eski adıyla OutoLLM), verilerinizi yükleyip saniyeler içinde profesyonel makine öğrenmesi modelleri oluşturmanızı sağlayan **Streamlit** ve **PyCaret** tabanlı Türkçe bir web uygulamasıdır. Veri bilimci olmanıza gerek kalmadan; veri temizliğinden keşifçi veri analizine (EDA), model eğitiminden tahmin (inference) aşamasına kadar tüm makine öğrenmesi süreçlerini 6 basit sekmede yönetebilirsiniz.

## ✨ Öne Çıkan Özellikler

- 📊 **6 Sekmeli Akıcı Süreç:** Yükleme → Temizlik → Eğitim → Değerlendirme → Tahmin → EDA
- 🤖 **Desteklenen Görevler:** Sınıflandırma (Classification), Regresyon (Regression) ve Kümeleme (Clustering).
- 🧹 **Gelişmiş Veri Temizliği:** Duplikat kayıtların tespiti, IQR yöntemiyle aykırı değer kırpma ve eksik veri doldurma algoritmaları.
- ⚙️ **Optuna Bayesyen Optimizasyon:** Sadece standart model eğitimi değil, verinizin istatistiklerine özel geliştirilmiş dinamik hiperparametre önerileri ve Optuna entegrasyonu ile "Uzman Mod" optimizasyonu.
- 📉 **Kapsamlı EDA (Keşifçi Veri Analizi):** Pearson korelasyon ısı haritaları, dağılım grafikleri, eksik veri haritaları ve kategorik analizler için hazır interaktif grafikler.
- 💾 **Model Yönetimi (Registry):** Eğitilen modelleri yanlarındaki detaylı JSON kartları (.pkl + .json) ile birlikte kaydetme ve daha sonra kullanmak üzere yükleyebilme.

## 🛠️ Kullanılan Teknolojiler

- **Arayüz:** [Streamlit](https://streamlit.io/)
- **AutoML Motoru:** [PyCaret 3.x](https://pycaret.org/)
- **Veri İşleme:** Pandas, NumPy, SciPy
- **Optimizasyon:** [Optuna](https://optuna.org/)
- **Görselleştirme:** Çoğunlukla yerleşik Streamlit chart fonksiyonları (sadeliği korumak için).

## 📂 Dizin Yapısı

Proje modüler bir mimari üzerine inşa edilmiştir:

```text
AutoLLM/
├── automl_app/
│   ├── app.py               # Uygulamayı çalıştıran ana orkestratör
│   ├── config.py            # Sabitler, model ve metrik listeleri
│   ├── data_module.py       # Veri IO, temizlik, duplikat ve aykırı değer tespiti
│   ├── feature_module.py    # EDA ve keşifsel veri analizi modülü
│   ├── train_module.py      # PyCaret model eğitim pipeline'ı
│   ├── inference_module.py  # Model yükleme, doğrulama ve yeni tahmin yapma
│   ├── optimization_module.py # Dinamik hiperparametre ve Optuna arama motoru
│   ├── ui_components.py     # CSS, sidebar, step-indicator ve görsel bileşenler
│   ├── logger.py            # Merkezi logging ve yan panel log gösterimi
│   └── saved_models/        # Eğitilen modellerin saklandığı dizin
└── PROJE_BAĞLAM.md          # Geliştiriciler için proje tasarım defteri
```

## 🚀 Kurulum & Çalıştırma

### 1. Gereksinimleri Yükleme
Projenin ihtiyaç duyduğu bağımlılıkları sanal bir ortamda (virtual environment) kurmanız tavsiye edilir:

```bash
pip install streamlit>=1.32.0 pandas>=2.0.0 pycaret>=3.3.0 scipy>=1.11.0 optuna>=3.6.0 openpyxl
```
*(İsteğe bağlı: Yükleme sekmesindeki detaylı rapor için `ydata-profiling` kurabilirsiniz).*

### 2. Uygulamayı Başlatma
Terminalinizden `automl_app` dizinine gidin ve Streamlit sunucusunu çalıştırın:

```bash
cd automl_app
streamlit run app.py
```

Tarayıcınızda otomatik olarak **http://localhost:8501** adresinde uygulamanız açılacaktır.

## 🕹️ Nasıl Kullanılır?

1. **Yükleme:** Elinizdeki CSV veya Excel dosyasını sisteme sürükleyip bırakın.
2. **Temizlik:** Eksik verileri doldurun, aykırı değerleri tespit edip kırpın.
3. **Eğitim:** Hedef sütununuzu seçin. Uygulama sizin için sınıflandırma mı yoksa regresyon mu yapması gerektiğini otomatik anlar (Kümeleme isterseniz de seçebilirsiniz). Uzman modunu açarak Optuna ile hiperparametre optimizasyonu başlatabilirsiniz.
4. **Değerlendirme:** Oluşan Liderlik Tablosundan (Leaderboard) en iyi modeli seçip grafiklerini (Confusion Matrix vb.) inceleyin ve modeli diske indirin/kaydedin.
5. **Tahmin:** Eğitip kaydettiğiniz modeli seçin, yeni veriler yükleyip tahmin (inference) yaptırın.
6. **EDA:** Verinizin dağılımlarını ve sütunlar arası korelasyonlarını tek tıkla analiz edin.

---
*Geliştiriciler için Not: Yeni özellikler eklemek veya mimariyi incelemek isterseniz lütfen `PROJE_BAĞLAM.md` dosyasını okuyunuz.*
