import logging
import time
import itertools
from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import RFECV
from joblib import Parallel, delayed

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DynamicFeatureSelector")

class DynamicFeatureSelector:
    """
    Üretim standartlarında (production-ready) dinamik Özellik Seçimi (Feature Selection) sınıfı.
    
    Özellik sayısı 'threshold' değerinden küçükse Brute-Force (Tüm Kombinasyonlar),
    büyükse Recursive Feature Elimination with Cross-Validation (RFECV) kullanır.
    
    Attributes:
        threshold (int): Brute-Force için maksimum özellik sayısı sınırı (Varsayılan: 15).
        cv_folds (int): K-Fold Cross-Validation için katlama sayısı (Varsayılan: 5).
        n_jobs (int): Paralel işlem için kullanılacak CPU çekirdek sayısı (-1 tüm çekirdekler).
        random_state (int): Tekrar edilebilirlik için rastgelelik tohumu.
    """
    
    def __init__(self, threshold: int = 15, cv_folds: int = 5, n_jobs: int = -1, random_state: int = 42):
        self.threshold = threshold
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Temel modelimiz Random Forest Regressor (R2 metriği ile değerlendirilecek)
        # Brute-force paralel çalışacağı için modelin kendi n_jobs değerini 1 tutuyoruz.
        self.model = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=1)
        
    def _evaluate_subset(self, X: pd.DataFrame, y: pd.Series, features: Tuple[str, ...]) -> Tuple[Tuple[str, ...], float]:
        """Belirli bir özellik kombinasyonunu K-Fold CV ile eğitip R2 skorunu döndürür."""
        X_subset = X[list(features)]
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.model, X_subset, y, cv=cv, scoring="r2", n_jobs=1)
        return features, float(np.mean(scores))

    def _run_brute_force(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], float]:
        """Tüm olası kombinasyonları joblib multiprocessing ile paralel değerlendirir."""
        logger.info(f"Brute-Force (Tüm Kombinasyonlar) başlatıldı. Özellik sayısı: {X.shape[1]}")
        all_features = X.columns.tolist()
        
        # 1'den N'e kadar tüm olası özellik kombinasyonlarını oluştur
        combinations = []
        for i in range(1, len(all_features) + 1):
            combinations.extend(itertools.combinations(all_features, i))
            
        logger.info(f"Toplam {len(combinations)} kombinasyon {self.n_jobs if self.n_jobs != -1 else 'tüm'} CPU çekirdeği ile değerlendiriliyor...")
        
        # joblib ile tüm kombinasyonları paralel olarak değerlendir
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_subset)(X, y, combo) for combo in combinations
        )
        
        # En yüksek R2 skoruna sahip kombinasyonu bul
        best_features, best_score = max(results, key=lambda item: item[1])
        logger.info(f"Brute-Force tamamlandı. En iyi R2: {best_score:.4f}")
        
        return list(best_features), best_score

    def _run_rfe(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], float]:
        """Cross-Validation destekli Recursive Feature Elimination (RFECV) çalıştırır."""
        logger.info(f"Recursive Feature Elimination (RFECV) başlatıldı. Özellik sayısı: {X.shape[1]}")
        
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # RFECV optimum özellik sayısını kendi belirler
        selector = RFECV(
            estimator=self.model,
            step=1,
            cv=cv,
            scoring="r2",
            n_jobs=self.n_jobs,
            min_features_to_select=1
        )
        
        selector.fit(X, y)
        selected_features = X.columns[selector.support_].tolist()
        
        # sklearn sürümüne göre cross validation sonuçlarını çek
        if hasattr(selector, "cv_results_"):
            best_score = float(np.max(selector.cv_results_["mean_test_score"]))
        elif hasattr(selector, "grid_scores_"):
            best_score = float(np.max(selector.grid_scores_))
        else:
            # Fallback
            _, best_score = self._evaluate_subset(X, y, tuple(selected_features))
            
        logger.info(f"RFECV tamamlandı. Seçilen özellik sayısı: {len(selected_features)}. En iyi R2: {best_score:.4f}")
        
        return selected_features, best_score

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Dinamik feature selection sürecini yöneten ana metot.
        
        Args:
            X (pd.DataFrame): Özellik (Feature) matrisi.
            y (pd.Series): Hedef (Target) değişken.
            
        Returns:
            pd.DataFrame: İşlem özetini, en iyi skoru ve seçilen listeyi içeren rapor.
        """
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise TypeError("X pandas DataFrame, y ise pandas Series formatında olmalıdır.")
            
        start_time = time.time()
        num_features = X.shape[1]
        
        if num_features == 0:
            raise ValueError("Veri setinde hiçbir özellik bulunamadı.")
            
        logger.info(f"Dinamik Feature Selection Başlatıldı. Toplam Değişken: {num_features}")
        
        # Dinamik Karar Mekanizması
        if num_features < self.threshold:
            method_used = "Brute-Force (All Combinations)"
            best_features, best_score = self._run_brute_force(X, y)
        else:
            method_used = "RFECV (Recursive Feature Elimination)"
            best_features, best_score = self._run_rfe(X, y)
            
        time_spent = time.time() - start_time
        logger.info(f"İşlem {time_spent:.2f} saniyede tamamlandı.")
        
        # Temiz ve okunabilir Pandas DataFrame Raporu oluştur
        report_df = pd.DataFrame([{
            "Kullanılan Yöntem": method_used,
            "Başlangıçtaki Özellik Sayısı": num_features,
            "Seçilen Özellik Sayısı": len(best_features),
            "En İyi R2 Skoru": round(best_score, 4),
            "Harcanan Süre (sn)": round(time_spent, 2),
            "Seçilen Özellikler": ", ".join(best_features)
        }])
        
        return report_df

# Örnek Kullanım Testi
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    
    # 1. Senaryo: 15'ten küçük özellik (Brute-Force tetiklenir)
    print("\\n--- Senaryo 1: Brute-Force Testi ---")
    X_small, y_small = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    selector = DynamicFeatureSelector(threshold=15, cv_folds=3)
    rapor_small = selector.select_features(pd.DataFrame(X_small, columns=[f"feat_{i}" for i in range(5)]), pd.Series(y_small))
    print(rapor_small.T)
    
    # 2. Senaryo: 15'ten büyük özellik (RFECV tetiklenir)
    print("\\n--- Senaryo 2: RFECV Testi ---")
    X_large, y_large = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
    rapor_large = selector.select_features(pd.DataFrame(X_large, columns=[f"feat_{i}" for i in range(20)]), pd.Series(y_large))
    print(rapor_large.T)
