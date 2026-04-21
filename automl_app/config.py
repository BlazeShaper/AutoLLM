MAX_FILE_MB = 50

# Standart mod için hızlı modeller
DEFAULT_CLASSIFICATION_MODELS = ["lr", "dt", "rf", "knn", "nb"]
DEFAULT_REGRESSION_MODELS = ["lr", "dt", "rf", "knn", "en"]

# Uzman mod için genel modeller
ALL_CLASSIFICATION_MODELS = ["lr", "knn", "nb", "dt", "svm", "rbfsvm", "gpc", "mlp", "ridge", "rf", "qda", "ada", "gbc", "lda", "et", "xgboost", "lightgbm", "dummy"]
ALL_REGRESSION_MODELS = ["lr", "lasso", "ridge", "en", "lar", "llar", "omp", "br", "ard", "par", "ransac", "tr", "huber", "kr", "svm", "knn", "dt", "rf", "et", "ada", "gbr", "mlp", "xgboost", "lightgbm", "dummy"]

# Metrikler
CLASSIFICATION_METRICS = ["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC"]
REGRESSION_METRICS = ["MAE", "MSE", "RMSE", "R2", "RMSLE", "MAPE"]

# Eksik veri doldurma yöntemleri
NUMERIC_IMPUTATION_METHODS = ["mean", "median", "zero", "knn"]
CATEGORICAL_IMPUTATION_METHODS = ["mode", "constant"]
