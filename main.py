import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

def create_minimal_features(df):
    df = df.copy()
    
    df['dt'] = pd.to_datetime(df['dt'])
    df['day_of_week'] = df['dt'].dt.dayofweek
    df['day_of_year'] = df['dt'].dt.dayofyear
    
    if 'price_p05' in df.columns and 'price_p95' in df.columns:
        df['price_center'] = (df['price_p05'] + df['price_p95']) / 2
        df['price_width'] = df['price_p95'] - df['price_p05']
    
    return df

def create_essential_lags(df, is_train=True):
    if 'product_id' not in df.columns:
        return df
    
    df = df.sort_values(['product_id', 'dt']).reset_index(drop=True)
    
    for lag in [1, 7]:
        df[f'n_stores_lag_{lag}'] = df.groupby('product_id')['n_stores'].shift(lag)
    
    df['n_stores_rolling_mean_7'] = df.groupby('product_id')['n_stores'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    
    if is_train and 'price_center' in df.columns:
        for lag in [1, 7]:
            df[f'price_center_lag_{lag}'] = df.groupby('product_id')['price_center'].shift(lag)
            df[f'price_width_lag_{lag}'] = df.groupby('product_id')['price_width'].shift(lag)
    
    return df

class SimpleIntervalPredictor:    
    def __init__(self):
        self.models_lower = []
        self.models_upper = []
        self.models_center = []
        self.models_width = []
        
    def fit_simple(self, X, y_lower, y_upper, y_center, y_width):        
        rf_center = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        rf_center.fit(X, y_center)
        self.models_center.append(rf_center)
        
        rf_width = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=30,
            min_samples_leaf=15,
            random_state=42,
            n_jobs=-1
        )
        rf_width.fit(X, np.maximum(y_width, 0))
        self.models_width.append(rf_width)
        
        lgb_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 5,
            'num_leaves': 20,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'random_state': 42,
            'verbose': -1
        }
        
        lgb_lower = lgb.LGBMRegressor(
            objective='quantile',
            alpha=0.05,
            **lgb_params
        )
        lgb_lower.fit(X, y_lower)
        self.models_lower.append(lgb_lower)
        
        lgb_upper = lgb.LGBMRegressor(
            objective='quantile',
            alpha=0.95,
            **lgb_params
        )
        lgb_upper.fit(X, y_upper)
        self.models_upper.append(lgb_upper)
        
        rf_lower = RandomForestRegressor(
            n_estimators=80,
            max_depth=7,
            min_samples_split=25,
            min_samples_leaf=12,
            random_state=43,
            n_jobs=-1
        )
        rf_lower.fit(X, y_lower)
        self.models_lower.append(rf_lower)
        
        rf_upper = RandomForestRegressor(
            n_estimators=80,
            max_depth=7,
            min_samples_split=25,
            min_samples_leaf=12,
            random_state=43,
            n_jobs=-1
        )
        rf_upper.fit(X, y_upper)
        self.models_upper.append(rf_upper)
    
    def predict_simple(self, X):
        """Простое предсказание"""
        center_preds = []
        for model in self.models_center:
            center_preds.append(model.predict(X))
        pred_center = np.mean(center_preds, axis=0)
        
        width_preds = []
        for model in self.models_width:
            width_preds.append(model.predict(X))
        pred_width = np.mean(width_preds, axis=0)
        pred_width = np.maximum(pred_width, 0)
        
        pred_lower_from_center = pred_center - pred_width / 2
        pred_upper_from_center = pred_center + pred_width / 2
        
        lower_preds = []
        for model in self.models_lower:
            lower_preds.append(model.predict(X))
        pred_lower_direct = np.mean(lower_preds, axis=0)
        
        upper_preds = []
        for model in self.models_upper:
            upper_preds.append(model.predict(X))
        pred_upper_direct = np.mean(upper_preds, axis=0)
        
        pred_lower = (pred_lower_from_center + pred_lower_direct) / 2
        pred_upper = (pred_upper_from_center + pred_upper_direct) / 2
        
        return pred_lower, pred_upper, pred_center, pred_width

def postprocessing(pred_lower, pred_upper, train_data):    
    pred_lower = np.maximum(pred_lower, 0)
    pred_upper = np.maximum(pred_upper, pred_lower + 0.01)
    
    epsilon = 0.1
    width = pred_upper - pred_lower
    mask_narrow = width < epsilon
    if mask_narrow.any():
        expansion = (epsilon - width[mask_narrow]) / 2
        pred_lower[mask_narrow] -= expansion
        pred_upper[mask_narrow] += expansion
    
    pred_lower = np.minimum(pred_lower, pred_upper)
    pred_upper = np.maximum(pred_lower, pred_upper)
    
    if 'price_p05' in train_data.columns and 'price_p95' in train_data.columns:
        global_p05_median = train_data['price_p05'].median()
        global_p95_median = train_data['price_p95'].median()
        global_width_median = train_data['price_p95'].median() - train_data['price_p05'].median()
        
        current_center = (pred_lower + pred_upper) / 2
        current_width = pred_upper - pred_lower
        
        alpha = 0.1
        
        center_correction = global_p05_median + global_width_median/2 - np.median(current_center)
        pred_lower = pred_lower + center_correction * alpha
        pred_upper = pred_upper + center_correction * alpha
        
        width_ratio = current_width / (global_width_median + 1e-7)
        mask_too_narrow = width_ratio < 0.5
        mask_too_wide = width_ratio > 2.0
        
        if mask_too_narrow.any():
            pred_lower[mask_too_narrow] -= global_width_median * 0.1 * alpha
            pred_upper[mask_too_narrow] += global_width_median * 0.1 * alpha
        
        if mask_too_wide.any():
            pred_lower[mask_too_wide] += global_width_median * 0.1 * alpha
            pred_upper[mask_too_wide] -= global_width_median * 0.1 * alpha
    
    pred_lower = np.maximum(pred_lower, 0)
    pred_upper = np.maximum(pred_upper, pred_lower + 0.01)
    pred_lower = np.minimum(pred_lower, pred_upper)
    pred_upper = np.maximum(pred_lower, pred_upper)
    
    return pred_lower, pred_upper

def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    import os
    
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    predictions.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path

def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # Загрузка данных
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    if 'row_id' not in test.columns:
        test['row_id'] = np.arange(len(test))
    
    # Предобработка данных
    train = create_minimal_features(train)
    test = create_minimal_features(test)
    
    train = create_essential_lags(train, is_train=True)
    test = create_essential_lags(test, is_train=False)
    
    # Определение признаков
    base_features = [
        'n_stores', 'precpt', 'avg_temperature', 'avg_humidity', 
        'avg_wind_level', 'holiday_flag', 'activity_flag',
        'management_group_id', 'first_category_id', 'second_category_id', 
        'third_category_id', 'dow', 'day_of_month', 'week_of_year', 'month',
        'day_of_week', 'day_of_year'
    ]
    
    lag_features = [col for col in train.columns if 'lag' in col or 'rolling' in col]
    lag_features = [col for col in lag_features if col in test.columns]
    
    all_features = base_features + lag_features
    
    seen = set()
    feature_cols = []
    for col in all_features:
        if col in train.columns and col in test.columns and col not in seen:
            seen.add(col)
            feature_cols.append(col)
    
    # Заполнение пропусков
    for col in feature_cols:
        if col in train.columns:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
        
        if col in test.columns:
            if col in train.columns:
                median_val = train[col].median()
            else:
                median_val = test[col].median()
            test[col] = test[col].fillna(median_val)
    
    # Подготовка данных для обучения
    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    
    y_lower = train['price_p05'].values
    y_upper = train['price_p95'].values
    y_center = (y_lower + y_upper) / 2
    y_width = y_upper - y_lower
    
    # Обучение и предсказание
    predictor = SimpleIntervalPredictor()
    predictor.fit_simple(X_train, y_lower, y_upper, y_center, y_width)
    
    pred_lower, pred_upper, pred_center, pred_width = predictor.predict_simple(X_test)
    
    # Постобработка
    pred_lower_final, pred_upper_final = postprocessing(pred_lower, pred_upper, train)
    
    # Создание submission
    submission = pd.DataFrame({
        'row_id': test['row_id'].values,
        'price_p05': pred_lower_final,
        'price_p95': pred_upper_final
    })
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)

if __name__ == "__main__":
    main()
