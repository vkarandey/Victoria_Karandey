import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import re
import warnings
from tqdm import tqdm
import gc
import math

warnings.filterwarnings('ignore')
np.random.seed(993)


def create_submission(submission_df):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission_df.to_csv(submission_path, index=False)
    
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
    train = pd.read_csv('data/train-2.csv')
    test = pd.read_csv('data/test.csv')
    
    if 'product_locale' in train.columns:
        train = train.drop('product_locale', axis=1)
    if 'product_locale' in test.columns:
        test = test.drop('product_locale', axis=1)
    
    print(f"Train: {train.shape}, Test: {test.shape}")
    
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Очистка текста
    text_cols = ['query', 'product_title', 'product_description', 'product_bullet_point']
    for col in text_cols:
        if col in train.columns:
            train[col] = train[col].fillna('').apply(clean_text)
        if col in test.columns:
            test[col] = test[col].fillna('').apply(clean_text)
    
    # Создание корпуса для IDF
    corpus = train['product_title'].tolist() + test['product_title'].tolist()
    vectorizer = CountVectorizer(min_df=2, max_features=10000)
    X_counts = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()
    
    # Вычисление IDF
    N = len(corpus)
    df_counts = np.array((X_counts > 0).sum(axis=0)).flatten()
    idf_values = np.log((N + 1) / (df_counts + 1)) + 1
    idf_dict = dict(zip(vocab, idf_values))
    
    def compute_bm25(query, doc, idf_dict, avg_dl=100, k1=1.5, b=0.75):
        query_terms = query.split()
        doc_terms = doc.split()
        
        if not query_terms or not doc_terms:
            return 0.0
        
        score = 0.0
        doc_len = len(doc_terms)
        
        for term in query_terms:
            if term in idf_dict:
                tf = doc_terms.count(term)
                if tf > 0:
                    idf = idf_dict[term]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / avg_dl))
                    score += idf * (numerator / denominator)
        
        return score
    
    # Вычисление BM25
    for df in [train, test]:
        df['bm25_title'] = [compute_bm25(q, t, idf_dict) for q, t in 
                           tqdm(zip(df['query'], df['product_title']), total=len(df), desc="BM25 Title")]
        
        df['bm25_desc'] = [compute_bm25(q, d, idf_dict) for q, d in 
                          tqdm(zip(df['query'], df['product_description']), total=len(df), desc="BM25 Desc")]
        
        if 'product_bullet_point' in df.columns:
            df['bm25_bullet'] = [compute_bm25(q, b, idf_dict) for q, b in 
                               tqdm(zip(df['query'], df['product_bullet_point']), total=len(df), desc="BM25 Bullet")]
    
    def create_simple_match_features(df):
        features = pd.DataFrame(index=df.index)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Match features"):
            query = row['query']
            title = row['product_title']
            
            query_words = set(query.split())
            title_words = set(title.split())
            
            # Jaccard similarity
            if query_words and title_words:
                intersection = len(query_words & title_words)
                union = len(query_words | title_words)
                features.loc[idx, 'jaccard'] = intersection / union if union > 0 else 0
            else:
                features.loc[idx, 'jaccard'] = 0
            
            # Процент слов запроса в названии
            if query_words:
                features.loc[idx, 'query_in_title'] = len(query_words & title_words) / len(query_words)
            else:
                features.loc[idx, 'query_in_title'] = 0
            
            # Количество точных совпадений
            features.loc[idx, 'exact_matches'] = len(query_words & title_words)
            
            # хотя бы одно совпадение
            features.loc[idx, 'any_match'] = 1 if query_words & title_words else 0
        
        return features
    
    train_match = create_simple_match_features(train)
    test_match = create_simple_match_features(test)
    
    # Базовые текстовые признаки
    for df in [train, test]:
        df['query_len'] = df['query'].apply(len)
        df['title_len'] = df['product_title'].apply(len)
        df['desc_len'] = df['product_description'].apply(len)
        
        df['query_words'] = df['query'].apply(lambda x: len(x.split()))
        df['title_words'] = df['product_title'].apply(lambda x: len(x.split()))
        
        df['title_query_ratio'] = df['title_len'] / (df['query_len'] + 1)
        df['words_query_ratio'] = df['title_words'] / (df['query_words'] + 1)
    
    # Объединенный текст для TF-IDF
    train['combined_text'] = train['query'] + " " + train['product_title']
    test['combined_text'] = test['query'] + " " + test['product_title']
    
    all_text = pd.concat([train['combined_text'], test['combined_text']])
    
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=2,
        analyzer='word',
        use_idf=True,
        smooth_idf=True
    )
    vectorizer.fit(all_text)
    
    train_tfidf = vectorizer.transform(train['combined_text'])
    test_tfidf = vectorizer.transform(test['combined_text'])
    
    # Уменьшение размерности
    svd = TruncatedSVD(n_components=50, random_state=993, n_iter=10)
    train_tfidf_svd = svd.fit_transform(train_tfidf)
    test_tfidf_svd = svd.transform(test_tfidf)
    
    tfidf_cols = [f'tfidf_{i}' for i in range(50)]
    train_tfidf_df = pd.DataFrame(train_tfidf_svd, columns=tfidf_cols, index=train.index)
    test_tfidf_df = pd.DataFrame(test_tfidf_svd, columns=tfidf_cols, index=test.index)
    
    # Бренды
    if 'product_brand' in train.columns:
        brand_freq = train['product_brand'].value_counts()
        train['brand_freq'] = train['product_brand'].map(brand_freq)
        test['brand_freq'] = test['product_brand'].map(brand_freq).fillna(0)
        
        known_brands = set(brand_freq[brand_freq > 10].index)
        train['is_known_brand'] = train['product_brand'].apply(lambda x: 1 if x in known_brands else 0)
        test['is_known_brand'] = test['product_brand'].apply(lambda x: 1 if x in known_brands else 0)
    
    # Статистики по запросам
    train['products_per_query'] = train.groupby('query_id')['product_id'].transform('count')
    train['product_rank'] = train.groupby('query_id').cumcount() + 1
    train['is_first'] = (train['product_rank'] == 1).astype(int)
    
    test['products_per_query'] = test.groupby('query_id')['product_id'].transform('count')
    test['product_rank'] = test.groupby('query_id').cumcount() + 1
    test['is_first'] = (test['product_rank'] == 1).astype(int)
    
    # Сбор фичей
    feature_cols = [
        'bm25_title', 'bm25_desc',
        'query_len', 'title_len',
        'query_words', 'title_words',
        'title_query_ratio', 'words_query_ratio',
        'brand_freq', 'is_known_brand',
        'products_per_query', 'product_rank', 'is_first',
    ]
    
    if 'bm25_bullet' in train.columns and 'bm25_bullet' in test.columns:
        feature_cols.append('bm25_bullet')
    
    X_train = pd.concat([train[feature_cols], train_match, train_tfidf_df], axis=1)
    X_test = pd.concat([test[feature_cols], test_match, test_tfidf_df], axis=1)
    
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    y_train = train['relevance'].values
    groups = train['query_id'].values
    
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  Признаков: {X_train.shape[1]}")
    
    # Параметры модели
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10],
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'num_threads': 4,
        'seed': 993,
        'max_depth': 8,
        'min_data_in_leaf': 20,
        'min_sum_hessian_in_leaf': 0.001,
        'label_gain': [0, 1, 3, 7],
        'early_stopping_round': 50,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_gain_to_split': 0.01,
        'max_bin': 255,
    }
    
    def compute_ndcg(y_true, y_pred, query_ids, k=10):
        ndcg_scores = []
        unique_queries = np.unique(query_ids)
        
        for query_id in unique_queries:
            mask = query_ids == query_id
            if mask.sum() == 0:
                continue
                
            query_true = y_true[mask]
            query_pred = y_pred[mask]
            
            sorted_idx = np.argsort(-query_pred)[:k]
            sorted_true = query_true[sorted_idx]
            
            dcg = 0.0
            for i, rel in enumerate(sorted_true, 1):
                gain = (2 ** rel) - 1
                discount = np.log2(i + 1)
                dcg += gain / discount
            
            ideal_sorted = np.sort(query_true)[::-1][:k]
            idcg = 0.0
            for i, rel in enumerate(ideal_sorted, 1):
                gain = (2 ** rel) - 1
                discount = np.log2(i + 1)
                idcg += gain / discount
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    # Кросс-валидация
    n_folds = 5
    group_kfold = GroupKFold(n_splits=n_folds)
    
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    fold_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X_train, y_train, groups)):
        print(f"\nФолд {fold + 1}/{n_folds}")
        
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]
        query_ids_tr = groups[train_idx]
        query_ids_val = groups[val_idx]
        
        _, counts_tr = np.unique(query_ids_tr, return_counts=True)
        _, counts_val = np.unique(query_ids_val, return_counts=True)
        
        print(f"  Train: {len(X_tr)} samples")
        print(f"  Valid: {len(X_val)} samples")
        
        train_dataset = lgb.Dataset(X_tr, label=y_tr, group=counts_tr)
        val_dataset = lgb.Dataset(X_val, label=y_val, group=counts_val, reference=train_dataset)
        
        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=1000,
            valid_sets=[val_dataset],
            valid_names=['val'],
            callbacks=[
                lgb.log_evaluation(100),
                lgb.early_stopping(50, verbose=False)
            ]
        )
        
        models.append(model)
        
        val_pred = model.predict(X_val)
        oof_predictions[val_idx] = val_pred
        
        fold_ndcg = compute_ndcg(y_val, val_pred, query_ids_val, k=10)
        fold_scores.append(fold_ndcg)
        print(f"  nDCG@10: {fold_ndcg:.4f}")
        
        test_pred = model.predict(X_test)
        test_predictions += test_pred / n_folds
        
        del X_tr, X_val, y_tr, y_val, train_dataset, val_dataset
        gc.collect()
    
    # Нормализация предсказаний
    test_with_pred = test.copy()
    test_with_pred['prediction'] = test_predictions
    
    def simple_normalize(group):
        if len(group) > 1:
            min_val, max_val = group.min(), group.max()
            if max_val > min_val:
                return (group - min_val) / (max_val - min_val)
        return group
    
    normalized_preds = test_with_pred.groupby('query_id')['prediction'].transform(simple_normalize)
    
    # Создание submission
    submission = pd.DataFrame({
        'id': test['id'],
        'prediction': normalized_preds
    })
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()


