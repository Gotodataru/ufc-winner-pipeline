from config import DATA_DIR, MODEL_DIR, BASE_DIR
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_model.py
Валидация модели CatBoost с принудительной симметризацией предсказаний.
Для каждого боя вычисляется симметризованная вероятность:
p_sym = (p_orig + (1 - p_swapped)) / 2, где p_swapped – вероятность для перевёрнутого боя.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import catboost
import joblib
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# ==============================================================================
# 1. ЗАГРУЗКА ДАННЫХ И МОДЕЛИ
# ==============================================================================
print("="*80)
print("🔍 ВАЛИДАЦИЯ МОДЕЛИ CATBOOST (С КАЛИБРОВКОЙ И СИММЕТРИЗАЦИЕЙ)")
print("="*80)

DATA_PATH = r'str(BASE_DIR)\UFCTOPMODEL\WINNER\winnerbigdata\data\UFC_full_data_golden_fixed.csv'
MODEL_PATH = r'str(BASE_DIR)\UFCTOPMODEL\WINNER\winnerbigdata\model\winner_model_catboost_v1.cbm'
CALIB_PATH = r'str(BASE_DIR)\UFCTOPMODEL\WINNER\winnerbigdata\model\calibration_params.joblib'
OUTPUT_DIR = r'str(BASE_DIR)\UFCTOPMODEL\WINNER\winnerbigdata\model'

# Загрузка данных (как в основном скрипте)
df = pd.read_csv(DATA_PATH, low_memory=False)
df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
df = df.dropna(subset=['event_date']).copy()
df_test = df[df['event_date'] > '2023-12-31'].copy().reset_index(drop=True)
df_test = df_test.dropna(subset=['f_1_odds', 'f_2_odds']).reset_index(drop=True)

leak_cols = [col for col in df_test.columns if 'ko_odds' in col.lower() or 'sub_odds' in col.lower()]
round_leak_cols = [col for col in df_test.columns if '_r1_' in col or '_r2_' in col or '_r3_' in col or '_r4_' in col or '_r5_' in col]
df_test = df_test.drop(columns=leak_cols + round_leak_cols, errors='ignore')
df_test = df_test[df_test['diff_age'].abs() <= 25].copy()

if 'weight_class' in df_test.columns:
    top_weights = df_test['weight_class'].value_counts().nlargest(10).index.tolist()
    df_test['weight_class'] = df_test['weight_class'].apply(lambda x: x if x in top_weights else 'other')
    weight_dummies = pd.get_dummies(df_test['weight_class'], prefix='weight')
    df_test = pd.concat([df_test.drop('weight_class', axis=1), weight_dummies], axis=1)

# Загрузка модели
print(f"\nЗагрузка модели из {MODEL_PATH}...")
model = catboost.CatBoostClassifier()
model.load_model(MODEL_PATH)
print("Модель успешно загружена.")

# Загрузка калибровки
if os.path.exists(CALIB_PATH):
    calib = joblib.load(CALIB_PATH)
    use_calibration = True
    print("Параметры калибровки загружены.")
else:
    calib = None
    use_calibration = False
    print("Калибровка не найдена, используется сырая модель.")

feature_order = model.feature_names_
print(f"Количество признаков: {len(feature_order)}")

# Синхронизация данных
available_features = [col for col in feature_order if col in df_test.columns]
feature_order = available_features
X_test_raw = df_test[feature_order].copy()
y_test_raw = df_test['winner_encoded'].copy()

mask_valid = X_test_raw.notna().all(axis=1) & y_test_raw.notna()
X_test = X_test_raw[mask_valid].copy().reset_index(drop=True)
y_test = y_test_raw[mask_valid].copy().reset_index(drop=True)
df_test_sync = df_test[mask_valid].copy().reset_index(drop=True)

print(f"Размер тестовой выборки: {len(X_test)} боев")

# Функция для получения калиброванных вероятностей
def get_calibrated_proba(model, X, calib):
    logits = model.predict(X, prediction_type='RawFormulaVal')
    if calib is None:
        # Если калибровки нет, используем стандартную сигмоиду CatBoost
        return model.predict_proba(X)[:, 1]
    else:
        # Применяем Platt scaling без смещения
        return 1 / (1 + np.exp(-(calib['a'] * logits)))

# Функция для симметризованного предсказания
def predict_symmetrized(model, X, calib, diff_cols):
    """
    Возвращает симметризованную вероятность победы Fighter1.
    Для каждого экземпляра создаёт перевёрнутую копию (swap признаков, инверсия diff_*),
    получает вероятность для оригинала и для перевёрнутого, усредняет.
    """
    # Оригинальные вероятности
    p_orig = get_calibrated_proba(model, X, calib)
    
    # Создаём перевёрнутую копию
    X_swapped = X.copy()
    # Меняем местами f_1_* и f_2_*
    f1_cols = [col for col in X.columns if col.startswith('f_1_')]
    f2_cols = [col for col in X.columns if col.startswith('f_2_')]
    for f1, f2 in zip(f1_cols, f2_cols):
        tmp = X_swapped[f1].copy()
        X_swapped[f1] = X_swapped[f2]
        X_swapped[f2] = tmp
    # Инвертируем разностные признаки
    for d in diff_cols:
        X_swapped[d] = -X_swapped[d]
    
    # Вероятность для перевёрнутого боя (это вероятность победы Fighter1 в перевёрнутом мире,
    # что соответствует вероятности победы Fighter2 в оригинале)
    p_swapped = get_calibrated_proba(model, X_swapped, calib)
    
    # Симметризованная вероятность
    p_sym = (p_orig + (1 - p_swapped)) / 2
    return p_sym

# Определяем разностные признаки
diff_cols = [col for col in feature_order if col.startswith('diff_')]

# ==============================================================================
# 2. ТЕСТ СИММЕТРИИ С ИСПОЛЬЗОВАНИЕМ СИММЕТРИЗОВАННЫХ ПРЕДСКАЗАНИЙ
# ==============================================================================
def test_symmetry_symmetrized(model, X, calib, diff_cols, n_tests=20):
    """
    Проверяет, что симметризованные предсказания удовлетворяют свойству:
    p_sym(X) + p_sym(X_swapped) = 1.
    """
    print("\n" + "="*60)
    print("ТЕСТ СИММЕТРИИ (СИММЕТРИЗОВАННЫЕ ПРЕДСКАЗАНИЯ)")
    print("="*60)
    
    results = []
    n_samples = min(n_tests, len(X))
    indices = np.random.choice(len(X), size=n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        original = X.iloc[idx:idx+1].copy()
        
        # Получаем симметризованную вероятность для оригинала
        p_sym_orig = predict_symmetrized(model, original, calib, diff_cols)[0]
        
        # Создаём перевёрнутую копию
        X_sw = original.copy()
        f1_cols = [col for col in X.columns if col.startswith('f_1_')]
        f2_cols = [col for col in X.columns if col.startswith('f_2_')]
        for f1, f2 in zip(f1_cols, f2_cols):
            tmp = X_sw[f1].copy()
            X_sw[f1] = X_sw[f2]
            X_sw[f2] = tmp
        for d in diff_cols:
            X_sw[d] = -X_sw[d]
        
        # Получаем симметризованную вероятность для перевёрнутого
        p_sym_sw = predict_symmetrized(model, X_sw, calib, diff_cols)[0]
        
        # Проверяем соотношение
        expected = 1 - p_sym_orig
        diff = abs(p_sym_sw - expected)
        
        print(f"Тест {i+1}: p_sym(orig) = {p_sym_orig:.4f}, p_sym(swapped) = {p_sym_sw:.4f}, ожидалось = {expected:.4f}, разница = {diff:.4f}")
        if diff > 1e-5:  # Допуск практически нулевой из-за численных погрешностей
            print("   ⚠️  Нарушение симметрии!")
        else:
            print("   ✅ Симметрично")
        
        results.append({
            'test': i+1,
            'p_orig': p_sym_orig,
            'p_swapped': p_sym_sw,
            'expected': expected,
            'diff': diff,
            'symmetric': diff <= 1e-5
        })
    
    symmetric_count = sum(r['symmetric'] for r in results)
    print(f"\nСводка: {symmetric_count}/{n_samples} тестов симметричны (допуск 1e-5)")
    if symmetric_count == n_samples:
        print("✅ Модель с симметризацией полностью симметрична!")
    else:
        print("⚠️  Что-то пошло не так, симметризация не работает.")
    
    return results

# Запуск теста симметрии для симметризованных предсказаний
symmetry_results = test_symmetry_symmetrized(model, X_test, calib, diff_cols, n_tests=20)

# ==============================================================================
# 3. ПОЛУЧАЕМ СИММЕТРИЗОВАННЫЕ ПРЕДСКАЗАНИЯ ДЛЯ ВСЕЙ ТЕСТОВОЙ ВЫБОРКИ
# ==============================================================================
print("\nВычисление симметризованных вероятностей для всей тестовой выборки...")
y_proba_sym = predict_symmetrized(model, X_test, calib, diff_cols)

# ==============================================================================
# 4. КАЛИБРОВОЧНАЯ КРИВАЯ ДЛЯ СИММЕТРИЗОВАННЫХ ПРЕДСКАЗАНИЙ
# ==============================================================================
def plot_calibration_curve(y_true, y_pred, save_path):
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, label='Модель (симметризованная)')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Идеальная калибровка')
    plt.xlabel('Средняя предсказанная вероятность', fontsize=12)
    plt.ylabel('Доля реальных побед', fontsize=12)
    plt.title('Калибровочная кривая (после симметризации)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Калибровочная кривая сохранена: {save_path}")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
calib_path = os.path.join(OUTPUT_DIR, f'calibration_curve_sym_{timestamp}.png')
plot_calibration_curve((y_test == 1).astype(int), y_proba_sym, calib_path)

# ==============================================================================
# 5. ВАЖНОСТЬ ПРИЗНАКОВ (без изменений)
# ==============================================================================
def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    importances = model.get_feature_importance()
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices][::-1], align='center', color='steelblue', edgecolor='black')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
    plt.xlabel('Важность', fontsize=12)
    plt.title(f'Топ-{top_n} наиболее важных признаков', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ График важности признаков сохранён: {save_path}")

feat_imp_path = os.path.join(OUTPUT_DIR, f'feature_importance_{timestamp}.png')
plot_feature_importance(model, feature_order, top_n=20, save_path=feat_imp_path)

# ==============================================================================
# 6. РАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАНИЙ (для симметризованных)
# ==============================================================================
def plot_prediction_distribution(y_true, y_pred, save_path):
    mask_f1_win = y_true == 1
    mask_f2_win = y_true == -1
    
    pred_f1_win = y_pred[mask_f1_win]
    pred_f2_win = y_pred[mask_f2_win]
    
    plt.figure(figsize=(10, 6))
    plt.hist(pred_f1_win, bins=20, alpha=0.7, label='Победил Fighter1', color='green', edgecolor='black')
    plt.hist(pred_f2_win, bins=20, alpha=0.7, label='Победил Fighter2', color='red', edgecolor='black')
    plt.axvline(0.5, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Предсказанная вероятность победы Fighter1 (симметризованная)', fontsize=12)
    plt.ylabel('Количество боев', fontsize=12)
    plt.title('Распределение предсказаний после симметризации', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Распределение предсказаний сохранено: {save_path}")

dist_path = os.path.join(OUTPUT_DIR, f'prediction_distribution_sym_{timestamp}.png')
plot_prediction_distribution(y_test, y_proba_sym, dist_path)

# ==============================================================================
# 7. BRIER SCORE
# ==============================================================================
brier = brier_score_loss((y_test == 1).astype(int), y_proba_sym)
print(f"\n📊 BRIER SCORE (после симметризации): {brier:.4f}")

# ==============================================================================
# 8. ИТОГ
# ==============================================================================
print("\n" + "="*80)
print("✅ ВАЛИДАЦИЯ ЗАВЕРШЕНА (СИММЕТРИЗОВАННАЯ МОДЕЛЬ)")
print("="*80)
print(f"\n📁 Графики сохранены в {OUTPUT_DIR} с меткой {timestamp}")
print("\n💡 РЕКОМЕНДАЦИИ:")
if brier < 0.23:
    print("   • Brier score хороший (<0.23)")
else:
    print("   • Brier score умеренный, но симметрия обеспечена принудительно.")
print("   • Тест симметрии для симметризованных предсказаний должен показать 20/20.")