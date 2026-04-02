#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ПОЛНЫЙ ТРЕНИНГ МОДЕЛИ UFC CATBOOST (УГЛУБЛЁННАЯ ВЕРСИЯ)
- Аугментация для симметрии (train + val)
- Усиленные параметры (depth=14, iterations=7000, lr=0.01, l2=10)
- Калибровка вероятностей (Platt scaling без смещения для сохранения симметрии)
- Сохранение модели и параметров калибровки
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, brier_score_loss
)
from datetime import datetime
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. КОНФИГУРАЦИЯ
# ==============================================================================
CONFIG = {
    'data_path': r'D:\BETTING\UFCTOPMODEL\WINNER\winnerbigdata\data\UFC_full_data_golden_fixed.csv',
    'output_dir': r'D:\BETTING\UFCTOPMODEL\WINNER\winnerbigdata\model',
    'min_date': '2016-01-01',
    'train_end': '2022-12-31',
    'val_end': '2023-12-31',
    'edge_threshold': 0.05,
    'catboost_params': {
        'iterations': 7000,
        'learning_rate': 0.01,
        'depth': 14,
        'l2_leaf_reg': 10,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': 100,
        'early_stopping_rounds': 500,
        'use_best_model': True,
        'random_strength': 1.0,
        'border_count': 128,
        'task_type': 'CPU',
    }
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ==============================================================================
# 2. ТОП-40 ПРИЗНАКОВ (разностные + симметричные)
# ==============================================================================
TOP_40_FEATURES = [
    'diff_age', 'diff_sapm_5', 'diff_footwork_5', 'diff_str_def_5', 'diff_timing_5',
    'diff_td_avg_5', 'diff_str_acc_5', 'diff_sub_avg_5', 'diff_speed_5',
    'diff_physical_strength_5', 'diff_punching_power_5', 'diff_chin_5',
    'diff_dynamika_5', 'diff_cardio_5', 'diff_td_def_5', 'diff_td_acc_5',
    'diff_ctrl_ratio_5', 'diff_slpm_5', 'diff_fight_number',
    'diff_head_acc_5', 'diff_body_acc_5', 'diff_leg_acc_5',
    'diff_distance_acc_5', 'diff_clinch_acc_5', 'diff_ground_acc_5',
    'title_fight', 'num_rounds',
    'weight_Bantamweight', 'weight_Featherweight', 'weight_Flyweight',
    'weight_Heavyweight', 'weight_Light Heavyweight', 'weight_Lightweight',
    'weight_Middleweight', 'weight_Welterweight', 'weight_Women\'s Bantamweight',
    'weight_Women\'s Featherweight', 'weight_Women\'s Flyweight', 'weight_Women\'s Strawweight'
]

# ==============================================================================
# 3. ЗАГРУЗКА И ФИЛЬТРАЦИЯ ДАННЫХ
# ==============================================================================
print("="*80)
print("🚀 ЗАГРУЗКА ДАННЫХ С 2016 ГОДА ДЛЯ CATBOOST")
print("="*80)

df = pd.read_csv(CONFIG['data_path'], low_memory=False)
df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
df = df.dropna(subset=['event_date']).copy()
df = df[df['event_date'] >= CONFIG['min_date']].copy()

print(f"✓ Загружено {len(df):,} боев с {CONFIG['min_date']} по {df['event_date'].max().strftime('%Y-%m-%d')}")

class_balance = df['winner_encoded'].value_counts(normalize=True) * 100
print(f"✓ Баланс классов: Победа бойца 1 = {class_balance.get(1, 0):.1f}%, Победа бойца 2 = {class_balance.get(-1, 0):.1f}%")

# ==============================================================================
# 4. ПОДГОТОВКА ПРИЗНАКОВ (УДАЛЕНИЕ УТЕЧЕК, ЗАПОЛНЕНИЕ ПРОПУСКОВ, OHE)
# ==============================================================================
print("\n[2/6] Подготовка признаков (ТОЛЬКО топ-40 без утечек)...")

# Удаление явных утечек (коэффициенты, раунды)
leak_cols = [col for col in df.columns if 'odds' in col.lower() or ('_r' in col.lower() and any(x in col for x in ['_r1_', '_r2_', '_r3_', '_r4_', '_r5_']))]
if leak_cols:
    df = df.drop(columns=leak_cols)
    print(f"  → Удалены колонки с коэффициентами и раундами: {len(leak_cols)} шт")

# Фильтрация аномалий возраста
df = df[df['diff_age'].abs() <= 25].copy()
print(f"  → Отфильтрованы аномалии возраста, осталось {len(df):,} боев")

# Заполнение пропусков медианой
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64] and df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# One-hot encoding весовых категорий
if 'weight_class' in df.columns:
    top_weights = df['weight_class'].value_counts().nlargest(10).index.tolist()
    df['weight_class'] = df['weight_class'].apply(lambda x: x if x in top_weights else 'other')
    weight_dummies = pd.get_dummies(df['weight_class'], prefix='weight')
    df = pd.concat([df.drop('weight_class', axis=1), weight_dummies], axis=1)
    print(f"  → One-hot encoding весовых категорий: {len(weight_dummies.columns)} колонок")

# Оставляем только нужные колонки
required_cols = ['event_date', 'winner_encoded'] + TOP_40_FEATURES
available_cols = [col for col in required_cols if col in df.columns]
df = df[available_cols].copy()
print(f"✓ Отфильтровано до {len(available_cols) - 2} признаков (из топ-40)")

# Удаление строковых колонок (если остались)
string_cols = df.select_dtypes(include=['object']).columns.tolist()
if string_cols:
    df = df.drop(columns=string_cols)

# ==============================================================================
# 5. ВРЕМЕННОЙ СПЛИТ
# ==============================================================================
print("\n[3/6] Временной сплит (без утечки будущего)...")

df = df.sort_values('event_date').reset_index(drop=True)
train_mask = df['event_date'] <= CONFIG['train_end']
val_mask = (df['event_date'] > CONFIG['train_end']) & (df['event_date'] <= CONFIG['val_end'])
test_mask = df['event_date'] > CONFIG['val_end']

X_train = df.loc[train_mask].copy()
y_train = df.loc[train_mask, 'winner_encoded'].copy()
X_val = df.loc[val_mask].copy()
y_val = df.loc[val_mask, 'winner_encoded'].copy()
X_test = df.loc[test_mask].copy()
y_test = df.loc[test_mask, 'winner_encoded'].copy()

feature_cols = [col for col in X_train.columns if col not in ['event_date', 'winner_encoded']]
feature_cols = [col for col in feature_cols if np.issubdtype(X_train[col].dtype, np.number)]

X_train = X_train[feature_cols]
X_val = X_val[feature_cols]
X_test = X_test[feature_cols]

print(f"✓ Train:   {len(X_train):,} боев с {len(feature_cols)} признаками")
print(f"✓ Val:     {len(X_val):,} боев")
print(f"✓ Test:    {len(X_test):,} боев — unseen данные!")

# ==============================================================================
# 6. АУГМЕНТАЦИЯ ТРЕНИРОВОЧНЫХ ДАННЫХ ДЛЯ СИММЕТРИИ
# ==============================================================================
print("\n[4/7] Аугментация тренировочных данных для обеспечения симметрии...")

diff_cols = [col for col in feature_cols if col.startswith('diff_')]

# Создаём перевёрнутые копии
X_train_inv = X_train.copy()
X_train_inv[diff_cols] = -X_train_inv[diff_cols]
y_train_inv = -y_train

X_train_aug = pd.concat([X_train, X_train_inv], axis=0, ignore_index=True)
y_train_aug = pd.concat([y_train, y_train_inv], axis=0, ignore_index=True)

# Перемешиваем
shuffle_idx = np.random.permutation(len(X_train_aug))
X_train_aug = X_train_aug.iloc[shuffle_idx].reset_index(drop=True)
y_train_aug = y_train_aug.iloc[shuffle_idx].reset_index(drop=True)

print(f"✓ Размер тренировочного набора после аугментации: {len(X_train_aug)} боев")

# ==============================================================================
# 7. ПОДГОТОВКА ВАЛИДАЦИОННОЙ ВЫБОРКИ ДЛЯ ОБУЧЕНИЯ (АУГМЕНТИРОВАННАЯ)
# ==============================================================================
X_val_inv = X_val.copy()
X_val_inv[diff_cols] = -X_val_inv[diff_cols]
y_val_inv = -y_val

X_val_aug = pd.concat([X_val, X_val_inv], axis=0, ignore_index=True)
y_val_aug = pd.concat([y_val, y_val_inv], axis=0, ignore_index=True)

shuffle_idx_val = np.random.permutation(len(X_val_aug))
X_val_aug = X_val_aug.iloc[shuffle_idx_val].reset_index(drop=True)
y_val_aug = y_val_aug.iloc[shuffle_idx_val].reset_index(drop=True)

val_pool = Pool(X_val_aug, y_val_aug)
print(f"✓ Валидационный набор аугментирован: {len(X_val_aug)} боев")

# ==============================================================================
# 8. ОБУЧЕНИЕ БАЗОВОЙ МОДЕЛИ CATBOOST (УГЛУБЛЁННЫЕ ПАРАМЕТРЫ)
# ==============================================================================
print("\n[5/7] Обучение базовой модели CatBoost (углублённые параметры)...")

model = CatBoostClassifier(**CONFIG['catboost_params'])

model.fit(
    X_train_aug, y_train_aug,
    eval_set=val_pool,
    verbose=100
)

# ==============================================================================
# 9. КАЛИБРОВКА ВЕРОЯТНОСТЕЙ (PLATT SCALING БЕЗ СМЕЩЕНИЯ)
# ==============================================================================
print("\n[6/7] Калибровка вероятностей (Platt scaling без смещения) на валидационной выборке...")

# Получаем сырые значения формулы (logits) для валидации (оригинальной)
val_logits = model.predict(X_val, prediction_type='RawFormulaVal')
y_val_binary = (y_val == 1).astype(int)

# Обучаем логистическую регрессию без свободного члена (fit_intercept=False)
platt = LogisticRegression(C=1e9, solver='lbfgs', fit_intercept=False)
platt.fit(val_logits.reshape(-1, 1), y_val_binary)

calib_a = platt.coef_[0][0]
calib_b = 0.0  # intercept = 0, чтобы сохранить симметрию

# Функция калибровки
def calibrate_proba(logits, a, b):
    return 1 / (1 + np.exp(-(a * logits + b)))

# Оценка калибровки на валидации
val_proba_raw = model.predict_proba(X_val)[:, 1]
val_proba_calib = calibrate_proba(val_logits, calib_a, calib_b)
brier_raw = brier_score_loss(y_val_binary, val_proba_raw)
brier_calib = brier_score_loss(y_val_binary, val_proba_calib)
print(f"✓ Brier score на валидации: raw = {brier_raw:.4f}, calibrated = {brier_calib:.4f}")

# ==============================================================================
# 10. ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ С КАЛИБРОВКОЙ
# ==============================================================================
print("\n[7/7] Оценка качества на тестовой выборке (2024-2025)...")

test_logits = model.predict(X_test, prediction_type='RawFormulaVal')
y_test_proba_calib = calibrate_proba(test_logits, calib_a, calib_b)
y_test_pred_calib = (y_test_proba_calib > 0.5).astype(int) * 2 - 1

# Метрики
test_acc = accuracy_score(y_test, y_test_pred_calib)
test_bal_acc = balanced_accuracy_score(y_test, y_test_pred_calib)
test_auc = roc_auc_score(y_test, y_test_proba_calib)
test_brier = brier_score_loss((y_test == 1).astype(int), y_test_proba_calib)

print(f"\n📊 МЕТРИКИ КАЧЕСТВА НА ТЕСТЕ (2024-2025):")
print(f"   Accuracy:          {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   Balanced Accuracy: {test_bal_acc:.4f} ({test_bal_acc*100:.2f}%)")
print(f"   AUC-ROC:           {test_auc:.4f}")
print(f"   Brier Score:       {test_brier:.4f}")

# Прибыльность стратегии
def calculate_profitability(y_true, y_proba, edge_threshold=0.05):
    y_true_binary = (y_true == 1).astype(int)
    edge = np.abs(y_proba - 0.5)
    valid_bets = edge > edge_threshold
    if valid_bets.sum() == 0:
        return {'total_bets': 0, 'win_rate': 0, 'profit_units': 0, 'roi': 0}
    bets_outcome = np.where(
        ((y_proba > 0.5) & (y_true_binary == 1)) | ((y_proba <= 0.5) & (y_true_binary == 0)),
        1, -1
    )
    valid_outcomes = bets_outcome[valid_bets]
    total_bets = len(valid_outcomes)
    wins = (valid_outcomes == 1).sum()
    win_rate = wins / total_bets if total_bets > 0 else 0
    profit_units = valid_outcomes.sum()
    roi = (profit_units / total_bets) * 100 if total_bets > 0 else 0
    return {
        'total_bets': total_bets,
        'win_rate': win_rate,
        'profit_units': profit_units,
        'roi': roi
    }

test_profit = calculate_profitability(y_test, y_test_proba_calib, CONFIG['edge_threshold'])
print(f"\n💰 ПРИБЫЛЬНОСТЬ СТРАТЕГИИ (ставка при edge > {CONFIG['edge_threshold']*100:.0f}%):")
print(f"Тест (2024-2025): {test_profit['total_bets']} ставок | Win Rate: {test_profit['win_rate']*100:.1f}% | Прибыль: {test_profit['profit_units']:+.2f} units | ROI: {test_profit['roi']:+.2f}%")

# ==============================================================================
# 11. СОХРАНЕНИЕ МОДЕЛИ И ПАРАМЕТРОВ КАЛИБРОВКИ
# ==============================================================================
print("\n[8/7] Сохранение модели и параметров калибровки...")

model_path = os.path.join(CONFIG['output_dir'], 'winner_model_catboost_v1.cbm')
model.save_model(model_path, format='cbm')
print(f"✅ Базовая модель сохранена: {model_path}")

calib_params = {'a': calib_a, 'b': calib_b}
calib_path = os.path.join(CONFIG['output_dir'], 'calibration_params.joblib')
joblib.dump(calib_params, calib_path)
print(f"✅ Параметры калибровки сохранены: {calib_path}")

# ==============================================================================
# 12. СОХРАНЕНИЕ ОТЧЁТА
# ==============================================================================
report = f"""
МОДЕЛЬ ПРЕДСКАЗАНИЯ ПОБЕДИТЕЛЯ UFC — CATBOOST (УГЛУБЛЁННАЯ ВЕРСИЯ)
Дата обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
ИСТОЧНИК ДАННЫХ:
  • Период: {CONFIG['min_date']} — {df['event_date'].max().strftime('%Y-%m-%d')}
  • Боев: {len(df):,}
  • Признаков: {len(feature_cols)}

МОДЕЛЬ:
  • Итераций (best): {model.get_best_iteration() + 1}
  • Параметры: {CONFIG['catboost_params']}
  • Аугментация train: да
  • Аугментация val: да
  • Калибровка: Platt scaling без смещения (a={calib_a:.4f}, b=0)

РЕЗУЛЬТАТЫ НА ТЕСТЕ (2024-2025):
  • Accuracy:    {test_acc*100:.2f}%
  • AUC-ROC:     {test_auc:.4f}
  • Brier Score: {test_brier:.4f}
  • Прибыль (edge>5%): {test_profit['profit_units']:+.2f} units
  • ROI:         {test_profit['roi']:+.2f}%
  • Ставок:      {test_profit['total_bets']}
  • Win Rate:    {test_profit['win_rate']*100:.1f}%

ТОП-10 ПРИЗНАКОВ ПО ВАЖНОСТИ:
"""
importance = model.get_feature_importance()
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance / importance.sum() * 100
}).sort_values('importance', ascending=False).head(10)

for i, (_, row) in enumerate(importance_df.iterrows(), 1):
    report += f"  {i:2d}. {row['feature']:40s} | {row['importance']:5.2f}%\n"

report += """
================================================================================
ИНСТРУКЦИЯ ПО ЗАГРУЗКЕ:
  1. Загрузите базовую модель: 
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model('winner_model_catboost_v1.cbm')
  2. Загрузите параметры калибровки:
        import joblib
        calib = joblib.load('calibration_params.joblib')
  3. Для получения калиброванных вероятностей:
        logits = model.predict(X, prediction_type='RawFormulaVal')
        proba = 1 / (1 + np.exp(-(calib['a'] * logits)))
   (b=0, поэтому можно не использовать)
"""

report_path = os.path.join(CONFIG['output_dir'], 'catboost_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"✓ Отчет сохранен: {report_path}")

print("\n" + "="*80)
print("✅ МОДЕЛЬ УСПЕШНО ОБУЧЕНА С УЧЁТОМ СИММЕТРИИ И КАЛИБРОВКИ")
print("="*80)