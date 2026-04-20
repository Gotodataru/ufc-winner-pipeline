from config import DATA_DIR, MODEL_DIR, BASE_DIR
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_and_test_candidates.py
1. Загружает последний файл оптимизации (optimization_results_*.csv).
2. Отбирает комбинации для аутсайдеров (min_odds >= 2.5, bets >= 30).
3. Берёт топ-5 по прибыли (можно изменить критерий).
4. Для каждой комбинации проводит бэктест на тестовом периоде 2024–2025.
5. Выводит сравнительную таблицу и сохраняет её.
"""

import os
import sys
import pandas as pd
import numpy as np
import catboost
import joblib
from datetime import datetime
import glob

# ===================== НАСТРОЙКИ =====================
DATA_PATH = r'str(BASE_DIR)\UFCTOPMODEL\WINNER\winnerbigdata\data\UFC_full_data_golden_fixed.csv'
MODEL_PATH = r'str(BASE_DIR)\UFCTOPMODEL\WINNER\winnerbigdata\model\winner_model_catboost_v1.cbm'
CALIB_PATH = r'str(BASE_DIR)\UFCTOPMODEL\WINNER\winnerbigdata\model\calibration_params.joblib'
OPTIMIZATION_DIR = r'str(BASE_DIR)\UFCTOPMODEL\WINNER\winnerbigdata\model'

# Период тестирования (можно менять)
TEST_START_DATE = '2024-01-01'
TEST_END_DATE = None  # до конца данных

# Параметры отбора кандидатов
MIN_ODDS_THRESHOLD = 2.5        # минимальный коэффициент (для аутсайдеров)
MIN_BETS_VALIDATION = 30        # минимум ставок на валидации для статистики
TOP_N_CANDIDATES = 5             # сколько кандидатов взять
SORT_BY = 'profit_pct'           # критерий сортировки: 'profit_pct' или 'profit_drawdown_ratio'

# Размер ставки для тестирования (оставляем 1% для сравнения)
BET_SIZE_PCT = 1.0

# ===================== ЗАГРУЗКА ПОСЛЕДНЕГО ФАЙЛА ОПТИМИЗАЦИИ =====================
print("=" * 80)
print("🔍 ПОИСК ЛУЧШИХ КАНДИДАТОВ ДЛЯ АУТСАЙДЕРОВ")
print("=" * 80)

# Ищем самый свежий файл optimization_results_*.csv
opt_files = glob.glob(os.path.join(OPTIMIZATION_DIR, 'optimization_results_*.csv'))
if not opt_files:
    print("❌ Не найден ни один файл optimization_results_*.csv")
    sys.exit(1)
latest_opt = max(opt_files, key=os.path.getctime)
print(f"Загружаем результаты оптимизации: {latest_opt}")

df_opt = pd.read_csv(latest_opt)

# Фильтруем: только с min_odds >= порога и достаточным числом ставок
df_filtered = df_opt[(df_opt['min_odds'] >= MIN_ODDS_THRESHOLD) & (df_opt['bets'] >= MIN_BETS_VALIDATION)].copy()
if df_filtered.empty:
    print(f"❌ Нет комбинаций с min_odds >= {MIN_ODDS_THRESHOLD} и ставок >= {MIN_BETS_VALIDATION}")
    sys.exit(1)

print(f"Найдено комбинаций после фильтрации: {len(df_filtered)}")

# Сортируем по заданному критерию
df_sorted = df_filtered.sort_values(SORT_BY, ascending=False)
candidates = df_sorted.head(TOP_N_CANDIDATES)

print(f"\nТоп-{TOP_N_CANDIDATES} кандидатов (по {SORT_BY}):")
print(candidates[['min_odds', 'max_odds', 'min_model_prob', 'min_edge', 'min_ev',
                  'bets', 'profit_pct', 'max_drawdown', 'profit_drawdown_ratio']].to_string(index=False))

# ===================== ЗАГРУЗКА МОДЕЛИ И ТЕСТОВЫХ ДАННЫХ =====================
print("\n" + "=" * 80)
print("ЗАГРУЗКА МОДЕЛИ И ТЕСТОВЫХ ДАННЫХ (2024–2025)")
print("=" * 80)

# Загрузка данных
df = pd.read_csv(DATA_PATH, low_memory=False)
df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
df = df.dropna(subset=['event_date']).copy()

# Оставляем только тестовый период
df_test = df[df['event_date'] >= TEST_START_DATE].copy()
if TEST_END_DATE:
    df_test = df_test[df_test['event_date'] <= TEST_END_DATE].copy()
df_test = df_test.dropna(subset=['f_1_odds', 'f_2_odds']).reset_index(drop=True)

print(f"Период тестирования: {df_test['event_date'].min().date()} — {df_test['event_date'].max().date()}")
print(f"Всего боев в периоде: {len(df_test)}")

# Удаление колонок-утечек
leak_cols = [col for col in df_test.columns if 'ko_odds' in col.lower() or 'sub_odds' in col.lower()]
round_leak_cols = [col for col in df_test.columns if '_r1_' in col or '_r2_' in col or '_r3_' in col or '_r4_' in col or '_r5_' in col]
df_test = df_test.drop(columns=leak_cols + round_leak_cols, errors='ignore')
df_test = df_test[df_test['diff_age'].abs() <= 25].copy()

# One-hot encoding весовых категорий
if 'weight_class' in df_test.columns:
    top_weights = df_test['weight_class'].value_counts().nlargest(10).index.tolist()
    df_test['weight_class'] = df_test['weight_class'].apply(lambda x: x if x in top_weights else 'other')
    weight_dummies = pd.get_dummies(df_test['weight_class'], prefix='weight')
    df_test = pd.concat([df_test.drop('weight_class', axis=1), weight_dummies], axis=1)

# Загрузка модели
print("\nЗагрузка модели...")
model = catboost.CatBoostClassifier()
model.load_model(MODEL_PATH)
print("Модель загружена.")

# Загрузка калибровки
if os.path.exists(CALIB_PATH):
    calib = joblib.load(CALIB_PATH)
    print("Параметры калибровки загружены.")
else:
    calib = None
    print("Калибровка не найдена, используются сырые вероятности.")

feature_order = model.feature_names_
available_features = [col for col in feature_order if col in df_test.columns]
feature_order = available_features

X_test_raw = df_test[feature_order].copy()
y_test_raw = df_test['winner_encoded'].copy()
odds_1_raw = df_test['f_1_odds'].copy()
odds_2_raw = df_test['f_2_odds'].copy()

mask_valid = X_test_raw.notna().all(axis=1) & y_test_raw.notna() & odds_1_raw.notna() & odds_2_raw.notna()
X_test = X_test_raw[mask_valid].copy().reset_index(drop=True)
y_test = y_test_raw[mask_valid].copy().reset_index(drop=True)
odds_1 = odds_1_raw[mask_valid].copy().reset_index(drop=True)
odds_2 = odds_2_raw[mask_valid].copy().reset_index(drop=True)

print(f"Боев после очистки: {len(X_test)}")

# Функции для вероятностей
def get_calibrated_proba(model, X, calib):
    logits = model.predict(X, prediction_type='RawFormulaVal')
    if calib is None:
        return model.predict_proba(X)[:, 1]
    else:
        return 1 / (1 + np.exp(-(calib['a'] * logits)))

def predict_symmetrized(model, X, calib, diff_cols):
    p_orig = get_calibrated_proba(model, X, calib)
    X_swapped = X.copy()
    f1_cols = [col for col in X.columns if col.startswith('f_1_')]
    f2_cols = [col for col in X.columns if col.startswith('f_2_')]
    for f1, f2 in zip(f1_cols, f2_cols):
        tmp = X_swapped[f1].copy()
        X_swapped[f1] = X_swapped[f2]
        X_swapped[f2] = tmp
    for d in diff_cols:
        X_swapped[d] = -X_swapped[d]
    p_swapped = get_calibrated_proba(model, X_swapped, calib)
    return (p_orig + (1 - p_swapped)) / 2

diff_cols = [col for col in feature_order if col.startswith('diff_')]
print("Вычисление симметризованных вероятностей...")
y_proba = predict_symmetrized(model, X_test, calib, diff_cols)

# Букмекерские вероятности и EV
book_prob_1 = 1 / odds_1
book_prob_2 = 1 / odds_2
ev_1 = (y_proba * odds_1) - 1
ev_2 = ((1 - y_proba) * odds_2) - 1
edge_1 = y_proba - book_prob_1
edge_2 = (1 - y_proba) - book_prob_2

print("✅ Данные готовы.\n")

# ===================== ФУНКЦИЯ БЭКТЕСТА ДЛЯ ОДНОЙ КОМБИНАЦИИ =====================
def backtest_candidate(params, name):
    signal_f1 = (
        (odds_1 >= params['min_odds']) &
        (odds_1 <= params['max_odds']) &
        (y_proba >= params['min_model_prob']) &
        (edge_1 >= params['min_edge']) &
        (ev_1 >= params['min_ev'])
    )
    signal_f2 = (
        (odds_2 >= params['min_odds']) &
        (odds_2 <= params['max_odds']) &
        ((1 - y_proba) >= params['min_model_prob']) &
        (edge_2 >= params['min_edge']) &
        (ev_2 >= params['min_ev'])
    )
    valid_bet = signal_f1 | signal_f2
    total_bets = valid_bet.sum()
    
    if total_bets == 0:
        return {
            'candidate': name,
            'bets': 0,
            'profit_pct': 0.0,
            'roi': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'avg_odds': 0.0,
            'avg_ev': 0.0
        }
    
    bet_size = BET_SIZE_PCT / 100.0
    odds_for_bet = np.where(signal_f1, odds_1, np.where(signal_f2, odds_2, 1.0))
    bet_outcome = np.where(
        (signal_f1 & (y_test == 1)) | (signal_f2 & (y_test == -1)),
        1,
        np.where(valid_bet, -1, 0)
    )
    
    profit_per_bet = np.where(
        bet_outcome == 1,
        bet_size * (odds_for_bet - 1),
        np.where(bet_outcome == -1, -bet_size, 0)
    )
    
    total_profit_pct = profit_per_bet.sum() * 100
    roi = (profit_per_bet.sum() / total_bets) * 100 if total_bets > 0 else 0
    win_rate = (bet_outcome == 1).sum() / total_bets * 100
    
    cumulative = np.cumsum(profit_per_bet * 100)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - rolling_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
    
    avg_odds = odds_for_bet[valid_bet].mean()
    avg_ev = np.where(signal_f1, ev_1, np.where(signal_f2, ev_2, 0))[valid_bet].mean() * 100
    
    return {
        'candidate': name,
        'bets': total_bets,
        'profit_pct': total_profit_pct,
        'roi': roi,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'avg_odds': avg_odds,
        'avg_ev': avg_ev
    }

# ===================== ТЕСТИРОВАНИЕ КАНДИДАТОВ =====================
print("=" * 80)
print("ТЕСТИРОВАНИЕ КАНДИДАТОВ НА ПЕРИОДЕ 2024–2025")
print("=" * 80)

test_results = []
for idx, row in candidates.iterrows():
    # Формируем словарь параметров (добавляем bet_size_pct, который не был в optimisation)
    params = {
        'min_odds': row['min_odds'],
        'max_odds': row['max_odds'],
        'min_model_prob': row['min_model_prob'],
        'min_edge': row['min_edge'],
        'min_ev': row['min_ev'],
    }
    # Создаём имя кандидата из ключевых параметров
    name = (f"odds={row['min_odds']:.1f}-{row['max_odds']:.0f}_"
            f"prob={row['min_model_prob']:.2f}_"
            f"edge={row['min_edge']:.2f}_ev={row['min_ev']:.2f}")
    
    print(f"\nТестирование: {name}")
    res = backtest_candidate(params, name)
    test_results.append(res)
    print(f"  Ставок: {res['bets']}, Прибыль: {res['profit_pct']:.2f}%, ROI: {res['roi']:.2f}%, "
          f"Винрейт: {res['win_rate']:.1f}%, Просадка: {res['max_drawdown']:.2f}%")

# ===================== ИТОГОВАЯ ТАБЛИЦА =====================
results_df = pd.DataFrame(test_results)
results_df = results_df.sort_values('profit_pct', ascending=False)

print("\n" + "=" * 80)
print("ИТОГОВАЯ ТАБЛИЦА (ТЕСТ 2024–2025)")
print("=" * 80)
print(results_df.to_string(index=False, float_format='%.2f'))

# Сохранение
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = os.path.join(OPTIMIZATION_DIR, f'candidates_test_{timestamp}.csv')
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ Результаты сохранены в {output_path}")

# Сравнение с текущими лучшими стратегиями (если хотите)
print("\n" + "=" * 80)
print("ДЛЯ СРАВНЕНИЯ: текущие стратегии на тесте 2024–2025")
print("medium_high_value : +20.53% / -12.46% просадка")
print("high_odds_strict  : +18.43% / -13.57%")
print("high_odds_expanded: +16.97% / -13.00%")
print("=" * 80)