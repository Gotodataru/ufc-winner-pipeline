#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_value_bets.py
Бэктест стратегий value betting из value_bet_filters.py на исторических данных.
Использует симметризованные вероятности модели CatBoost.
"""

import os
import sys
import pandas as pd
import numpy as np
import catboost
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Импортируем конфигурации стратегий
from value_bet_filters import FILTER_CONFIGS, ACTIVE_STRATEGY

# ===================== НАСТРОЙКИ =====================
DATA_PATH = r'D:\BETTING\UFCTOPMODEL\WINNER\winnerbigdata\data\UFC_full_data_golden_fixed.csv'
MODEL_PATH = r'D:\BETTING\UFCTOPMODEL\WINNER\winnerbigdata\model\winner_model_catboost_v1.cbm'
CALIB_PATH = r'D:\BETTING\UFCTOPMODEL\WINNER\winnerbigdata\model\calibration_params.joblib'
OUTPUT_DIR = r'D:\BETTING\UFCTOPMODEL\WINNER\winnerbigdata\model'

# Период бэктеста (можно изменить)
TEST_START_DATE = '2024-01-01'
TEST_END_DATE = None  # None = до конца данных

# Какие стратегии тестировать: 'active' только активную, 'all' все из FILTER_CONFIGS
TEST_STRATEGIES = 'all'  # или 'active'

# Минимальное количество ставок для вывода стратегии
MIN_BETS = 10

# Начальный банк (для графика)
INITIAL_BANKROLL = 100.0

# ===================== ЗАГРУЗКА ДАННЫХ И МОДЕЛИ =====================
print("=" * 80)
print("🔍 БЭКТЕСТ СТРАТЕГИЙ VALUE BETTING")
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

# Удаление колонок-утечек (как в optimize_filters.py)
leak_cols = [col for col in df_test.columns if 'ko_odds' in col.lower() or 'sub_odds' in col.lower()]
round_leak_cols = [col for col in df_test.columns if '_r1_' in col or '_r2_' in col or '_r3_' in col or '_r4_' in col or '_r5_' in col]
df_test = df_test.drop(columns=leak_cols + round_leak_cols, errors='ignore')
df_test = df_test[df_test['diff_age'].abs() <= 25].copy()

# One-hot encoding весовых категорий (если ещё не сделано)
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
    use_calibration = True
    print("Параметры калибровки загружены.")
else:
    calib = None
    use_calibration = False
    print("Калибровка не найдена, используются сырые вероятности.")

feature_order = model.feature_names_
print(f"Количество признаков: {len(feature_order)}")

# Синхронизация признаков
available_features = [col for col in feature_order if col in df_test.columns]
if len(available_features) < len(feature_order):
    missing = set(feature_order) - set(available_features)
    print(f"Внимание: отсутствуют признаки: {missing}")
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

# ===================== ФУНКЦИЯ БЭКТЕСТА =====================
def backtest_strategy(params, name):
    """
    params: dict с ключами min_odds, max_odds, min_model_prob, min_edge, min_ev, bet_size_pct
    """
    # Сигналы для Fighter 1 и Fighter 2
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
            'strategy': name,
            'bets': 0,
            'profit_pct': 0.0,
            'roi': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'avg_odds': 0.0,
            'avg_ev': 0.0
        }
    
    # Определяем исходы
    bet_size = params['bet_size_pct'] / 100.0  # в долях от банка
    odds_for_bet = np.where(signal_f1, odds_1, np.where(signal_f2, odds_2, 1.0))
    bet_outcome = np.where(
        (signal_f1 & (y_test == 1)) | (signal_f2 & (y_test == -1)),
        1,
        np.where(valid_bet, -1, 0)
    )
    
    # Прибыль каждой ставки в % от банка
    profit_per_bet = np.where(
        bet_outcome == 1,
        bet_size * (odds_for_bet - 1),
        np.where(bet_outcome == -1, -bet_size, 0)
    )
    
    total_profit_pct = profit_per_bet.sum() * 100  # в процентах от начального банка
    roi = (profit_per_bet.sum() / total_bets) * 100 if total_bets > 0 else 0
    
    win_rate = (bet_outcome == 1).sum() / total_bets * 100
    
    # Просадка
    cumulative = np.cumsum(profit_per_bet * 100)  # в процентах
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - rolling_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
    
    # Средние показатели
    avg_odds = odds_for_bet[valid_bet].mean()
    avg_ev = np.where(signal_f1, ev_1, np.where(signal_f2, ev_2, 0))[valid_bet].mean() * 100
    
    return {
        'strategy': name,
        'bets': total_bets,
        'profit_pct': total_profit_pct,
        'roi': roi,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'avg_odds': avg_odds,
        'avg_ev': avg_ev
    }

# ===================== ЗАПУСК БЭКТЕСТА =====================
strategies_to_test = []
if TEST_STRATEGIES == 'active':
    if ACTIVE_STRATEGY not in FILTER_CONFIGS:
        print(f"Ошибка: активная стратегия '{ACTIVE_STRATEGY}' не найдена в FILTER_CONFIGS")
        sys.exit(1)
    strategies_to_test = [(ACTIVE_STRATEGY, FILTER_CONFIGS[ACTIVE_STRATEGY])]
else:
    strategies_to_test = [(name, cfg) for name, cfg in FILTER_CONFIGS.items()]

results = []
for name, cfg in strategies_to_test:
    print(f"Тестирование стратегии: {name} - {cfg.get('name', '')}")
    res = backtest_strategy(cfg, name)
    results.append(res)
    print(f"  Ставок: {res['bets']}, Прибыль: {res['profit_pct']:.2f}%, ROI: {res['roi']:.2f}%, "
          f"Винрейт: {res['win_rate']:.1f}%, Просадка: {res['max_drawdown']:.2f}%")
    print()

# Создаём DataFrame результатов
results_df = pd.DataFrame(results)
results_df = results_df[results_df['bets'] >= MIN_BETS].sort_values('profit_pct', ascending=False)

# Вывод
print("\n" + "=" * 80)
print(f"ИТОГИ БЭКТЕСТА (минимум ставок: {MIN_BETS})")
print("=" * 80)
print(results_df.to_string(index=False, float_format='%.2f'))

# Сохранение
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_csv = os.path.join(OUTPUT_DIR, f'backtest_results_{timestamp}.csv')
results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\n✅ Результаты сохранены в {output_csv}")

# ===================== ГРАФИК (если есть matplotlib) =====================
try:
    import matplotlib.pyplot as plt
    
    # Для лучшей стратегии строим график роста банка
    if len(results_df) > 0:
        best_strat = results_df.iloc[0]['strategy']
        cfg = FILTER_CONFIGS[best_strat]
        print(f"\n📈 Построение графика для стратегии: {best_strat}")
        
        # Повторно получаем сигналы для лучшей стратегии
        signal_f1 = (
            (odds_1 >= cfg['min_odds']) & (odds_1 <= cfg['max_odds']) &
            (y_proba >= cfg['min_model_prob']) &
            (edge_1 >= cfg['min_edge']) & (ev_1 >= cfg['min_ev'])
        )
        signal_f2 = (
            (odds_2 >= cfg['min_odds']) & (odds_2 <= cfg['max_odds']) &
            ((1 - y_proba) >= cfg['min_model_prob']) &
            (edge_2 >= cfg['min_edge']) & (ev_2 >= cfg['min_ev'])
        )
        valid_bet = signal_f1 | signal_f2
        
        bet_size = cfg['bet_size_pct'] / 100.0
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
        
        cumulative = np.cumsum(profit_per_bet * 100) + INITIAL_BANKROLL
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative, label=f'{best_strat} ({cfg["name"]})', color=cfg.get('color', 'blue'))
        plt.axhline(y=INITIAL_BANKROLL, color='gray', linestyle='--', alpha=0.5)
        plt.title('Рост банка при ставке 1% от текущего (simulated)')
        plt.xlabel('Номер ставки')
        plt.ylabel('Банк, %')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        graph_path = os.path.join(OUTPUT_DIR, f'backtest_chart_{timestamp}.png')
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✅ График сохранён в {graph_path}")
except ImportError:
    print("\n⚠️ matplotlib не установлен, график не построен.")

print("\n" + "=" * 80)