from config import DATA_DIR, MODEL_DIR, BASE_DIR
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
optimize_filters.py
Расширенный поиск оптимальных параметров стратегий value betting на валидационном периоде (до 2024).
Без утечек: используется только период до 2023-12-31.
"""

import pandas as pd
import numpy as np
import joblib
import catboost
from datetime import datetime
import os
import itertools
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. ЗАГРУЗКА ДАННЫХ И МОДЕЛИ
# ==============================================================================
DATA_PATH = r'str(BASE_DIR)\UFCTOPMODEL\WINNER\winnerbigdata\data\UFC_full_data_golden_fixed.csv'
MODEL_PATH = r'str(BASE_DIR)\UFCTOPMODEL\WINNER\winnerbigdata\model\winner_model_catboost_v1.cbm'
CALIB_PATH = r'str(BASE_DIR)\UFCTOPMODEL\WINNER\winnerbigdata\model\calibration_params.joblib'
OUTPUT_DIR = r'str(BASE_DIR)\UFCTOPMODEL\WINNER\winnerbigdata\model'

print("="*80)
print("🔍 ОПТИМИЗАЦИЯ ПАРАМЕТРОВ СТРАТЕГИЙ НА ВАЛИДАЦИОННОМ ПЕРИОДЕ (до 2024)")
print("="*80)

# Загрузка данных
df = pd.read_csv(DATA_PATH, low_memory=False)
df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
df = df.dropna(subset=['event_date']).copy()

# Оставляем только бои до 2024 года (валидационный период)
df_val = df[df['event_date'] <= '2023-12-31'].copy().reset_index(drop=True)
df_val = df_val.dropna(subset=['f_1_odds', 'f_2_odds']).reset_index(drop=True)

# Удаление утечек
leak_cols = [col for col in df_val.columns if 'ko_odds' in col.lower() or 'sub_odds' in col.lower()]
round_leak_cols = [col for col in df_val.columns if '_r1_' in col or '_r2_' in col or '_r3_' in col or '_r4_' in col or '_r5_' in col]
df_val = df_val.drop(columns=leak_cols + round_leak_cols, errors='ignore')
df_val = df_val[df_val['diff_age'].abs() <= 25].copy()

# One-hot encoding весовых категорий
if 'weight_class' in df_val.columns:
    top_weights = df_val['weight_class'].value_counts().nlargest(10).index.tolist()
    df_val['weight_class'] = df_val['weight_class'].apply(lambda x: x if x in top_weights else 'other')
    weight_dummies = pd.get_dummies(df_val['weight_class'], prefix='weight')
    df_val = pd.concat([df_val.drop('weight_class', axis=1), weight_dummies], axis=1)

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
available_features = [col for col in feature_order if col in df_val.columns]
feature_order = available_features
X_val_raw = df_val[feature_order].copy()
y_val_raw = df_val['winner_encoded'].copy()
odds_1_raw = df_val['f_1_odds'].copy()
odds_2_raw = df_val['f_2_odds'].copy()

mask_valid = X_val_raw.notna().all(axis=1) & y_val_raw.notna() & odds_1_raw.notna() & odds_2_raw.notna()
X_val = X_val_raw[mask_valid].copy().reset_index(drop=True)
y_val = y_val_raw[mask_valid].copy().reset_index(drop=True)
odds_1 = odds_1_raw[mask_valid].copy().reset_index(drop=True)
odds_2 = odds_2_raw[mask_valid].copy().reset_index(drop=True)

# Функция для получения калиброванных вероятностей (с симметризацией)
def get_calibrated_proba(model, X, calib):
    logits = model.predict(X, prediction_type='RawFormulaVal')
    if calib is None:
        return model.predict_proba(X)[:, 1]
    else:
        return 1 / (1 + np.exp(-(calib['a'] * logits)))

# Функция для симметризованного предсказания
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
print("Вычисление симметризованных вероятностей для валидации...")
y_proba = predict_symmetrized(model, X_val, calib, diff_cols)

# Рассчитываем букмекерские вероятности и EV
bookmaker_prob_1 = 1 / odds_1
bookmaker_prob_2 = 1 / odds_2
ev_fighter_1 = (y_proba * odds_1) - 1
ev_fighter_2 = ((1 - y_proba) * odds_2) - 1
edge_fighter_1 = y_proba - bookmaker_prob_1
edge_fighter_2 = (1 - y_proba) - bookmaker_prob_2

print(f"✅ Данные подготовлены: {len(X_val)} боев на валидации (до 2024)")

# ==============================================================================
# 2. РАСШИРЕННАЯ СЕТКА ПАРАМЕТРОВ
# ==============================================================================
param_grid = {
    'min_odds': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    'max_odds': [6.0, 10.0, 15.0, 20.0],
    'min_model_prob': [0.30, 0.40, 0.50, 0.60],
    'min_edge': [0.03, 0.06, 0.10, 0.15],
    'min_ev': [0.03, 0.06, 0.10, 0.15, 0.20],
}

# Все комбинации
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(f"Всего комбинаций для перебора: {len(combinations)}")

# ==============================================================================
# 3. ФУНКЦИЯ ДЛЯ РАСЧЁТА МЕТРИК ПО ЗАДАННЫМ ПАРАМЕТРАМ
# ==============================================================================
def evaluate_params(params):
    # Применяем фильтры
    signal_f1 = (
        (odds_1 >= params['min_odds']) &
        (odds_1 <= params['max_odds']) &
        (y_proba >= params['min_model_prob']) &
        (edge_fighter_1 >= params['min_edge']) &
        (ev_fighter_1 >= params['min_ev'])
    )
    signal_f2 = (
        (odds_2 >= params['min_odds']) &
        (odds_2 <= params['max_odds']) &
        ((1 - y_proba) >= params['min_model_prob']) &
        (edge_fighter_2 >= params['min_edge']) &
        (ev_fighter_2 >= params['min_ev'])
    )
    valid_bet = signal_f1 | signal_f2
    total_bets = valid_bet.sum()
    if total_bets == 0:
        return {
            'bets': 0,
            'profit_pct': 0,
            'win_rate': 0,
            'roi': 0,
            'max_drawdown': 0,
            'avg_odds': 0,
            'avg_ev': 0,
            'profit_drawdown_ratio': 0
        }
    
    # Определяем исходы
    bet_size = 0.01  # фиксированный 1% для сравнения
    odds_for_bet = np.where(signal_f1, odds_1, np.where(signal_f2, odds_2, 1.0))
    bet_outcome = np.where(
        (signal_f1 & (y_val == 1)) | (signal_f2 & (y_val == -1)),
        1,
        np.where(valid_bet, -1, 0)
    )
    profit = np.where(
        bet_outcome == 1,
        bet_size * (odds_for_bet - 1),
        np.where(bet_outcome == -1, -bet_size, 0)
    )
    
    win_rate = (bet_outcome == 1).sum() / total_bets * 100
    avg_odds = odds_for_bet[valid_bet].mean()
    total_profit = profit.sum() * 100  # в % от банка
    roi = total_profit / total_bets if total_bets > 0 else 0  # ROI в % (средний доход на ставку)
    
    # Просадка
    cumulative = np.cumsum(profit * 100)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - rolling_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
    
    # Средний EV
    avg_ev = np.where(
        signal_f1, ev_fighter_1,
        np.where(signal_f2, ev_fighter_2, 0)
    )[valid_bet].mean() * 100
    
    # Отношение прибыль/просадка (по модулю)
    profit_drawdown_ratio = total_profit / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'bets': total_bets,
        'profit_pct': total_profit,
        'win_rate': win_rate,
        'roi': roi,
        'max_drawdown': max_drawdown,
        'avg_odds': avg_odds,
        'avg_ev': avg_ev,
        'profit_drawdown_ratio': profit_drawdown_ratio
    }

# ==============================================================================
# 4. ПЕРЕБОР ВСЕХ КОМБИНАЦИЙ
# ==============================================================================
results = []
total = len(combinations)
for i, params in enumerate(combinations):
    if (i+1) % 100 == 0:
        print(f"Обработано {i+1}/{total} комбинаций...")
    metrics = evaluate_params(params)
    results.append({
        **params,
        **metrics
    })

# Преобразуем в DataFrame
results_df = pd.DataFrame(results)

# ==============================================================================
# 5. ФИЛЬТРАЦИЯ И СОРТИРОВКА
# ==============================================================================
# Оставляем только комбинации с достаточным числом ставок (например, >=20)
min_bets = 20
df_filtered = results_df[results_df['bets'] >= min_bets].copy()

if len(df_filtered) == 0:
    print(f"Нет комбинаций с количеством ставок >= {min_bets}. Уменьшите порог.")
    exit()

# ==============================================================================
# 6. ВЫВОД ТОП-20 ПО РАЗНЫМ КРИТЕРИЯМ
# ==============================================================================
print("\n" + "="*80)
print(f"ТОП-20 КОМБИНАЦИЙ ПО ПРИБЫЛИ (ставок >= {min_bets})")
print("="*80)
top_by_profit = df_filtered.sort_values('profit_pct', ascending=False).head(20)
print(top_by_profit[['min_odds', 'max_odds', 'min_model_prob', 'min_edge', 'min_ev',
                     'bets', 'profit_pct', 'win_rate', 'max_drawdown', 'roi', 'profit_drawdown_ratio']].to_string(index=False))

print("\n" + "="*80)
print(f"ТОП-20 КОМБИНАЦИЙ ПО СООТНОШЕНИЮ ПРИБЫЛЬ/ПРОСАДКА (ставок >= {min_bets})")
print("="*80)
top_by_ratio = df_filtered.sort_values('profit_drawdown_ratio', ascending=False).head(20)
print(top_by_ratio[['min_odds', 'max_odds', 'min_model_prob', 'min_edge', 'min_ev',
                    'bets', 'profit_pct', 'win_rate', 'max_drawdown', 'roi', 'profit_drawdown_ratio']].to_string(index=False))

# ==============================================================================
# 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ==============================================================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = os.path.join(OUTPUT_DIR, f'optimization_results_{timestamp}.csv')
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ Все результаты сохранены в {output_path}")

# Сохраняем топ-20 отдельно
top20_profit_path = os.path.join(OUTPUT_DIR, f'optimization_top20_profit_{timestamp}.csv')
top_by_profit.to_csv(top20_profit_path, index=False, encoding='utf-8-sig')
print(f"✅ Топ-20 по прибыли сохранён в {top20_profit_path}")

top20_ratio_path = os.path.join(OUTPUT_DIR, f'optimization_top20_ratio_{timestamp}.csv')
top_by_ratio.to_csv(top20_ratio_path, index=False, encoding='utf-8-sig')
print(f"✅ Топ-20 по соотношению прибыль/просадка сохранён в {top20_ratio_path}")

print("\n" + "="*80)
print("💡 РЕКОМЕНДАЦИИ ПО ВЫБОРУ ПАРАМЕТРОВ")
print("="*80)
print("""
- Выбирайте комбинации с достаточным количеством ставок (>30 для стабильности).
- Обратите внимание на стратегии с высоким соотношением прибыль/просадка.
- Избегайте экстремально низких min_odds (1.5–2.0) — они могут давать много ставок, но часто убыточны.
- Проверьте выбранные 2–3 комбинации на тестовом периоде 2024–2025 с помощью backtest_value_bets.py.
""")