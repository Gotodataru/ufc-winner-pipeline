import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from catboost import CatBoostClassifier
import joblib
import sys
from pathlib import Path
from datetime import datetime

# Настройка путей
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config import DATA_DIR, MODEL_DIR

# Настройки стиля графика
plt.style.use('seaborn-v0_8-darkgrid') # Или 'ggplot', если seaborn не установлен

def load_data_and_model():
    """Загрузка данных и модели"""
    print("🔄 Загрузка данных и модели...")
    
    # Путь к данным (используем тот же файл, что и при обучении)
    data_path = DATA_DIR / 'UFC_full_data_golden_fixed.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"Файл {data_path} не найден!")
    
    df = pd.read_csv(data_path, low_memory=False)
    df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    df = df.dropna(subset=['event_date']).sort_values('event_date').reset_index(drop=True)
    
    # Фильтр дат (только тестовый период: после 2023-12-31)
    test_start = '2024-01-01'
    df_test = df[df['event_date'] >= test_start].copy()
    
    if len(df_test) == 0:
        raise ValueError(f"Нет данных для теста после {test_start}")

    # Загрузка модели
    model_path = MODEL_DIR / 'winner_model_catboost_v1.cbm'
    calib_path = MODEL_DIR / 'calibration_params.joblib'
    
    if not model_path.exists() or not calib_path.exists():
        raise FileNotFoundError("Модель или файлы калибровки не найдены. Сначала обучите модель.")
        
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    calib_params = joblib.load(str(calib_path))
    
    return df_test, model, calib_params

def prepare_features(df):
    """Подготовка признаков (копия логики из train_winner_model.py)"""
    # Удаление утечек
    leak_cols = [col for col in df.columns if 'odds' in col.lower() or ('_r' in col.lower() and any(x in col for x in ['_r1_', '_r2_', '_r3_', '_r4_', '_r5_']))]
    df_clean = df.drop(columns=leak_cols, errors='ignore')
    
    # Топ признаки (упрощенный список, модель сама отберет нужные)
    # В реальном продакшене лучше хранить список фич в конфиге модели
    feature_cols = [col for col in df_clean.columns if col.startswith('diff_') or col.startswith('weight_') or col in ['title_fight', 'num_rounds', 'age']]
    
    # Заполнение пропусков
    for col in feature_cols:
        if df_clean[col].dtype in [np.float64, np.int64]:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    # One-Hot Encoding для весов (если есть категориальная колонка weight_class)
    if 'weight_class' in df_clean.columns:
        df_clean = pd.get_dummies(df_clean, columns=['weight_class'], prefix='weight')
        # Приводим к тем же колонкам, что были при обучении (насколько возможно)
        # Для простоты берем все числовые колонки кроме таргета и даты
        feature_cols = [c for c in df_clean.columns if c not in ['event_date', 'winner_encoded', 'fighter_1', 'fighter_2']]
        
    return df_clean, feature_cols

def calculate_backtest_metrics(df, model, calib_params, feature_cols, edge_threshold=0.05):
    """Прогон модели и расчет метрик"""
    print("📊 Прогон бэктеста...")
    
    X = df[feature_cols].fillna(0)
    y_true = df['winner_encoded'].values
    dates = df['event_date'].values
    
    # Предсказания
    logits = model.predict(X, prediction_type='RawFormulaVal')
    proba = 1 / (1 + np.exp(-(calib_params['a'] * logits + calib_params['b'])))
    
    # Логика ставок
    predictions = (proba > 0.5).astype(int) * 2 - 1 # 1 или -1
    edges = np.abs(proba - 0.5)
    mask_bet = edges > edge_threshold
    
    # Расчет P&L
    # Если поставили на 1 и выиграл 1 -> +1. Если проиграл -> -1.
    # Если поставили на -1 (бойца 2) и выиграл -1 (боец 2) -> +1.
    
    pnl = []
    cumulative_pnl = []
    current_pnl = 0
    bets_count = 0
    wins = 0
    
    bet_outcomes = []
    bet_dates = []
    
    for i in range(len(df)):
        if mask_bet[i]:
            bet_val = 1 # Ставка 1 единица
            outcome = 0
            
            # Проверка исхода
            if predictions[i] == y_true[i]:
                outcome = 1 # Win
                wins += 1
            else:
                outcome = -1 # Loss
                
            current_pnl += outcome
            bets_count += 1
            
            bet_outcomes.append(outcome)
            bet_dates.append(dates[i])
        
        pnl.append(current_pnl)
        cumulative_pnl.append(current_pnl)
        
    # Метрики
    total_bets = bets_count
    win_rate = wins / total_bets if total_bets > 0 else 0
    total_profit = current_pnl
    roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0
    
    # Accuracy только по сделанным ставкам
    if total_bets > 0:
        # Пересчитываем accuracy только по бетам
        acc_mask = [mask_bet[i] for i in range(len(df))]
        y_pred_bets = predictions[mask_bet]
        y_true_bets = y_true[mask_bet]
        accuracy = np.sum(y_pred_bets == y_true_bets) / len(y_true_bets)
    else:
        accuracy = 0
        
    metrics = {
        'total_bets': total_bets,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'roi': roi,
        'accuracy': accuracy
    }
    
    return bet_dates, bet_outcomes, cumulative_pnl, metrics

def plot_results(bet_dates, cumulative_pnl, metrics):
    """Построение графика"""
    print("🎨 Отрисовка графика...")
    
    if len(bet_dates) == 0:
        print("⚠️ Нет ставок для отображения графика.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Конвертируем даты для matplotlib
    x_dates = pd.to_datetime(bet_dates)
    
    # График прибыли
    ax.plot(x_dates, cumulative_pnl, color='#2E86AB', linewidth=2, label='Cumulative Profit (Units)')
    ax.fill_between(x_dates, cumulative_pnl, alpha=0.3, color='#2E86AB')
    
    # Линия нуля
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    
    # Оформление
    ax.set_title(f'UFC Model Backtest Results (Edge > 5%)\nTotal Profit: {metrics["total_profit"]:+.1f} | ROI: {metrics["roi"]:+.2f}% | WinRate: {metrics["win_rate"]*100:.1f}%', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Fight Date', fontsize=12)
    ax.set_ylabel('Profit (Units)', fontsize=12)
    
    # Форматирование дат
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Текстовая плашка с метриками
    textstr = (
        f"📊 Metrics Summary:\n"
        f"• Total Bets: {metrics['total_bets']}\n"
        f"• Win Rate: {metrics['win_rate']*100:.1f}%\n"
        f"• Accuracy: {metrics['accuracy']*100:.1f}%\n"
        f"• Total Profit: {metrics['total_profit']:+.1f} units\n"
        f"• ROI: {metrics['roi']:+.2f}%"
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=1)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, family='monospace')
    
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    # Сохранение
    output_path = ROOT_DIR / "backtest_result" / f"backtest_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ График сохранен: {output_path}")
    plt.show()

if __name__ == "__main__":
    try:
        df_test, model, calib_params = load_data_and_model()
        df_prep, feature_cols = prepare_features(df_test)
        
        # Важно: нужно убедиться, что колонки в df_prep совпадают с теми, на которых училась модель
        # Модель CatBoost умеет работать с названиями колонок, но лучше явно отфильтровать
        common_cols = [c for c in feature_cols if c in model.feature_names_]
        # Добавим недостающие нулями, если модель ожидает больше
        final_cols = model.feature_names_
        X_final = df_prep.reindex(columns=final_cols, fill_value=0)
        
        # Перепишем функцию расчета для работы с готовым DataFrame
        # (Упрощенная версия для запуска)
        X = X_final.fillna(0)
        y_true = df_test['winner_encoded'].values
        dates = df_test['event_date'].values
        
        logits = model.predict(X, prediction_type='RawFormulaVal')
        proba = 1 / (1 + np.exp(-(calib_params['a'] * logits + calib_params['b'])))
        
        predictions = (proba > 0.5).astype(int) * 2 - 1
        edges = np.abs(proba - 0.5)
        mask_bet = edges > 0.05
        
        cum_pnl = []
        current = 0
        bet_d = []
        
        for i in range(len(df_test)):
            if mask_bet[i]:
                if predictions[i] == y_true[i]:
                    current += 1
                else:
                    current -= 1
                bet_d.append(dates[i])
            cum_pnl.append(current)
            
        # Собираем метрики вручную для графика
        total_bets = len(bet_d)
        wins = sum(1 for i in range(len(df_test)) if mask_bet[i] and predictions[i] == y_true[i])
        wr = wins/total_bets if total_bets else 0
        profit = cum_pnl[-1] if cum_pnl else 0
        roi = (profit/total_bets)*100 if total_bets else 0
        
        # Accuracy на всех тестах (для справки) или только на ставках? Обычно на ставках.
        acc = wr # В данном контексте winrate и accuracy на отобранных ставках совпадают
        
        mets = {'total_bets': total_bets, 'win_rate': wr, 'total_profit': profit, 'roi': roi, 'accuracy': acc}
        
        plot_results(bet_d, [cum_pnl[i] for i in range(len(df_test)) if mask_bet[i]], mets) # Тут логика накопления только по бетам нужна в plot_results, исправим ниже
        
        # Исправленный вызов с правильным накоплением только по моментам ставок
        real_cum = []
        curr = 0
        for i in range(len(df_test)):
            if mask_bet[i]:
                if predictions[i] == y_true[i]: curr += 1
                else: curr -= 1
                real_cum.append(curr)
        
        plot_results(pd.to_datetime(bet_d), real_cum, mets)

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()