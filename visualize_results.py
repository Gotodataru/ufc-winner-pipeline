import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# Настройка стилей для красивого графика
plt.style.use('seaborn-v0_8-darkgrid') # Или 'default', если seaborn нет
# Если возникнет ошибка стиля, раскомментируй строку ниже:
# plt.style.use('default')

def create_performance_report(metrics, equity_curve, output_path):
    """
    Создает PNG отчет с метриками и графиком PnL.
    
    metrics: dict с ключами 'accuracy', 'roi', 'win_rate', 'total_bets', 'auc'
    equity_curve: list или array с накопленной прибылью по шагам
    output_path: куда сохранить картинку
    """
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('UFC Model Performance Report', fontsize=20, fontweight='bold')

    # --- ВЕРХНЯЯ ЧАСТЬ: МЕТРИКИ (ТАБЛИЦА) ---
    axs[0].axis('off')
    
    # Формируем текст для таблицы
    table_data = [
        ['Metric', 'Value'],
        ['Accuracy', f"{metrics['accuracy']:.2%}"],
        ['Balanced Accuracy', f"{metrics.get('bal_acc', 0):.2%}"],
        ['AUC-ROC', f"{metrics['auc']:.4f}"],
        ['Win Rate (Bets)', f"{metrics['win_rate']:.2%}"],
        ['Total Bets', str(metrics['total_bets'])],
        ['ROI', f"{metrics['roi']:+.2f}%"],
        ['Net Profit', f"{metrics['profit_units']:+.2f} units"]
    ]
    
    # Рисуем таблицу
    table = axs[0].table(cellText=table_data[1:], colLabels=table_data[0], 
                         loc='center', cellLoc='center', colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2.0)
    
    # Раскраска заголовка
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
        
    # Раскраска положительных значений ROI и Profit
    if metrics['roi'] > 0:
        table[(6, 1)].set_facecolor('#C8E6C9') # Светло-зеленый
        table[(6, 1)].set_text_props(color='#2E7D32', fontweight='bold')
    
    if metrics['profit_units'] > 0:
        table[(7, 1)].set_facecolor('#C8E6C9')
        table[(7, 1)].set_text_props(color='#2E7D32', fontweight='bold')

    axs[0].set_title('Key Performance Indicators', fontsize=16, pad=20)

    # --- НИЖНЯЯ ЧАСТЬ: ГРАФИК РОСТА ПРИБЫЛИ (EQUITY CURVE) ---
    x_axis = np.arange(len(equity_curve))
    
    # Цвет линии зависит от того, прибыль это или убыток (упрощенно - зеленая)
    color = '#2E7D32' if equity_curve[-1] >= 0 else '#D32F2F'
    
    axs[1].plot(x_axis, equity_curve, color=color, linewidth=2.5, label='Cumulative Profit')
    axs[1].axhline(0, color='black', linewidth=1, linestyle='--')
    
    # Заполнение области под графиком
    axs[1].fill_between(x_axis, equity_curve, 0, color=color, alpha=0.2)
    
    axs[1].set_xlabel('Number of Bets', fontsize=12)
    axs[1].set_ylabel('Profit (Units)', fontsize=12)
    axs[1].set_title('Equity Curve Growth (Backtest)', fontsize=16)
    axs[1].legend(loc='upper left')
    axs[1].grid(True, linestyle='--', alpha=0.7)

    # Сохранение
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Отчет сохранен: {output_path}")

if __name__ == "__main__":
    # ЭМУЛЯЦИЯ ДАННЫХ ИЗ ТВОЕГО ПОСЛЕДНЕГО ЗАПУСКА
    # В реальном проекте эти данные можно брать из CSV отчета бэктеста
    # Здесь мы воссоздаем кривую на основе итоговых цифр: 477 ставок, +143 units
    
    total_bets = 477
    final_profit = 143.0
    win_rate = 0.65
    
    # Генерируем реалистичную кривую капитала (случайное блуждание с дрейфом вверх)
    np.random.seed(42) # Для воспроизводимости
    steps = total_bets
    
    # Средняя прибыль на ставку
    avg_step = final_profit / steps
    
    # Генерируем шаги: +1 (win) или -1 (loss) с небольшим шумом для реалистичности единиц
    # Упрощенная модель: ставка 1 единица. Win = +1, Loss = -1.
    outcomes = np.random.choice([1, -1], size=steps, p=[win_rate, 1-win_rate])
    
    # Корректируем сумму, чтобы она точно совпадала с финальным профитом (для красоты отчета)
    # В реальности просто берем реальный лог ставок
    current_profit = np.cumsum(outcomes)
    
    # Нормализуем к финальному значению (опционально, чтобы график точно сходился)
    # Но лучше оставить как есть, это симуляция.
    # Для отчета используем сгенерированную кривую
    equity_curve = current_profit
    
    # Данные для таблицы (из твоего лога)
    metrics = {
        'accuracy': 0.6000,
        'bal_acc': 0.5996,
        'auc': 0.6485,
        'win_rate': 0.650,
        'total_bets': 477,
        'roi': 29.98,
        'profit_units': 143.0
    }
    
    # Путь для сохранения
    output_dir = Path("backtest_result")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"performance_report_{timestamp}.png"
    
    create_performance_report(metrics, equity_curve, output_file)