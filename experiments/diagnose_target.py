from config import DATA_DIR, MODEL_DIR, BASE_DIR
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ДИАГНОСТИКА ЦЕЛЕВОЙ ПЕРЕМЕННОЙ — выявление проблемы с балансом классов
"""

import pandas as pd
import numpy as np

DATA_PATH = r'str(BASE_DIR)\UFCTOPMODEL\WINNER\winnerbigdata\data\UFC_full_data_golden.csv'

print("="*80)
print("🔍 ДИАГНОСТИКА ЦЕЛЕВОЙ ПЕРЕМЕННОЙ 'winner_encoded'")
print("="*80)

# Загружаем ТОЛЬКО целевую переменную и дату для быстрой диагностики
df = pd.read_csv(
    DATA_PATH,
    usecols=['event_date', 'winner_encoded'],
    parse_dates=['event_date'],
    low_memory=False
)

print(f"\n📊 Исходные данные:")
print(f"   Всего боев: {len(df):,}")
print(f"   Тип данных winner_encoded: {df['winner_encoded'].dtype}")
print(f"   Уникальные значения: {df['winner_encoded'].unique()}")
print(f"   Распределение:\n{df['winner_encoded'].value_counts().to_string()}")

# Проверяем наличие строковых значений
if df['winner_encoded'].dtype == 'object':
    print("\n⚠️  ВНИМАНИЕ: колонка 'winner_encoded' имеет тип object (строка)!")
    print("   Возможные значения как строки:")
    print(df['winner_encoded'].value_counts(dropna=False).head(10).to_string())

# Проверяем наличие пропусков
nulls = df['winner_encoded'].isna().sum()
if nulls > 0:
    print(f"\n⚠️  Пропусков в winner_encoded: {nulls:,} ({nulls/len(df)*100:.2f}%)")

# Анализ по годам (выявление аномалий во времени)
df['year'] = df['event_date'].dt.year
yearly_dist = df.groupby('year')['winner_encoded'].value_counts().unstack(fill_value=0)
print(f"\n📅 Распределение по годам (топ-10 последних лет):")
print(yearly_dist.tail(10).to_string())

# Сохраняем полный отчет
df['winner_encoded'].value_counts(dropna=False).to_csv(
    r'str(BASE_DIR)\UFCTOPMODEL\WINNER\winnerbigdata\data\winner_encoded_distribution.csv',
    header=True,
    index_label='value'
)

print("\n" + "="*80)
print("💡 РЕКОМЕНДАЦИИ:")
print("="*80)

# Определяем проблему
unique_vals = df['winner_encoded'].dropna().unique()
if len(unique_vals) == 1:
    print("❌ КРИТИЧЕСКАЯ ОШИБКА: в данных только ОДНО значение целевой переменной!")
    print("   Возможные причины:")
    print("   1. Ошибка при генерации датасета — все бои закодированы как победа бойца 1")
    print("   2. winner_encoded хранится как строка '1' вместо числа 1 или -1")
    print("   3. Признаки бойца 1 и бойца 2 перепутаны местами в исходных данных")
elif set(unique_vals) == {1.0}:
    print("❌ ОШИБКА: только победы бойца 1 (1.0) — нет побед бойца 2 (-1.0)")
    print("   Требуется перегенерация датасета с правильной кодировкой победителя")
elif set(unique_vals) == {1, 0}:
    print("⚠️  Внимание: winner_encoded содержит 1 и 0 вместо 1 и -1")
    print("   Возможно, 0 = победа бойца 2, а не ничья. Требуется уточнение логики кодировки")
else:
    print("✅ Целевая переменная содержит множественные классы — проблема в фильтрации")
    print("   Проверьте логику: df = df[df['winner_encoded'].isin([1, -1])]")
    print("   Возможно, -1 хранится как строка '-1' или как 0")

print("\n📁 Отчет сохранен: winner_encoded_distribution.csv")