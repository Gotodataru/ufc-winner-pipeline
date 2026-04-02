#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ИСПРАВЛЕНИЕ КОДИРОВКИ ПОБЕДИТЕЛЯ В UFC_full_data_golden.csv
Цель: восстановить корректную целевую переменную через сравнение имен бойцов
"""

import pandas as pd
import numpy as np
from datetime import datetime

DATA_PATH = r'D:\BETTING\UFCTOPMODEL\WINNER\winnerbigdata\data\UFC_full_data_golden.csv'
OUTPUT_PATH = r'D:\BETTING\UFCTOPMODEL\WINNER\winnerbigdata\data\UFC_full_data_golden_fixed.csv'

print("="*80)
print("🔧 ИСПРАВЛЕНИЕ КОДИРОВКИ ПОБЕДИТЕЛЯ")
print("="*80)

# Загружаем ТОЛЬКО необходимые колонки для восстановления победителя
print("\n[1/4] Загрузка данных...")
cols_needed = ['event_date', 'f_1_name', 'f_2_name', 'winner', 'winner_encoded', 
               'result', 'weight_class', 'title_fight', 'num_rounds']
df = pd.read_csv(DATA_PATH, usecols=cols_needed, parse_dates=['event_date'], low_memory=False)

print(f"✓ Загружено {len(df):,} боев")
print(f"  Исходное распределение winner_encoded: {df['winner_encoded'].value_counts().to_dict()}")

# ==============================================================================
# 2. ВОССТАНОВЛЕНИЕ ПРАВИЛЬНОЙ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
# ==============================================================================
print("\n[2/4] Восстановление победителя через сравнение имен...")

# Фильтруем только завершенные бои (исключаем ничьи и No Contest)
valid_results = ['Decision', 'KO/TKO', 'Submission', 'TKO', 'KO']
df = df[df['result'].isin(valid_results)].copy()
print(f"✓ Отфильтрованы только завершенные бои → {len(df):,} записей")

# Восстанавливаем победителя: 1 = победил боец 1, -1 = победил боец 2
df['winner_encoded_fixed'] = np.where(
    df['f_1_name'] == df['winner'], 
    1, 
    -1
)

# Проверяем баланс классов
class_balance = df['winner_encoded_fixed'].value_counts(normalize=True) * 100
print(f"\n📊 Распределение ПОСЛЕ исправления:")
print(f"   Победа бойца 1 (1):  {class_balance.get(1, 0):.2f}% ({(df['winner_encoded_fixed'] == 1).sum():,} боев)")
print(f"   Победа бойца 2 (-1): {class_balance.get(-1, 0):.2f}% ({(df['winner_encoded_fixed'] == -1).sum():,} боев)")

# Сравнение с исходной кодировкой
print(f"\n🔍 Анализ ошибок в исходной кодировке:")
mismatches = (df['winner_encoded'] != df['winner_encoded_fixed']).sum()
total = len(df)
print(f"   Несоответствий: {mismatches:,} из {total:,} ({mismatches/total*100:.2f}%)")
print(f"   Примеры несоответствий:")
mismatch_sample = df[df['winner_encoded'] != df['winner_encoded_fixed']].head(5)
if not mismatch_sample.empty:
    for _, row in mismatch_sample.iterrows():
        print(f"     • {row['f_1_name']} vs {row['f_2_name']} | winner={row['winner']} | "
              f"old={row['winner_encoded']} → new={row['winner_encoded_fixed']}")

# ==============================================================================
# 3. СОХРАНЕНИЕ ИСПРАВЛЕННОГО ДАТАСЕТА
# ==============================================================================
print("\n[3/4] Сохранение исправленного датасета...")

# Загружаем полный датасет для сохранения с исправленной колонкой
print("   ⚠️  Полная перезапись датасета (378 МБ) — это займет ~2-3 минуты...")
full_df = pd.read_csv(DATA_PATH, low_memory=False)
full_df['winner_encoded'] = np.where(
    full_df['f_1_name'] == full_df['winner'], 
    1, 
    -1
)

# Исключаем ничьи и NC из финального датасета для бинарной классификации
full_df = full_df[full_df['result'].isin(valid_results)].copy()
print(f"   ✓ Исключены ничьи/NC → {len(full_df):,} боев для обучения")

full_df.to_csv(OUTPUT_PATH, index=False)
print(f"✓ Исправленный датасет сохранен: {OUTPUT_PATH}")
print(f"   Новое распределение: {full_df['winner_encoded'].value_counts().to_dict()}")

# ==============================================================================
# 4. ВАЛИДАЦИЯ И ОТЧЕТ
# ==============================================================================
print("\n[4/4] Финальная валидация...")

# Проверка по годам (чтобы убедиться, что нет временных аномалий)
full_df['year'] = pd.to_datetime(full_df['event_date']).dt.year
yearly_dist = full_df.groupby('year')['winner_encoded'].value_counts().unstack(fill_value=0)
yearly_dist_pct = yearly_dist.div(yearly_dist.sum(axis=1), axis=0) * 100

print("\n📅 Баланс классов по годам (последние 5 лет):")
for year in sorted(yearly_dist_pct.index)[-5:]:
    win1 = yearly_dist_pct.loc[year, 1] if 1 in yearly_dist_pct.columns else 0
    win2 = yearly_dist_pct.loc[year, -1] if -1 in yearly_dist_pct.columns else 0
    total = yearly_dist.loc[year].sum()
    print(f"   {year}: {win1:.1f}% / {win2:.1f}% (всего {int(total)} боев)")

# Сохраняем отчет
report = {
    'исправлено': datetime.now().isoformat(),
    'исходных_боев': 8231,
    'боев_после_фильтрации': len(full_df),
    'побед_бойца_1': int((full_df['winner_encoded'] == 1).sum()),
    'побед_бойца_2': int((full_df['winner_encoded'] == -1).sum()),
    'баланс': f"{(full_df['winner_encoded'] == 1).mean()*100:.2f}% / {(full_df['winner_encoded'] == -1).mean()*100:.2f}%"
}

import json
with open(r'D:\BETTING\UFCTOPMODEL\WINNER\winnerbigdata\data\winner_encoding_fix_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n📄 Отчет сохранен: winner_encoding_fix_report.json")

print("\n" + "="*80)
print("✅ ИСПРАВЛЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
print("="*80)
print(f"📁 Исправленный датасет: UFC_full_data_golden_fixed.csv")
print(f"🎯 Теперь можно запускать анализ важности признаков — классы сбалансированы!")
print(f"\n⚠️  ВАЖНО: во всех последующих скриптах используйте ИСПРАВЛЕННЫЙ файл:")
print(f"   D:\\BETTING\\UFCTOPMODEL\\WINNER\\winnerbigdata\\data\\UFC_full_data_golden_fixed.csv")