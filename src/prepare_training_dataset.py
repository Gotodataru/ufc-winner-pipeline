# D:\BETTING\UFCTOPMODEL\WINNER\winnerbigdata\data\prepare_training_dataset.py
import pandas as pd
import numpy as np
import json
from datetime import datetime

def prepare_training_dataset():
    print("🚀 Подготовка датасета для обучения модели предсказания победителя UFC")
    print("="*70)
    
    # 1. Загрузка данных с экономией памяти
    print("\n[1/5] Загрузка данных...")
    cols_to_load = [
        'event_date', 'weight_class', 'title_fight', 'num_rounds',
        'f_1_age', 'f_2_age', 'f_1_fight_number', 'f_2_fight_number',
        'winner_encoded'
    ]
    
    # Добавляем все разностные признаки за 5 боев
    diff_features_5 = [col for col in pd.read_csv(
        r'D:\BETTING\UFCTOPMODEL\WINNER\winnerbigdata\data\UFC_full_data_golden.csv', 
        nrows=0
    ).columns if col.startswith('diff_') and col.endswith('_5')]
    
    cols_to_load.extend(diff_features_5)
    
    df = pd.read_csv(
        r'D:\BETTING\UFCTOPMODEL\WINNER\winnerbigdata\data\UFC_full_data_golden.csv',
        usecols=cols_to_load,
        parse_dates=['event_date'],
        low_memory=False
    )
    
    print(f"✓ Загружено {len(df)} боев, {len(cols_to_load)} колонок")
    
    # 2. Фильтрация утечек и некачественных данных
    print("\n[2/5] Фильтрация утечек и пропусков...")
    
    # Исключаем коэффициенты (если случайно попали)
    leak_columns = [col for col in df.columns if 'odds' in col.lower() or 'ko_odds' in col.lower() or 'sub_odds' in col.lower()]
    if leak_columns:
        df = df.drop(columns=leak_columns)
        print(f"  → Удалено {len(leak_columns)} колонок с коэффициентами")
    
    # Исключаем статистику раундов (утечка из будущего)
    round_leak_cols = [col for col in df.columns if '_r1_' in col or '_r2_' in col or '_r3_' in col or '_r4_' in col or '_r5_' in col]
    if round_leak_cols:
        df = df.drop(columns=round_leak_cols)
        print(f"  → Удалено {len(round_leak_cols)} колонок со статистикой раундов (утечка)")
    
    # Обработка выбросов в возрасте
    df = df[(df['f_1_age'].between(18, 45)) & (df['f_2_age'].between(18, 45))]
    df['diff_age'] = df['f_1_age'] - df['f_2_age']
    df = df.drop(columns=['f_1_age', 'f_2_age'])
    print(f"  → Отфильтровано выбросов по возрасту, осталось {len(df)} боев")
    
    # Заполнение пропусков в разностных признаках
    diff_cols = [col for col in df.columns if col.startswith('diff_')]
    missing_pct = df[diff_cols].isna().mean() * 100
    cols_to_drop = missing_pct[missing_pct > 30].index.tolist()
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  → Удалено {len(cols_to_drop)} признаков с >30% пропусков")
    
    # Заполняем оставшиеся пропуски медианой
    for col in diff_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # 3. Кодирование категориальных признаков
    print("\n[3/5] Кодирование категориальных признаков...")
    
    # Весовые категории → one-hot encoding (топ-10 + остальные = 'other')
    top_weight_classes = df['weight_class'].value_counts().nlargest(10).index.tolist()
    df['weight_class'] = df['weight_class'].apply(lambda x: x if x in top_weight_classes else 'other')
    weight_dummies = pd.get_dummies(df['weight_class'], prefix='weight')
    df = pd.concat([df.drop('weight_class', axis=1), weight_dummies], axis=1)
    print(f"  → One-hot encoding для {len(weight_dummies.columns)} весовых категорий")
    
    # 4. Валидация целевой переменной
    print("\n[4/5] Валидация целевой переменной...")
    df = df[df['winner_encoded'].isin([1, -1])]  # исключаем ничьи (0)
    print(f"  → Исключены ничьи, осталось {len(df)} боев для бинарной классификации")
    
    # 5. Сортировка по дате (обязательно для временной валидации!)
    df = df.sort_values('event_date').reset_index(drop=True)
    
    # 6. Сохранение метаданных
    print("\n[5/5] Сохранение метаданных и финального датасета...")
    
    feature_metadata = {
        'created_at': datetime.now().isoformat(),
        'total_rows': len(df),
        'total_features': len(df.columns) - 2,  # минус event_date и winner_encoded
        'target_column': 'winner_encoded',
        'temporal_column': 'event_date',
        'feature_groups': {
            'striking_diff_5': [col for col in df.columns if 'slpm' in col or 'str_acc' in col or 'sapm' in col or 'str_def' in col],
            'grappling_diff_5': [col for col in df.columns if 'td_avg' in col or 'td_acc' in col or 'td_def' in col or 'sub_avg' in col or 'ctrl_ratio' in col],
            'physical_diff_5': [col for col in df.columns if 'physical_strength' in col or 'speed' in col or 'chin' in col or 'cardio' in col],
            'zone_accuracy_diff_5': [col for col in df.columns if 'head_acc' in col or 'body_acc' in col or 'leg_acc' in col or 'clinch_acc' in col or 'ground_acc' in col],
            'context': ['diff_age', 'diff_fight_number', 'title_fight', 'num_rounds'] + [col for col in df.columns if col.startswith('weight_')]
        },
        'excluded_features': {
            'bookmaker_odds': leak_columns,
            'round_stats': round_leak_cols,
            'high_missing': cols_to_drop
        },
        'validation_strategy': 'temporal_split: train < 2023-01-01, val 2023, test 2024-2025'
    }
    
    # Сохраняем датасет и метаданные
    output_path = r'D:\BETTING\UFCTOPMODEL\WINNER\winnerbigdata\data\training_dataset.csv'
    metadata_path = r'D:\BETTING\UFCTOPMODEL\WINNER\winnerbigdata\data\feature_metadata.json'
    
    df.to_csv(output_path, index=False)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(feature_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Датасет подготовлен и сохранен:")
    print(f"   → {output_path}")
    print(f"   → {len(df)} строк × {len(df.columns)} колонок")
    print(f"   → {len(df.columns)-2} признаков + 1 целевая переменная + 1 дата")
    print(f"\n📊 Состав признаков:")
    for group, features in feature_metadata['feature_groups'].items():
        print(f"   • {group}: {len(features)} признаков")
    
    return df, feature_metadata

if __name__ == "__main__":
    df, meta = prepare_training_dataset()