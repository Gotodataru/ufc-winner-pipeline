#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — полный раннер для UFC value betting
Использует стратегию medium_high_value (переходная зона 2.8–5.0),
которая показала лучшие результаты на бэктесте: +20.5% при просадке -12.5%.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import requests
import pandas as pd
import numpy as np
import catboost
import joblib
from bs4 import BeautifulSoup

# Импортируем конфигурации стратегий, но активную стратегию не используем
from value_bet_filters import FILTER_CONFIGS

# ===================== ЗАГРУЗКА .ENV =====================
ROOT = Path(__file__).resolve().parent
env_path = ROOT / ".env"
if env_path.exists():
    try:
        content = env_path.read_text(encoding="utf-8-sig").strip()
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()
        print(f"✅ .env загружен: {env_path}")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки .env: {e}")
else:
    print(f"⚠️ .env не найден: {env_path}")

# ===================== НАСТРОЙКА ЛОГИРОВАНИЯ =====================
UPCOMING_DIR = ROOT / "upcoming_fights"
UPCOMING_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(UPCOMING_DIR / "runner.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ufc_runner")

# ===================== ПРОВЕРКА API КЛЮЧА =====================
API_KEY = os.getenv("ODDS_API_KEY", "").strip()
if not API_KEY:
    logger.error("❌ ODDS_API_KEY не найден в переменных окружения!")
    sys.exit(1)
logger.info(f"✅ API ключ загружен (длина: {len(API_KEY)})")

# ===================== ПУТИ К МОДЕЛИ И ДАННЫМ =====================
MODEL_PATH = ROOT / "model" / "winner_model_catboost_v1.cbm"
CALIB_PATH = ROOT / "model" / "calibration_params.joblib"
DATA_PATH = ROOT / "data" / "UFC_full_data_golden_fixed.csv"

if not MODEL_PATH.exists():
    logger.error(f"❌ Модель не найдена: {MODEL_PATH}")
    sys.exit(1)

logger.info("Загрузка модели...")
model = catboost.CatBoostClassifier()
model.load_model(str(MODEL_PATH))
logger.info(f"✅ Модель загружена, признаков: {len(model.feature_names_)}")

if CALIB_PATH.exists():
    calib = joblib.load(CALIB_PATH)
    logger.info("✅ Параметры калибровки загружены.")
else:
    calib = None
    logger.warning("⚠️ Калибровка не найдена, используются сырые вероятности.")

feature_order = model.feature_names_
logger.info(f"Ожидаемый порядок признаков: {feature_order[:5]}...")

# ===================== ЗАГРУЗКА ДАТАСЕТА =====================
logger.info(f"Загрузка датасета из {DATA_PATH}...")
if not DATA_PATH.exists():
    logger.error(f"❌ Датасет не найден: {DATA_PATH}")
    sys.exit(1)

df_full = pd.read_csv(DATA_PATH, low_memory=False)
df_full['event_date'] = pd.to_datetime(df_full['event_date'], errors='coerce')
df_full = df_full.dropna(subset=['event_date']).copy()
logger.info(f"✅ Датасет загружен, записей: {len(df_full)}")

# ===================== ПОСТРОЕНИЕ СЛОВАРЯ ПОСЛЕДНИХ СТАТИСТИК БОЙЦОВ =====================
def build_fighter_last_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Для каждого бойца находит его самый последний бой (по дате) и сохраняет
    все необходимые для модели признаки (возраст, скользящие средние за 5 боёв и т.д.)
    """
    fighter_stats = {}
    # Все колонки датасета
    all_cols = df.columns.tolist()

    # Функция для извлечения имени бойца и его признаков из строки
    def process_fighter(row, fighter_prefix):
        # Определяем имя бойца
        if fighter_prefix == 1:
            name_col = 'f_1_name'
        else:
            name_col = 'f_2_name'
        if name_col not in row or pd.isna(row[name_col]):
            return
        name = str(row[name_col]).strip().lower()
        if not name or name == 'nan':
            return

        # Собираем признаки для этого бойца из строки
        stats = {}
        # Добавляем возраст (колонки f_1_age / f_2_age)
        age_col = f'f_{fighter_prefix}_age'
        if age_col in row:
            stats['age'] = row[age_col]

        # Добавляем fight_number
        fn_col = f'f_{fighter_prefix}_fight_number'
        if fn_col in row:
            stats['fight_number'] = row[fn_col]

        # Добавляем скользящие средние за 5 боёв (колонки вида sapm_5_f_1 и т.д.)
        # Список интересующих нас базовых признаков (из diff_* в feature_order)
        diff_features = [f for f in feature_order if f.startswith('diff_')]
        base_features = [f.replace('diff_', '') for f in diff_features if f != 'diff_fight_number']
        # Добавляем также age и fight_number (они уже могли быть добавлены)
        for base in base_features:
            if base in ['age', 'fight_number']:
                continue  # уже обработаны
            # Ищем колонку вида {base}_5_f_{fighter_prefix}
            col_candidate = f'{base}_5_f_{fighter_prefix}'
            if col_candidate in row:
                stats[base] = row[col_candidate]
            else:
                # Возможно, колонка называется по-другому, но пока так
                pass

        # Добавляем weight_class
        if 'weight_class' in row:
            stats['weight_class'] = row['weight_class']

        # Добавляем дату боя (для сравнения свежести)
        stats['date'] = row['event_date']

        return name, stats

    # Перебираем все строки датасета (можно оптимизировать, но для 8k строк нормально)
    for idx, row in df.iterrows():
        # Обрабатываем первого бойца
        res = process_fighter(row, 1)
        if res:
            name, stats = res
            if name not in fighter_stats or fighter_stats[name]['date'] < stats['date']:
                fighter_stats[name] = stats

        # Обрабатываем второго бойца
        res = process_fighter(row, 2)
        if res:
            name, stats = res
            if name not in fighter_stats or fighter_stats[name]['date'] < stats['date']:
                fighter_stats[name] = stats

    logger.info(f"✅ Построен словарь для {len(fighter_stats)} уникальных бойцов")
    return fighter_stats

logger.info("Построение словаря последних статистик бойцов...")
fighter_last_stats = build_fighter_last_stats(df_full)

# ===================== ФУНКЦИЯ ПОДГОТОВКИ ПРИЗНАКОВ =====================
def get_fighter_stats(fighter_name: str) -> Dict[str, Any]:
    """
    Возвращает последние известные статистики бойца (словарь).
    Если боец не найден, возвращает пустой словарь (признаки будут нулевыми).
    """
    key = fighter_name.strip().lower()
    if key in fighter_last_stats:
        return fighter_last_stats[key]
    else:
        # Пытаемся найти по частичному совпадению (например, если в датасете имя записано иначе)
        for name, stats in fighter_last_stats.items():
            if key in name or name in key:
                logger.info(f"   Частичное совпадение: '{fighter_name}' -> '{name}'")
                return stats
        logger.warning(f"   Боец '{fighter_name}' не найден, используются нули")
        return {}

def prepare_features(fighter1_name: str, fighter2_name: str, event_date: datetime) -> pd.DataFrame:
    """
    Возвращает DataFrame с одной строкой, содержащей все признаки в порядке feature_order.
    Использует последние известные статистики бойцов из словаря.
    """
    stats1 = get_fighter_stats(fighter1_name)
    stats2 = get_fighter_stats(fighter2_name)

    # Создаём словарь для признаков
    data = {}

    # Вычисляем все diff_* признаки
    diff_features = [f for f in feature_order if f.startswith('diff_')]
    for diff_feat in diff_features:
        base = diff_feat.replace('diff_', '')
        val1 = stats1.get(base, 0.0)
        val2 = stats2.get(base, 0.0)
        data[diff_feat] = val1 - val2

    # Добавляем title_fight и num_rounds (пока заглушки)
    data['title_fight'] = 0
    data['num_rounds'] = 3  # можно уточнять по типу боя позже

    # Весовые категории (one-hot)
    weight_class1 = stats1.get('weight_class', '')
    weight_class2 = stats2.get('weight_class', '')
    # Берём первую непустую категорию (обычно они совпадают)
    weight = weight_class1 or weight_class2
    # Все возможные весовые категории из feature_order
    weight_cols = [f for f in feature_order if f.startswith('weight_')]
    for wcol in weight_cols:
        expected = wcol.replace('weight_', '')
        data[wcol] = 1.0 if weight == expected else 0.0

    # Создаём DataFrame и убеждаемся, что колонки идут в правильном порядке
    df = pd.DataFrame([data])
    df = df[feature_order]
    return df

# ===================== ФУНКЦИИ ДЛЯ МОДЕЛИ =====================
def get_calibrated_proba(model, X, calib):
    logits = model.predict(X, prediction_type='RawFormulaVal')
    if calib is None:
        return model.predict_proba(X)[:, 1]
    else:
        return 1 / (1 + np.exp(-(calib['a'] * logits)))

def predict_symmetrized(model, X, calib, diff_cols, f1_cols, f2_cols):
    """Симметризованная вероятность победы Fighter1."""
    p_orig = get_calibrated_proba(model, X, calib)
    X_swapped = X.copy()
    for f1, f2 in zip(f1_cols, f2_cols):
        tmp = X_swapped[f1].copy()
        X_swapped[f1] = X_swapped[f2]
        X_swapped[f2] = tmp
    for d in diff_cols:
        X_swapped[d] = -X_swapped[d]
    p_swapped = get_calibrated_proba(model, X_swapped, calib)
    return (p_orig + (1 - p_swapped)) / 2

# ===================== ПАРСИНГ UFCSTATS =====================
def scrape_upcoming_fights() -> pd.DataFrame:
    logger.info("Парсинг боёв с ufcstats.com...")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    try:
        resp = session.get("http://ufcstats.com/statistics/events/upcoming", timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Ошибка загрузки ивентов: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    rows = soup.select("tr.b-statistics__table-row")
    today = datetime.now(timezone.utc).date()
    events = []

    for row in rows[1:]:
        link = row.select_one("a.b-link_style_black")
        if not link:
            continue
        event_id = link["href"].split("/")[-1]
        event_name = link.get_text(strip=True)
        date_elem = row.select_one("span.b-statistics__date")
        if date_elem:
            try:
                event_date = datetime.strptime(date_elem.get_text(strip=True), "%B %d, %Y").replace(tzinfo=timezone.utc)
                if event_date.date() > today:
                    events.append((event_id, event_name, event_date))
            except:
                continue

    if not events:
        logger.warning("Нет предстоящих ивентов")
        return pd.DataFrame()

    events.sort(key=lambda x: x[2])
    event_id, event_name, event_date = events[0]
    logger.info(f"✅ Выбран ивент: {event_name} ({event_date.date()})")

    try:
        resp = session.get(f"http://ufcstats.com/event-details/{event_id}", timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Ошибка загрузки ивента: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    fights = []

    for row in soup.select("tr.b-fight-details__table-row")[1:]:
        fighters = row.select("a.b-link_style_black")
        if len(fighters) >= 2:
            f1 = fighters[0].get_text(strip=True)
            f2 = fighters[1].get_text(strip=True)
            fights.append({
                "fighter_1": f1,
                "fighter_2": f2,
                "event_name": event_name,
                "event_date": event_date.isoformat(),
                "f1_lastname": f1.split()[-1].lower(),
                "f2_lastname": f2.split()[-1].lower(),
                "event_date_obj": event_date
            })

    logger.info(f"✅ Спарсено {len(fights)} боёв")
    return pd.DataFrame(fights) if fights else pd.DataFrame()

# ===================== ЗАГРУЗКА КОЭФФИЦИЕНТОВ =====================
def fetch_real_odds(api_key: str) -> List[Dict]:
    logger.info("Загрузка коэффициентов из Odds API...")
    url = f"https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds?apiKey={api_key}&regions=us&markets=h2h&oddsFormat=decimal"
    try:
        resp = requests.get(url, timeout=15)
        logger.info(f"Статус ответа API: {resp.status_code}")
        if resp.status_code != 200:
            logger.error(f"Тело ответа: {resp.text[:200]}")
            resp.raise_for_status()
        events = resp.json()
        logger.info(f"✅ Загружено {len(events)} ивентов")
        return events
    except Exception as e:
        logger.error(f"Ошибка Odds API: {e}")
        return []

# ===================== СОПОСТАВЛЕНИЕ БОЁВ =====================
def match_fight_with_odds(fight: Dict, odds_events: List[Dict]) -> Tuple[Optional[float], Optional[float], str]:
    fight_date = fight["event_date_obj"].date()
    f1_lastname = fight["f1_lastname"]
    f2_lastname = fight["f2_lastname"]

    for event in odds_events:
        try:
            event_date = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00")).date()
            if abs((event_date - fight_date).days) > 1:
                continue
            for bookmaker in event.get("bookmakers", [])[:2]:
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = market.get("outcomes", [])
                    if len(outcomes) < 2:
                        continue
                    o1_name = outcomes[0]["name"].strip()
                    o2_name = outcomes[1]["name"].strip()
                    o1_lastname = o1_name.split()[-1].lower()
                    o2_lastname = o2_name.split()[-1].lower()
                    o1_odds = float(outcomes[0]["price"])
                    o2_odds = float(outcomes[1]["price"])

                    if ((f1_lastname in o1_lastname or o1_lastname in f1_lastname) and
                        (f2_lastname in o2_lastname or o2_lastname in f2_lastname)):
                        return o1_odds, o2_odds, f"{o1_name} vs {o2_name}"
                    if ((f1_lastname in o2_lastname or o2_lastname in f1_lastname) and
                        (f2_lastname in o1_lastname or o1_lastname in f2_lastname)):
                        return o2_odds, o1_odds, f"{o2_name} vs {o1_name}"
        except:
            continue
    return None, None, "не найдено"

# ===================== ФОРМАТИРОВАНИЕ СООБЩЕНИЯ =====================
def format_bet_message(fight: Dict, bet: Dict) -> str:
    event_name = fight["event_name"]
    event_date = fight["event_date"]
    f1 = fight["fighter_1"]
    f2 = fight["fighter_2"]
    odds1 = fight["odds1"]
    odds2 = fight["odds2"]
    prob_f1 = fight["model_prob_f1"]
    prob_f2 = fight["model_prob_f2"]

    # fair odds (без маржи)
    imp1 = 1 / odds1
    imp2 = 1 / odds2
    fair_prob1 = imp1 / (imp1 + imp2)
    fair_prob2 = 1 - fair_prob1

    ev1 = prob_f1 * odds1 - 1
    ev2 = prob_f2 * odds2 - 1

    # Определяем, на кого ставка
    if bet["position"] == "P1":
        value_fighter = f1
        ev_value = ev1
    else:
        value_fighter = f2
        ev_value = ev2

    # Форматируем дату как в примере (без Z, но можно оставить)
    try:
        dt = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
        dt_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except:
        dt_str = event_date

    return (
        f"🏆 {event_name}\n"
        f"📅 {dt_str}\n"
        f"🥊 {f1} ({prob_f1*100:.1f}%) @{odds1:.2f} - {f2} ({prob_f2*100:.1f}%) @{odds2:.2f}\n"
        f"📊 Вер. букмекера (без маржи): {f1} {fair_prob1*100:.1f}%, {f2} {fair_prob2*100:.1f}%\n"
        f"📊 EV: {f1} {ev1*100:+.1f}%, {f2} {ev2*100:+.1f}%\n"
        f"🎯 ✅ VALUE {value_fighter}\n"
        "----------------------------------------"
    )

# ===================== ОСНОВНАЯ ЛОГИКА =====================
def main():
    logger.info("="*70)
    logger.info("🚀 РАННЕР С МОДЕЛЬЮ И РЕАЛЬНЫМИ ПРИЗНАКАМИ")
    logger.info("="*70)

    # --- Жёстко выбираем лучшую стратегию medium_high_value ---
    STRATEGY_NAME = 'medium_high_value'
    if STRATEGY_NAME not in FILTER_CONFIGS:
        logger.error(f"❌ Стратегия '{STRATEGY_NAME}' не найдена в FILTER_CONFIGS!")
        sys.exit(1)
    cfg = FILTER_CONFIGS[STRATEGY_NAME]
    logger.info(f"✅ Используется стратегия: {STRATEGY_NAME} - {cfg['name']}")
    logger.info(f"   Параметры: min_odds={cfg['min_odds']}, max_odds={cfg['max_odds']}, "
                f"min_model_prob={cfg['min_model_prob']}, min_edge={cfg['min_edge']}, "
                f"min_ev={cfg['min_ev']}, bet_size={cfg['bet_size_pct']}%")

    # 1. Парсинг боёв
    upcoming_df = scrape_upcoming_fights()
    if upcoming_df.empty:
        logger.warning("Нет боёв для анализа")
        sys.exit(0)

    # 2. Загрузка коэффициентов
    odds_events = fetch_real_odds(API_KEY)
    if not odds_events:
        logger.error("❌ Не удалось загрузить коэффициенты")
        sys.exit(1)

    # 3. Сопоставление боёв с коэффициентами
    fights_with_odds = []
    logger.info("\n🔍 Сопоставление боёв...")
    for _, fight in upcoming_df.iterrows():
        odds1, odds2, source = match_fight_with_odds(fight.to_dict(), odds_events)
        if odds1 is None or odds2 is None:
            logger.warning(f"⚠️ Кф не найдены: {fight['fighter_1']} vs {fight['fighter_2']}")
            continue
        # Отсев подозрительных дефолтных 3.20
        if abs(odds1 - 3.20) < 0.01 or abs(odds2 - 3.20) < 0.01:
            logger.warning(f"⚠️ Подозрительные кф (3.20) — пропускаем")
            continue
        fight_dict = fight.to_dict() | {"odds1": odds1, "odds2": odds2, "odds_source": source}
        fights_with_odds.append(fight_dict)
        logger.info(f"✅ {fight['fighter_1']} @{odds1:.2f} vs {fight['fighter_2']} @{odds2:.2f}")

    if not fights_with_odds:
        logger.error("❌ Нет боёв с валидными кф")
        sys.exit(1)

    # Определяем diff_cols, f1_cols, f2_cols из feature_order
    diff_cols = [col for col in feature_order if col.startswith('diff_')]
    f1_cols = [col for col in feature_order if col.startswith('f_1_')]
    f2_cols = [col for col in feature_order if col.startswith('f_2_')]

    # 4. Расчёт вероятностей моделью
    logger.info("\n🔮 Расчёт вероятностей модели (симметризованные)...")
    for fight in fights_with_odds:
        X = prepare_features(fight["fighter_1"], fight["fighter_2"], fight["event_date_obj"])
        prob_f1 = predict_symmetrized(model, X, calib, diff_cols, f1_cols, f2_cols)[0]
        prob_f2 = 1 - prob_f1
        fight["model_prob_f1"] = prob_f1
        fight["model_prob_f2"] = prob_f2
        logger.info(f"   {fight['fighter_1']} vs {fight['fighter_2']} : prob_f1 = {prob_f1:.3f}")

    # 5. Фильтрация value-ставок по выбранной стратегии
    value_bets = []
    for fight in fights_with_odds:
        odds1 = fight["odds1"]
        odds2 = fight["odds2"]
        prob_f1 = fight["model_prob_f1"]
        prob_f2 = fight["model_prob_f2"]

        ev1 = prob_f1 * odds1 - 1
        ev2 = prob_f2 * odds2 - 1
        edge1 = prob_f1 - 1/odds1
        edge2 = prob_f2 - 1/odds2

        # Условия для P1
        if (cfg["min_odds"] <= odds1 <= cfg["max_odds"] and
            prob_f1 >= cfg["min_model_prob"] and
            edge1 >= cfg["min_edge"] and
            ev1 >= cfg["min_ev"]):
            value_bets.append((fight, {
                "fighter": fight["fighter_1"],
                "position": "P1",
                "odds": odds1,
                "model_prob": prob_f1,
                "ev": ev1,
                "bet_size_pct": cfg["bet_size_pct"]
            }))

        # Условия для P2
        if (cfg["min_odds"] <= odds2 <= cfg["max_odds"] and
            prob_f2 >= cfg["min_model_prob"] and
            edge2 >= cfg["min_edge"] and
            ev2 >= cfg["min_ev"]):
            value_bets.append((fight, {
                "fighter": fight["fighter_2"],
                "position": "P2",
                "odds": odds2,
                "model_prob": prob_f2,
                "ev": ev2,
                "bet_size_pct": cfg["bet_size_pct"]
            }))

    # 6. Сохранение результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = UPCOMING_DIR / f"value_bets_{timestamp}.txt"

    if value_bets:
        with open(txt_path, "w", encoding="utf-8") as f:
            for fight, bet in value_bets:
                line = format_bet_message(fight, bet)
                f.write(line + "\n")
        logger.info(f"\n✅ Найдено {len(value_bets)} велью-ставок → {txt_path}")
        # Дублируем в консоль
        for fight, bet in value_bets:
            print(format_bet_message(fight, bet))
    else:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Велью-ставок не найдено\n")
        logger.info("\n⚠️ Велью-ставок не найдено")

    logger.info("="*70)
    logger.info(f"📊 Итоги: {len(upcoming_df)} боёв → {len(fights_with_odds)} с кф → {len(value_bets)} ставок")
    logger.info("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Необработанная ошибка: {e}")
        sys.exit(1)