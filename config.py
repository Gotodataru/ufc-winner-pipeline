import os
from pathlib import Path

# Автоматически определяем корень проекта независимо от ОС (Windows/Mac/Linux)
BASE_DIR = Path(__file__).resolve().parent

# Пути к данным и моделям (теперь относительные)
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Создаем папки, если их нет (чтобы код не падал при первом запуске)
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)