# 📌 Chunker DataFrame

Этот проект реализует функцию `chunk_dataframe` для разбиения `pandas.DataFrame` на части фиксированного размера.

## 📦 Установка

1. **Клонируйте репозиторий**:
```bash
git clone https://github.com/ailchenkodev/dataengineering-assignment-python.git
cd dataengineering-assignment-python
```
2. **Создайте и активируйте виртуальное окружение (рекомендуется)**:
```bash 
python -m venv venv
source venv/bin/activate  # Для macOS/Linux
venv\Scripts\activate     # Для Windows
```
3. **Установите зависимости**:

```bash
pip install -r requirements.txt
```
🚀 **Использование**
Пример разбиения pandas.DataFrame на чанки по 3 строки:
```python
import pandas as pd
from chunker.chunker import chunk_dataframe

    df = pd.DataFrame(
        {
            "dt": [
                "2023-01-01 00:00:02",
                "2023-01-01 00:00:02",
                "2023-01-01 00:00:02",
                "2023-01-01 00:00:01",
                "2023-01-01 00:00:01",
                "2023-01-01 00:00:03",
            ]
        }
     )
    
    for i in chunk_dataframe(df, 2, 'dt', True):
        print(i[::]["dt"])
```
🔹 **Выходные данные**:

Для чанка размером между 1 и 2  - результат 3 чанка:
```
2023-01-01 00:00:01
2023-01-01 00:00:01

2023-01-01 00:00:02
2023-01-01 00:00:02
2023-01-01 00:00:02

2023-01-01 00:00:03
```

Для чанка размером между 3 и 5 - результат 2 чанка:
```
2023-01-01 00:00:01
2023-01-01 00:00:01
2023-01-01 00:00:02
2023-01-01 00:00:02
2023-01-01 00:00:02

2023-01-01 00:00:03
```

Для чанка размером от 6 и выше - результат весь фрейм:
```
2023-01-01 00:00:01
2023-01-01 00:00:01
2023-01-01 00:00:02
2023-01-01 00:00:02
2023-01-01 00:00:02
2023-01-01 00:00:03
```

🛠 **Тестирование**

Проект использует pytest для тестирования.

Запуск тестов:
```bash
pytest tests/
```

При успешном прохождении тестов вы увидите:

![Pytest](https://img.shields.io/badge/tests-passing-brightgreen)
```
============================= test session starts =============================
collected 12 items

tests/test_chunker.py::test_empty_dataframe PASSED                                                                                                                           [  8%] 
tests/test_chunker.py::test_small_chunk_size PASSED                                                                                                                          [ 16%] 
tests/test_chunker.py::test_medium_chunk_size PASSED                                                                                                                         [ 25%]
tests/test_chunker.py::test_large_chunk_size PASSED                                                                                                                          [ 33%] 
tests/test_chunker.py::test_generator_behavior PASSED                                                                                                                        [ 41%] 
tests/test_chunker.py::test_memory_usage PASSED                                                                                                                              [ 50%]
tests/test_chunker.py::test_parametrized_chunk_sizes[2-3] PASSED                                                                                                             [ 58%] 
tests/test_chunker.py::test_parametrized_chunk_sizes[3-2] PASSED                                                                                                             [ 66%] 
tests/test_chunker.py::test_parametrized_chunk_sizes[6-1] PASSED                                                                                                             [ 75%]
tests/test_chunker.py::test_sorting_behavior PASSED                                                                                                                          [ 83%] 
tests/test_chunker.py::test_non_datetime_column PASSED                                                                                                                       [ 91%] 
tests/test_chunker.py::test_data_integrity PASSED                                                                                                                            [100%]

=============================================================================== 12 passed in 0.07s ================================================================================ 
(                                                [100%]

============================== 2 passed in 0.12s ==============================
```
🏗 **Структура проекта**

```
project_root/
│── chunker/                  # Основной модуль с логикой чанкинга
│   ├── __init__.py
│   ├── chunker.py             # Реализация функции chunk_dataframe
│── tests/                     # Юнит-тесты
│   ├── __init__.py
│   ├── test_chunker.py        # Тестирование chunk_dataframe
│── requirements.txt           # Список зависимостей
│── pyproject.toml             # Конфигурация проекта
│── main.py                    # Тестовый запуск кода
│── README.md                  # Документация проекта
```
📜 **Лицензия**

Этот проект распространяется по лицензии MIT.
Автор: ailchenkodev

📌 **Ссылки**:

[Документация Pandas](https://pandas.pydata.org/docs/)
![Логотип](https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg)

---

### ✅ **Что включено в `README.md`:**
- 🔹 **Название проекта**  
- 📦 **Как установить**  
- 🚀 **Как использовать** (пример кода + вывод)  
- 🛠 **Как запустить тесты**  
- 🏗 **Структура проекта**  
- 📜 **Лицензия**  
- 📌 **Ссылки**
