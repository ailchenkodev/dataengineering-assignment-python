import pytest
import pandas as pd
from chunker import chunk_dataframe
import sys


@pytest.fixture
def sample_df():
    dates = pd.date_range("2023-01-01 00:00:01", "2023-01-01 00:00:03", freq="s")
    return pd.DataFrame({"dt": dates.repeat([2, 3, 1])})


def test_empty_dataframe():
    empty_df = pd.DataFrame(columns=['dt'])
    chunks = list(chunk_dataframe(empty_df, chunk_size=2))
    assert len(chunks) == 0, "Пустой DataFrame должен возвращать 0 чанков"


def test_small_chunk_size(sample_df):
    # При chunk_size=2 ожидаем, что каждая группа будет своим чанкем, т.к. каждая группа по датам уже >=1
    chunks = list(chunk_dataframe(sample_df, chunk_size=2))
    assert len(chunks) == 3, "Ожидается 3 чанка при chunk_size=2"

    # Проверяем, что каждая группа (уникальная дата) полностью входит в один чанк.
    dates_in_chunks = [set(chunk['dt'].unique()) for chunk in chunks]
    for i in range(len(dates_in_chunks) - 1):
        for j in range(i + 1, len(dates_in_chunks)):
            overlap = dates_in_chunks[i] & dates_in_chunks[j]
            assert len(overlap) == 0, f"Даты {overlap} встречаются в более чем одном чанке"


def test_medium_chunk_size(sample_df):
    # При chunk_size=3 ожидается, что группы 00:00:01 (2 строки) и 00:00:02 (3 строки) объединятся в один чанк (итого 5 строк),
    # а оставшаяся группа (00:00:03) станет вторым чанком.
    chunks = list(chunk_dataframe(sample_df, chunk_size=3))
    assert len(chunks) == 2, "Ожидается 2 чанка при chunk_size=3"

    # Проверка содержимого чанков по количеству строк
    assert len(chunks[0]) >= 3, "Первый чанк должен содержать не менее 3 строк"
    assert len(chunks[1]) >= 1, "Второй чанк должен содержать хотя бы 1 строку"

    # Дополнительно: проверим, что в первом чанке присутствуют две уникальные даты, а во втором – одна
    unique_dates_first = chunks[0]['dt'].unique()
    unique_dates_second = chunks[1]['dt'].unique()
    assert len(unique_dates_first) == 2, "Первый чанк должен содержать 2 уникальные даты"
    assert len(unique_dates_second) == 1, "Второй чанк должен содержать 1 уникальную дату"


def test_large_chunk_size(sample_df):
    # При chunk_size=6 (или больше) ожидается, что весь DataFrame возвращается одним чанкем.
    chunks = list(chunk_dataframe(sample_df, chunk_size=6))
    assert len(chunks) == 1, "Ожидается 1 чанк при chunk_size>=6"
    assert len(chunks[0]) == len(sample_df), "Чанк должен содержать все строки исходного DataFrame"


def test_generator_behavior(sample_df):
    # Проверяем, что функция возвращает генератор
    chunks = chunk_dataframe(sample_df, chunk_size=2)
    assert hasattr(chunks, '__iter__'), "Результат должен быть итератором"
    assert hasattr(chunks, '__next__'), "Результат должен поддерживать метод __next__"

    # Последовательное получение чанков
    chunk1 = next(chunks)
    assert isinstance(chunk1, pd.DataFrame), "Первый элемент генератора должен быть DataFrame"
    remaining_chunks = list(chunks)
    # Исходя из sample_df и chunk_size=2, всего должно быть 3 чанка
    assert len(remaining_chunks) == 2, "После первого next() должно остаться 2 чанка"


def test_memory_usage(sample_df):
    # Тест, проверяющий, что размер генератора намного меньше размера данных.
    big_df = pd.concat([sample_df] * 1000)
    chunks_gen = chunk_dataframe(big_df, chunk_size=100)
    gen_size = sys.getsizeof(chunks_gen)
    data_size = big_df.memory_usage(deep=True).sum()
    assert gen_size < data_size / 100, "Размер генератора должен быть менее 1% от размера исходных данных"


@pytest.mark.parametrize("chunk_size,expected_chunks", [
    (2, 3),  # Маленький размер чанка
    (3, 2),  # Средний размер чанка
    (6, 1),  # Большой размер чанка
])
def test_parametrized_chunk_sizes(sample_df, chunk_size, expected_chunks):
    chunks = list(chunk_dataframe(sample_df, chunk_size=chunk_size))
    assert len(chunks) == expected_chunks, f"При chunk_size={chunk_size} ожидается {expected_chunks} чанка"


def test_sorting_behavior(sample_df):
    # Тест проверяет, что при sort=True внутри каждого чанка строки отсортированы по dt.
    chunks = list(chunk_dataframe(sample_df, chunk_size=6, sort=True))
    # Если весь DataFrame возвращается одним чанкем, проверим его сортировку.
    for chunk in chunks:
        sorted_chunk = chunk.sort_values(by='dt')
        pd.testing.assert_frame_equal(
            chunk.reset_index(drop=True),
            sorted_chunk.reset_index(drop=True),
            check_like=True
        )


def test_non_datetime_column():
    # Тест для случая, когда столбец dt имеет строковый тип
    df = pd.DataFrame({
        "dt": ["2023-01-01 00:00:01", "2023-01-01 00:00:01",
               "2023-01-01 00:00:02", "2023-01-01 00:00:02",
               "2023-01-01 00:00:02", "2023-01-01 00:00:03"]
    })
    chunks = list(chunk_dataframe(df, chunk_size=3))
    # Ожидаем 2 чанка, аналогично поведению для datetime
    assert len(chunks) == 2, "Ожидается 2 чанка для строковых дат при chunk_size=3"
    # Проверяем, что группировка по строковому типу отработала корректно
    dates_in_first = set(chunks[0]['dt'].unique())
    dates_in_second = set(chunks[1]['dt'].unique())
    assert len(dates_in_first.intersection(dates_in_second)) == 0, "Группы не должны пересекаться"


def test_data_integrity(sample_df):
    # Проверяет, что объединение всех чанков (без изменения порядка) дает исходный DataFrame
    chunks = list(chunk_dataframe(sample_df, chunk_size=2))
    concatenated = pd.concat(chunks).sort_index()
    # Сортировка по индексу для сравнения с исходным
    pd.testing.assert_frame_equal(concatenated.reset_index(drop=True), sample_df.reset_index(drop=True))


