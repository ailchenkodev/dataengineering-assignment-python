import pandas as pd
from typing import Generator


def chunk_dataframe(
        df: pd.DataFrame,
        chunk_size: int,
        dt_column: str = 'dt',
        sort: bool = False
) -> Generator[pd.DataFrame, None, None]:
    """
    Разбивает DataFrame на чанки по группам дат с целостным переносом строк,
    принадлежащих одной и той же дате. Чанк формируется накоплением групп до тех
    пор, пока общее число строк не достигнет или не превысит chunk_size. Если в конце
    остаются группы, не дотягивающие до chunk_size, они возвращаются как последний чанк.

    Args:
        df: Исходный DataFrame.
        chunk_size: Желаемый минимальный размер чанка (в строках).
        dt_column: Название столбца с датой/временем.
        sort: Если True, внутри каждого чанка выполняется сортировка по dt_column.

    Returns:
        Генератор, выдающий чанки DataFrame.
    """
    if df.empty:
        return
    if len(df) <= chunk_size:
        # Даже если фрейм возвращается целиком, сортируем его, если требуется.
        yield df.sort_values(by=dt_column) if sort else df
        return

    # Определяем, является ли столбец датой, и подготавливаем значения для группировки.
    is_dt = pd.api.types.is_datetime64_any_dtype(df[dt_column])
    if is_dt:
        # Преобразуем даты в строки с точностью до секунд для корректной группировки.
        grouping_values = df[dt_column].dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        grouping_values = df[dt_column]

    # Группируем DataFrame по значениям столбца.
    gb = df.groupby(grouping_values, sort=sort)
    date_sizes = gb.size()  # Series: индекс – уникальное значение, значение – число строк.
    groups = gb.groups  # Словарь: ключ – уникальное значение, значение – индексы строк.

    # Формируем границы чанков: список списков ключей (дат),
    # где каждый список соответствует одному чанку.
    chunk_boundaries = []
    current_chunk_dates = []
    current_size = 0

    for date, size in date_sizes.items():
        current_chunk_dates.append(date)
        current_size += size
        # Как только накопленный размер достигает или превышает chunk_size, фиксируем чанк.
        if current_size >= chunk_size:
            chunk_boundaries.append(current_chunk_dates)
            current_chunk_dates = []
            current_size = 0

    # Если после цикла остались группы, возвращаем их как последний чанк.
    if current_chunk_dates:
        chunk_boundaries.append(current_chunk_dates)

    # Функция для формирования чанка DataFrame по списку ключей.
    def get_chunk(dates):
        indexes = []
        for d in dates:
            indexes.extend(groups[d])
        chunk = df.loc[indexes]
        # Если требуется, сортируем по dt_column.
        if sort:
            chunk = chunk.sort_values(by=dt_column)
        return chunk

    for dates in chunk_boundaries:
        yield get_chunk(dates)