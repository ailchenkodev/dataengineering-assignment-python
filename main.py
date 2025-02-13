import pandas as pd
from chunker import chunk_dataframe
import pytest
import sys


def run_chunker_example():

    dfs = pd.date_range("2023-01-01 00:00:00", "2023-01-01 00:00:05", freq="s")
    df = pd.DataFrame({"dt": dfs.repeat(3)})
    df.head(10)

    # print(df[::]["dt"])

    # df = pd.DataFrame(
    #     {
    #         "dt": [
    #             "2023-01-01 00:00:02",
    #             "2023-01-01 00:00:02",
    #             "2023-01-01 00:00:02",
    #             "2023-01-01 00:00:01",
    #             "2023-01-01 00:00:01",
    #             "2023-01-01 00:00:03",
    #         ]
    #     }
    #  )

    for i in chunk_dataframe(df, 4, 'dt', True):
        print(i[::]["dt"])

def run_tests():
    """Запуск всех тестов с помощью pytest."""
    exit_code = pytest.main(["-v", "tests"])
    sys.exit(exit_code)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Запуск примера или тестов.")
    parser.add_argument('--test', action='store_true', help="Запуск юнит-тестов")

    args = parser.parse_args()

    if args.test:
        run_tests()
    else:
        run_chunker_example()

