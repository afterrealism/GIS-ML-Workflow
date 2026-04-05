import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from mlwkf.utlities import (
    get_csv_columns,
    read_dataframe_from_csv,
    flatten,
    create_chunked_target,
    get_formated_dataframe,
)


class TestGetCsvColumns:

    def test_returns_column_names(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("col_a,col_b,col_c\n1,2,3\n4,5,6\n")
        result = get_csv_columns(str(csv_path))
        assert result == ["col_a", "col_b", "col_c"]

    def test_single_column(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("only_col\n1\n2\n")
        result = get_csv_columns(str(csv_path))
        assert result == ["only_col"]


class TestReadDataframeFromCsv:

    def test_reads_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b\n1.0,2.0\n3.0,4.0\n")
        df = read_dataframe_from_csv(str(csv_path))
        assert len(df) == 2
        assert df.dtypes["a"] == np.float32

    def test_drops_nan_rows(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b\n1.0,2.0\n,4.0\n5.0,6.0\n")
        df = read_dataframe_from_csv(str(csv_path))
        assert len(df) == 2

    def test_drops_inf_rows(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [4.0, 5.0, 6.0]})
        df.to_csv(csv_path, index=False)
        result = read_dataframe_from_csv(str(csv_path))
        assert len(result) == 2

    def test_drops_negative_9999_rows(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1.0, -9999.0, 3.0], "b": [4.0, 5.0, 6.0]})
        df.to_csv(csv_path, index=False)
        result = read_dataframe_from_csv(str(csv_path))
        assert len(result) == 2

    def test_resets_index(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1.0,2.0\n,4.0\n5.0,6.0\n")
        df = read_dataframe_from_csv(str(csv_path))
        assert list(df.index) == list(range(len(df)))


class TestFlatten:

    def test_flat_list(self):
        assert flatten([1, 2, 3]) == [1, 2, 3]

    def test_nested_list(self):
        assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_deeply_nested(self):
        assert flatten([[1, [2, 3]], [4]]) == [1, 2, 3, 4]

    def test_single_element(self):
        assert flatten([1]) == [1]

    def test_scalar(self):
        assert flatten(5) == [5]


class TestCreateChunkedTarget:

    def test_even_chunks(self):
        result = create_chunked_target([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]

    def test_uneven_chunks(self):
        result = create_chunked_target([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_chunk_size_larger_than_list(self):
        result = create_chunked_target([1, 2], 10)
        assert result == [[1, 2]]

    def test_empty_list(self):
        result = create_chunked_target([], 5)
        assert result == []


class TestGetFormatedDataframe:

    def test_casts_to_float32(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = get_formated_dataframe(df)
        assert result.dtypes["a"] == np.float32

    def test_removes_nan(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        result = get_formated_dataframe(df)
        assert len(result) == 2

    def test_removes_inf(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [4.0, 5.0, 6.0]})
        result = get_formated_dataframe(df)
        assert len(result) == 2

    def test_index_reset(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        result = get_formated_dataframe(df)
        assert list(result.index) == list(range(len(result)))
