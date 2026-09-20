import pytest

from covertype.data import (
    CLASS_NAMES,
    DataError,
    continuous_only,
    invert_one_hot,
    parse_csv,
)

CSV = (
    "Elevation,Slope,Wilderness_Area1,Wilderness_Area2,Soil_Type1,Soil_Type2,Cover_Type\n"
    "2596,3,1,0,1,0,5\n"
    "2590,2,0,1,0,1,5\n"
    "2804,9,1,0,0,1,2\n"
)


def test_features_and_labels_are_separated():
    dataset = parse_csv(CSV)
    assert len(dataset) == 3
    assert "Cover_Type" not in dataset.columns
    assert dataset.y == (5, 5, 2)


def test_class_counts_include_every_declared_class():
    counts = parse_csv(CSV).class_counts()
    assert set(counts) == set(CLASS_NAMES)
    assert counts[5] == 2 and counts[1] == 0


def test_imbalance_ratio_ignores_absent_classes():
    assert parse_csv(CSV).imbalance_ratio() == 2.0


def test_a_missing_target_column_is_refused():
    with pytest.raises(DataError, match="Cover_Type"):
        parse_csv("A,B\n1,2\n")


def test_an_unparseable_row_is_refused():
    with pytest.raises(DataError, match="unparseable"):
        parse_csv("Elevation,Cover_Type\nhigh,5\n")


def test_a_file_with_no_rows_is_refused():
    with pytest.raises(DataError, match="no usable rows"):
        parse_csv("Elevation,Cover_Type\n")


def test_one_hot_inversion_produces_a_code_per_row():
    assert invert_one_hot(parse_csv(CSV), "Wilderness_Area") == [1, 2, 1]


def test_inverting_a_block_that_is_not_one_hot_is_refused():
    # Two active indicators means collapsing would invent a category.
    broken = "Wilderness_Area1,Wilderness_Area2,Cover_Type\n1,1,5\n"
    with pytest.raises(DataError, match="exactly one active indicator"):
        invert_one_hot(parse_csv(broken), "Wilderness_Area")


def test_inverting_an_all_zero_block_is_refused():
    broken = "Wilderness_Area1,Wilderness_Area2,Cover_Type\n0,0,5\n"
    with pytest.raises(DataError, match="exactly one active indicator"):
        invert_one_hot(parse_csv(broken), "Wilderness_Area")


def test_inverting_an_unknown_prefix_is_refused():
    with pytest.raises(DataError, match="no columns starting"):
        invert_one_hot(parse_csv(CSV), "Aspect")


def test_continuous_only_keeps_the_measurements():
    dataset = continuous_only(parse_csv(CSV))
    assert dataset.columns == ("Elevation", "Slope")


def test_continuous_only_on_a_frame_without_them_is_refused():
    with pytest.raises(DataError, match="no continuous columns"):
        continuous_only(parse_csv("Soil_Type1,Cover_Type\n1,5\n"))


def test_subset_keeps_rows_and_labels_aligned():
    dataset = parse_csv(CSV).subset([2, 0])
    assert dataset.y == (2, 5)
    assert dataset.x[0][0] == 2804.0


def test_select_reorders_and_narrows_columns():
    dataset = parse_csv(CSV).select(["Slope", "Elevation"])
    assert dataset.columns == ("Slope", "Elevation")
    assert dataset.x[0] == (3.0, 2596.0)
