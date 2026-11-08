# tests/test_make_folds.py
from src.make_folds import folder_to_label


def test_folder_names_map_to_labels():
    # Check the folder-name to label rule used when the data index is built
    cases = [
        ("ECG Images of Myocardial Infarction Patients (240x12=2880)", "MI"),
        (
            "ECG Images of Patient that have abnormal heartbeat (233x12=2796)",
            "Abnormal",
        ),
        (
            "ECG Images of Patient that have History of MI (172x12=2064)",
            "History_of_MI",
        ),
        ("Normal Person ECG Images (284x12=3408)", "Normal"),
    ]
    for folder_name, expected in cases:
        assert folder_to_label(folder_name) == expected


def test_abnormal_is_not_mistaken_for_normal():
    # The word "abnormal" contains "normal", so the order of the checks matters
    assert folder_to_label("Patient with abnormal heartbeat") == "Abnormal"


def test_unknown_folder_is_ignored():
    # Folders like "cache" must not become a class
    assert folder_to_label("cache") is None
