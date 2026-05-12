from pathlib import Path

BASE = Path(__file__).parent.parent
DATA = BASE / "datasets"
RESULTS = BASE / "results"
FIGS = RESULTS / "figures"
TABLES = RESULTS / "tables"

DATASETS = {
    "NSL-KDD": {
        "path": DATA / "NSLKDD" / "KDD.csv",
        "label_col": "class",
        "benign_labels": [0, "0"],
        "drop_cols": ["id"],
        "type": "kdd",
    },
    "CICIDS-2017": {
        "path": DATA / "CICIDS2017" / "TrafficLabelling_",
        "label_col": "Label",
        "benign_labels": ["BENIGN"],
        "drop_cols": ["Flow ID", "Source IP", "Source Port", "Destination IP",
                      "Destination Port", "Timestamp"],
        "type": "cicids17",
        "sample_n": 250_000,
    },
    "CICIDS-2018": {
        "path": DATA / "CICIDS2018" / "datasetcsv.csv",
        "label_col": "class",
        "benign_labels": [0, "0"],
        "drop_cols": ["Timestamp"],
        "type": "cicids18",
        "sample_n": 150_000,
    },
    "UNSW-NB15": {
        "path": DATA / "UNSW-NB15" / "UNSW-NB15.csv",
        "label_col": "label",
        "benign_labels": [0, "0"],
        "drop_cols": [],
        "type": "unsw",
        "nrows": 300_000,
        "sample_n": 200_000,
    },
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = -1

ABLATION_SETS = ["conventional", "entropy_only", "combined"]
MODELS = ["LightGBM", "RandomForest"]
