import numpy as np
import os
import pandas as pd
import pickle
import sys
import xgboost as xgb
from dmatrix2np import dmatrix_to_numpy
from itertools import repeat
from multiprocessing import Pool, cpu_count, freeze_support
from scipy.sparse import coo_matrix
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    f1_score,
    confusion_matrix
)

from model_classifyAppRisk import process_files
import argparse

def predict(DATA_FOLDER,MAPPING_FILE,MODEL):

    DATA_FOLDER = DATA_FOLDER
    MAPPING_FILE = MAPPING_FILE
    MODEL = MODEL

    print("Predicting...")
    with open(
            MODEL,
        "rb"
    ) as f:
        dict = pickle.load(f)


    filepaths = [
            os.path.join(DATA_FOLDER, f)
            for f in os.listdir(DATA_FOLDER)
            if f.endswith(".txt")
        ]


    feature_mapping = pd.read_csv(MAPPING_FILE)

    valid_features = set(
            feature_mapping["feature_number"].values
        )
    max_feature = max(valid_features)

    type_lookup = np.empty(
        max_feature + 1,
        dtype=object
    )

    subtype_lookup = np.empty(
        max_feature + 1,
        dtype=object
    )
    feature_numbers = sorted(valid_features)
    feature_to_col = {
        feature_num: idx
        for idx, feature_num in enumerate(feature_numbers)
    }
    NUM_FEATURES = len(feature_to_col)
    feature_lookup = (
        feature_mapping
        .set_index("feature_number")
        [["feature_type", "feature_subtype"]]
        .to_dict("index")
    )
    for feature_num, info in feature_lookup.items():
        type_lookup[feature_num] = info["feature_type"]
        subtype_lookup[feature_num] = info["feature_subtype"]

    with (Pool(cpu_count()) as pool):
        results = pool.starmap(process_files, zip(filepaths, repeat(feature_to_col),
                                                  repeat(valid_features), repeat(type_lookup), repeat(subtype_lookup)))


    print("Building sparse matrix...")

    all_sparse_data = []
    all_sparse_rows = []
    all_sparse_cols = []
    all_scores = []
    all_targets = []
    all_engineered = []

    all_sparse_data = np.concatenate(
        [np.asarray(r[0], dtype=np.float32) for r in results]
    )

    all_sparse_rows = np.concatenate(
        [np.asarray(r[1], dtype=np.int32) for r in results]
    )

    all_sparse_cols = np.concatenate(
        [np.asarray(r[2], dtype=np.int32) for r in results]
    )

    all_score = np.concatenate(
        [np.asarray(r[5], dtype=np.float32) for r in results]
    )

    for r in results:
        all_engineered.extend(r[3])

    all_targets = np.concatenate(
        [np.asarray(r[4], dtype=np.int32) for r in results]
    )

    row_counts = np.array(
        [r[6] for r in results],
        dtype=np.int64
    )

    num_rows = int(row_counts.sum())

    X_sparse = coo_matrix(

        (
            all_sparse_data,

            (
                all_sparse_rows,
                all_sparse_cols
            )
        ),

        shape=(
            num_rows,
            NUM_FEATURES
        ),

        dtype=np.float32

    ).tocsr()

    base_feature_names = (
        feature_mapping
        .sort_values("feature_number")
        ["feature_name"]
        .tolist()
    )


    model = dict["model"]
    threshold = dict["threshold"]

    training_feature_names = dict["feature_names"]
    saved_engineered_columns = dict["engineered_columns"]

    engineered_df = pd.DataFrame(
        all_engineered
    )


    engineered_df = engineered_df.reindex(
        columns=saved_engineered_columns,
        fill_value=0
    )

    engineered_feature_names = (
        engineered_df.columns.tolist()
    )

    engineered_sparse = coo_matrix(
        engineered_df.fillna(0).astype(np.float32).values
    )

    X_final = hstack([
        X_sparse,
        engineered_sparse
    ]).tocsr()

    y = np.array(
        all_targets
    )

    print("Dataset Shape:", X_final.shape)

    class_weights = {0: 2, 1: 1}
    sample_weights = np.array([class_weights[class_id] for class_id in y])

    feature_names = (
            base_feature_names +
            engineered_feature_names
    )
    dtest = xgb.DMatrix(
        X_final,
        label=y,
        feature_names=feature_names,
        weight=sample_weights
    )
    pred_scores = model.predict(dtest)
    predictions = (pred_scores >= threshold).astype(int)
    print(
        "Accuracy:",
        accuracy_score(y, predictions)

    )
    print(
        "Precision:",
        precision_score(y, predictions))

    print(
        "Recall:",
        recall_score(y, predictions))

    cm = confusion_matrix(y, predictions)
    print(cm)
    print("LOW_RISK:", np.sum(y == 0))
    print("HIGH_RISK:", np.sum(y == 1))

    for i, p in enumerate(predictions):
        label = (
            "HIGH RISK"
            if p >= threshold
            else "LOW RISK"
        )

        print(
            f"Application {i}: {label} "
            f"(label={p:.4f})"
        )
def create_arg_parser():
    # Creates and returns the ArgumentParser object

    parser = argparse.ArgumentParser(description='Description of your app.')
    parser.add_argument('testInputDirectory',
                    help='Path to the raw test input directory.')
    parser.add_argument('mappingFile',
                    help='Path to the mapping file.')
    parser.add_argument('modelLoadDir',
                        help='Path to the load tuned Model file.')
    return parser

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("too few arguments")
        sys.exit(1)

    freeze_support()
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    # Check if the file exists
    if not os.path.exists(os.path.join(parsed_args.mappingFile)):
        print(f"Error: File '{os.path.join(parsed_args.mappingFile)}' not found.")
        sys.exit(1)
    predict(os.path.join(parsed_args.testInputDirectory),os.path.join(parsed_args.mappingFile),os.path.join(parsed_args.modelLoadDir))