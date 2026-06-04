from scipy.sparse import hstack
import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from dmatrix2np import dmatrix_to_numpy
from multiprocessing import Pool, cpu_count, freeze_support
from itertools import repeat
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from array import *
from collections import Counter
from hyperopt import (
    STATUS_OK,
    Trials,
    fmin,
    hp,
    tpe
)
from sklearn.metrics import (
    fbeta_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    f1_score,
    balanced_accuracy_score
)
import sys

def objective(space):

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_delta_step=int(space['max_delta_step']),
        scale_pos_weight=scale_pos_weight,
        n_estimators=int(space['n_estimators']),
        max_depth=int(space['max_depth']),
        learning_rate=space['learning_rate'],
        gamma=space['gamma'],
        min_child_weight=space['min_child_weight'],
        colsample_bytree=space['colsample_bytree'],
        subsample=space['subsample'],
        eval_metric='logloss',
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )
    sample_weights = np.where(
        GLOBAL_Y_TRAIN == 0,
        2,
        1
    )
    model.fit(
        GLOBAL_X_TRAIN,
        GLOBAL_Y_TRAIN,
        eval_set=[
            (GLOBAL_X_VALIDATE,
             GLOBAL_Y_VALIDATE)
        ],
        verbose=False,
        sample_weight=sample_weights
    )

    pred = model.predict(
        GLOBAL_X_VALIDATE
    )

    return {
        'loss': -fbeta_score(
            GLOBAL_Y_VALIDATE,
            pred,beta=2
        ),
        'status': STATUS_OK
    }

def evaluate_model(y_test, predictions):

    print("\n======================")
    print("MODEL PERFORMANCE")
    print("======================")

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    print(
        "Accuracy:",
        accuracy_score(y_test, predictions)
    )

    print(
        "Precision:",
        precision
    )

    print(
        "Recall:",
        recall
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

def save_model(model,precision,recall,MODEL_OUTPUT,feature_names):

    os.makedirs(MODEL_OUTPUT, exist_ok=True)

    if precision < 0.90 or recall < 0.90:
        raise ValueError(
            "Model does not meet assignment requirements"
        )
    else:
        with open(MODEL_OUTPUT + "/malware_model.pkl", "wb") as f:

            pickle.dump(
                {
                    "model": model,
                    "feature_names": feature_names,
                    "engineered_columns":
                        engineered_df.columns.tolist(),
                    "threshold":0.7
                },
                f
            )

    print("\nModel saved.")


def process_files(
    filepath,
    feature_to_col,
    valid_features,
    type_lookup,
    subtype_lookup
):

    sparse_rows = []

    sparse_cols = []

    sparse_data = []

    engineered_rows = []

    risk_scores = []

    labels = []

    with open(filepath, encoding="utf-8") as f:

        lines = f.readlines()

    for row_id, line in enumerate(lines):

        parts = line.strip().split()

        risk_score = float(parts[0])

        risk_scores.append(risk_score)

        label = int(risk_score >= 0.30)

        labels.append(label)

        features = [
            x.split(":")
            for x in parts[1:]
            if ":" in x
        ]

        if not features:
            continue

        # =================================================
        # VECTORIZE FEATURES
        # =================================================

        feature_nums = np.array(
            [int(x[0]) for x in features],
            dtype=np.int32
        )

        values = np.array(
            [float(x[1]) for x in features],
            dtype=np.float32
        )

        # =================================================
        # FILTER VALID FEATURES
        # =================================================

        valid_mask = np.isin(
            feature_nums,
            list(valid_features)
        )

        feature_nums = feature_nums[
            valid_mask
        ]

        values = values[
            valid_mask
        ]

        if len(feature_nums) == 0:
            continue

        # =================================================
        # SPARSE MATRIX ENTRIES
        # =================================================

        cols = np.array(
            [
                feature_to_col[f]
                for f in feature_nums
            ],
            dtype=np.int32
        )

        rows = np.full(
            len(cols),
            row_id,
            dtype=np.int32
        )

        sparse_rows.extend(rows)

        sparse_cols.extend(cols)

        sparse_data.extend(values)

        # =================================================
        # FEATURE ENGINEERING
        # =================================================

        total_features = len(values)

        total_frequency = values.sum()

        feature_density = (
            total_features /
            len(feature_to_col)
        )


        feature_variance = values.var()

        probs = values / values.sum()

        feature_entropy = entropy(probs)

        # =================================================
        # FEATURE TYPE COUNTS
        # =================================================
        feature_types = type_lookup[feature_nums]

        feature_subtypes = subtype_lookup[feature_nums]

        engineered = {

            "total_features":
                total_features,

            "feature_density":
                feature_density,

            "feature_variance":
                feature_variance,

            "feature_entropy":
                feature_entropy
        }

        # VECTOR COUNT TYPES
        feature_types = np.asarray(
            feature_types,
            dtype=str
        )

        feature_subtypes = np.asarray(
            feature_subtypes,
            dtype=str
        )
        unique_types, counts = np.unique(
            feature_types,
            return_counts=True
        )

        engineered.update({

            f"type_count_{k}": v
            for k, v in zip(
                unique_types,
                counts
            )
        })

        unique_subtypes, counts = np.unique(
            feature_subtypes,
            return_counts=True
        )

        engineered.update({

            f"subtype_count_{k}": v
            for k, v in zip(
                unique_subtypes,
                counts
            )
        })

        engineered_rows.append(
            engineered
        )

    return (

        sparse_data,

        sparse_rows,

        sparse_cols,

        engineered_rows,

        labels,

        risk_scores,

        len(labels)
    )


def plot_feature_imp(model,FI_OUTPUT):

    importance = model.get_score(
        importance_type="gain"
    )

    importance_df = (
        pd.DataFrame(
            importance.items(),
            columns=["feature", "gain"]
        )
        .sort_values(
            "gain",
            ascending=False
        )
        .head(20)
    )

    plt.figure(figsize=(12, 8))

    plt.barh(
        importance_df["feature"][::-1],
        importance_df["gain"][::-1]
    )

    plt.xlabel("Gain")
    plt.ylabel("Feature")
    plt.title("Top 20 Malware Features")

    plt.tight_layout()

    plt.savefig(
        FI_OUTPUT + "feature_importance.png",
        dpi=300
    )

# =====================================================
# HYPERPARAMETER TUNING
# =====================================================

def tune_hyper_parameters():

    # =====================================================
    # HYPERPARAMETER SPACE
    # =====================================================

    space = {
        #'scale_pos_weight':[1, scale_pos_weight * 0.5, scale_pos_weight, scale_pos_weight * 1.5],
        'max_delta_step':hp.quniform('max_delta_step', 1, 10, 1),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'gamma': hp.uniform('gamma', 0, 5),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'n_estimators': hp.quniform('n_estimators', 100, 300, 25)
    }

    # =====================================================
    # HYPERPARAMETER SEARCH
    # =====================================================

    trials = Trials()

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=2,  # lower for speed
        trials=trials
    )

    print("Best Parameters:", best)
    return best


def write_to_file(X_sparse,base_feature_names,y,APP_RISK_DATA_FILE):
    os.makedirs("..\\input\\processed", exist_ok=True)

    df = pd.DataFrame.sparse.from_spmatrix(
        X_sparse,
        columns=base_feature_names
    )

    df["label"] = y
    df = df.fillna(0)
    df.to_csv(
        APP_RISK_DATA_FILE,
        index=False
    )
    X_sparse = coo_matrix(df).tocsr()
    print("app_risk_dataset.csv created.")

# =====================================================
# LOAD FEATURE MAPPING AND DATA
# =====================================================
def load_data(MAPPING_FILE,DATA_FOLDER):

    global feature_mapping

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

    feature_lookup = (
        feature_mapping
        .set_index("feature_number")
        [["feature_type", "feature_subtype"]]
        .to_dict("index")
    )

    for feature_num, info in feature_lookup.items():
        type_lookup[feature_num] = info["feature_type"]
        subtype_lookup[feature_num] = info["feature_subtype"]



    feature_numbers = sorted(valid_features)
    feature_to_col = {
         feature_num: idx
         for idx, feature_num in enumerate(feature_numbers)
    }

    global NUM_FEATURES

    NUM_FEATURES = len(feature_to_col)

    # =====================================================
    # MULTIPROCESS FILE LOADING
    # =====================================================

    print("Loading files in parallel...")

    filepaths = [
        os.path.join(DATA_FOLDER, f)
        for f in os.listdir(DATA_FOLDER)
        if f.endswith(".txt")
    ]


    with (Pool(cpu_count()) as pool):
        results = pool.starmap(process_files, zip(filepaths, repeat(feature_to_col),
        repeat(valid_features),repeat(type_lookup),repeat(subtype_lookup)))

    return results

def merge_data(results):
    print("Building sparse matrix...")

    all_sparse_data = []
    all_sparse_rows = []
    all_sparse_cols = []
    all_targets = []
    all_score = []
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

    all_targets = np.concatenate(
        [np.asarray(r[4], dtype=np.int32) for r in results]
    )

    for r in results:
        all_engineered.extend(r[3])

    all_score = np.concatenate(
        [np.asarray(r[5], dtype=np.float32) for r in results]
    )

    row_counts = np.array(
        [r[6] for r in results],
        dtype=np.int64
    )

    num_rows = int(row_counts.sum())

    global X_sparse

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

    global engineered_df

    engineered_df = pd.DataFrame(
        all_engineered
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
    return (X_final,y)

# =====================================================
# MODEL GENERATION
# =====================================================

def model_data(DATA_FOLDER,MAPPING_FILE,MODEL_OUTPUT,FI_OUTPUT,APP_RISK_DATA_FILE):

    global GLOBAL_X_TRAIN
    global GLOBAL_Y_TRAIN
    global GLOBAL_X_VALIDATE
    global GLOBAL_Y_VALIDATE

    # =====================================================
    # CONFIG
    # =====================================================



    # DATA_FOLDER = "..\\input\\raw"
    # MAPPING_FILE = "..\\config\\feature_name_to_number_mapping.csv"
    # MODEL_OUTPUT = "..\\outputs\\models\\malware_model.pkl"

    DATA_FOLDER = DATA_FOLDER
    MAPPING_FILE = MAPPING_FILE
    MODEL_OUTPUT = MODEL_OUTPUT
    # =====================================================
    # LOAD FEATURE MAPPING
    # =====================================================

    results = load_data(MAPPING_FILE,DATA_FOLDER)

    # =====================================================
    # BUILD SPARSE MATRIX
    # =====================================================

    X_final,y = merge_data(results)

    # Undersampling majority for rebalancing

    # undersample = RandomUnderSampler(sampling_strategy='majority')
    # X_final_under, y_under = undersample.fit_resample(X_final, y)
    # print("Undersampled class distribution:", Counter(y_under))
    # oversample = RandomOverSampler(sampling_strategy='minority')
    # X_final_over, y_over = oversample.fit_resample(X_final, y)
    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_final,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train, X_validate, y_train, y_validate = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_temp
    )
    counter = Counter(y_train)

    global scale_pos_weight
    scale_pos_weight= counter[0] / counter[1]

    GLOBAL_X_TRAIN = X_train

    GLOBAL_Y_TRAIN = np.asarray(
        y_train,
        dtype=np.float32
    )

    GLOBAL_X_VALIDATE = X_validate

    GLOBAL_Y_VALIDATE = np.asarray(
        y_validate,
        dtype=np.int32
    )

    # =====================================================
    # XGBOOST DMATRIX (FASTER)
    # =====================================================

    base_feature_names = (
        feature_mapping
        .sort_values("feature_number")
        ["feature_name"]
        .tolist()
    )

    engineered_feature_names = (
        engineered_df.columns.tolist()
    )

    feature_names = (
            base_feature_names +
            engineered_feature_names
    )

    write_to_file(X_sparse,base_feature_names,y,APP_RISK_DATA_FILE)
    class_weights = {0: 2, 1: 1}
    sample_weights = np.array([class_weights[class_id] for class_id in GLOBAL_Y_TRAIN])
    dtrain = xgb.DMatrix(
        GLOBAL_X_TRAIN,
        label=GLOBAL_Y_TRAIN,
        feature_names=feature_names,
        weight=sample_weights
    )
    class_weights = {0: 2, 1: 1}
    sample_weights = np.array([class_weights[class_id] for class_id in y_test])
    dtest = xgb.DMatrix(
        X_test,
        label=y_test,
        feature_names=feature_names,
        weight=sample_weights
    )
    class_weights = {0: 2, 1: 1}
    sample_weights = np.array([class_weights[class_id] for class_id in GLOBAL_Y_VALIDATE])
    dvalidate = xgb.DMatrix(
        GLOBAL_X_VALIDATE,
        label=GLOBAL_Y_VALIDATE,
        feature_names=feature_names,
        weight=sample_weights
    )

    evals = [(dvalidate, "validation")]

    best = tune_hyper_parameters()

    # =====================================================
    # FINAL MODEL
    # =====================================================
    #

    params = {
        'objective':'binary:logistic',
        'max_delta_step':int(best['max_delta_step']),
        'scale_pos_weight':2.0*scale_pos_weight,
        'max_depth': int(best['max_depth']),
        'learning_rate': best['learning_rate'],
        'gamma': best['gamma'],
        'min_child_weight': best['min_child_weight'],
        'colsample_bytree': best['colsample_bytree'],
        'subsample': best['subsample'],
        'eval_metric': 'logloss',
        #'eval_metric':'aucpr',
        'tree_method': 'hist',
        'n_jobs': -1,
        'random_state': 42
    }

    # =====================================================
    # TRAIN FINAL MODEL
    # =====================================================

    print("Training model...")
    num_rounds = int(best['n_estimators'])
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=evals,
        early_stopping_rounds=20
    )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================
    # tree_df = model.trees_to_dataframe()
    #
    # print(
    #     tree_df.groupby("Feature")
    #     .size()
    #     .sort_values(
    #         ascending=False
    #     )
    #     .head(20)
    # )

    plot_feature_imp(model,FI_OUTPUT)

    # =====================================================
    # PREDICT
    # =====================================================

    pred = model.predict(dtest)
    predictions = (pred >= 0.3).astype(int)

    # =====================================================
    # EVALUATION
    # =====================================================
    pred_validate = model.predict(dvalidate)
    predictions_validate = (pred_validate >= 0.3).astype(int)
    evaluate_model(y_validate, predictions_validate)

    # =====================================================
    # SAVE FINAL MODEL
    # =====================================================
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    save_model(model,precision,recall,MODEL_OUTPUT,base_feature_names)

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("too few arguments")
        sys.exit(1)
    freeze_support()
    model_data(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])