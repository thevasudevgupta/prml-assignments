"""
# Pattern Recognition and Machine Learning

## Data Contest Submission

Submitted By:
* Aniruddha Gandhewar (ME18B181)
* Vasudev Gupta (ME18B182)
"""

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.decomposition import PCA

SEED = 9


def load_data(train_file1: str, test_file1: str, train_file2: str, test_file2: str):
  df1 = pd.read_csv(train_file1, index_col=0, header=0)
  X1 = df1.values.T[:, :-2]
  co1 = df1.iloc[-2,:].values.astype(np.int32)
  co2 = df1.iloc[-1,:].values.astype(np.int32)

  df1_test = pd.read_csv(test_file1, index_col=0, header=0)
  X1_test = df1_test.values.T

  df2 = pd.read_csv(train_file2, index_col=0, header=0)
  X2 = df2.values.T[:, :-4]
  co3 = df2.iloc[-4, :].values.astype(np.int32)
  co4 = df2.iloc[-3, :].values.astype(np.int32)
  co5 = df2.iloc[-2, :].values.astype(np.int32)
  co6 = df2.iloc[-1, :].values.astype(np.int32)

  df2_test = pd.read_csv(test_file2, index_col=0, header=0)
  X2_test = df2_test.values.T

  return X1, X2, (co1, co2, co3, co4, co5, co6), X1_test, X2_test


DEFAULT_CONFIGS_1 = {
  "RFE__n_features_to_select": 60,
  "BGC__SVC__kernel": "linear",
  "BGC__SVC__n_estimators": 100,
  "BGC__SVC__max_samples": 1.0,
  "BGC__SVC__max_features": 1.0,
}


def train_and_predict_1(X, y, X_test, configs={}):
  cfg = DEFAULT_CONFIGS_1.copy()
  cfg.update(configs)
  print(cfg)

  pca = PCA(n_components=0.9, svd_solver="full", random_state=SEED)
  X = pca.fit_transform(X)
  X_test = pca.transform(X_test)

  rfe = RFE(SVC(kernel="linear", random_state=SEED), step=1, n_features_to_select=cfg["RFE__n_features_to_select"])
  X = rfe.fit_transform(X, y)
  X_test = rfe.transform(X_test)

  bgc_svc = SVC(kernel=cfg["BGC__SVC__kernel"], random_state=SEED)
  model = BaggingClassifier(
    bgc_svc,
    n_estimators=cfg["RFE__n_features_to_select"],
    max_samples=cfg["BGC__SVC__max_samples"],
    max_features=cfg["BGC__SVC__max_features"],
    random_state=SEED,
  )
  model.fit(X, y)

  y_pred = model.predict(X_test)
  return y_pred


DEFAULT_CONFIGS_2 = {
    "SelectKBest__k": 3000,
    "PCA__n_components": 300,
    "RFE__SVC__C": 0.1,
    "RFE__SVC__gamma": 1,
    "RFE__SVC__kernel": 'linear',
    "RFE__n_features_to_select": 10,
    "BaggingClassifier__SVC__C": 0.1,
    "BaggingClassifier__SVC__gamma": 1,
    "BaggingClassifier__SVC__kernel": "linear",
    "BaggingClassifier__n_estimators": 100,
}


def train_and_predict_2(X, y, X_test, configs={}):
  cfg = DEFAULT_CONFIGS_2.copy()
  cfg.update(configs)
  print(cfg)

  features_selector = SelectKBest(k=cfg["SelectKBest__k"])
  features_selector.fit(X, y)

  ftrs = features_selector.get_support()
  X = X[:, ftrs]
  X_test = X_test[:, ftrs]

  pca = PCA(n_components=cfg["PCA__n_components"], random_state=SEED)
  X = pca.fit_transform(X)
  X_test = pca.transform(X_test)

  rfe_svc = SVC(
      C=cfg["RFE__SVC__C"],
      gamma=cfg["RFE__SVC__gamma"],
      kernel=cfg["RFE__SVC__kernel"],
      random_state=SEED,
  )
  rfe = RFE(rfe_svc, n_features_to_select=cfg["RFE__n_features_to_select"])

  X = rfe.fit_transform(X, y)
  X_test = rfe.transform(X_test)

  bgc_svc = SVC(
      C=cfg["BaggingClassifier__SVC__C"],
      gamma=cfg["BaggingClassifier__SVC__gamma"],
      kernel=cfg["BaggingClassifier__SVC__kernel"],
      random_state=SEED,
  )
  bgc = BaggingClassifier(
      base_estimator=bgc_svc,
      n_estimators=cfg["BaggingClassifier__n_estimators"],
      random_state=SEED,
  )
  bgc.fit(X, y)

  return bgc.predict(X_test)


def make_submission(co1_pred, co2_pred, co3_pred, co4_pred, co5_pred, co6_pred):
  predictions = np.concatenate([co1_pred, co2_pred, co3_pred, co4_pred, co5_pred, co6_pred])
  predictions = predictions.astype(np.int32)
  submission = pd.DataFrame({"Id": np.arange(len(predictions)), "Predicted": predictions})
  return submission


if __name__ == '__main__':

  train_file1, test_file1 = "Dataset_1_Training.csv", "Dataset_1_Testing.csv"
  train_file2, test_file2 = "Dataset_2_Training.csv", "Dataset_2_Testing.csv"

  X1, X2, targets, X1_test, X2_test = load_data(train_file1, test_file1, train_file2, test_file2)
  co1, co2, co3, co4, co5, co6 = targets

  print("dataset-1:", X1.shape, co1.shape, co2.shape, X1_test.shape)
  print("dataset-2:", X2.shape, co3.shape, co4.shape, co5.shape, co6.shape, X2_test.shape)

  co1_pred = train_and_predict_1(X1, co1, X1_test)
  print("co1_pred_shape:", co1_pred.shape)

  configs = {
    "RFE__n_features_to_select": 70,
    "BGC__SVC__kernel": "rbf",
    "BGC__SVC__n_estimators": 75,
    "BGC__SVC__max_samples": 0.75,
    "BGC__SVC__max_features": 0.9,
  }
  co2_pred = train_and_predict_1(X1, co2, X1_test, configs=configs)
  print("co2_pred_shape:", co2_pred.shape)

  configs = {
    "RFE__n_features_to_select": 75,
    "BGC__SVC__kernel": "rbf",
    "BGC__SVC__max_samples": 0.75,
  }
  co3_pred = train_and_predict_1(X2, co3, X2_test, configs=configs)
  print("co3_pred_shape:", co3_pred.shape)

  configs = {"RFE__n_features_to_select": 100}
  co4_pred = train_and_predict_2(X2, co4, X2_test, configs=configs)
  print("co4_pred_shape:", co4_pred.shape)

  configs = {"RFE__n_features_to_select": 105}
  co5_pred = train_and_predict_2(X2, co5, X2_test, configs=configs)
  print("co5_pred_shape:", co5_pred.shape)

  configs = {"BaggingClassifier__SVC__kernel": "rbf", "RFE__n_features_to_select": 95}
  co6_pred = train_and_predict_2(X2, co6, X2_test, configs=configs)
  print("co6_pred_shape:", co6_pred.shape)

  submission = make_submission(co1_pred, co2_pred, co3_pred, co4_pred, co5_pred, co6_pred)

  print("saving submission at ME18B181_ME18B182.csv")
  submission.to_csv("ME18B181_ME18B182.csv", index=False)
