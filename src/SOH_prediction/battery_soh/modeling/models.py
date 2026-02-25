# -*- coding: utf-8 -*-
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def fit_predict_single_regularized_model(X_train, y_train, X_test, y_test,random_state=42):
    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    X_train_p = pre.fit_transform(X_train)
    X_test_p = pre.transform(X_test)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=5000,
        max_depth=7,
        min_samples_leaf=10,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=80,
        random_state=42
    )

    model.fit(X_train_p, y_train)

    y_pred = model.predict(X_test_p)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return y_pred, mse, mae

def get_oof_preds(model, X_train, y_train, X_test, groups=None, n_splits=5, random_state=42):
    if groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        split_iter = cv.split(X_train, y_train, groups=groups)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = cv.split(X_train, y_train)

    oof_train = np.zeros((X_train.shape[0],), dtype=float)
    oof_test_folds = np.zeros((n_splits, X_test.shape[0]), dtype=float)

    for i, (tr_idx, va_idx) in enumerate(split_iter):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr = y_train[tr_idx]

        model.fit(X_tr, y_tr)
        oof_train[va_idx] = model.predict(X_va)
        oof_test_folds[i, :] = model.predict(X_test)

    oof_test = oof_test_folds.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

def fit_predict_stacked_ensemble(X_train, y_train, X_test, y_test, groups_train=None, n_splits=5, random_state=42):
    pre = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    X_train_p = pre.fit_transform(X_train)
    X_test_p = pre.transform(X_test)

    xgb = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=2.0, reg_lambda=1.0, random_state=101, n_jobs=-1)
    lgb = LGBMRegressor(n_estimators=1200, learning_rate=0.02, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=101, n_jobs=-1)
    rf = RandomForestRegressor(n_estimators=600, max_depth=14, min_samples_leaf=2, random_state=101, n_jobs=-1)

    base_models = [xgb, lgb, rf]

    oof_tr_list, oof_te_list = [], []
    for m in base_models:
        oof_tr, oof_te = get_oof_preds(m, X_train_p, y_train, X_test_p, groups=groups_train, n_splits=n_splits, random_state=random_state)
        oof_tr_list.append(oof_tr)
        oof_te_list.append(oof_te)

    X_meta_train = np.concatenate(oof_tr_list, axis=1)
    X_meta_test = np.concatenate(oof_te_list, axis=1)

    meta_model = Ridge(alpha=5.0, random_state=random_state)
    meta_model.fit(X_meta_train, y_train)

    y_pred = meta_model.predict(X_meta_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return y_pred, mse, mae