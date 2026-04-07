import os
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    check_scoring,
    roc_auc_score,
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
import optuna
import warnings

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import has_fit_parameter

from patient_utils import make_patient_weights  # noqa: E402

# Ray: disable deprecated env override behavior
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

# Optuna experimental warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def subsample_by_patient(indices, groups, max_windows_per_patient, rng):
    """Cap the number of windows contributed by each patient in one fold."""
    indices = np.asarray(indices)
    groups = np.asarray(groups)
    selected = []
    fold_groups = groups[indices]

    for group in np.unique(fold_groups):
        patient_idx = indices[fold_groups == group]
        take = min(max_windows_per_patient, patient_idx.shape[0])
        if take == patient_idx.shape[0]:
            chosen = patient_idx
        else:
            chosen = rng.choice(patient_idx, size=take, replace=False)
        selected.append(np.sort(chosen))

    return np.sort(np.concatenate(selected))


def get_fit_kwargs(model, sample_weight):
    """Route sample weights to the estimator, including pipeline-wrapped models."""
    if isinstance(model, Pipeline):
        step_name, step_estimator = model.steps[-1]
        if has_fit_parameter(step_estimator, "sample_weight"):
            return {f"{step_name}__sample_weight": sample_weight}
        return {}

    if has_fit_parameter(model, "sample_weight"):
        return {"sample_weight": sample_weight}

    return {}


def predict_scores(model, X, multiclass=False):
    """Return continuous scores and the matching hard-decision threshold.

    For multiclass, returns the full probability matrix and hard predictions
    as (proba_matrix, predictions) with threshold=None.
    """
    if multiclass:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
        else:
            proba = None
        preds = model.predict(X)
        return proba, preds

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
        if scores.ndim == 2:
            scores = scores[:, 1]
        threshold = 0.5
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        threshold = 0.0
    else:
        scores = model.predict(X)
        threshold = 0.5

    return scores, threshold


def compute_patient_balanced_metrics(model, X, y, groups, task="binary"):
    """Compute validation metrics under equal patient contribution."""
    weights = make_patient_weights(groups)

    if task == "multiclass":
        proba, y_pred = predict_scores(model, X, multiclass=True)
        metrics = {
            "patient_balanced_accuracy": balanced_accuracy_score(
                y, y_pred, sample_weight=weights
            ),
        }
        if proba is not None and np.unique(y).shape[0] > 1:
            try:
                metrics["patient_auc"] = roc_auc_score(
                    y,
                    proba,
                    multi_class="ovr",
                    average="macro",
                    sample_weight=weights,
                )
            except ValueError:
                metrics["patient_auc"] = np.nan
        else:
            metrics["patient_auc"] = np.nan
        return metrics

    scores, threshold = predict_scores(model, X)
    y_pred = (scores >= threshold).astype(int)

    metrics = {
        "patient_auc": np.nan,
        "patient_balanced_accuracy": balanced_accuracy_score(
            y, y_pred, sample_weight=weights
        ),
    }

    if np.unique(y).shape[0] > 1:
        metrics["patient_auc"] = roc_auc_score(y, scores, sample_weight=weights)

    return metrics


def train_cv(
    config,
    X,
    y,
    groups,
    folds,
    estimator,
    scoring,
    balance_mode,
    max_windows_per_patient,
    task="binary",
):
    """Train one hyperparameter trial across all prepared CV folds."""

    summary = {}
    metrics = {}

    base = clone(estimator).set_params(**config)

    if not isinstance(y, np.ndarray):
        y = y.to_numpy()
    if not isinstance(groups, np.ndarray):
        groups = groups.to_numpy()

    # Loop over all repeated CV folds
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fit_train_idx = train_idx
        if balance_mode == "subsample":
            # Keep validation untouched; only the fitting subset is balanced.
            rng = np.random.default_rng(42 + fold_idx)
            fit_train_idx = subsample_by_patient(
                train_idx, groups, max_windows_per_patient, rng
            )

        X_tr, X_val = X[fit_train_idx], X[val_idx]
        y_tr, y_val = y[fit_train_idx], y[val_idx]
        groups_tr, groups_val = groups[fit_train_idx], groups[val_idx]

        model = clone(base)  # fresh model per fold
        fit_kwargs = {}
        if balance_mode == "weight":
            fit_kwargs = get_fit_kwargs(model, make_patient_weights(groups_tr))
        model.fit(X_tr, y_tr, **fit_kwargs)

        scores_train = scoring(model, X_tr, y_tr)
        scores_val = scoring(model, X_val, y_val)
        patient_train = compute_patient_balanced_metrics(
            model, X_tr, y_tr, groups_tr, task=task
        )
        patient_val = compute_patient_balanced_metrics(
            model, X_val, y_val, groups_val, task=task
        )
        scores_train = scores_train | patient_train
        scores_val = scores_val | patient_val

        for key in scores_train.keys():
            if fold_idx == 0:
                summary[f"train_{key}"] = [scores_train[key]]
                summary[f"val_{key}"] = [scores_val[key]]

            else:
                summary[f"train_{key}"].append(scores_train[key])
                summary[f"val_{key}"].append(scores_val[key])

            metrics[f"mean_val_{key}"] = float(np.nanmean(summary[f"val_{key}"]))
            metrics[f"mean_train_{key}"] = float(np.nanmean(summary[f"train_{key}"]))

            metrics[f"std_val_{key}"] = float(np.nanstd(summary[f"val_{key}"]))
            metrics[f"std_train_{key}"] = float(np.nanstd(summary[f"train_{key}"]))
            metrics[f"gap_{key}"] = (
                metrics[f"mean_train_{key}"] - metrics[f"mean_val_{key}"]
            )

        metrics["fold"] = fold_idx

        tune.report(metrics)


class RayAdaptiveRepeatedCVSearch:
    """
    RayTune-based adaptive repeated cross-validation search.
    Reproduces caret adaptive resampling behavior:
        - repeated CV (default: 5x10)
        - per-fold adaptive pruning
        - minimum folds before pruning (grace_period)
        - AUC + Balanced Accuracy evaluation
        - StandardScaler preprocessing
        - final best estimator refit on full data
    """

    def __init__(
        self,
        estimator,
        search_space,
        cv=None,
        grace_period=7,
        reduction_factor=2,
        num_samples=50,
        scoring="roc_auc",
        rank_metric="mean_val_auc",
        mode="max",
        store_path="./ray_results",
        model_name="base",
        balance_mode="none",
        max_windows_per_patient=50,
        task="binary",
    ):
        """
        estimator: sklearn estimator class (e.g. RandomForestClassifier)
        search_space: Ray Tune search space dict
        cv: custom CV folds
        grace_period: min CV folds before pruning (caret min=10)
        reduction_factor: ASHA aggression
        num_samples: number of hyperparameter trials
        """
        self.estimator = estimator
        self.search_space = search_space
        self.cv = (
            cv if cv is not None else RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
        )
        self.grace_period = grace_period
        self.reduction_factor = reduction_factor
        self.num_samples = num_samples
        self.scoring = check_scoring(estimator, scoring=scoring)
        self.mode = mode
        self.rank_metric = rank_metric
        self.task = task

        self.folds = None
        self.best_config = None
        self.best_estimator_ = None
        self.analysis = None
        self.store_path = store_path
        self.model_name = model_name
        self.balance_mode = balance_mode
        self.max_windows_per_patient = max_windows_per_patient

        if isinstance(self.cv, list):
            self.n_splits, self.n_repeats = self.cv[0], self.cv[1]

    def _build_folds(self, X, y, groups=None):
        if groups is None:
            self.folds = list(self.cv.split(X, y))
        else:
            self.folds = list(self._make_repeated_stratified_group_kfold(X, y, groups))

    def _make_repeated_stratified_group_kfold(self, X, y, groups, random_state=42):
        n_splits = self.n_splits
        n_repeats = self.n_repeats

        for r in range(n_repeats):
            sgkf = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state + r
            )
            for train_idx, val_idx in sgkf.split(X, y, groups):
                yield train_idx, val_idx

    def fit(self, X, y, groups=None):

        self._build_folds(X, y, groups=groups)

        folds = self.folds
        estimator = self.estimator

        # -------------------------------
        # Training function for Ray Tune
        # -------------------------------

        # -------------------------------
        # Ray Tune scheduler (caret-like pruning)
        # -------------------------------
        scheduler = ASHAScheduler(
            max_t=len(self.folds),
            grace_period=self.grace_period,
            reduction_factor=self.reduction_factor,
        )

        # -------------------------------
        # Run Ray Tune
        # -------------------------------
        searcher = OptunaSearch(
            sampler=optuna.samplers.TPESampler(
                multivariate=True,  # better exploration
                group=True,  # improves correlated parameters
                n_startup_trials=10,
                constant_liar=True,
            ),
        )
        # searcher = ConcurrencyLimiter(searcher, max_concurrent=os.cpu_count())
        trainable = tune.with_parameters(
            train_cv,
            X=X,
            y=y,
            groups=groups,
            folds=folds,
            estimator=estimator,
            scoring=self.scoring,
            balance_mode=self.balance_mode,
            max_windows_per_patient=self.max_windows_per_patient,
            task=self.task,
        )
        tuner = tune.Tuner(
            trainable,
            param_space=self.search_space,
            tune_config=tune.TuneConfig(
                metric=self.rank_metric,
                mode=self.mode,
                scheduler=scheduler,
                search_alg=searcher,
                num_samples=self.num_samples,
            ),
            run_config=tune.RunConfig(
                name=self.model_name,
                verbose=1,
                failure_config=tune.FailureConfig(fail_fast=False),
                storage_path=self.store_path,
            ),
        )

        self.results = tuner.fit()

        # -------------------------------------------------------
        # Extract best config
        # -------------------------------------------------------
        self.best_config = self.results.get_best_result(
            metric=self.rank_metric, mode=self.mode
        ).config

        # -------------------------------------------------------
        # Fit final best estimator on full dataset
        # -------------------------------------------------------
        self.best_estimator_ = clone(estimator).set_params(**self.best_config)
        fit_X, fit_y = X, y
        fit_groups = groups

        if fit_groups is not None and not isinstance(fit_groups, np.ndarray):
            fit_groups = fit_groups.to_numpy()
        if fit_groups is not None and self.balance_mode == "subsample":
            # Apply the same balancing policy used in CV before fitting the
            # persisted final estimator.
            full_idx = subsample_by_patient(
                np.arange(len(y)),
                fit_groups,
                self.max_windows_per_patient,
                np.random.default_rng(42),
            )
            fit_X = X[full_idx]
            fit_y = y[full_idx]
            fit_groups = fit_groups[full_idx]

        fit_kwargs = {}
        if fit_groups is not None and self.balance_mode == "weight":
            fit_kwargs = get_fit_kwargs(
                self.best_estimator_, make_patient_weights(fit_groups)
            )

        self.best_estimator_.fit(fit_X, fit_y, **fit_kwargs)

        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        if hasattr(self.best_estimator_, "predict_proba"):
            return self.best_estimator_.predict_proba(X)[:, 1]
        return self.best_estimator_.decision_function(X)

    def get_best_config(self):
        return self.best_config

    def get_best_score(self):
        return self.analysis.best_result.get(self.metric)

    def cv_results_(self):
        return self.results.get_dataframe().sort_values(
            by=self.rank_metric, ascending=(self.mode == "min")
        )
