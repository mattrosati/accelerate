import os
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import check_scoring
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
import optuna
import warnings

# Ray: disable deprecated env override behavior
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

# Optuna experimental warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def train_cv(config, X, y, folds, estimator, scoring):

    model = estimator.set_params(**config)
    summary = {}
    metrics = {}
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()
    # Loop over all repeated CV folds
    for fold_idx, (train_idx, val_idx) in enumerate(folds):

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)

        scores_train = scoring(estimator, X_tr, y_tr)
        scores_val = scoring(estimator, X_val, y_val)

        if fold_idx == 0:
            for key in scores_train.keys():
                summary[f"train_{key}"] = [scores_train[key]]
                summary[f"val_{key}"] = [scores_val[key]]

                metrics[f"mean_val_{key}"] = float(np.mean([scores_val[key]]))
                metrics[f"mean_train_{key}"] = float(np.mean([scores_train[key]]))

                metrics[f"std_val_{key}"] = 0.0
                metrics[f"std_train_{key}"] = 0.0
        else:
            for key in scores_train.keys():
                summary[f"train_{key}"].append(scores_train[key])
                summary[f"val_{key}"].append(scores_val[key])

                metrics[f"mean_val_{key}"] = float(np.mean([scores_val[key]]))
                metrics[f"mean_train_{key}"] = float(np.mean([scores_train[key]]))

                metrics[f"std_val_{key}"] = float(np.std(summary[f"val_{key}"]))
                metrics[f"std_train_{key}"] = float(np.std(summary[f"train_{key}"]))

        fold_summary = metrics | {"fold": fold_idx}

        tune.report(metrics=fold_summary)


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
        grace_period=6,
        reduction_factor=2,
        num_samples=50,
        scoring="roc_auc",
        rank_metric="mean_val_auc",
        mode="max",
        store_path="./ray_results",
        model_name="base",
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

        self.folds = None
        self.best_config = None
        self.best_estimator_ = None
        self.analysis = None
        self.store_path = store_path
        self.model_name = model_name

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
                n_splits=n_splits, shuffle=True, random_state=random_state
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
            folds=folds,
            estimator=estimator,
            scoring=self.scoring,
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
        self.best_estimator_ = estimator.set_params(**self.best_config)
        self.best_estimator_.fit(X, y)

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
