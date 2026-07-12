from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from .features import HierarchicalFeatureTransformer


DEFAULT_ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


@dataclass
class RidgeModelBundle:
    transformer: HierarchicalFeatureTransformer
    model: Ridge
    feature_columns: list
    subdir_col: str | None
    interaction_columns: list
    alpha: float

    def predict(self, frame):
        return self.model.predict(self.transformer.transform(frame))

    def weight_table(self):
        names = self.transformer.get_feature_names_out()
        return pd.DataFrame({"feature": names, "standardized_weight": self.model.coef_})


def reliability_weights(frame, n_audio_col="n_audio", cap=5.0):
    if n_audio_col not in frame.columns:
        return np.ones(len(frame), dtype=np.float64)
    counts = pd.to_numeric(frame[n_audio_col], errors="coerce").fillna(1.0).clip(lower=1.0)
    weights = np.sqrt(np.minimum(counts.to_numpy(dtype=np.float64), cap))
    return weights / weights.mean()


def fit_ridge(
    frame,
    target_col,
    feature_cols,
    alpha,
    subdir_col=None,
    interaction_cols=None,
    use_reliability_weights=True,
    weight_cap=5.0,
):
    transformer = HierarchicalFeatureTransformer(feature_cols, subdir_col, interaction_cols or [])
    x = transformer.fit_transform(frame)
    y = frame[target_col].to_numpy(dtype=np.float64)
    weights = reliability_weights(frame, cap=weight_cap) if use_reliability_weights else None
    model = Ridge(alpha=float(alpha))
    model.fit(x, y, sample_weight=weights)
    return RidgeModelBundle(
        transformer=transformer,
        model=model,
        feature_columns=list(feature_cols),
        subdir_col=subdir_col,
        interaction_columns=list(interaction_cols or []),
        alpha=float(alpha),
    )


def _inner_select_alpha(
    frame,
    target_col,
    group_col,
    feature_cols,
    subdir_col,
    interaction_cols,
    alphas,
    weight_cap,
):
    groups = frame[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if unique_groups.size < 3:
        return float(alphas[len(alphas) // 2])

    splitter = GroupKFold(n_splits=min(5, unique_groups.size))
    scores = []
    for alpha in alphas:
        predictions = np.full(len(frame), np.nan, dtype=np.float64)
        for train_idx, test_idx in splitter.split(frame, groups=groups):
            train = frame.iloc[train_idx]
            test = frame.iloc[test_idx]
            bundle = fit_ridge(
                train,
                target_col,
                feature_cols,
                alpha,
                subdir_col,
                interaction_cols,
                use_reliability_weights=True,
                weight_cap=weight_cap,
            )
            predictions[test_idx] = bundle.predict(test)
        scores.append((mean_absolute_error(frame[target_col], predictions), float(alpha)))
    return min(scores, key=lambda item: item[0])[1]


def nested_leave_one_device_out(
    frame,
    target_col,
    device_col,
    feature_cols,
    subdir_col=None,
    interaction_cols=None,
    alphas=DEFAULT_ALPHAS,
    weight_cap=5.0,
):
    working = frame.dropna(subset=[target_col, device_col]).reset_index(drop=True)
    groups = working[device_col].astype(str).to_numpy()
    splitter = LeaveOneGroupOut()
    predictions = np.full(len(working), np.nan, dtype=np.float64)
    fold_alphas = []

    for train_idx, test_idx in splitter.split(working, groups=groups):
        train = working.iloc[train_idx].reset_index(drop=True)
        test = working.iloc[test_idx]
        alpha = _inner_select_alpha(
            train,
            target_col,
            device_col,
            feature_cols,
            subdir_col,
            interaction_cols or [],
            alphas,
            weight_cap,
        )
        bundle = fit_ridge(
            train,
            target_col,
            feature_cols,
            alpha,
            subdir_col,
            interaction_cols,
            use_reliability_weights=True,
            weight_cap=weight_cap,
        )
        predictions[test_idx] = bundle.predict(test)
        fold_alphas.append({"held_out_device": str(test.iloc[0][device_col]), "alpha": alpha})

    final_alpha = _inner_select_alpha(
        working,
        target_col,
        device_col,
        feature_cols,
        subdir_col,
        interaction_cols or [],
        alphas,
        weight_cap,
    )
    final_bundle = fit_ridge(
        working,
        target_col,
        feature_cols,
        final_alpha,
        subdir_col,
        interaction_cols,
        use_reliability_weights=True,
        weight_cap=weight_cap,
    )
    result = working.copy()
    result["predicted_score_oof"] = predictions
    return result, final_bundle, pd.DataFrame(fold_alphas)


def build_overall_table(subdir_prediction_df, overall_labels, device_col, subdir_col):
    pivot = subdir_prediction_df.pivot_table(
        index=device_col,
        columns=subdir_col,
        values="predicted_score_oof",
        aggfunc="mean",
    )
    pivot.columns = [f"predicted_subdir_{column}" for column in pivot.columns]
    pivot = pivot.reset_index()
    return pivot.merge(overall_labels, on=device_col, how="inner")


def pivot_new_subdir_predictions(prediction_df, device_col, subdir_col, score_col="predicted_subdir_score"):
    pivot = prediction_df.pivot_table(index=device_col, columns=subdir_col, values=score_col, aggfunc="mean")
    pivot.columns = [f"predicted_subdir_{column}" for column in pivot.columns]
    return pivot.reset_index()
