import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


RECOMMENDED_FEATURE_COLUMNS = [
    "kl_pooled_audio_mean_device_mean",
    "kl_pooled_audio_var_device_mean",
    "kl_pooled_audio_mean_between_audio_std",
    "js_pooled_audio_mean_device_mean",
    "js_pooled_audio_var_device_mean",
    "js_pooled_audio_mean_between_audio_std",
    "l2_audio_median_device_mean",
    "l2_audio_median_between_audio_std",
    "cos_sim_audio_median_device_mean",
    "cos_sim_audio_median_between_audio_std",
]


class HierarchicalFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols, subdir_col=None, interaction_cols=None):
        self.numeric_cols = numeric_cols
        self.subdir_col = subdir_col
        self.interaction_cols = interaction_cols

    def fit(self, x, y=None):
        frame = pd.DataFrame(x).copy()
        self.numeric_cols_ = list(self.numeric_cols)
        numeric = frame[self.numeric_cols_].apply(pd.to_numeric, errors="coerce")
        self.medians_ = numeric.median().fillna(0.0)
        imputed = numeric.fillna(self.medians_)
        self.means_ = imputed.mean()
        self.scales_ = imputed.std(ddof=0).replace(0.0, 1.0).fillna(1.0)

        if self.subdir_col:
            self.categories_ = sorted(frame[self.subdir_col].fillna("<missing>").astype(str).unique().tolist())
        else:
            self.categories_ = []

        requested = list(self.interaction_cols or [])
        self.interaction_cols_ = [col for col in requested if col in self.numeric_cols_]
        return self

    def transform(self, x):
        frame = pd.DataFrame(x).copy()
        numeric = frame[self.numeric_cols_].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(self.medians_)
        z = ((numeric - self.means_) / self.scales_).to_numpy(dtype=np.float64)
        blocks = [z]

        if self.categories_:
            values = frame[self.subdir_col].fillna("<missing>").astype(str).to_numpy()
            one_hot = np.column_stack([(values == category).astype(np.float64) for category in self.categories_[1:]])
            if one_hot.shape[1]:
                blocks.append(one_hot)
                for col in self.interaction_cols_:
                    index = self.numeric_cols_.index(col)
                    blocks.append(z[:, [index]] * one_hot)
        return np.column_stack(blocks)

    def get_feature_names_out(self):
        names = list(self.numeric_cols_)
        if self.categories_:
            category_names = [f"{self.subdir_col}={category}" for category in self.categories_[1:]]
            names.extend(category_names)
            for col in self.interaction_cols_:
                names.extend([f"{col}*{name}" for name in category_names])
        return np.asarray(names, dtype=object)


def default_feature_columns(df, device_col, subdir_col, target_cols=None):
    recommended = [col for col in RECOMMENDED_FEATURE_COLUMNS if col in df.columns]
    if len(recommended) >= 4:
        return recommended
    excluded = {device_col, subdir_col, "n_audio", "n_windows_total"}
    excluded.update(target_cols or [])
    return [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
