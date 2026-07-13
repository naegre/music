from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MetricSpec:
    key: str
    mean_column: str
    variance_column: str
    transform: str
    direction: float
    role: str


@dataclass(frozen=True)
class ColumnConfig:
    subdir: str = "subdir"
    device: str = "name"
    audio_id: str = "x"
    n_windows: str = "n_windows"
    kl_mean: str = "kl_mean"
    kl_var: str = "kl_var"
    js_mean: str = "js_mean"
    js_var: str = "js_var"
    l2_mean: str = "l2_mean"
    l2_var: str = "l2_var"
    cosim_mean: str = "cosim_mean"
    cosim_var: str = "cosim_var"

    @property
    def metric_specs(self):
        return (
            MetricSpec("l2", self.l2_mean, self.l2_var, "log1p", -1.0, "primary"),
            MetricSpec("cosim", self.cosim_mean, self.cosim_var, "fisher", 1.0, "primary"),
            MetricSpec("kl", self.kl_mean, self.kl_var, "log1p", -1.0, "auxiliary"),
            MetricSpec("js", self.js_mean, self.js_var, "log1p", -1.0, "auxiliary"),
        )

    @property
    def required_columns(self):
        columns = [self.subdir, self.device, self.audio_id, self.n_windows]
        for spec in self.metric_specs:
            columns.extend([spec.mean_column, spec.variance_column])
        return columns

    def to_dict(self):
        return asdict(self)
