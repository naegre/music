from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ColumnConfig:
    device: str = "name"
    subdir: str = "subdir"
    test_file: str = "test_file"
    original_file: str = "original_file"
    audio_id: str = "audio_index"
    time_slice: str = "time_segment"
    kl_mean: str = "kl_mean"
    kl_var: str = "kl_var"
    js_mean: str = "js_mean"
    js_var: str = "js_var"
    l2: str = "l2"
    cos_sim: str = "cos_sim"

    @property
    def metric_columns(self):
        return [self.kl_mean, self.kl_var, self.js_mean, self.js_var, self.l2, self.cos_sim]

    @property
    def audio_group_columns(self):
        return [self.device, self.subdir, self.audio_id, self.original_file]

    @property
    def device_subdir_group_columns(self):
        return [self.device, self.subdir]

    def to_dict(self):
        return asdict(self)
