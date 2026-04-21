from hydra.core.config_store import ConfigStore

from machineunlearning.config.schemas import (
    DatasetCfg,
    ModelCfg,
    OptCfg,
    StrategyCfg,
    TrainConfig,
    UnlearnConfig,
)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="train_schema", node=TrainConfig)
    cs.store(name="unlearn_schema", node=UnlearnConfig)
    cs.store(group="dataset", name="_schema", node=DatasetCfg)
    cs.store(group="model", name="_schema", node=ModelCfg)
    cs.store(group="optimizer", name="_schema", node=OptCfg)
    cs.store(group="strategy", name="_schema", node=StrategyCfg)


__all__ = [
    "DatasetCfg",
    "ModelCfg",
    "OptCfg",
    "StrategyCfg",
    "TrainConfig",
    "UnlearnConfig",
    "register_configs",
]
