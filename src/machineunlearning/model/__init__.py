from machineunlearning.model.models import (
    MLP,
    LRTorchNet,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    SimpleCNN,
)

MODEL_REGISTRY = {
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "MLP": MLP,
    "SimpleCNN": SimpleCNN,
    "LRTorchNet": LRTorchNet,
}
