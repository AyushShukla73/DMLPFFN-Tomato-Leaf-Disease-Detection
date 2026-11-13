import os
from dataclasses import dataclass, field

@dataclass
class Config:
    # Paths
    data_root: str = os.environ.get('DATA_ROOT', 'tomato_leaf')
    save_dir: str = os.environ.get('SAVE_DIR', './outputs')

    # Data
    pca_components: int = 128
    remove_bands: list = field(default_factory=lambda: [(1350,1460), (1800,1950)])
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    patch_size: int = 9
    stride: int = 4

    # Model
    hidden_dims: list = field(default_factory=lambda: [2048, 1024, 512])
    num_classes: int = 4  # healthy, asymptomatic, early, late
    dropout: float = 0.6

    # Training
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 250
    patience: int = 25
    seed: int = 42

cfg = Config()
