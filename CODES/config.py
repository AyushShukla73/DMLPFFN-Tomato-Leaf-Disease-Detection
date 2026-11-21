import os
from dataclasses import dataclass, field

@dataclass
class Config:
    # Paths
    data_root: str = os.environ.get('DATA_ROOT', 'tomato_leaf/tomato_leaf')
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
    dropout: float = 0.7
    label_smoothing: float = 0.1

    # Training
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 5e-4
    epochs: int = 250
    patience: int = 25
    seed: int = 42

    # Device
    device: str = 'cuda'

    # Optimizer
    optimizer: str = 'AdamW'
    loss_function: str = 'CrossEntropyLoss'
    scheduler: str = 'CosineAnnealingWarmRestarts'
    scheduler_params: dict = field(default_factory=lambda: {})

    # Augmentation (Phase 3)
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0

    # Class balancing
    use_focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

cfg = Config()
