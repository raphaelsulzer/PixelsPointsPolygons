import torch
from pydantic import BaseModel
from dataclasses import dataclass
from enum import Enum

from pathlib import Path
from typing import Optional


class Pix2PolyConfig(BaseModel):
    """Parsable version of the Pix2Poly CFG class."""

    img_path: Path
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset: Path
    train_dataset_dir: Path
    val_dataset_dir: Path
    test_images_dir: Path
    train_ddp: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    load_model: bool = False
    n_vertices: int
    sinkhorn_iterations: int = 100
    max_len: int
    img_size: int
    input_size: int = 224
    patch_size: int = 8
    input_height: int = 224
    input_width: int = 224
    num_bins: int = 224
    label_smoothing: float = 0.0
    vertex_loss_weight: float = 1.0
    perm_loss_weight: float = 10.0
    shuffle_tokens: bool = False
    batch_size: int = 24
    start_epoch: int = 0
    num_epochs: int = 500
    milestone: int = 0
    save_best: bool = True
    save_latest: bool = True
    save_every: int = 10
    val_every: int = 1
    model_name: str = "vit_small_patch8_224_dino"
    num_patches: int = int((224 // 8) ** 2)
    lr: float = 4e-4
    weight_decay: float = 1e-4
    generation_steps: int
    run_eval: bool = False
    experiment_name: str
    checkpoint_path: Path
    pad_idx: Optional[int] = None  # code can set this from the tokenizer

class DatasetType(str, Enum):
    INRIA = "inria"
    SPACENET = "spacenet" 
    WHU_BUILDINGS = "whu_buildings"
    MASS_ROADS = "mass_roads"

class TrainingConfig(BaseModel):
    """Configuration for model training."""
    experiment_name: str
    dataset_type: DatasetType
    train_dataset_dir: Path
    val_dataset_dir: Path 
    test_images_dir: Path
    checkpoint_path: Optional[Path]
    
    # Model params
    model_name: str = "resnet18"
    input_height: int = 320
    input_width: int = 320
    num_bins: int = 32
    num_patches: int = 100
    n_vertices: int = 100
    
    # Training params
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_tokens: bool = True
    start_epoch: int = 0
    
    # Checkpointing
    save_best: bool = True
    save_latest: bool = True 
    save_every: int = 5
    val_every: int = 1
    
    # Loss weights
    vertex_loss_weight: float = 1.0
    perm_loss_weight: float = 10.0
    
    # Model architecture
    encoder_dim: int = 256
    decoder_dim: int = 256
    decoder_heads: int = 8
    decoder_layers: int = 6
    sinkhorn_iterations: int = 20
    
    # Generation
    generation_steps: int = 256
    top_k: int = 0
    top_p: float = 1.0
    
    # Additional params
    milestone: int = 0
    img_size: int = 224
    
    class Config:
        arbitrary_types_allowed = True