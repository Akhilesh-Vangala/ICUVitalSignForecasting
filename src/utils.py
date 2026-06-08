"""Utility functions for ICU vital sign forecasting."""

import torch
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Any


def setup_logging(log_dir: str = 'logs'):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def save_config(config: Dict[str, Any], filepath: str):
    """Save configuration to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         epoch: int, loss: float, filepath: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)


def load_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         filepath: str, device: str = 'cpu'):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
