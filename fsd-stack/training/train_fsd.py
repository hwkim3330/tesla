"""
Tesla FSD Training Script

Train the complete FSD network on driving datasets.

Usage:
    python train_fsd.py --config configs/fsd_config.yaml

Supports:
- Multi-GPU training
- Mixed precision
- Wandb/Tensorboard logging
- Checkpointing
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class FSDTrainer:
    """
    Trainer for Tesla FSD network.

    Handles:
    - Data loading
    - Training loop
    - Validation
    - Checkpointing
    - Logging
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_logging()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

    def _setup_model(self):
        """Initialize model."""
        from models.e2e import FSDNetwork

        model_cfg = self.config['model']

        self.model = FSDNetwork(
            backbone=model_cfg['backbone'],
            pretrained=model_cfg['pretrained'],
            bev_size=tuple(model_cfg['bev_size']),
            bev_range=model_cfg['bev_range'],
            num_cameras=model_cfg.get('num_cameras', 6),
            temporal_frames=model_cfg.get('temporal_frames', 4),
        )

        self.model = self.model.to(self.device)

        # Multi-GPU
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        print(f"Model initialized on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")

    def _setup_data(self):
        """Setup data loaders."""
        data_cfg = self.config['data']

        # For demo, use synthetic data
        # In production, use actual dataset loaders
        self.train_loader = self._create_synthetic_loader(
            batch_size=data_cfg['batch_size'],
            num_samples=1000,
        )
        self.val_loader = self._create_synthetic_loader(
            batch_size=data_cfg['batch_size'],
            num_samples=100,
        )

        print(f"Data loaders initialized")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

    def _create_synthetic_loader(
        self,
        batch_size: int,
        num_samples: int,
    ) -> DataLoader:
        """Create synthetic data loader for testing."""
        from torch.utils.data import Dataset

        class SyntheticDataset(Dataset):
            def __init__(self, num_samples, img_size=(480, 640), num_cameras=6):
                self.num_samples = num_samples
                self.img_size = img_size
                self.num_cameras = num_cameras

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                # Synthetic multi-camera images
                images = torch.randn(self.num_cameras, 3, *self.img_size)

                # Camera parameters
                intrinsics = torch.eye(3).unsqueeze(0).expand(self.num_cameras, 3, 3).clone()
                intrinsics[:, 0, 0] = self.img_size[1]  # fx
                intrinsics[:, 1, 1] = self.img_size[0]  # fy
                intrinsics[:, 0, 2] = self.img_size[1] / 2
                intrinsics[:, 1, 2] = self.img_size[0] / 2

                extrinsics = torch.eye(4).unsqueeze(0).expand(self.num_cameras, 4, 4).clone()

                # Targets (simplified)
                targets = {
                    'trajectory': torch.randn(10, 3),
                    'steering': torch.randn(1),
                    'throttle': torch.rand(1),
                    'brake': torch.rand(1),
                }

                return {
                    'images': images,
                    'intrinsics': intrinsics,
                    'extrinsics': extrinsics,
                    'targets': targets,
                }

        dataset = SyntheticDataset(num_samples)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        train_cfg = self.config['training']

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay'],
        )

        # Scheduler with warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=train_cfg['warmup_epochs'],
        )

        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg['epochs'] - train_cfg['warmup_epochs'],
            eta_min=train_cfg['min_lr'],
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[train_cfg['warmup_epochs']],
        )

        # Mixed precision
        self.scaler = GradScaler() if train_cfg['precision'] == 16 else None

    def _setup_logging(self):
        """Setup logging."""
        log_cfg = self.config['logging']

        self.use_wandb = log_cfg.get('use_wandb', False)
        self.use_tensorboard = log_cfg.get('use_tensorboard', True)

        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=log_cfg['project'],
                    config=self.config,
                )
            except ImportError:
                print("wandb not installed, skipping")
                self.use_wandb = False

        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    def train(self):
        """Main training loop."""
        train_cfg = self.config['training']
        num_epochs = train_cfg['epochs']

        print(f"\nStarting training for {num_epochs} epochs...")

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            # Train epoch
            train_loss = self._train_epoch()

            # Validate
            if (epoch + 1) % self.config['evaluation']['eval_every'] == 0:
                val_metrics = self._validate()
                self._log_metrics(val_metrics, prefix='val')

            # Save checkpoint
            if (epoch + 1) % train_cfg['save_every'] == 0:
                self._save_checkpoint()

            # Update scheduler
            self.scheduler.step()

            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")

        print("\nTraining complete!")

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")

        for batch in pbar:
            # Move to device
            images = batch['images'].to(self.device)
            intrinsics = batch['intrinsics'].to(self.device)
            extrinsics = batch['extrinsics'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}

            # Forward pass
            self.optimizer.zero_grad()

            if self.scaler:
                with autocast():
                    outputs = self.model(images, intrinsics, extrinsics)
                    loss = self._compute_loss(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, intrinsics, extrinsics)
                loss = self._compute_loss(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

            # Log
            if self.global_step % self.config['logging']['log_every'] == 0:
                self._log_metrics({'train_loss': loss.item()}, prefix='train')

        return total_loss / len(self.train_loader)

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute training loss."""
        loss_cfg = self.config['losses']
        total_loss = 0.0

        # Trajectory loss
        if 'trajectory' in outputs and 'trajectory' in targets:
            traj_loss = F.smooth_l1_loss(outputs['trajectory'], targets['trajectory'])
            total_loss += loss_cfg['trajectory_loss'] * traj_loss

        # Control losses
        if 'steering' in outputs and 'steering' in targets:
            steer_loss = F.mse_loss(outputs['steering'], targets['steering'].squeeze())
            total_loss += loss_cfg['steering_loss'] * steer_loss

        if 'throttle' in outputs and 'throttle' in targets:
            throttle_loss = F.mse_loss(outputs['throttle'], targets['throttle'].squeeze())
            total_loss += loss_cfg['throttle_loss'] * throttle_loss

        if 'brake' in outputs and 'brake' in targets:
            brake_loss = F.mse_loss(outputs['brake'], targets['brake'].squeeze())
            total_loss += loss_cfg['brake_loss'] * brake_loss

        return total_loss

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        metrics = {'val_loss': 0.0}

        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch['images'].to(self.device)
            intrinsics = batch['intrinsics'].to(self.device)
            extrinsics = batch['extrinsics'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}

            outputs = self.model(images, intrinsics, extrinsics)
            loss = self._compute_loss(outputs, targets)
            metrics['val_loss'] += loss.item()

        metrics['val_loss'] /= len(self.val_loader)
        return metrics

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ''):
        """Log metrics to wandb/tensorboard."""
        for key, value in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key

            if self.use_tensorboard:
                self.writer.add_scalar(full_key, value, self.global_step)

            if self.use_wandb:
                import wandb
                wandb.log({full_key: value}, step=self.global_step)

    def _save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }

        path = checkpoint_dir / f"checkpoint_epoch{self.epoch+1}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description='Train Tesla FSD Network')
    parser.add_argument('--config', type=str, default='configs/fsd_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create trainer
    trainer = FSDTrainer(config)

    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.epoch = checkpoint['epoch']
        trainer.global_step = checkpoint['global_step']
        print(f"Resumed from {args.resume}")

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
