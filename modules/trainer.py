import os
from pathlib import Path

import copy
import glob
import threading

import torch
import torch.nn.functional as F

import wandb

from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, device, ssim_module, 
                 checkpoint_dir, scheduler_patience=5, max_checkpoints=10):
        """
        Initialize trainer with combined training, evaluation and scheduling capabilities.
        
        Args:
            model: VQVAE model
            optimizer: Optimizer
            device: Training device
            ssim_module: SSIM loss calculator
            checkpoint_dir: Directory to save checkpoints
            scheduler_patience: Epochs to wait before reducing LR
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.ssim_module = ssim_module
        self.best_ssim = -float('inf')

        self.initialize = True
        self.initialize_counter = 0

        self.current_epoch = 0
        self.last_eval_epoch = 0
        self.eval_interval = 10
        
        # Initialize checkpoint manager
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
        # Initialize LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=scheduler_patience, 
            verbose=True, min_lr=1e-6
        )
        
        # Initialize running metrics
        self.reset_running_metrics()

    def load_checkpoint(self, checkpoint_path=None):
        """
        Load model checkpoint.
        If no path specified, tries to load latest checkpoint.
        """
        if checkpoint_path is None:
            # Try to find latest checkpoint
            checkpoints = glob.glob(str(self.checkpoint_dir / 'checkpoint_epoch_*.pth'))
            if not checkpoints:
                print("No checkpoints found.")
                return False
                
            # Get latest checkpoint by epoch number
            latest_checkpoint = max(checkpoints, 
                key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
            checkpoint_path = latest_checkpoint
            
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_ssim = checkpoint.get('best_ssim', self.best_ssim)
            
            print(f"Successfully loaded checkpoint from epoch {self.current_epoch}")
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return False

    def load_best_model(self):
        """Load the best performing model"""
        best_path = self.checkpoint_dir / 'best_model.pth'
        if best_path.exists():
            return self.load_checkpoint(best_path)
        return False

    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        """Set learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_running_metrics(self):
        """Reset running average metrics"""
        self.running_metrics = {
            'loss': 0,
            'ssim': 0,
            'mae': 0,
            'rmse': 0,
            'mape': 0,
            'psnr': 0,
            'perplexity': 0
        }
        self.batch_count = 0

    def update_running_metrics(self, metrics):
        """Update running average metrics"""
        self.batch_count += 1
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.running_metrics[key] = (self.running_metrics[key] * (self.batch_count - 1) + value) / self.batch_count

    def calculate_metrics(self, pred, target):
        """Calculate evaluation metrics"""
        mae = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        rmse = torch.sqrt(mse)
        epsilon = 1e-8
        mape = torch.mean(torch.abs((target - pred) / (target + epsilon))) * 100
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return mae, rmse, mape, psnr

    def calculate_losses(self, reconstruction, uncertainty, target):
        """Calculate all losses and metrics"""
        reconstruction_errors = F.mse_loss(reconstruction, target, reduction='none')
        recon_loss = (reconstruction_errors * uncertainty.detach()).mean()
        mae, rmse, mape, psnr = self.calculate_metrics(reconstruction, target)
        ssim_loss = 1 - self.ssim_module(reconstruction, target)
        return recon_loss, mae, rmse, mape, psnr, ssim_loss

    def save_checkpoint(self, epoch, ssim, is_best=False):
        """Save model checkpoint"""
        checkpoint_name = f'checkpoint_epoch_{epoch}.pth'
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ssim': ssim,
            'best_ssim': self.best_ssim
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.best_ssim = ssim
            print(f"\nNew best model saved! SSIM: {ssim:.4f}")

    def cleanup_old_checkpoints(self):
        """Keep only recent checkpoints plus best"""
        checkpoints = glob.glob(str(self.checkpoint_dir / 'checkpoint_epoch_*.pth'))
        if len(checkpoints) <= self.max_checkpoints:
            return
            
        checkpoint_epochs = []
        for ckpt in checkpoints:
            try:
                epoch = int(ckpt.split('_epoch_')[1].split('.')[0])
                checkpoint_epochs.append((epoch, ckpt))
            except:
                continue
                
        checkpoint_epochs.sort(reverse=True)
        to_keep = {ckpt for _, ckpt in checkpoint_epochs[:self.max_checkpoints]}
        to_keep.add(str(self.checkpoint_dir / 'best_model.pth'))
        
        for ckpt in checkpoints:
            if ckpt not in to_keep:
                try:
                    os.remove(ckpt)
                except Exception as e:
                    print(f"Error removing checkpoint {ckpt}: {e}")

    def start_evaluation(self, val_loader, state_dict):
        """Start background evaluation thread"""
        # Create a copy of model for evaluation to avoid affecting training
        eval_model = copy.deepcopy(self.model).to(self.device)
        eval_model.load_state_dict(state_dict)
        
        self.eval_model = eval_model
        self.current_val_loader = val_loader
        self.thread = threading.Thread(target=self.evaluation_loop)
        self.thread.start()

    def is_evaluating(self):
        """Check if evaluation is currently running"""
        return self.thread is not None and self.thread.is_alive()

    def get_eval_results(self):
        """Get evaluation results if available"""
        if self.thread and not self.thread.is_alive():
            try:
                results = self.eval_results
                self.thread = None
                self.eval_results = None
                return results
            except:
                return None
        return None

    def evaluation_loop(self):
        """Background evaluation process"""
        self.eval_model.eval()
        self.reset_running_metrics()
        
        try:
            with torch.no_grad():
                with tqdm(self.current_val_loader, desc='Evaluating') as pbar:
                    for batch_idx, (data, _) in enumerate(pbar):
                        [input_res, target_res] = data
                        input_res = input_res.to(self.device)
                        target_res = target_res.to(self.device)
                        
                        reconstruction, vq_loss, perplexity, uncertainty = self.eval_model.train_forward(input_res, target_res)
                        recon_loss, mae, rmse, mape, psnr, ssim_loss = self.calculate_losses(
                            reconstruction, uncertainty, target_res
                        )
                        
                        metrics = {
                            'loss': recon_loss,
                            'ssim': 1 - ssim_loss.item(),
                            'mae': mae,
                            'rmse': rmse,
                            'mape': mape,
                            'psnr': psnr,
                            'perplexity': perplexity
                        }
                        self.update_running_metrics(metrics)
                        
                        pbar.set_postfix({
                            'Avg SSIM': f'{self.running_metrics["ssim"]:.4f}',
                            'Avg MAE': f'{self.running_metrics["mae"]:.4f}',
                            'Avg PSNR': f'{self.running_metrics["psnr"]:.2f}'
                        })
            
            self.eval_results = self.running_metrics
            
        except Exception as e:
            print(f"Error in background evaluation: {str(e)}")
            self.eval_results = None

    def train_epoch(self, train_loader, val_loader, epoch):
        """Train for one epoch with background evaluation"""
        self.model.train()
        self.reset_running_metrics()
        
        # Start evaluation if needed
        if not self.is_evaluating() and epoch - self.last_eval_epoch >= self.eval_interval:
            print(f"\nStarting background evaluation at epoch {epoch}")
            self.evaluator.start_evaluation(val_loader, self.model.state_dict())
            self.last_eval_epoch = epoch

            print(f"\nStarting background evaluation at epoch {epoch}")
            self.start_evaluation(val_loader, self.model.state_dict())
        
        with tqdm(train_loader, desc=f'Epoch {epoch} (lr: {self.get_lr():.2e})') as pbar:
            for batch_idx, (data, _) in enumerate(pbar):
                # Warmup period
                if self.initialize and self.initialize_counter < 50 and batch_idx > 10:
                    self.initialize_counter += 1
                    if self.initialize_counter == 50:
                        self.initialize = False
                    break

                # Get data
                [input_res, target_res] = data
                input_res = input_res.to(self.device)
                target_res = target_res.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                reconstruction, vq_loss, perplexity, uncertainty = self.model.train_forward(input_res, target_res)
                
                # Calculate losses
                recon_loss, mae, rmse, mape, psnr, ssim_loss = self.calculate_losses(
                    reconstruction, uncertainty, target_res
                )
                
                # Total loss
                loss = recon_loss + vq_loss + 100 * ssim_loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                metrics = {
                    'loss': loss,
                    'ssim': 1 - ssim_loss.item(),
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'psnr': psnr,
                    'perplexity': perplexity
                }
                self.update_running_metrics(metrics)
                
                # Check for evaluation results
                eval_results = self.get_eval_results()
                if eval_results is not None:
                    self.scheduler.step(eval_results['ssim'])
                    if eval_results['ssim'] > self.best_ssim:
                        self.save_checkpoint(epoch, eval_results['ssim'], is_best=True)
                    
                    # Start new evaluation
                    print(f"\nStarting background evaluation at epoch {epoch}")
                    self.start_evaluation(val_loader, self.model.state_dict())
                
                # Update progress bar
                postfix = {
                    'Avg Loss': f'{self.running_metrics["loss"]:.4f}',
                    'Avg SSIM': f'{self.running_metrics["ssim"]:.4f}',
                    'Avg MAE': f'{self.running_metrics["mae"]:.4f}',
                    'Avg RMSE': f'{self.running_metrics["rmse"]:.4f}',
                    'Avg PSNR': f'{self.running_metrics["psnr"]:.2f}',
                }
                if eval_results:
                    postfix['Val SSIM'] = f"{eval_results['ssim']:.4f}"
                pbar.set_postfix(postfix)
                
                # Log to wandb
                wandb.log({
                    'epoch': epoch,
                    'loss': loss.item(),
                    'ssim': 1 - ssim_loss.item(),
                    'mae': mae.item(),
                    'rmse': rmse.item(),
                    'psnr': psnr.item(),
                    'mape': mape.item(),
                    'perplexity': perplexity.item(),
                    'lr': self.get_lr()
                })
                
        # Regular checkpoint saving
        if epoch % 10 == 0:
            self.save_checkpoint(epoch, self.running_metrics['ssim'])
            
        self.cleanup_old_checkpoints()
        
        return self.running_metrics['loss'], self.running_metrics['ssim']

    def cleanup(self):
        """Cleanup background evaluation resources"""
        if self.is_evaluating():
            self.thread.join()