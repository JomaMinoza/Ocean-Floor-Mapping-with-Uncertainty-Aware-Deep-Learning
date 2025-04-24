import os
from pathlib import Path
import copy
import glob
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import wandb
from tqdm import tqdm

from .uncertainty import BlockUncertaintyTracker

class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualAttentionBlock, self).__init__()

        # Trunk branch
        self.trunk = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(in_channels, in_channels, kernel_size=9, padding=0),
            nn.SiLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0),
            nn.SiLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0)
        )

        # Mask branch
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mask = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(in_channels, in_channels, kernel_size=9, padding=0),
            nn.SiLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0),
            nn.SiLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.sigmoid = nn.Sigmoid()
        self.silu = nn.SiLU()

    def forward(self, x):

        # Trunk branch
        trunk_output = self.trunk(x)

        # Mask branch
        mask = self.downsample(x)
        mask = self.mask(mask)
        mask = self.upsample(mask)
        mask = self.sigmoid(mask)

        # Attention + residual connection
        #print(x.shape, trunk_output.shape, mask.shape)
        out = x + mask * trunk_output
        return self.silu(out)   

class VectorQuantizer(nn.Module):
    def __init__(
            self,
            n_embeddings=1024,
            embedding_dim=1024,
            beta=0.25,
            use_ema=True,
            decay=0.99,
            eps=1e-5,
            initial_temperature=1.0,
            min_temperature=0.1,
            anneal_rate=0.001,
            diversity_factor=0.8
        ):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.use_ema = use_ema
        self.decay = decay
        self.eps = eps
        self.diversity_factor = diversity_factor

        # Temperature annealing parameters
        self.register_buffer('temperature', torch.tensor(initial_temperature))
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate

        # Initialize embeddings
        self.embeddings = nn.Parameter(torch.randn(n_embeddings, embedding_dim))
        nn.init.orthogonal_(self.embeddings)

        # Initialize EMA buffers if using EMA
        if self.use_ema:
            self.register_buffer('ema_cluster_size', torch.zeros(n_embeddings))
            self.register_buffer('ema_w', torch.zeros_like(self.embeddings))
            self.register_buffer('usage_count', torch.zeros(n_embeddings))

    def _compute_distances(self, flat_x):
        """Compute distances between input vectors and codebook embeddings"""
        return (torch.sum(flat_x**2, dim=1, keepdim=True) +
                torch.sum(self.embeddings**2, dim=1) -
                2 * torch.matmul(flat_x, self.embeddings.t()))

    def _get_soft_assignments(self, distances):
        """Compute temperature-scaled soft assignments"""
        return F.softmax(-distances / self.temperature, dim=1)

    def _get_hard_assignments(self, distances):
        """Compute hard assignments using argmin"""
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self.n_embeddings,
            device=distances.device
        )
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        return encodings

    def _update_ema(self, encodings, flat_x):
        """Update EMA parameters for codebook learning"""
        with torch.no_grad():
            # Update cluster size EMA
            self.ema_cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, encodings.sum(0)
            )

            # Compute normalized cluster size
            n = self.ema_cluster_size.sum()
            cluster_size = ((self.ema_cluster_size + self.eps) /
                          (n + self.n_embeddings * self.eps) * n)

            # Update embedding EMA
            dw = torch.matmul(encodings.t(), flat_x)
            self.ema_w.data.mul_(self.decay).add_(1 - self.decay, dw)

            # Update embeddings based on EMA
            self.embeddings.data = self.ema_w / cluster_size.unsqueeze(1)

    def _update_codebook_usage(self, encodings):
        """Track codebook usage statistics"""
        with torch.no_grad():
            self.usage_count.mul_(0.99).add_(encodings.sum(0), alpha=0.01)

    def _compute_losses(self, x, quantized, encodings):
        """Compute all loss terms"""
        # Commitment and codebook losses
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())

        # Codebook regularization
        reg_loss = torch.sum(self.embeddings**2)

        # Diversity losses
        avg_probs = torch.mean(encodings, dim=0)
        entropy_loss = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        usage_prob = self.usage_count / (self.usage_count.sum() + 1e-5)
        diversity_loss = -torch.sum(usage_prob * torch.log(usage_prob + 1e-10))

        # Calculate perplexity
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Combine losses
        loss = (q_latent_loss +
                self.beta * e_latent_loss +
                reg_loss +
                self.diversity_factor * (entropy_loss + diversity_loss))

        return loss, perplexity

    def _anneal_temperature(self):
        """Update temperature parameter"""
        self.temperature.data = torch.max(
            self.temperature * (1. - self.anneal_rate),
            torch.tensor(self.min_temperature, device=self.temperature.device)
        )

    def _quantize(self, encodings):
        """Convert encodings to quantized vectors"""
        return torch.matmul(encodings, self.embeddings)

    def training_step(self, x):
        """Forward pass during training using soft assignments"""
        # Reshape input
        flat_x = x.reshape(-1, self.embedding_dim)

        # Compute distances and soft assignments
        distances = self._compute_distances(flat_x)
        encodings = self._get_soft_assignments(distances)

        # Quantize
        quantized = self._quantize(encodings)
        quantized = quantized.view(x.shape)

        # Update tracking statistics
        if self.use_ema:
            self._update_ema(encodings, flat_x)
            self._update_codebook_usage(encodings)

        # Compute losses
        loss, perplexity = self._compute_losses(x, quantized, encodings)

        # Anneal temperature
        self._anneal_temperature()

        # Straight through estimator
        quantized = x + (quantized - x).detach()

        return quantized, loss, perplexity

    def inference_step(self, x):
        """Forward pass during inference using hard assignments"""
        # Reshape input
        flat_x = x.reshape(-1, self.embedding_dim)

        # Compute distances and hard assignments
        distances = self._compute_distances(flat_x)
        encodings = self._get_hard_assignments(distances)

        # Quantize
        quantized = self._quantize(encodings)
        quantized = quantized.view(x.shape)

        return quantized

    def forward(self, x):
        """Main forward pass, dispatches between training and inference"""
        if self.training:
            return self.training_step(x)
        else:
            return self.inference_step(x)

    def get_codebook_stats(self):
        """Return current codebook statistics"""
        stats = {
            'temperature': self.temperature.item(),
            'perplexity': torch.exp(-torch.sum(
                self.usage_count * torch.log(self.usage_count + 1e-10)
            )).item(),
            'usage_entropy': -torch.sum(
                self.usage_count * torch.log(self.usage_count + 1e-10)
            ).item(),
        }
        return stats

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, embedding_dim):
        super().__init__()
        
        # Initial convolution while maintaining spatial dims
        self.input_conv = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(in_channels, hidden_dims[0], 9, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.SiLU()
        )
        
        # Process through hidden dimensions
        layers = []
        current_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.extend([
                nn.ReflectionPad2d(1),
                nn.Conv2d(current_channels, h_dim, 3, stride=1, padding=0),
                nn.BatchNorm2d(h_dim),
                nn.SiLU()
            ])
            current_channels = h_dim
            
        self.hidden_layers = nn.Sequential(*layers)

        # Attention processing
        self.process = nn.Sequential(
            ResidualAttentionBlock(hidden_dims[-1]),
            ResidualAttentionBlock(hidden_dims[-1]),
            ResidualAttentionBlock(hidden_dims[-1])
        )

        # Project to embedding dimension
        self.project = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dims[-1], embedding_dim, 3, padding=0),
            nn.BatchNorm2d(embedding_dim)
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.hidden_layers(x)
        x = self.process(x)
        return self.project(x)

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dims, out_channels):
        super().__init__()
        current_channels = embedding_dim
        hidden_dims = hidden_dims[::-1]  # Reverse hidden dims for decoder

        # Initial processing at 32x32
        decoder_layers = []
        for h_dim in hidden_dims:
            decoder_layers.extend([
                nn.ReflectionPad2d(1),
                nn.Conv2d(current_channels, h_dim, kernel_size=3, padding=0),
                nn.BatchNorm2d(h_dim),
                ResidualAttentionBlock(h_dim)
            ])
            current_channels = h_dim

        self.initial_decoder = nn.Sequential(*decoder_layers)

        # Upsampling block to 64x64
        self.upsample = nn.Sequential(
            ResidualAttentionBlock(hidden_dims[-1]),
            ResidualAttentionBlock(hidden_dims[-1]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=0),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.SiLU()
        )

        # Final output layer
        self.final = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=9, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)

    def _forward_impl(self, x):
        # Initial processing at 32x32
        x = self.initial_decoder(x)
        
        # Upsample to 64x64
        x = self.upsample(x)
        
        # Final conv to output channels
        return self.final(x)

class VQVAE(nn.Module):

    def __init__(self, in_channels=1, hidden_dims=[64, 128, 256],
                 num_embeddings=512, embedding_dim=256, alpha=0.05, block_size=4):
        super().__init__()

        self.use_checkpointing = True
        
        # Initialize block-wise uncertainty tracking
        self.uncertainty_tracker = BlockUncertaintyTracker(
            block_size=block_size,
            alpha=alpha,
            decay=0.99,
            eps=1e-5
        )

        # Main model components
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim
        )

        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            out_channels=in_channels
        )

        self.vq = VectorQuantizer(
            n_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            beta=0.25,
            use_ema=True,
            decay=0.99,
            eps=1e-5,
            initial_temperature=1.0,
            min_temperature=0.1,
            anneal_rate=0.001,
            diversity_factor=0.8
        )

    def calibrate(self, loader, n_batches=100):
        """Calibrate uncertainty tracker with representative data"""
        self.eval()  # Set to eval mode during calibration
        self.uncertainty_tracker.calibrate(loader, n_batches)

    def forward(self, x):
        """Standard forward pass"""
        # Encode
        z = self.encoder(x)

        # Vector quantization
        if self.training:
            z_q, loss, perplexity = self.vq(z)
        else:
            z_q = self.vq(z)

        # Decode
        reconstructed = self.decoder(z_q)

        if self.training:
            return reconstructed, loss, perplexity
        return reconstructed

    def train_forward(self, x, y):
        """Training forward pass with uncertainty tracking"""
        # Get reconstruction
        reconstruction, loss, perplexity = self.forward(x)

        # Calculate reconstruction error
        error = torch.abs(reconstruction - y)

        # Update uncertainty tracker
        self.uncertainty_tracker.update(error)

        # Get uncertainty map for loss weighting
        uncertainty_map = self.uncertainty_tracker.get_uncertainty(error)

        return reconstruction, loss, perplexity, uncertainty_map

    def predict_with_uncertainty(self, x, confidence_level=0.95):
        """
        Forward pass with calibrated uncertainty bounds
        Args:
            x: Input tensor
            confidence_level: Confidence level for bounds (default: 0.95)
        Returns:
            tuple: (reconstruction, lower_bounds, upper_bounds)
        """
        if not self.uncertainty_tracker.calibrated:
            print("Warning: Model not calibrated. Call calibrate() first.")
            return self.forward(x), None, None

        self.eval()
        with torch.no_grad():
            # Get reconstruction
            reconstruction = self.forward(x)
            
            # Get calibrated uncertainty bounds
            lower_bounds, upper_bounds = self.uncertainty_tracker.get_bounds(
                reconstruction, confidence_level
            )
            
            return reconstruction, lower_bounds, upper_bounds

    def save_calibration(self, path):
        """Save calibration statistics"""
        if not self.uncertainty_tracker.calibrated:
            print("Warning: No calibration to save.")
            return
            
        torch.save({
            'block_scale_means': self.uncertainty_tracker.block_scale_means,
            'block_scale_stds': self.uncertainty_tracker.block_scale_stds,
            'calibrated': True
        }, path)

    def load_calibration(self, path):
        """Load calibration statistics"""
        calibration = torch.load(path)
        self.uncertainty_tracker.block_scale_means = calibration['block_scale_means']
        self.uncertainty_tracker.block_scale_stds = calibration['block_scale_stds']
        self.uncertainty_tracker.calibrated = calibration['calibrated']

class Trainer:
    def __init__(self, model, optimizer, device, ssim_module,
                 checkpoint_dir, eval_interval=10, scheduler_patience=5, max_checkpoints=10):
        """
        Initialize trainer with combined training, evaluation and scheduling capabilities.
        
        Args:
            model: VQVAE model
            optimizer: Optimizer 
            device: Training device
            ssim_module: SSIM loss calculator
            checkpoint_dir: Directory to save checkpoints
            eval_interval: Epochs between evaluations
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
        self.eval_interval = eval_interval

        # Initialize checkpoint manager
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints

        # Initialize evaluation tracking
        self.thread = None
        self.eval_results = None
        self.current_val_loader = None
        self.eval_model = None

        # Initialize LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=scheduler_patience,
            verbose=True, min_lr=1e-6
        )

        # Initialize block-wise uncertainty tracking
        self.block_size = self.model.uncertainty_tracker.block_size
        self.reset_running_metrics()

    def reset_running_metrics(self):
        """Reset running average metrics"""
        self.running_metrics = {
            'loss': 0,
            'ssim': 0,
            'mae': 0,
            'rmse': 0,
            'mape': 0,
            'psnr': 0,
            'perplexity': 0,
            'mean_block_mse': 0,
            'max_block_mse': 0
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

    def calculate_block_metrics(self, reconstruction, target):
        """Calculate metrics for each block"""
        B, C, H, W = reconstruction.shape
        
        # Use unfold to get blocks
        unfold = torch.nn.Unfold(kernel_size=self.block_size, stride=self.block_size)
        recon_blocks = unfold(reconstruction)  # [B, C*block_size*block_size, num_blocks]
        target_blocks = unfold(target)
        
        # Reshape to [B, num_blocks, block_area]
        recon_blocks = recon_blocks.transpose(1, 2)
        target_blocks = target_blocks.transpose(1, 2)
        
        # Calculate per-block metrics
        block_mse = torch.mean((recon_blocks - target_blocks)**2, dim=-1)  # [B, num_blocks]
        block_mae = torch.mean(torch.abs(recon_blocks - target_blocks), dim=-1)
        
        return {
            'block_mse': block_mse,
            'block_mae': block_mae
        }

    def calculate_losses(self, reconstruction, uncertainty_map, target):
        """Calculate losses with block-wise uncertainty"""
        # Basic reconstruction error
        reconstruction_errors = F.mse_loss(reconstruction, target, reduction='none')
        
        # Weight errors by uncertainty map
        weighted_errors = reconstruction_errors * uncertainty_map.detach()
        recon_loss = weighted_errors.mean()
        
        # Calculate other metrics
        mae, rmse, mape, psnr = self.calculate_metrics(reconstruction, target)
        ssim_loss = 1 - self.ssim_module(reconstruction, target)
        
        # Calculate block-wise metrics
        block_metrics = self.calculate_block_metrics(reconstruction, target)
        
        return recon_loss, mae, rmse, mape, psnr, ssim_loss, block_metrics

    def load_checkpoint(self, checkpoint_path=None):
        """Load model checkpoint including calibration state"""
        if checkpoint_path is None:
            checkpoints = glob.glob(str(self.checkpoint_dir / 'checkpoint_epoch_*.pth'))
            if not checkpoints:
                print("No checkpoints found.")
                return False
                
            latest_checkpoint = max(checkpoints,
                key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
            checkpoint_path = latest_checkpoint
            
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            
            self.model.load_state_dict(checkpoint['state_dict'], strict = False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                
            # Load calibration state if available
            if 'calibration_state' in checkpoint:
                cal_state = checkpoint['calibration_state']
                if cal_state['calibrated']:
                    self.model.uncertainty_tracker.block_scale_means = cal_state['block_scale_means']
                    self.model.uncertainty_tracker.block_scale_stds = cal_state['block_scale_stds']
                    self.model.uncertainty_tracker.calibrated = True
                    print("Loaded calibration state")
                
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

    def save_checkpoint(self, epoch, ssim, is_best=False):
        """Save model checkpoint including calibration state"""
        checkpoint_name = f'checkpoint_epoch_{epoch}.pth'
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ssim': ssim,
            'best_ssim': self.best_ssim,

            # Save calibration state
            'calibration_state': {
                'block_scale_means': self.model.uncertainty_tracker.block_scale_means 
                    if hasattr(self.model.uncertainty_tracker, 'block_scale_means') else None,
                'block_scale_stds': self.model.uncertainty_tracker.block_scale_stds 
                    if hasattr(self.model.uncertainty_tracker, 'block_scale_stds') else None,
                'calibrated': self.model.uncertainty_tracker.calibrated 
                    if hasattr(self.model.uncertainty_tracker, 'calibrated') else False
            }
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

    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

    def set_lr(self, lr):
        """Set learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def start_evaluation(self, val_loader, state_dict):
        """Start background evaluation thread"""
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
        """Background evaluation process with uncertainty bounds"""
        self.eval_model.eval()
        self.reset_running_metrics()
        
        try:
            with torch.no_grad():
                with tqdm(self.current_val_loader, desc='Evaluating') as pbar:
                    for batch_idx, (data, _) in enumerate(pbar):
                        [_, input_res, target_res] = data
                        input_res = input_res.to(self.device)
                        target_res = target_res.to(self.device)
                        
                        # Get reconstruction with uncertainty bounds if model is calibrated
                        if self.eval_model.uncertainty_tracker.calibrated:
                            reconstruction, lower_bounds, upper_bounds = self.eval_model.predict_with_uncertainty(
                                input_res, confidence_level=0.95
                            )
                            uncertainty_width = (upper_bounds - lower_bounds).mean()
                        else:
                            reconstruction, vq_loss, perplexity, uncertainty_map = self.eval_model.train_forward(
                                input_res, target_res
                            )
                            uncertainty_width = uncertainty_map.mean()
                        
                        # Calculate all losses and metrics
                        recon_loss, mae, rmse, mape, psnr, ssim_loss, block_metrics = self.calculate_losses(
                            reconstruction, 
                            uncertainty_map if not self.eval_model.uncertainty_tracker.calibrated else torch.ones_like(reconstruction),
                            target_res
                        )
                        
                        metrics = {
                            'loss': recon_loss,
                            'ssim': 1 - ssim_loss.item(),
                            'mae': mae,
                            'rmse': rmse,
                            'mape': mape,
                            'psnr': psnr,
                            'perplexity': perplexity if 'perplexity' in locals() else 0.0,
                            'mean_block_mse': block_metrics['block_mse'].mean().item(),
                            'max_block_mse': block_metrics['block_mse'].max().item(),
                            'uncertainty_width': uncertainty_width.item()
                        }
                        
                        self.update_running_metrics(metrics)
                        
                        # Update progress bar with uncertainty info
                        pbar.set_postfix({
                            'Avg SSIM': f'{self.running_metrics["ssim"]:.4f}',
                            'Avg MAE': f'{self.running_metrics["mae"]:.4f}',
                            'Avg PSNR': f'{self.running_metrics["psnr"]:.2f}',
                            'Uncertainty': f'{self.running_metrics["uncertainty_width"]:.4f}'
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
            self.start_evaluation(val_loader, self.model.state_dict())
            self.last_eval_epoch = epoch
            
        with tqdm(train_loader, desc=f'Epoch {epoch} (lr: {self.get_lr():.2e})') as pbar:
            for batch_idx, (data, _) in enumerate(pbar):
                # Warmup period
                if self.initialize and self.initialize_counter < 50 and batch_idx > 10:
                    self.initialize_counter += 1
                    if self.initialize_counter == 50:
                        self.initialize = False
                    break
                    
                # Get data
                [_, input_res, target_res] = data
                input_res = input_res.to(self.device)
                target_res = target_res.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                reconstruction, vq_loss, perplexity, uncertainty_map = self.model.train_forward(
                    input_res, target_res
                )
                
                # Calculate all losses and metrics
                recon_loss, mae, rmse, mape, psnr, ssim_loss, block_metrics = self.calculate_losses(
                    reconstruction, uncertainty_map, target_res
                )
                
                # Total loss
                loss = recon_loss + 0.001 * vq_loss + 10 * ssim_loss 
                value  = recon_loss + 0.1 * vq_loss + 10 * ssim_loss 
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics including block-wise metrics
                metrics = {
                    'loss': value,
                    'ssim': 1 - ssim_loss.item(),
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'psnr': psnr,
                    'perplexity': perplexity,
                    'mean_block_mse': block_metrics['block_mse'].mean().item(),
                    'max_block_mse': block_metrics['block_mse'].max().item()
                }
                
                self.update_running_metrics(metrics)
                
                # Check for evaluation results
                #eval_results = self.get_eval_results()
                eval_results = None
                if eval_results is not None:
                    self.scheduler.step(eval_results['ssim'])
                    if eval_results['ssim'] > self.best_ssim:
                        self.save_checkpoint(epoch, eval_results['ssim'], is_best=True)
                        
                # Update progress bar
                postfix = {
                    'Avg Loss': f'{self.running_metrics["loss"]:.4f}',
                    'Avg SSIM': f'{self.running_metrics["ssim"]:.4f}',
                    'Avg MAE': f'{self.running_metrics["mae"]:.4f}',
                    'Avg RMSE': f'{self.running_metrics["rmse"]:.4f}',
                    'Avg MAPE': f'{self.running_metrics["mape"]:.4f}',
                    'Avg PSNR': f'{self.running_metrics["psnr"]:.2f}',
                    'Avg Perplexity': f'{self.running_metrics["perplexity"]:.4f}',
                    'Max Block MSE': f'{self.running_metrics["max_block_mse"]:.4f}',
                    'Mean Block MSE': f'{self.running_metrics["mean_block_mse"]:.4f}',
                    'Best SSIM': f'{self.best_ssim:.4f}'
                }
                
                if eval_results:
                    postfix['Val SSIM'] = f"{eval_results['ssim']:.4f}"
                pbar.set_postfix(postfix)
                
                # Log to wandb
                wandb.log({
                    'epoch': epoch,
                    'loss': value.item(),
                    'recon_loss': recon_loss.item(),
                    'vq_loss': vq_loss.item(),
                    'ssim': 1 - ssim_loss.item(),
                    'mae': mae.item(),
                    'rmse': rmse.item(),
                    'psnr': psnr.item(),
                    'mape': mape.item(),
                    'perplexity': perplexity.item(),
                    'mean_block_mse': metrics['mean_block_mse'],
                    'max_block_mse': metrics['max_block_mse'],
                    'lr': self.get_lr()
                })
        

        current_ssim = self.running_metrics['ssim']
        if current_ssim > self.best_ssim:
            self.best_ssim = current_ssim
            self.save_checkpoint(epoch, current_ssim, is_best=True)


        # Regular checkpoint saving
        if epoch % 10 == 0:
            self.save_checkpoint(epoch, self.running_metrics['ssim'])
            
        self.cleanup_old_checkpoints()
        
        return self.running_metrics['loss'], self.running_metrics['ssim']

    def cleanup(self):
        """Cleanup background evaluation resources"""
        if self.is_evaluating():
            self.thread.join()