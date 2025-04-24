import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class BlockUncertaintyTracker(nn.Module):
    def __init__(self, block_size=4, alpha=0.1, decay=0.99, eps=1e-5):
        super().__init__()
        self.block_size = block_size
        self.decay = decay
        self.alpha = alpha
        self.eps = eps

        # Initialize unfold layer for block extraction
        self.unfold = nn.Unfold(kernel_size=block_size, stride=block_size)

        # Register buffers with initial values
        self.register_buffer('ema_errors', None)
        self.register_buffer('ema_quantile', None)

        self.num_blocks_h = None
        self.num_blocks_w = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Calibration statistics
        self.calibrated = False
        self.block_means = []
        self.block_stds = []
        self.block_scale_means = None
        self.block_scale_stds = None

    def _initialize_buffers(self, h, w, device):
        """Initialize EMA buffers based on number of blocks in image"""
        self.num_blocks_h = h // self.block_size
        self.num_blocks_w = w // self.block_size
        num_blocks = self.num_blocks_h * self.num_blocks_w

        # Initialize buffers on the correct device
        self.ema_errors = torch.zeros(num_blocks, device=device)
        self.ema_quantile = torch.zeros(num_blocks, device=device)

    def update(self, current_errors):
        """Update EMA of errors and quantiles for each block"""
        B, C, H, W = current_errors.shape
        device = current_errors.device

        # Initialize buffers if not done yet
        if self.ema_errors is None:
            self._initialize_buffers(H, W, device)

        # Unfold into blocks
        blocks = self.unfold(current_errors)
        block_errors = blocks.transpose(1, 2)
        block_errors = block_errors.reshape(-1, self.num_blocks_h * self.num_blocks_w, self.block_size * self.block_size)

        with torch.no_grad():
            # Compute mean error per block
            block_mean_errors = block_errors.mean(dim=-1)  # [B, num_blocks]

            # Update EMA errors for each block
            block_means = block_mean_errors.mean(dim=0)  # Average across batch
            block_means = block_means.to(device)  # Ensure on correct device
            self.ema_errors = self.ema_errors.to(device)  # Ensure on correct device
            self.ema_errors.mul_(self.decay).add_(block_means * (1 - self.decay))

            # Update quantiles for each block
            block_quantiles = torch.quantile(block_errors, 1 - self.alpha, dim=-1)  # [B, num_blocks]
            quantile_means = block_quantiles.mean(dim=0)  # Average across batch
            quantile_means = quantile_means.to(device)  # Ensure on correct device
            self.ema_quantile = self.ema_quantile.to(device)  # Ensure on correct device
            self.ema_quantile.mul_(self.decay).add_(quantile_means * (1 - self.decay))

    def get_uncertainty(self, errors):
        """Calculate block-wise uncertainty scores with improved interpolation"""
        device = errors.device
        B, C, H, W = errors.shape

        # Ensure buffers are on correct device
        self.ema_errors = self.ema_errors.to(device)
        self.ema_quantile = self.ema_quantile.to(device)

        # Create a more detailed uncertainty map
        uncertainty_map = torch.zeros_like(errors, dtype=torch.float32)

        for i in range(self.num_blocks_h):
            for j in range(self.num_blocks_w):
                # Compute block index
                block_idx = i * self.num_blocks_w + j

                # Get block boundaries
                start_h = i * self.block_size
                start_w = j * self.block_size

                # Extract uncertainty for this block
                block_uncertainty = self.ema_quantile[block_idx] / (self.ema_errors[block_idx] + self.eps)

                # Fill the block with the computed uncertainty
                uncertainty_map[:, :,
                    start_h:start_h+self.block_size,
                    start_w:start_w+self.block_size] = block_uncertainty

        return uncertainty_map

    def get_bounds(self, x, confidence_level=0.95):
        """
        Get prediction bounds with improved spatial uncertainty handling

        Args:
            x: Input tensor [B, C, H, W]
            confidence_level: Confidence level for bounds
        Returns:
            tuple: (lower_bounds, upper_bounds)
        """
        if not self.calibrated:
            print("Warning: Model not calibrated. Bounds may be inaccurate.")
            return None, None

        # Calculate z-score based on confidence level
        z_scores = {
            0.99: 2.576,
            0.95: 1.96,
            0.90: 1.645,
            0.85: 1.440
        }
        z_score = z_scores.get(confidence_level, 1.96)

        # Compute bounds with spatially-aware uncertainty
        B, C, H, W = x.shape

        # Create bounds tensors
        lower_bounds = x.clone()
        upper_bounds = x.clone()

        for i in range(self.num_blocks_h):
            for j in range(self.num_blocks_w):
                # Compute block index
                block_idx = i * self.num_blocks_w + j

                # Get block boundaries
                start_h = i * self.block_size
                start_w = j * self.block_size

                # Get block-specific uncertainty
                block_mean = self.block_scale_means[block_idx]
                block_std = self.block_scale_stds[block_idx]

                # Compute block-specific bounds
                block_uncertainty = z_score * block_std

                # Apply bounds to the specific block
                block_slice = (
                    slice(None),  # batch dimension
                    slice(None),  # channel dimension
                    slice(start_h, start_h+self.block_size),  # height
                    slice(start_w, start_w+self.block_size)   # width
                )

                # Adjust bounds while maintaining valid range
                lower_bounds[block_slice] = torch.clamp(
                    x[block_slice] - block_uncertainty,
                    min=0.0
                )
                upper_bounds[block_slice] = torch.clamp(
                    x[block_slice] + block_uncertainty,
                    max=1.0
                )

        return lower_bounds, upper_bounds

    def calibrate(self, model, loader, n_batches=100):
        """
        Calibrate uncertainty bounds using representative data
        Args:
            loader: DataLoader with calibration data
            n_batches: Number of batches to use for calibration
        """
        print("Calibrating uncertainty bounds...")
        self.block_means = []
        self.block_stds = []

        # Create progress bar
        pbar = tqdm(total=n_batches, desc="Calibrating", unit="batch")

        model.eval()

        with torch.no_grad():
            for batch_idx, (data_list, _) in enumerate(loader):
                if batch_idx >= n_batches:
                    break

                # Get data
                [_, input_res, target_res] = data_list
                input_res = input_res.to(self.device)
                target_res = target_res.to(self.device)

                output_res = model(input_res)

                # Get errors for each block
                errors = self._get_block_errors(output_res, target_res)

                # Store statistics
                self.block_means.append(errors.mean(dim=0))
                self.block_stds.append(errors.std(dim=0))

                # Update progress bar
                pbar.update(1)
                # Show current statistics
                current_means = [f"B{i}:{self.block_means[-1][i]:.3f}" for i in range(min(3, len(self.block_means[-1])))]
                pbar.set_postfix_str(f"Mean Errors: {', '.join(current_means)}...")

        pbar.close()

        # Calculate final statistics
        self.block_scale_means = torch.stack(self.block_means).mean(dim=0)
        self.block_scale_stds = torch.stack(self.block_stds).mean(dim=0)

        self.calibrated = True
        print("\nCalibration complete!")
        print(f"Number of blocks: {len(self.block_scale_means)}")
        print(f"Mean block error range: [{self.block_scale_means.min():.4f}, {self.block_scale_means.max():.4f}]")
        print(f"Mean block std range: [{self.block_scale_stds.min():.4f}, {self.block_scale_stds.max():.4f}]")

    def _get_block_errors(self, x, y):
        """Get errors for each block"""
        error = torch.abs(x - y)
        blocks = self.unfold(error)
        block_errors = blocks.transpose(1, 2)
        return block_errors.mean(dim=-1)  # Average within each block