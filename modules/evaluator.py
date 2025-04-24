import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from pytorch_msssim import SSIM

class MultiModelEvaluator:
    def __init__(self, models_dict, device):
        """
        Args:
            models_dict: Dictionary of model_name: model pairs
            device: torch device
        """
        self.device = device
        self.models = {name: model.to(device).eval() for name, model in models_dict.items()}
        self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=1).to(device)
        self.methods = ['nearest', 'bilinear', 'bicubic']

        # Track which models support uncertainty
        self.uncertainty_capable = {
            name: hasattr(model, 'predict_with_uncertainty')
            for name, model in self.models.items()
        }

    def calculate_metrics(self, pred, target):
        """Calculate basic reconstruction metrics"""
        if isinstance(pred, tuple) or isinstance(pred, list):
            pred = pred[-1]

        mse = F.mse_loss(pred, target).item()
        rmse = np.sqrt(mse)
        mae = F.l1_loss(pred, target).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        ssim = self.ssim_module(pred, target).item()

        return ssim, psnr, mse, rmse, mae

    def calculate_uncertainty_metrics(self, pred, lower, upper, target):
        """Calculate uncertainty-specific metrics"""
        # Handle case where bounds are None
        if lower is None or upper is None:
            return None, None, None

        try:
            # Calculate average uncertainty width
            uncertainty_width = (upper - lower).mean().item()

            # Calculate calibration error (% of true values within bounds)
            in_bounds = ((target >= lower) & (target <= upper)).float().mean().item()
            calibration_error = abs(in_bounds - 0.95)  # Assuming 95% confidence bounds

            calibration_uncertainty_error_score = calibration_error * uncertainty_width

            return uncertainty_width, calibration_uncertainty_error_score

        except Exception as e:
            print(f"Error calculating uncertainty metrics: {str(e)}")
            return None, None, None

    def evaluate(self, dataloader):
        # Initialize metrics dictionaries
        metrics = {
            name: {
                'ssim': [], 'psnr': [], 'mse': [], 'rmse': [], 'mae': [],
                'uncertainty_width': [], 'calibration_error': [],
                'locations': []
            }
            for name in self.models.keys()
        }

        baseline_metrics = {
            method: {
                'ssim': [], 'psnr': [], 'mse': [], 'rmse': [], 'mae': [], 'locations': []
            }
            for method in self.methods
        }

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating Models"):
                [_, low_res_32, high_res], metadata = batch
                low_res = low_res_32.to(self.device)
                high_res = high_res.to(self.device)

                # Evaluate each model
                for name, model in self.models.items():
                    # Handle models with uncertainty capabilities
                    if self.uncertainty_capable[name]:
                        reconstruction, lower_bounds, upper_bounds = model.predict_with_uncertainty(
                            low_res, confidence_level=0.95
                        )

                        # Calculate basic metrics
                        ssim, psnr, mse, rmse, mae = self.calculate_metrics(reconstruction, high_res)

                        # Calculate uncertainty metrics
                        u_width, cal_error = self.calculate_uncertainty_metrics(
                            reconstruction, lower_bounds, upper_bounds, high_res
                        )

                        # Store uncertainty metrics
                        metrics[name]['uncertainty_width'].extend([u_width] * len(metadata['location_name']))
                        metrics[name]['calibration_error'].extend([cal_error] * len(metadata['location_name']))

                    else:
                        # Standard evaluation for models without uncertainty
                        output = model(low_res)
                        ssim, psnr, mse, mae = self.calculate_metrics(output, high_res)

                        # Fill uncertainty metrics with None
                        metrics[name]['uncertainty_width'].extend([None] * len(metadata['location_name']))
                        metrics[name]['calibration_error'].extend([None] * len(metadata['location_name']))

                    # Store basic metrics
                    for metric_name, value in zip(['ssim', 'psnr', 'mse', 'rmse', 'mae'],
                                                [ssim, psnr, mse, rmse, mae]):
                        metrics[name][metric_name].extend([value] * len(metadata['location_name']))
                    metrics[name]['locations'].extend(metadata['location_name'])

                # Evaluate baseline methods
                for method in self.methods:
                    upscaled = F.interpolate(
                        low_res, size=(64, 64), mode=method,
                        align_corners=False if method != 'nearest' else None
                    )
                    ssim, psnr, mse, rmse, mae = self.calculate_metrics(upscaled, high_res)

                    for metric_name, value in zip(['ssim', 'psnr', 'mse', 'rmse', 'mae'],
                                                [ssim, psnr, mse, rmse, mae]):
                        baseline_metrics[method][metric_name].extend([value] * len(metadata['location_name']))
                    baseline_metrics[method]['locations'].extend(metadata['location_name'])

        #self.print_comparative_results(metrics, baseline_metrics)
        return metrics, baseline_metrics

    def format_metrics_line(self, name, metrics_dict, include_uncertainty=False):
        """Format a line of metrics for printing"""
        basic_metrics = (
            f"{name:<15} | "
            f"SSIM: {metrics_dict['ssim']:>6.4f} | "
            f"PSNR: {metrics_dict['psnr']:>7.4f} | "
            f"MSE: {metrics_dict['mse']:>7.4f} | "
            f"RMSE: {metrics_dict['rmse']:>7.4f} | "
            f"MAE: {metrics_dict['mae']:>7.4f}"
        )

        if include_uncertainty and None not in [
            metrics_dict.get('uncertainty_width'),
            metrics_dict.get('calibration_error'),
        ]:
            uncertainty_metrics = (
                f" | UWidth: {metrics_dict['uncertainty_width']:>7.4f} | "
                f"CalErr: {metrics_dict['calibration_error']:>7.4f} | "
            )
            return basic_metrics + uncertainty_metrics

        return basic_metrics

    def print_comparative_results(self, models_metrics, baseline_metrics):
        print(f"\n{'='*120}")
        print(f"32x32 to 64x64 Results")
        print(f"{'='*120}\n")

        # Overall Performance
        print("Overall Performance:")
        print("-" * 120)

        # Models results
        for model_name, model_metrics in models_metrics.items():
            model_means = {
                k: np.mean([v for v in vals if v is not None])
                for k, vals in model_metrics.items()
                if k != 'locations'
            }
            print(self.format_metrics_line(
                model_name,
                model_means,
                include_uncertainty=self.uncertainty_capable[model_name]
            ))

        # Baseline results
        for method in self.methods:
            baseline_means = {
                k: np.mean(v)
                for k, v in baseline_metrics[method].items()
                if k != 'locations'
            }
            print(self.format_metrics_line(method.capitalize(), baseline_means))

        # By Location
        print("\nPerformance by Location:")
        print("-" * 120)

        locations = np.unique(list(models_metrics.values())[0]['locations'])
        for location in locations:
            print(f"\n{location}:")

            # Models results by location
            for model_name, model_metrics in models_metrics.items():
                mask = np.array(model_metrics['locations']) == location
                loc_means = {
                    k: np.mean([v for v in np.array(vals)[mask] if v is not None])
                    for k, vals in model_metrics.items()
                    if k != 'locations'
                }
                print(self.format_metrics_line(
                    f"  {model_name}",
                    loc_means,
                    include_uncertainty=self.uncertainty_capable[model_name]
                ))

            # Baseline results by location
            for method in self.methods:
                mask = np.array(baseline_metrics[method]['locations']) == location
                loc_means = {
                    k: np.mean(np.array(v)[mask])
                    for k, v in baseline_metrics[method].items()
                    if k != 'locations'
                }
                print(self.format_metrics_line(f"  {method.capitalize()}", loc_means))