import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional, Dict
from matplotlib.gridspec import GridSpec
from pathlib import Path

class BathymetryVisualizer:
    """Enhanced visualization class for bathymetry model comparisons and uncertainty analysis"""
    
    def __init__(self, model_loader=None, device='cuda', figsize=(10, 8)):
        self.model_loader = model_loader
        self.device = device
        self.figsize = figsize
        self.default_view = {'elev': 30, 'azim': 225, 'dist': 11}
        self.colors = {
            'uncertainty': 'lightbrown',
            'prediction': 'brown',
            'target': 'black',
            'input': 'gray',
            'bilinear': 'green',
            'nearest': 'red',
            'bicubic': 'purple'
        }

    def _convert_to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert input tensor/array to numpy and remove batch/channel dimensions"""
        if isinstance(tensor, torch.Tensor):
            return tensor.squeeze().detach().cpu().numpy()
        return tensor.squeeze()

    def _denormalize(self, data: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
        """Denormalize data using provided statistics"""
        return data
        denormed_data = data * (stats['max'] - stats['min']) + stats['min']
        return denormed_data

    def _create_meshgrid(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create meshgrid for 3D plotting"""
        h, w = data.shape
        x = np.linspace(0, w-1, w)
        y = np.linspace(0, h-1, h)
        return np.meshgrid(x, y)

    def _plot_surface(self, ax: plt.Axes, data: np.ndarray, title: str,
                     uncertainty: bool = False, lower_bound: Optional[np.ndarray] = None,
                     upper_bound: Optional[np.ndarray] = None, annotation_point: Optional[Tuple[int, int]] = None):
        """Plot 3D surface with optional uncertainty"""
        X, Y = self._create_meshgrid(data)
        
        if not uncertainty:
            surf = ax.plot_surface(X, Y, data, cmap='viridis',
                                 linewidth=0, antialiased=True)
            plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        else:
            # Plot uncertainty bounds
            ax.plot_surface(X, Y, lower_bound, color='gray', alpha=0.15,
                          linewidth=0, antialiased=True)
            ax.plot_surface(X, Y, data, color='blue', alpha=0.3,
                          linewidth=0, antialiased=True)
            ax.plot_surface(X, Y, upper_bound, color='gray', alpha=0.15,
                          linewidth=0, antialiased=True)
            
            # Add legend
            legend_patches = [
                plt.Rectangle((0, 0), 1, 1, fc='blue', alpha=alpha)
                for alpha in [0.2, 0.5, 0.1]
            ]
            ax.legend(legend_patches, ['Lower Bound', 'Prediction', 'Upper Bound'],
                     loc='upper right')
            

        # Annotate a specific point
        if annotation_point:
            x_idx, y_idx = annotation_point
            z_val = data[y_idx, x_idx]  

            # Project annotation point to X-Y plane
            x_proj = X[y_idx, x_idx]
            y_proj = Y[y_idx, x_idx]

            ax.scatter(X[y_idx, x_idx], Y[y_idx, x_idx], z_val, color='red', s=50)  # Mark the point

            # Draw a vertical line from the point to the lower bound
            if uncertainty:

                lower_z = lower_bound[y_idx, x_idx]
                upper_z = upper_bound[y_idx, x_idx]

                uncertainty_range = upper_z - lower_z

                ax.plot([x_proj, x_proj], [y_proj, y_proj], [lower_z, upper_z], color='red', linestyle='-')  # Solid line for range

                annotation_text = f"({x_idx}, {y_idx}): Pred={z_val:.2f}, Unc={uncertainty_range:.2f}"

            ax.scatter(x_proj, y_proj, z_val, color='red', s=50)  
            ax.legend()  # Show the annotation point in the legend
            
        ax.set_title(title)
        ax.view_init(elev=self.default_view['elev'], azim=self.default_view['azim'])
        ax.dist = self.default_view['dist']
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Depth')

    def _interpolate(self, x: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
        """Interpolate tensor to target size using specified mode"""
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        x = x.unsqueeze(0).unsqueeze(0)
        return F.interpolate(x, size=size, mode=mode, align_corners=False if mode != 'nearest' else None)

    def compare_interpolations(self, low_res: Union[torch.Tensor, np.ndarray],
                             high_res: Union[torch.Tensor, np.ndarray],
                             denorm_stats: Optional[Dict[str, float]] = None,
                             plot_type: str = '3d') -> plt.Figure:
        """Compare different interpolation methods"""
        # Convert inputs
        low_res = torch.from_numpy(self._convert_to_numpy(low_res)).to(self.device)
        high_res = self._convert_to_numpy(high_res)
        target_size = high_res.shape

        # Generate interpolated versions
        methods = ['nearest', 'bilinear', 'bicubic']
        interpolated = {}
        for method in methods:
            interp = self._convert_to_numpy(self._interpolate(low_res, target_size, method))
            interpolated[method] = interp

        # Denormalize if needed
        if denorm_stats:
            high_res = self._denormalize(high_res, denorm_stats)
            low_res = self._denormalize(self._convert_to_numpy(low_res), denorm_stats)
            interpolated = {k: self._denormalize(v, denorm_stats) 
                          for k, v in interpolated.items()}

        if plot_type == '3d':
            fig = self._plot_interpolation_3d(low_res, interpolated, high_res)
        else:
            fig = self._plot_interpolation_2d(low_res, interpolated, high_res)

        return fig

    def _plot_interpolation_3d(self, low_res, interpolated, high_res):
        """Create 3D comparison plot of interpolation methods"""
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 3)

        # Plot low-res input
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        self._plot_surface(ax1, low_res, 'Low Resolution Input')

        # Plot interpolated versions
        for idx, (method, data) in enumerate(interpolated.items()):
            ax = fig.add_subplot(gs[0, idx+1], projection='3d')
            self._plot_surface(ax, data, f'{method.capitalize()} Interpolation')

        # Plot ground truth
        ax_truth = fig.add_subplot(gs[1, 1], projection='3d')
        self._plot_surface(ax_truth, high_res, 'Ground Truth')

        plt.tight_layout()
        return fig

    def _plot_interpolation_2d(self, low_res, interpolated, high_res):
        """Create 2D comparison plot of interpolation methods"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 4)

        # Plot low-res input
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(low_res, cmap='viridis')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title('Low Resolution Input')

        # Plot interpolated versions and error maps
        for idx, (method, data) in enumerate(interpolated.items()):
            # Interpolation result
            ax = fig.add_subplot(gs[0, idx+1])
            im = ax.imshow(data, cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'{method.capitalize()}')

            # Error map
            ax_error = fig.add_subplot(gs[1, idx+1])
            error = np.abs(data - high_res)
            im_error = ax_error.imshow(error, cmap='Reds')
            plt.colorbar(im_error, ax=ax_error)
            ax_error.set_title(f'Error Map')

        # Plot ground truth
        ax_truth = fig.add_subplot(gs[0, -1])
        im_truth = ax_truth.imshow(high_res, cmap='viridis')
        plt.colorbar(im_truth, ax=ax_truth)
        ax_truth.set_title('Ground Truth')

        plt.tight_layout()
        return fig
    
    def compare_models(self, input_data: Union[torch.Tensor, np.ndarray],
                      models: Dict[str, Dict[str, str]],
                      target: Optional[Union[torch.Tensor, np.ndarray]] = None,
                      denorm_stats: Optional[Dict[str, float]] = None,
                      plot_type: str = '3d') -> plt.Figure:
        """Compare different model outputs with uncertainty"""
        if self.model_loader is None:
            raise ValueError("ModelLoader required for model comparison")

        # Get predictions from each model
        predictions = {}
        uncertainties = {}
        
        for name, model_info in models.items():
            model = self.model_loader.get_model(
                model_info['type']
            )
            model.calibrated = True
            model.eval()
            
            with torch.no_grad():
                input_tensor = torch.from_numpy(self._convert_to_numpy(input_data)).unsqueeze(0).unsqueeze(0).to(self.device)
                if hasattr(model, 'predict_with_uncertainty'):
                    pred, lower, upper = model.predict_with_uncertainty(input_tensor)
                    predictions[name] = self._convert_to_numpy(pred)
                    uncertainties[name] = {
                        'lower': self._convert_to_numpy(lower),
                        'upper': self._convert_to_numpy(upper)
                    }
                else:
                    pred = model(input_tensor)
                    predictions[name] = self._convert_to_numpy(pred)

        # Denormalize if needed
        if denorm_stats:
            input_data = self._denormalize(self._convert_to_numpy(input_data), denorm_stats)
            predictions = {k: self._denormalize(v, denorm_stats) for k, v in predictions.items()}
            if target is not None:
                target = self._denormalize(self._convert_to_numpy(target), denorm_stats)
            for model in uncertainties:
                uncertainties[model] = {
                    k: self._denormalize(v, denorm_stats)
                    for k, v in uncertainties[model].items()
                }

        if plot_type == '3d':
            return self._plot_model_comparison_3d(input_data, predictions, target, uncertainties)
        else:
            return self._plot_model_comparison_2d(input_data, predictions, target, uncertainties)

    def _plot_model_comparison_3d(self, input_data, predictions, target, uncertainties):
        """Create 3D comparison plot of model outputs"""
        n_models = len(predictions)
        fig = plt.figure(figsize=(30, 20))
        gs = GridSpec(2, n_models + 1)

        # Plot input
        ax_input = fig.add_subplot(gs[0, 0], projection='3d')
        self._plot_surface(ax_input, input_data, 'Input')

        # Plot target if available
        if target is not None:
            ax_target = fig.add_subplot(gs[0, 1], projection='3d')
            self._plot_surface(ax_target, target, 'Ground Truth')


        annotation_point = (50, 50)

        # Plot model predictions
        for idx, (name, pred) in enumerate(predictions.items()):
            ax = fig.add_subplot(gs[1, idx], projection='3d')
            if name in uncertainties:
                self._plot_surface(ax, pred, f'{name}',
                                uncertainty=True,
                                lower_bound=uncertainties[name]['lower'],
                                upper_bound=uncertainties[name]['upper'],
                                annotation_point=annotation_point
                )
            else:
                self._plot_surface(ax, pred, f'{name}')

        plt.tight_layout()
        return fig

    def _plot_model_comparison_2d(self, input_data, predictions, target, uncertainties):
        """Create 2D comparison plot of model outputs"""
        n_models = len(predictions)
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(4, n_models + 1)

        # Plot input
        ax_input = fig.add_subplot(gs[0, 0])
        im_input = ax_input.imshow(input_data, cmap='viridis')
        plt.colorbar(im_input, ax=ax_input)
        ax_input.set_title('Input')

        # Plot target if available
        if target is not None:
            ax_target = fig.add_subplot(gs[0, 1])
            im_target = ax_target.imshow(target, cmap='viridis')
            plt.colorbar(im_target, ax=ax_target)
            ax_target.set_title('Ground Truth')

        # Plot predictions and uncertainties
        for idx, (name, pred) in enumerate(predictions.items()):
            # Prediction
            ax = fig.add_subplot(gs[1, idx])
            im = ax.imshow(pred, cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'{name}')

            # Error map if target available
            if target is not None:
                ax_error = fig.add_subplot(gs[2, idx])
                error = np.abs(pred - target)
                im_error = ax_error.imshow(error, cmap='Reds')
                plt.colorbar(im_error, ax=ax_error)
                ax_error.set_title('Error Map')

            # Uncertainty map if available
            if name in uncertainties:
                ax_uncert = fig.add_subplot(gs[3, idx])
                uncertainty = uncertainties[name]['upper'] - uncertainties[name]['lower']
                im_uncert = ax_uncert.imshow(uncertainty, cmap='Purples')
                plt.colorbar(im_uncert, ax=ax_uncert)
                ax_uncert.set_title('Uncertainty')


        plt.tight_layout()

    def visualize_model_uncertainty(self, 
                                  input_data: Union[torch.Tensor, np.ndarray],
                                  model_type: str,
                                  target: Optional[Union[torch.Tensor, np.ndarray]] = None,
                                  denorm_stats: Optional[Dict[str, float]] = None,
                                  plot_type: str = '3d') -> plt.Figure:
        """Create detailed uncertainty visualization for a single model"""
        if self.model_loader is None:
            raise ValueError("ModelLoader required for uncertainty visualization")

        # Load model and get prediction with uncertainty
        model = self.model_loader.get_model(model_type)
        model.calibrated = True
        model.eval()

        with torch.no_grad():
            input_tensor = torch.from_numpy(self._convert_to_numpy(input_data)).unsqueeze(0).unsqueeze(0).to(self.device)
            pred, lower, upper = model.predict_with_uncertainty(input_tensor)

        # Convert predictions to numpy
        pred = self._convert_to_numpy(pred)
        lower = self._convert_to_numpy(lower)
        upper = self._convert_to_numpy(upper)
        input_data = self._convert_to_numpy(input_data)
        if target is not None:
            target = self._convert_to_numpy(target)

        # Denormalize if needed
        if denorm_stats:
            pred = self._denormalize(pred, denorm_stats)
            lower = self._denormalize(lower, denorm_stats)
            upper = self._denormalize(upper, denorm_stats)
            input_data = self._denormalize(input_data, denorm_stats)
            if target is not None:
                target = self._denormalize(target, denorm_stats)

        if plot_type == '3d':
            return self._plot_uncertainty_3d(input_data, pred, lower, upper, target)
        else:
            return self._plot_uncertainty_2d(input_data, pred, lower, upper, target)

    def _plot_uncertainty_3d(self, input_data, pred, lower, upper, target=None):
        """Create detailed 3D uncertainty visualization"""
        fig = plt.figure(figsize=(30, 30))
        gs = GridSpec(3, 3)

        # Plot input
        ax_input = fig.add_subplot(gs[0, 0], projection='3d')
        self._plot_surface(ax_input, input_data, 'Input')

        ax_pred = fig.add_subplot(gs[0, 1], projection='3d')
        self._plot_surface(ax_pred, pred, 'Prediction',
                        uncertainty=False)

        # Plot prediction with uncertainty bounds
        ax_pred = fig.add_subplot(gs[0, 2], projection='3d')
        self._plot_surface(ax_pred, pred, 'Prediction with Uncertainty',
                        uncertainty=True, lower_bound=lower, upper_bound=upper)

        # Plot target if available
        if target is not None:
            ax_target = fig.add_subplot(gs[1, 0], projection='3d')
            self._plot_surface(ax_target, target, 'Ground Truth')

        # Plot uncertainty width
        ax_uncert = fig.add_subplot(gs[1, 1], projection='3d')
        uncertainty = upper - lower
        self._plot_surface(ax_uncert, uncertainty, 'Uncertainty Width')

        plt.tight_layout()
        return fig

    def _plot_uncertainty_2d(self, input_data, pred, lower, upper, target=None):
        """Create detailed 2D uncertainty visualization"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3)

        # Plot input and prediction
        ax_input = fig.add_subplot(gs[0, 0])
        im_input = ax_input.imshow(input_data, cmap='viridis')
        plt.colorbar(im_input, ax=ax_input)
        ax_input.set_title('Input')

        ax_pred = fig.add_subplot(gs[0, 1])
        im_pred = ax_pred.imshow(pred, cmap='viridis')
        plt.colorbar(im_pred, ax=ax_pred)
        ax_pred.set_title('Model Prediction')

        # Plot uncertainty bounds
        ax_lower = fig.add_subplot(gs[1, 0])
        im_lower = ax_lower.imshow(lower, cmap='viridis')
        plt.colorbar(im_lower, ax=ax_lower)
        ax_lower.set_title('Lower Bound')

        ax_upper = fig.add_subplot(gs[1, 1])
        im_upper = ax_upper.imshow(upper, cmap='viridis')
        plt.colorbar(im_upper, ax=ax_upper)
        ax_upper.set_title('Upper Bound')

        # Plot uncertainty width
        ax_uncert = fig.add_subplot(gs[1, 2])
        uncertainty = upper - lower
        im_uncert = ax_uncert.imshow(uncertainty, cmap='Purples')
        plt.colorbar(im_uncert, ax=ax_uncert)
        ax_uncert.set_title('Uncertainty Width')

        # Plot target if available
        if target is not None:
            ax_target = fig.add_subplot(gs[0, 2])
            im_target = ax_target.imshow(target, cmap='viridis')
            plt.colorbar(im_target, ax=ax_target)
            ax_target.set_title('Ground Truth')

        plt.tight_layout()
        return fig