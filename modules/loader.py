import torch
import torch.nn as nn
from typing import Dict, Optional, List
from enum import Enum

from .cnn import CNN
from .gan import UncertainESRGAN
from .vqvae import VQVAE

class ModelType(Enum):
    SRCNN = 'srcnn'
    GAN = 'gan'
    VQVAE = 'vqvae'

class ModelLoader:
    def __init__(self):
        # Base model configurations
        self.model_configs = {
            ModelType.SRCNN: {    
                "in_channels": 1,  
                "hidden_channels": 64,
                "num_residual_blocks": 8,
                "num_upsamples": 1,          
                "block_size": 4              
            },
            ModelType.GAN: {
                "in_channels": 1,
                "hidden_channels": 64,
                "num_rrdb_blocks": 8,
                "growth_channels": 32,
                "num_upsamples": 1,          
                "block_size": 4              
            },
            ModelType.VQVAE: {
                "in_channels": 1,
                "hidden_dims": [32, 64, 128, 256],
                "num_embeddings": 512,
                "embedding_dim": 256,
                "block_size": 4
            }
        }
        
        self.model_registry = {
            ModelType.SRCNN: CNN,
            ModelType.GAN: UncertainESRGAN,
            ModelType.VQVAE: VQVAE,
        }

        self.loaded_models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, model_type: str, checkpoint_path: str, config_overrides: Optional[Dict] = None) -> Optional[nn.Module]:
        """
        Load a model with its checkpoint and optional configuration overrides
        
        Args:
            model_type: Type of model to load ('srcnn', 'gan', or 'vqvae')
            checkpoint_path: Path to model checkpoint
            config_overrides: Optional dictionary of configuration overrides (e.g. {'block_size': 8})
        """
        try:
            # Convert string to enum
            model_type = ModelType(model_type.lower())
            
            # Get base config and apply overrides if provided
            model_config = self.model_configs[model_type].copy()
            if config_overrides:
                model_config.update(config_overrides)
            
            # Initialize model with potentially modified config
            model_class = self.model_registry[model_type]
            model = model_class(**model_config)
            model = torch.compile(model)
            model = model.to(self.device)

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)


            model.load_state_dict(checkpoint['state_dict'], strict = False)

            # Load uncertainty tracker state if available

            model.uncertainty_tracker.calibrated = checkpoint.get('calibrated', False)

            if model.uncertainty_tracker.calibrated:
                model.uncertainty_tracker.block_scale_means = checkpoint.get('block_scale_means', None)
                model.uncertainty_tracker.block_scale_stds = checkpoint.get('block_scale_stds', None)

            model.uncertainty_tracker.ema_errors = checkpoint['state_dict'].get(
                '_orig_mod.uncertainty_tracker.ema_errors',
                torch.zeros(model.uncertainty_tracker.block_size**2).to(self.device)
            )
            model.uncertainty_tracker.ema_quantile = checkpoint['state_dict'].get(
                '_orig_mod.uncertainty_tracker.ema_quantile',
                torch.zeros(model.uncertainty_tracker.block_size**2).to(self.device)
            )
            # Initialize buffers with potentially new block size
            model.uncertainty_tracker._initialize_buffers(64, 64, self.device)

            # Store model
            model.eval()
            self.loaded_models[model_type] = model
            
            return model

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def get_model(self, model_type: str) -> Optional[nn.Module]:
        """Get a loaded model by type"""
        try:
            model_type = ModelType(model_type.lower())
            return self.loaded_models.get(model_type)
        except:
            return None

    def unload_model(self, model_type: str) -> None:
        """Unload a specific model"""
        try:
            model_type = ModelType(model_type.lower())
            if model_type in self.loaded_models:
                del self.loaded_models[model_type]
                torch.cuda.empty_cache()
        except:
            pass

    def unload_all_models(self) -> None:
        """Unload all loaded models"""
        self.loaded_models.clear()
        torch.cuda.empty_cache()