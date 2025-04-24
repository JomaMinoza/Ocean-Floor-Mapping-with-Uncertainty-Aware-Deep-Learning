import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm

class BathyGEBCOSuperResolutionDataset(Dataset):
    """Dataset for super resolution with single-channel numpy arrays"""
    
    LOCATION_NAMES = {
        'loc1': 'Western Pacific Region',
        'loc2': 'Indian Ocean Basin',
        'loc3': 'Eastern Atlantic Coast',
        'loc4': 'South Pacific Region',
        'loc5': 'Eastern Pacific Basin',
        'loc6': 'North Atlantic Basin'
    }
    
    def __init__(self, base_dir, split_type='train', cfg=None, percentage=None, random_state=42):
        """
        Args:
            base_dir: Base directory containing train/test folders
            split_type: Type of split to use ('train' or 'test')
            cfg: Configuration dictionary with mean and std
            percentage: If provided, use only this percentage of data while maintaining stratification
            random_state: Random seed for stratification
        """
        assert split_type in ['train', 'test'], "split_type must be either 'train' or 'test'"
        
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / split_type
        self.split_type = split_type
        
        # Initialize paths
        self._setup_file_paths(percentage, random_state)
        
        # Initialize normalization parameters
        self._setup_normalization(cfg)

    def _setup_file_paths(self, percentage, random_state):
        """Set up file paths based on data splits and stratification"""
        if percentage is not None:
            self.files_df = self._get_stratified_sample(percentage, random_state)
            self.files_df = self.files_df[self.files_df['set'] == self.split_type]
        else:
            self.files_df = self._load_base_data()
            
        if self.files_df is not None:
            self._load_files_from_dataframe()
        else:
            self._load_all_files()
            
        self._print_file_summary()

    def _load_base_data(self):
        """Load base CSV data if available"""
        base_csv_path = self.base_dir / 'data.csv'
        if base_csv_path.exists():
            df = pd.read_csv(base_csv_path)
            return df[df['split'] == self.split_type]
        return None

    def _load_files_from_dataframe(self):
        """Load files based on DataFrame information"""
        self.res_v0_files = [self.data_dir/"source_x4"/f for f in self.files_df['filename']]
        self.res_v1_files = [self.data_dir/"source_x2"/f for f in self.files_df['filename']]
        self.res_v2_files = [self.data_dir/"target"/f for f in self.files_df['filename']]

    def _load_all_files(self):
        """Load all files from directories"""
        print("Warning: No data.csv found. Using all files without train/test split.")
        for path_name in ['source_x4', 'source_x2', 'target']:
            path = self.data_dir / path_name
            files = sorted(list(path.glob('*.npy')))
            setattr(self, f'res_{path_name.replace("source_", "v")}_files', files)

    def _print_file_summary(self):
        """Print summary of loaded files"""
        print(f"\nLoading {self.split_type} split:")
        for res in ['v0', 'v1', 'v2']:
            files = getattr(self, f'res_{res}_files')
            source = 'source_x4' if res == 'v0' else 'source_x2' if res == 'v1' else 'target'
            print(f"Found {len(files)} files in {source}")

        if self.files_df is not None:
            self._print_location_summary()

    def _print_location_summary(self):
        """Print summary of data distribution across locations"""
        print(f"\nSummary for {self.split_type} split:")
        total = 0
        for location, name in self.LOCATION_NAMES.items():
            loc_df = self.files_df[self.files_df['location'] == location]
            count = len(loc_df)
            if count > 0:
                total += count
                print(f"{name}: {count} files")
        print(f"Total files: {total}")

    def _get_stratified_sample(self, percentage, random_state):
        """Get or create stratified sample of the dataset"""
        base_csv_path = self.base_dir / 'data.csv'
        if not base_csv_path.exists():
            raise FileNotFoundError("data.csv not found in data directory")

        stratified_csv_path = self.base_dir / f'data_{self.split_type}_{float(percentage*100)}.csv'
        if stratified_csv_path.exists():
            return pd.read_csv(stratified_csv_path)

        return self._create_stratified_sample(base_csv_path, percentage, random_state, stratified_csv_path)

    def _create_stratified_sample(self, base_csv_path, percentage, random_state, save_path):
        """Create a new stratified sample"""
        df = pd.read_csv(base_csv_path)
        sampled_data = []

        for location in df['location'].unique():
            mask = (df['location'] == location) & (df['set'] == self.split_type)
            current_data = df[mask]
            
            if not current_data.empty:
                n_samples = int(np.ceil(len(current_data) * percentage))
                sampled = current_data.sample(n=n_samples, random_state=random_state)
                sampled_data.append(sampled)

        final_df = pd.concat(sampled_data, ignore_index=True)
        final_df.to_csv(save_path, index=False)
        return final_df

    def _setup_normalization(self, cfg):
        """Set up normalization parameters"""
        if cfg is not None:
            self.mean = cfg['mean']
            self.std = cfg['std']
            self.max = cfg.get('max', 0)
            self.min = cfg.get('min', 0)
        else:
            self._calculate_normalization_stats()

    def _calculate_normalization_stats(self):
        """Calculate normalization statistics from target data"""
        print("Calculating normalization statistics from target data...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        target_data = []
        progress_bar = tqdm(self.res_v2_files, leave=False)
        
        for file_path in progress_bar:
            target_arr = torch.from_numpy(np.load(file_path)).to(device)
            target_data.append(target_arr.flatten())
        
        target_data = torch.cat(target_data).to(torch.float32)
        
        self.mean = float(torch.mean(target_data))
        self.std = float(torch.std(target_data))
        self.max = float(torch.max(target_data))
        self.min = float(torch.min(target_data))
        
        print(f"Data statistics - Mean: {self.mean:.4f}, Std: {self.std:.4f}, "
              f"Max: {self.max:.4f}, Min: {self.min:.4f}")

    def load_and_resize(self, path, target_size=None):
        """Load, normalize, and optionally resize numpy array"""
        arr = np.load(path)
        
        # Apply normalization
        arr = (arr - self.mean) / (self.std + 1e-8)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        
        # Convert to torch tensor
        tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        
        # Resize if needed
        if target_size and tensor.shape[-2:] != target_size:
            tensor = F.interpolate(
                tensor,
                size=target_size,
                mode='bicubic',
                align_corners=False
            )
        
        return tensor.squeeze(0)  # Remove batch dimension
    
    def denormalize(self, tensor, stats=None):
        """
        Denormalize the tensor back to original scale using provided or default statistics
        
        Args:
            tensor (torch.Tensor): Normalized tensor to denormalize
            stats (tuple, optional): Tuple of (mean, std, max, min) statistics for denormalization.
                                   If not provided, uses dataset's default statistics.
        
        Returns:
            torch.Tensor: Denormalized tensor in original scale
        """
        if stats is None:
            mean, std, max_val, min_val = self.mean, self.std, self.max, self.min
        else:
            mean, std, max_val, min_val = stats

        # First denormalize from [0,1] range
        tensor = tensor * (max_val - min_val) + min_val
        
        # Then denormalize from standardization
        tensor = tensor * std + mean
        
        return tensor
    
    def get_stats(self):
        """Return normalization statistics"""
        return self.mean, self.std, self.max, self.min
    
    def get_data_stats(self, path):
        """Get statistics for a specific data file"""
        data = np.load(path)

        mean = np.mean(data)
        std = np.std(data)
        max_ = np.max(data)
        min_ = np.min(data)

        return mean, std, max_, min_

    def get_stats(self):
        """Return normalization statistics"""
        return self.mean, self.std, self.max, self.min
    
    def get_data_stats(self, path):

        data = np.load(path)

        mean = np.mean(data)
        std = np.std(data)
        max_ = np.max(data)
        min_ = np.min(data)

        return mean, std, max_, min_

    def __len__(self):
        return len(self.res_v2_files)

    def __getitem__(self, idx):
        # Load arrays and resize as needed
        res_v0 = self.load_and_resize(self.res_v0_files[idx], target_size=(16, 16))
        res_v1 = self.load_and_resize(self.res_v1_files[idx], target_size=(32, 32))
        res_v2 = self.load_and_resize(self.res_v2_files[idx], target_size=(64, 64))

        stats = self.get_data_stats(self.res_v2_files[idx])
        
        # Get metadata from DataFrame
        metadata = {}
        if self.files_df is not None:
            row = self.files_df.iloc[idx]
            metadata = {
                'stats': stats,
                'loc': row['location'],  
                'location_name': self.LOCATION_NAMES[row['location']], 
                'latitude': row['latitude'],  
                'longitude': row['longitude'], 
            }
        else:
            metadata = {
                'stats': stats,
                'loc': None,
                'location_name': None,
                'latitude': None,
                'longitude': None,
            }            
        
        return [res_v0, res_v1, res_v2], metadata