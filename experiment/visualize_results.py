import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_results(filepath='./experiment/results_acoustic.mat'):
    """Load saved results from .mat file"""
    mat = scipy.io.loadmat(filepath)
    return {
        'ground_truth': mat['ground_truth'],
        'reconstruction': mat['reconstruction'],
        'sparse_observation': mat['sparse_observation'],
        'observation_mask': mat['observation_mask'],
        'sampling_rate': float(mat['sampling_rate']),
        'rmse': float(mat['rmse'])
    }

def visualize_3d_field(data, title, slice_idx=10):
    """Visualize 3D field data by showing middle slice"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Calculate consistent color limits
    data_min = np.min(data)
    data_max = np.max(data)
    
    # XY slice (Z=slice_idx)
    im1 = axes[0].imshow(data[:, :, slice_idx], cmap='viridis', origin='lower', vmin=data_min, vmax=data_max)
    axes[0].set_title(f'{title} - XY slice (Z={slice_idx})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])
    
    # XZ slice (Y=slice_idx)
    im2 = axes[1].imshow(data[:, slice_idx, :], cmap='viridis', origin='lower', vmin=data_min, vmax=data_max)
    axes[1].set_title(f'{title} - XZ slice (Y={slice_idx})')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    plt.colorbar(im2, ax=axes[1])
    
    # YZ slice (X=slice_idx)
    im3 = axes[2].imshow(data[slice_idx, :, :], cmap='viridis', origin='lower', vmin=data_min, vmax=data_max)
    axes[2].set_title(f'{title} - YZ slice (X={slice_idx})')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    return fig

def visualize_comparison(results):
    """Create comparison visualization"""
    gt = results['ground_truth']
    recon = results['reconstruction']
    sparse = results['sparse_observation']
    mask = results['observation_mask']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    slice_idx = 10
    
    # Calculate consistent color limits for field data
    field_data = [gt, recon, sparse]
    field_min = -1e-5
    field_max = 2e-3
    
    # Ground Truth
    im1 = axes[0, 0].imshow(gt[slice_idx, :, :], cmap='viridis', vmin=field_min, vmax=field_max)
    axes[0, 0].set_title('Ground Truth')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Reconstruction
    im2 = axes[0, 1].imshow(recon[slice_idx, :, :], cmap='viridis', vmin=field_min, vmax=field_max)
    axes[0, 1].set_title('Reconstruction')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Sparse Observation
    im3 = axes[0, 2].imshow(sparse[slice_idx, :, :], cmap='viridis', vmin=field_min, vmax=field_max)
    axes[0, 2].set_title('Sparse Observation')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Observation Mask
    im4 = axes[1, 0].imshow(mask[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Observation Mask')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Error Map
    error = np.abs(recon - gt)
    im5 = axes[1, 1].imshow(error[:, :, slice_idx], cmap='hot', vmin=field_min, vmax=field_max)
    axes[1, 1].set_title('Absolute Error')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Difference (Reconstruction - Ground Truth)
    diff = recon - gt
    im6 = axes[1, 2].imshow(diff[slice_idx, :, :], cmap='RdBu_r', vmin=field_min, vmax=field_max)
    axes[1, 2].set_title('Difference (Recon - GT)')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.suptitle(f'Acoustic Field Reconstruction Results (Sampling Rate: {results["sampling_rate"]:.1%}, RMSE: {results["rmse"]:.4f})')
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Load results
    results = load_results()
    
    print(f"Sampling Rate: {results['sampling_rate']:.1%}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"Data shape: {results['ground_truth'].shape}")
    
    # Create visualizations
    fig1 = visualize_comparison(results)
    
    # Show plots
    plt.show()
    
    # Save plots
    fig1.savefig('./experiment/visualization_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to ./experiment/")
