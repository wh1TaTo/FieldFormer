import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_results(filepath='./experiment/results_acoustic.mat'):
    """Load saved results from .mat file"""
    mat = scipy.io.loadmat(filepath)
    results = {
        'ground_truth': mat['ground_truth'],
        'reconstruction': mat['reconstruction'],
        'sparse_observation': mat['sparse_observation'],
        'observation_mask': mat['observation_mask'],
        'sampling_rate': float(mat['sampling_rate']),
        'rmse': float(mat['rmse'])
    }
    # Add attention map if available
    if 'attention_map' in mat:
        results['attention_map'] = mat['attention_map']
    # Add distance bias if available
    if 'distance_bias' in mat:
        results['distance_bias'] = mat['distance_bias']
    return results

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
    im4 = axes[1, 0].imshow(mask[slice_idx, :, :], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Observation Mask')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Error Map
    error = np.abs(recon - gt)
    im5 = axes[1, 1].imshow(error[slice_idx, :, :], cmap='hot', vmin=field_min, vmax=field_max)
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

def visualize_attention_map(attention_map, top_k=5, figsize=(15, 10)):
    """Visualize attention map
    
    Args:
        attention_map: The attention map tensor (shape: [patches, patches] or similar)
        top_k: Number of top attention patterns to visualize
        figsize: Figure size
    """
    # Handle different attention map shapes
    if len(attention_map.shape) > 2:
        # For multi-head attention or other complex shapes, take the first head or flatten
        if attention_map.shape[0] > attention_map.shape[1]:
            # Assume shape is [num_heads, patches, patches] or similar
            attention_map = attention_map[0]  # Take first head
        else:
            # Flatten if needed
            attention_map = attention_map.reshape(attention_map.shape[0], -1)
    
    fig, axes = plt.subplots(2, (top_k + 1) // 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot the full attention map
    ax = axes[0]
    im = ax.imshow(attention_map, cmap='viridis', vmin=0, vmax=0.003, origin='lower')
    ax.set_title('Full Attention Map')
    plt.colorbar(im, ax=ax)
    
    # Plot top-k attention patterns (rows with highest sum)
    row_sums = attention_map.sum(axis=1)
    top_indices = np.argsort(row_sums)[-top_k:][::-1]
    
    for i, idx in enumerate(top_indices):
        if i + 1 < len(axes):
            ax = axes[i + 1]
            im = ax.plot(attention_map[idx], 'b-')
            ax.set_title(f'Top {i+1} Attention Pattern (Row {idx})')
            ax.set_xlabel('Target Position')
            ax.set_ylabel('Attention Weight')
            ax.grid(True, alpha=0.3)
    
    # Remove unused subplots
    for i in range(len(top_indices) + 1, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    return fig

def visualize_attention_heatmaps(attention_map, grid_shape=None, num_samples=4, figsize=(15, 15)):
    """Visualize attention map as heatmaps for selected query positions
    
    Args:
        attention_map: The attention map tensor
        grid_shape: Optional (Nx, Ny, Nz) grid shape to reshape patches
        num_samples: Number of sample query positions to visualize
        figsize: Figure size
    """
    patches = attention_map.shape[0]
    
    # Select sample positions to visualize
    if num_samples > patches:
        num_samples = patches
    sample_indices = np.linspace(0, patches - 1, num_samples, dtype=int)
    
    # Calculate grid dimensions if grid_shape is provided
    if grid_shape is not None and len(grid_shape) == 3:
        Nx, Ny, Nz = grid_shape
        grid_dim = int(np.sqrt(Nx * Ny * Nz))
        grid_size = int(np.ceil(np.sqrt(Nx * Ny * Nz)))
    else:
        # Find the minimum grid size that can hold all patches
        grid_size = int(np.ceil(np.sqrt(patches)))
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        # Get attention weights for this query position
        att_weights = attention_map[idx]
        
        # Pad to make it a perfect square if needed
        target_size = grid_size * grid_size
        if len(att_weights) < target_size:
            att_weights_padded = np.zeros(target_size)
            att_weights_padded[:len(att_weights)] = att_weights
            att_weights = att_weights_padded
        
        # Reshape to 2D grid for visualization
        att_grid = att_weights[:target_size].reshape(grid_size, grid_size)
        
        # Plot heatmap
        im = ax.imshow(att_grid, cmap='hot', interpolation='bilinear', origin='lower')
        ax.set_title(f'Attention Heatmap for Query Position {idx}')
        ax.set_xlabel('Key Position (Grid)')
        ax.set_ylabel('Key Position (Grid)')
        plt.colorbar(im, ax=ax, label='Attention Weight')
    
    plt.tight_layout()
    return fig

def visualize_distance_bias(distance_bias, top_k=5, figsize=(15, 10)):
    """Visualize distance bias map
    
    Args:
        distance_bias: The distance bias tensor (shape: [patches, patches])
        top_k: Number of top distance bias patterns to visualize
        figsize: Figure size
    """
    # Handle different distance bias shapes
    if len(distance_bias.shape) > 2:
        # For multi-head or other complex shapes, take the first slice
        if distance_bias.shape[0] > distance_bias.shape[1]:
            distance_bias = distance_bias[0]
        else:
            distance_bias = distance_bias.reshape(distance_bias.shape[0], -1)
    
    fig, axes = plt.subplots(2, (top_k + 1) // 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot the full distance bias map
    ax = axes[0]
    im = ax.imshow(distance_bias, cmap='RdBu_r', origin='lower', vmin=-0.04, vmax=0.00)
    ax.set_title('Full Distance Bias Map')
    plt.colorbar(im, ax=ax)
    
    # Plot top-k distance bias patterns (rows with largest magnitude)
    row_mags = np.abs(distance_bias).sum(axis=1)
    top_indices = np.argsort(row_mags)[-top_k:][::-1]
    
    for i, idx in enumerate(top_indices):
        if i + 1 < len(axes):
            ax = axes[i + 1]
            im = ax.plot(distance_bias[idx], 'b-')
            ax.set_title(f'Top {i+1} Distance Bias Pattern (Row {idx})')
            ax.set_xlabel('Target Position')
            ax.set_ylabel('Bias Value')
            ax.grid(True, alpha=0.3)
    
    # Remove unused subplots
    for i in range(len(top_indices) + 1, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    return fig

def visualize_distance_bias_heatmaps(distance_bias, grid_shape=None, num_samples=4, figsize=(15, 15)):
    """Visualize distance bias as heatmaps for selected query positions
    
    Args:
        distance_bias: The distance bias tensor
        grid_shape: Optional (Nx, Ny, Nz) grid shape to reshape patches
        num_samples: Number of sample query positions to visualize
        figsize: Figure size
    """
    patches = distance_bias.shape[0]
    
    # Select sample positions to visualize
    if num_samples > patches:
        num_samples = patches
    sample_indices = np.linspace(0, patches - 1, num_samples, dtype=int)
    
    # Calculate grid dimensions if grid_shape is provided
    if grid_shape is not None and len(grid_shape) == 3:
        Nx, Ny, Nz = grid_shape
        grid_size = int(np.ceil(np.sqrt(Nx * Ny * Nz)))
    else:
        # Find the minimum grid size that can hold all patches
        grid_size = int(np.ceil(np.sqrt(patches)))
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        # Get distance bias weights for this query position
        bias_weights = distance_bias[idx]
        
        # Pad to make it a perfect square if needed
        target_size = grid_size * grid_size
        if len(bias_weights) < target_size:
            bias_weights_padded = np.zeros(target_size)
            bias_weights_padded[:len(bias_weights)] = bias_weights
            bias_weights = bias_weights_padded
        
        # Reshape to 2D grid for visualization
        bias_grid = bias_weights[:target_size].reshape(grid_size, grid_size)
        
        # Plot heatmap
        im = ax.imshow(bias_grid, cmap='RdBu_r', interpolation='bilinear')
        ax.set_title(f'Distance Bias Heatmap for Query Position {idx}')
        ax.set_xlabel('Key Position (Grid)')
        ax.set_ylabel('Key Position (Grid)')
        plt.colorbar(im, ax=ax, label='Bias Value')
    
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
    
    # Visualize attention map if available
    if 'attention_map' in results:
        att_map = results['attention_map']
        print(f"Attention map shape: {att_map.shape}")
        
        # Create attention visualizations
        fig2 = visualize_attention_map(att_map)
        # fig3 = visualize_attention_heatmaps(att_map)
        
        # Save attention visualizations
        fig2.savefig('./experiment/attention_map.png', dpi=300, bbox_inches='tight')
        # fig3.savefig('./experiment/attention_heatmaps.png', dpi=300, bbox_inches='tight')
        print("Attention map visualizations saved")
    
    # Visualize distance bias if available
    if 'distance_bias' in results:
        dist_bias = results['distance_bias']
        print(f"Distance bias shape: {dist_bias.shape}")
        
        # Create distance bias visualizations
        fig4 = visualize_distance_bias(dist_bias)
        # fig5 = visualize_distance_bias_heatmaps(dist_bias)
        
        # Save distance bias visualizations
        fig4.savefig('./experiment/distance_bias_map.png', dpi=300, bbox_inches='tight')
        # fig5.savefig('./experiment/distance_bias_heatmaps.png', dpi=300, bbox_inches='tight')
        print("Distance bias visualizations saved")
    
    # Show plots
    plt.show()
    
    # Save comparison visualization
    fig1.savefig('./experiment/visualization_comparison.png', dpi=300, bbox_inches='tight')
    print("All visualizations saved to ./experiment/")
