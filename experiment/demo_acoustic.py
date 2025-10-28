
import numpy as np
import scipy
import torch
import random
import torch.optim as optim
import model.net as net
from tqdm import tqdm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rmse(x, x_hat):
    return torch.sqrt(torch.mean((x_hat - x) ** 2))


def train_and_evaluate(model, optimizer, loss_fn, x_ob, x_true, observation_tensor, denorm_max, TV=False):
    x_tensor = torch.FloatTensor(x_true).cuda()
    x_tensor = x_tensor.unsqueeze(0)

    x_ob_tensor = torch.FloatTensor(x_ob).cuda()
    x_ob_tensor = x_ob_tensor.unsqueeze(0)
    ot = torch.FloatTensor(observation_tensor).cuda()
    ot = ot.unsqueeze(0)

    rmse_min = 10
    train_loss_list = []
    rmse_list = []

    X_input = model.cut_tensor_into_sliding_patches(
        x_ob_tensor,
        subtensor_size=model.subtensor_size_tuple,
        stride=model.stride,
    )

    # Prepare normalization factor for denormalization during RMSE calc
    norm_tensor = None
    if isinstance(denorm_max, np.ndarray):
        norm_tensor = torch.from_numpy(denorm_max.astype('float32')).cuda().unsqueeze(0)

    for epoch in tqdm(range(num_epochs)):
        output, _ = model(X_input)
        loss = loss_fn(output, x_ob_tensor, ot, add_TV_regu=TV)
        train_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Denormalize model output back to original scale
        if norm_tensor is not None:
            output_denorm = output.detach() / norm_tensor
        else:
            output_denorm = output.detach() * denorm_max
        x_tensor_true = x_tensor.detach()
        total_rmse = get_rmse(output_denorm, x_tensor_true)
        rmse_list.append(total_rmse.item())
        total_rmse_np = total_rmse.cpu().numpy()
        if total_rmse_np < rmse_min:
            rmse_min = total_rmse_np
            print("--------epoch:", epoch, "loss:", loss.item(), ";min_rmse:", rmse_min)

    return rmse_min


if __name__ == "__main__":
    # hyperparameters
    num_epochs = 500
    learning_rate = 4e-3
    dropout_rate = 0.3
    sampling_rate = 0.3  # random sampling ratio
    subtensor_size = 10  # Can be int (same for all axes) or tuple (Sx, Sy, Sz)

    print("torch.cuda.is_available:", torch.cuda.is_available())
    set_random_seed(231)

    # 1) Load acoustic complex pressure and take magnitude
    mat = scipy.io.loadmat('./experiment/my_data/p_xhENV_synBTY_kraken_new.mat')
    field_raw = np.array(mat['square_p'])
    field_mag = np.abs(field_raw).astype('float64')

    # 2) Use original rectangular grid directly
    # 提供一个选项，可以对grid进行重采样
    resample_grid = True
    if resample_grid:
        target_shape = (72, 100, 100)
        if tuple(field_mag.shape) != target_shape:
            zoom_factors = [t / s for t, s in zip(target_shape, field_mag.shape)]
            field_resampled = zoom(field_mag, zoom_factors, order=1)
        else:
            field_resampled = field_mag
        x_true = field_resampled
        # 对重采样和原始grid进行可视化
        field_min = -1e-5
        field_max = 2e-3
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(field_mag[0, :, :], cmap='viridis', vmin=field_min, vmax=field_max)
        plt.title('Original Grid')
        plt.subplot(1, 2, 2)
        plt.imshow(field_resampled[0, :, :], cmap='viridis', vmin=field_min, vmax=field_max)
        plt.title('Resampled Grid')
        plt.show()
        
    else:
        x_true = field_mag
    N_x, N_y, N_z = x_true.shape

    # 3) Create random observation mask
    observation_tensor = (np.random.rand(N_x, N_y, N_z) < sampling_rate).astype('float64')

    # 4) Observed tensor and attenuation-based normalization
    x_ob = x_true * observation_tensor

    # Attenuation normalization (match dims: (N_x, N_y, N_z))
    r_range = [1, 10000]
    # Use N_y for the r grid length as in the provided design (len_val)
    r = np.linspace(r_range[0], r_range[1], N_y)
    amp = np.sqrt(8 * np.pi * r).astype('float64')
    # Build attenuation volume: repeat across X rows and Z slices
    atte_2d = np.tile(amp, (N_x, 1))  # (N_x, N_y)
    attenuation = np.repeat(atte_2d[:, :, np.newaxis], N_z, axis=2)  # (N_x, N_y, N_z)

    # Normalize observations by multiplying with attenuation
    x_ob_norm = x_ob * attenuation

    # 5) Model, optimizer, loss (pass rectangular grid size)
    model = net.FieldFormer_TAP(grid_size=(N_x, N_y, N_z), dropout_rate=dropout_rate, subtensor_size=subtensor_size, embed_dim=subtensor_size).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = net.loss_fn_mse

    # 6) Train and report RMSE
    tap_rmse_min = train_and_evaluate(
        model, optimizer, loss_fn, x_ob_norm, x_true, observation_tensor, attenuation, TV=False
    )
    print("Observation rate:", sampling_rate, " TAP RMSE:", tap_rmse_min)
    
    # 7) Get final reconstruction result
    X_input = model.cut_tensor_into_sliding_patches(
        torch.FloatTensor(x_ob_norm).cuda().unsqueeze(0),
        subtensor_size=model.subtensor_size_tuple,
        stride=model.stride,
    )
    with torch.no_grad():
        reconstruction, _ = model(X_input)
        # Inverse of normalization (divide by attenuation)
        reconstruction = reconstruction.squeeze(0).cpu().numpy() / attenuation
    
    # 8) Save results
    results = {
        'ground_truth': x_true,
        'reconstruction': reconstruction,
        'sparse_observation': x_ob,
        'observation_mask': observation_tensor,
        'sampling_rate': sampling_rate,
        'rmse': tap_rmse_min
    }
    scipy.io.savemat('./experiment/results_acoustic.mat', results)
    print("Results saved to ./experiment/results_acoustic.mat")


