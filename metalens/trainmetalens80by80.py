# %%
import os
import time
import copy
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import sparse
from scipy.sparse.linalg import spsolve
from transformers import get_cosine_schedule_with_warmup

from loss_gradient_adjointsmall_Fourier_size10width12 import *
from loss_gradient_adjointsmall_Fourier_size10width12 import build_laplacian_2d_PML
from unet_smallFourier import create_unet


# -----------------------------
# Utilities
# -----------------------------
def setup_cuda_path():
    cuda_lib = "/usr/local/cuda-12.1/lib64"
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib}:{current}" if current else cuda_lib
    print(f"LD_LIBRARY_PATH is set to: {os.environ['LD_LIBRARY_PATH']}")


def read_txt_file(file_path):
    data = np.loadtxt(file_path, delimiter="\t")
    return data.astype(np.float32)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def choose_sample_distance(epoch):
    if epoch < 50:
        return 50
    if epoch < 100:
        return 25
    return 10


def sample_target_position(epoch):
    sample_distance = choose_sample_distance(epoch)
    number_ratio_x = int(np.random.choice(np.arange(0, 101, sample_distance)))
    number_ratio_y = int(np.random.choice(np.arange(0, 101, sample_distance)))
    x_ratio = number_ratio_x / 100.0
    y_ratio = number_ratio_y / 100.0
    return x_ratio, y_ratio


def sample_preview_position(size_x, size_y, end_x, y_middle):
    ratio_x = np.random.choice([0.1, 0.3, 0.5, 0.7, 0.9]) + np.random.choice([-0.05, 0.0, 0.05])
    ratio_y = np.random.choice([0.1, 0.3, 0.5, 0.7, 0.9]) + np.random.choice([-0.05, 0.0, 0.05])

    ind_x = int(ratio_x * size_x + (end_x - size_x))
    ind_y = int(ratio_y * size_y + (y_middle - size_y / 2))
    return ind_x, ind_y


# -----------------------------
# Model components
# -----------------------------
class Binarization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, epoch):
        output1 = output + 1.7
        output1 = torch.clamp(output1, 1.0, 2.4)

        output_imag = torch.abs(torch.abs(output1 - 1.7) - 0.7)
        start_epoch = 2000
        scale = min(1.003 ** ((epoch - start_epoch) / 5), 100.0)

        output2 = output1 + 1j * output_imag * 0.01 * scale
        return output2


class StructureOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, bound_index, rows, cols, n_init, k0dx, dx, x_metasurface, ny, N_PML, n_bg):
        device = output.device
        n = torch.tensor(n_init, dtype=torch.complex64, device=device).clone()

        design_width = int(np.sum(cols))
        n_design_tiled = output[:, None].repeat(1, design_width)

        start_rows = np.where(rows == 1)[0][0]
        end_rows = np.where(rows == 1)[0][-1] + 1
        start_cols = np.where(cols == 1)[0][0]
        end_cols = np.where(cols == 1)[0][-1] + 1

        n[start_rows:end_rows, start_cols:end_cols] = n_design_tiled

        nx_lens = bound_index + 1 + N_PML
        n_lens = n[:, :nx_lens]

        A_scipy = build_laplacian_2d_PML(nx_lens, ny, N_PML, N_PML, N_PML, N_PML)
        A = torch.tensor(A_scipy.toarray(), dtype=torch.complex64, device=device)

        rows_diag = torch.arange(nx_lens * ny, device=device)
        indices_diag = torch.stack([rows_diag, rows_diag])
        values = (k0dx ** 2) * n_lens.permute(1, 0).reshape(-1) ** 2

        sparse_diag = torch.sparse_coo_tensor(
            indices_diag,
            values,
            (nx_lens * ny, nx_lens * ny),
            dtype=torch.complex64,
            device=device,
        )
        A = A + sparse_diag.to_dense()

        _ = k0dx / dx  # Not used in later codes.

        psi_in_2D = torch.zeros((ny, nx_lens), dtype=torch.complex64, device=device)
        psi_in_2D[40:-40, N_PML] = 1.0

        b_direct_lens = psi_in_2D[:, :nx_lens]
        psi_sca_lens = torch.linalg.solve(
            A,
            b_direct_lens.permute(1, 0).reshape(-1)
        ).reshape(nx_lens, ny).permute(1, 0)

        psi_tot_lens = psi_in_2D[:, :nx_lens] + psi_sca_lens
        return psi_tot_lens


class FigureOfMerit(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, psi_tot_lens, ind_x, ind_y, nx, ny, N_PML, k0dx, dx, x_metasurface, lambda_, n_bg, bound_index):
        device = psi_tot_lens.device

        ny0 = ny
        ny_pad = ny0 * 3
        dky = 2 * np.pi / (ny_pad * dx)

        if ny_pad % 2 == 1:
            ny_half = (ny_pad - 1) // 2
            ky = dky * torch.arange(-ny_half, ny_half + 1, device=device)
        else:
            ny_half = ny_pad // 2
            ky = dky * torch.arange(-ny_half, ny_half, device=device)

        k = n_bg * 2 * np.pi / lambda_
        kx_array = torch.sqrt((k**2 - ky**2).to(torch.complex64))

        psi_out_total_lens = psi_tot_lens[:, bound_index + 1]

        psi_out_total_lenspadding = torch.zeros((ny_pad, 1), dtype=torch.complex64, device=device)
        psi_out_total_lenspadding[ny0:2 * ny0] = psi_out_total_lens.reshape(-1, 1)

        x_metasurface_fft = torch.tensor(
            x_metasurface[bound_index + 1:] - x_metasurface[bound_index],
            dtype=torch.complex64,
            device=device,
        )

        fft_term = torch.fft.fftshift(torch.fft.fft(psi_out_total_lenspadding, dim=0))
        phase_term = torch.exp(1j * torch.matmul(kx_array.reshape(-1, 1), x_metasurface_fft.reshape(1, -1)))

        psi_out_all_afterpadding = torch.fft.ifft(
            torch.fft.ifftshift(phase_term * fft_term, dim=0),
            dim=0,
        )

        psi_out_all_after = psi_out_all_afterpadding[ny0:2 * ny0, :]

        aim_point_air_trans = aim_point_in_air(ind_x, ind_y, nx, ny0, N_PML, k0dx)
        aim_point_air_trans = np.abs(aim_point_air_trans[:, bound_index + 2:])
        aim_point_air_trans = (aim_point_air_trans / np.max(aim_point_air_trans)) ** 4
        _ = torch.tensor(aim_point_air_trans, dtype=torch.float32, device=device)

        FoM = -torch.abs(psi_out_all_after[ind_y, ind_x - bound_index]) ** 2
        print(f"FoM: {FoM}, position: ({ind_y}, {ind_x - bound_index}).")
        return FoM


# -----------------------------
# Preview visualization
# -----------------------------
@torch.no_grad()
def save_preview_figure(
    epoch,
    save_dir,
    unet_model,
    binarization,
    epoch_input_x,
    epoch_input_y,
    x_metasurface,
    y_metasurface,
    rows,
    cols,
    n_init,
    k0dx,
    dx,
    N_PML,
    nx,
    ny,
    size_x,
    size_y,
    end_x,
    y_middle,
):
    device = next(unet_model.parameters()).device

    ind_x_input = int(epoch_input_x)
    ind_y_input = int(epoch_input_y)

    output = unet_model(
        torch.tensor((ind_x_input,), dtype=torch.float32, device=device),
        torch.tensor((ind_y_input,), dtype=torch.float32, device=device),
    ).squeeze()
    output = binarization(output, epoch)

    p_opt = output.detach().cpu().numpy()

    n = copy.deepcopy(n_init)
    n_design_tiled = np.tile(p_opt[:, None], (1, np.sum(cols)))
    n[np.ix_(rows, cols)] = n_design_tiled
    n = np.real(n)

    ny0, nx0 = n.shape
    A = build_laplacian_2d_PML(nx0, ny0, N_PML, N_PML, N_PML, N_PML) + sparse.diags(
        (k0dx ** 2) * n.flatten("F") ** 2,
        0,
        shape=(nx0 * ny0, nx0 * ny0),
    )

    psi_in_2D = np.zeros((ny0, nx0), dtype=np.complex64)
    psi_in_2D[40:-40, N_PML] = 1.0
    psi_sca = spsolve(A, psi_in_2D.flatten("F")).reshape(ny0, nx0, order="F")
    psi_tot = psi_in_2D + psi_sca

    x_microns = x_metasurface * 1e6
    y_microns = y_metasurface * 1e6

    redblue = sns.diverging_palette(250, 10, n=256, as_cmap=True)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    im1 = axs[0].imshow(
        n,
        extent=[x_microns.min(), x_microns.max(), y_microns.min(), y_microns.max()],
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    axs[0].set_xlabel("x (micron)")
    axs[0].set_ylabel("y (micron)")
    axs[0].set_title("n(x, y)")
    fig.colorbar(im1, ax=axs[0], label="Refractive index")
    axs[0].axis("image")

    im2 = axs[1].imshow(
        np.abs(psi_tot) ** 2,
        extent=[x_microns.min(), x_microns.max(), y_microns.min(), y_microns.max()],
        cmap=redblue,
        aspect="auto",
        origin="lower",
    )
    fig.colorbar(im2, ax=axs[1], label=r"$|\Psi_{sca}|^2$")
    im2.set_clim(0, 20)
    axs[1].set_title(r"$|\Psi_{tot}|^2$")
    axs[1].set_xlabel("x ($\mu$m)")
    axs[1].set_ylabel("y ($\mu$m)")
    axs[1].axis("equal")

    b_adj = sparse.csr_matrix((ny0, nx0))
    ind_x = int(epoch_input_x)
    ind_y = int(epoch_input_y)
    b_adj[ind_y, ind_x] = 1

    kernel = np.ones((5, 5), np.uint8)
    densed_b_adj = cv2.dilate(b_adj.todense(), kernel, iterations=5)

    im3 = axs[2].imshow(
        densed_b_adj,
        extent=[x_microns.min(), x_microns.max(), y_microns.min(), y_microns.max()],
        cmap=redblue,
        aspect="auto",
        origin="lower",
    )
    fig.colorbar(im3, ax=axs[2], label=r"$|\Psi_{sca}|^2$")
    axs[2].set_title("Dilated b_adj")
    axs[2].set_xlabel("x ($\mu$m)")
    axs[2].set_ylabel("y ($\mu$m)")
    axs[2].axis("equal")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch}_x_{ind_x}_y_{ind_y}.png", dpi=200)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    setup_cuda_path()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    file_path = "/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/design_splitter/metlens_design/random_number.txt"
    data = read_txt_file(file_path)

    x_ratios = data[:, 0]
    y_ratios = data[:, 1]

    unet_model = create_unet(
        image_size=640,
        structure_size=301,
        classifier_use_fp16=False,
        classifier_width=32,
        classifier_depth=2,
        classifier_attention_resolutions="80,40,20",
        classifier_use_scale_shift_norm=False,
        classifier_resblock_updown=False,
    ).to(device)

    import torchsummary
    torchsummary.summary(unet_model, [(1,), (1,)], batch_size=1, device=device)

    binarization = Binarization().to(device)
    output_structure = StructureOutput().to(device)
    calculate_FoM = FigureOfMerit().to(device)

    optimizer = optim.AdamW(unet_model.parameters(), lr=5e-5)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=2000,
        num_training_steps=30001,
    )

    num_epochs = 30001
    time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    continue_training = False

    if continue_training:
        load_dir = "/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/design_splitter/metlens_design/results/2024-11-04_17-40-17_200_200_nonormalization"
        st_epoch, optimizer_load = load_checkpoint(unet_model, optimizer, load_dir, model_name="unet_model_epoch_10000")
        save_dir = "/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/design_splitter/metlens_design/results/2024-11-04_17-40-17_200_200_continue_minmaxnorm_gradient"
        ensure_dir(save_dir)
        start_epoch = st_epoch

        if not optimizer_load:
            optimizer = optim.AdamW(unet_model.parameters(), lr=2e-5)
            print("Initializing optimizer with lr=2e-5")
    else:
        save_dir = f"/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/design_splitter/metlens_design/results_new_adjoint_fourier/{time_stamp}_size10width12_x80y80"
        ensure_dir(save_dir)
        start_epoch = 0
        st_epoch = 0

    if "__file__" in globals():
        os.system(f"cp {__file__} {save_dir}/codebackup.py")

    log_file = open(f"{save_dir}/log.txt", "w")
    average_training_FoM = 0.0

    size_x = 80
    size_y = 80

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        x_ratio, y_ratio = sample_target_position(epoch)

        print("nx:", nx, "ny:", ny)
        print("D/dx:", D / dx, "D_extra/dx:", D_extra / dx, "W_tot/dx:", W_tot / dx)

        end_x = nx - 1 - N_PML - 5
        y_middle = int(ny / 2)
        start_x = end_x - size_x
        start_y = y_middle - size_y / 2

        ind_x = int(x_ratio * size_x + start_x)
        ind_y = int(y_ratio * size_y + start_y)

        ind_x_input = int(x_ratio * 640)
        ind_y_input = int(y_ratio * 640)

        print("data generation time", time.time() - start_time)

        optimizer.zero_grad()

        start_time = time.time()
        output = unet_model(
            torch.tensor((ind_x_input,), dtype=torch.float32, device=device),
            torch.tensor((ind_y_input,), dtype=torch.float32, device=device),
        ).squeeze()
        output = binarization(output, epoch)
        print("output.shape", output.shape)
        print("forward time", time.time() - start_time)

        start_time = time.time()
        bound_index = 37
        psi_tot_lens = output_structure(output, bound_index, rows, cols, n, k0dx, dx, x_metasurface, ny, N_PML, n_bg)
        loss_FoM = calculate_FoM(
            psi_tot_lens,
            ind_x,
            ind_y,
            nx,
            ny,
            N_PML,
            k0dx,
            dx,
            x_metasurface,
            lambda_,
            n_bg,
            bound_index,
        )
        loss = loss_FoM
        print("loss calculation time", time.time() - start_time)

        start_time = time.time()
        loss.backward()
        print("backward time", time.time() - start_time)

        optimizer.step()
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")
        average_training_FoM += float(loss.detach().cpu())

        if epoch % 20 == 0:
            save_number = 1
            if epoch % 100 == 0:
                save_number = 4
                torch.save(unet_model.state_dict(), f"{save_dir}/unet_model_epoch_{epoch}.pth")

                average_training_FoM /= 100.0
                log_file.write(f"Epoch [{epoch + 1}/{num_epochs}], Average Training FoM: {average_training_FoM:.4f}\n")
                log_file.flush()
                average_training_FoM = 0.0

            with torch.no_grad():
                for _ in range(save_number):
                    ind_x_prev, ind_y_prev = sample_preview_position(size_x, size_y, end_x, y_middle)

                    ind_x_input_prev = int(((float(ind_x_prev) - start_x) / size_x) * 640)
                    ind_y_input_prev = int(((float(ind_y_prev) - start_y) / size_y) * 640)

                    save_preview_figure(
                        epoch=epoch,
                        save_dir=save_dir,
                        unet_model=unet_model,
                        binarization=binarization,
                        epoch_input_x=ind_x_input_prev,
                        epoch_input_y=ind_y_input_prev,
                        x_metasurface=x_metasurface,
                        y_metasurface=y_metasurface,
                        rows=rows,
                        cols=cols,
                        n_init=n,
                        k0dx=k0dx,
                        dx=dx,
                        N_PML=N_PML,
                        nx=nx,
                        ny=ny,
                        size_x=size_x,
                        size_y=size_y,
                        end_x=end_x,
                        y_middle=y_middle,
                    )
                    
                    
                    