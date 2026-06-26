# %%
import os
import sys
import time

sys.path.append('/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/design_splitter/metlens_design')

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.sparse.linalg import spsolve

from loss_gradient_adjointsmall_Fourier_size10width12 import *
from loss_gradient_adjointsmall_Fourier_size10width12 import build_laplacian_2d_PML
from utils import *
from unet_smallFourier import create_unet

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["LD_LIBRARY_PATH"] = ":/usr/local/cuda-12.1/lib64" + os.environ.get("LD_LIBRARY_PATH", "")
print(f"LD_LIBRARY_PATH is set to: {os.environ['LD_LIBRARY_PATH']}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Model components
# -----------------------------
class Binarization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, epoch):
        output = output + 1.7
        output = torch.clamp(output, 1.0, 2.4)
        return output


class StructureOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, bound_index, rows, cols, n, k0dx, dx, x_metasurface, ny, N_PML, n_bg):
        device = output.device
        n = torch.tensor(n, dtype=torch.complex64, device=device).clone()

        n_design_tiled = output[:, None].repeat(1, cols.sum().item())

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

        kx = k0dx / dx
        x_metasurface = torch.tensor(x_metasurface, dtype=torch.complex64, device=device)

        psi_in = torch.exp(1j * kx * x_metasurface)
        psi_in_2D = psi_in.unsqueeze(0).repeat(ny, 1)

        b_direct_lens = -k0dx ** 2 * (n_lens ** 2 - n_bg ** 2) * psi_in_2D[:, :nx_lens]

        psi_sca_lens = torch.linalg.solve(
            A,
            b_direct_lens.permute(1, 0).reshape(-1),
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
        aim_point_air_trans = torch.tensor(aim_point_air_trans, dtype=torch.float32, device=device)

        sum_power_in_air = torch.sum(aim_point_air_trans)
        psi_out_all_after = torch.abs(psi_out_all_after)
        FoM = -torch.sum((torch.abs(psi_out_all_after) ** 2 * aim_point_air_trans / sum_power_in_air))

        print(f"FoM: {FoM}, position: ({ind_y}, {ind_x - bound_index}).")
        return FoM


# -----------------------------
# Helper functions
# -----------------------------
def process_full_grid_to_half(outputs, ny_grid=11, nx_full=11):
    efficiency = outputs[:, 11].reshape(ny_grid, nx_full).copy()
    digreebinary = outputs[:, 4].reshape(ny_grid, nx_full).copy()

    ind_y = outputs[:, 0].reshape(ny_grid, nx_full).copy()
    ind_x = outputs[:, 1].reshape(ny_grid, nx_full).copy()
    ind_max_y = outputs[:, 7].reshape(ny_grid, nx_full).copy()
    ind_max_x = outputs[:, 8].reshape(ny_grid, nx_full).copy()

    distances = np.sqrt((ind_max_x - ind_x) ** 2 + (ind_max_y - ind_y) ** 2) * (0.5 / 15)

    for r in range(ny_grid):
        if r % 2 == 1:
            efficiency[r, :] = efficiency[r, ::-1]
            digreebinary[r, :] = digreebinary[r, ::-1]
            distances[r, :] = distances[r, ::-1]
            ind_y[r, :] = ind_y[r, ::-1]
            ind_x[r, :] = ind_x[r, ::-1]
            ind_max_y[r, :] = ind_max_y[r, ::-1]
            ind_max_x[r, :] = ind_max_x[r, ::-1]

    nx_half = nx_full // 2 + 1
    flip_mask = np.zeros((ny_grid, nx_half), dtype=bool)

    for r in range(ny_grid):
        for c in range(nx_full // 2):
            c2 = nx_full - 1 - c
            if efficiency[r, c] < efficiency[r, c2]:
                efficiency[r, c] = efficiency[r, c2]
                digreebinary[r, c] = digreebinary[r, c2]
                distances[r, c] = distances[r, c2]
                ind_max_y[r, c] = ind_max_y[r, c2]
                ind_max_x[r, c] = ind_max_x[r, c2]
                flip_mask[r, c] = True
            else:
                efficiency[r, c2] = efficiency[r, c]
                digreebinary[r, c2] = digreebinary[r, c]
                distances[r, c2] = distances[r, c]
                ind_max_y[r, c2] = ind_max_y[r, c]
                ind_max_x[r, c2] = ind_max_x[r, c]

    efficiency_half = efficiency[:, :nx_half]
    digreebinary_half = digreebinary[:, :nx_half]
    distances_half = distances[:, :nx_half]

    x_vals = np.arange(0.0, 0.51, 0.1)
    y_vals = np.arange(0.0, 1.01, 0.1)
    X_half, Y_half = np.meshgrid(x_vals, y_vals)

    return efficiency_half, digreebinary_half, distances_half, X_half, Y_half, flip_mask


def build_model():
    return create_unet(
        image_size=640,
        structure_size=301,
        classifier_use_fp16=False,
        classifier_width=32,
        classifier_depth=2,
        classifier_attention_resolutions="80,40,20",
        classifier_use_scale_shift_norm=False,
        classifier_resblock_updown=False,
    ).to(device)


def compute_field_for_point(model, binarization, x_ratio, y_ratio, epoch):
    ind_x_input = int(x_ratio * 640)
    ind_y_input = int(y_ratio * 640)

    output, embedding = model(
        torch.tensor((ind_x_input,), dtype=torch.float32, device=device),
        torch.tensor((ind_y_input,), dtype=torch.float32, device=device),
        return_embedding=True,
    )
    output = output.squeeze()
    output = binarization(output, epoch)

    return output, embedding


def build_performance_case(
    output,
    rows,
    cols,
    n,
    nx,
    ny,
    N_PML,
    k0dx,
    dx,
    x_metasurface,
    ind_x,
    ind_y,
):
    p_opt = output.detach().cpu().numpy()

    # Binarization degree
    n_binary = (0.7 - np.abs(p_opt - 1.7)) / 0.7
    degreebinary = 1.0 - np.mean(n_binary)

    # Embed the predicted structure into the design region
    n_design_tiled = np.tile(p_opt[:, None], (1, np.sum(cols)))
    n[np.ix_(rows, cols)] = n_design_tiled

    # Full simulation domain
    n_large = np.ones((ny, nx))
    n_large[:, 10:] = n

    ny0, nx0 = n_large.shape
    A = build_laplacian_2d_PML(nx0, ny0, N_PML, N_PML, N_PML, N_PML) + \
        sparse.diags((k0dx ** 2) * n_large.flatten("F") ** 2, 0, shape=(nx0 * ny0, nx0 * ny0))

    psi_in_2D = np.zeros((ny0, nx0), dtype=np.complex64)
    psi_in_2D[40:-40, N_PML + 5] = 1.0

    # Solve
    start_time = time.time()
    psi_sca = spsolve(A, psi_in_2D.flatten("F")).reshape(ny0, nx0, order="F")
    psi_tot = psi_in_2D + psi_sca
    elapsed = time.time() - start_time

    # Metrics (kept for internal use; not printed except the requested one)
    power_incident = np.sum(np.abs(psi_in_2D[:, N_PML + 5]) ** 2)
    power_focal = np.sum(np.abs(psi_tot[ind_y - 3: ind_y + 4, ind_x]) ** 2)
    eff_perc = power_focal / power_incident * 100

    half_max = np.max(np.abs(psi_tot[40:-40, ind_x]) ** 2 / 30) / 2
    half_max_loc = np.where(np.abs(psi_tot[40:-40, ind_x]) ** 2 / 30 > half_max)[0]
    half_max_wid = (half_max_loc[-1] - half_max_loc[0]) * dx

    leng = 0.3e-6
    half_win = int((leng / 2) / dx)
    dx_local = x_metasurface[1] - x_metasurface[0]

    psi_transmission_total = np.zeros(nx)
    psi_poynt = np.zeros((ny, nx))
    for xi in range(0, nx - 1):
        psi_transmission_total[xi] = np.imag(np.vdot(psi_tot[:, xi], psi_tot[:, xi + 1])) / np.sin(kx * dx_local)
        psi_poynt[:, xi] = np.imag(np.conjugate(psi_tot[:, xi]) * psi_tot[:, xi + 1]) / np.sin(kx * dx_local)

    y_slice = slice(ind_y - half_win, ind_y + half_win + 1)
    psi_transmission_totalpoint = np.imag(
        np.vdot(psi_tot[y_slice, ind_x], psi_tot[y_slice, ind_x + 1])
    ) / np.sin(kx * dx_local)

    psi_transmission_totalinput = 445.94516036157154
    efficiencypoynti = psi_transmission_totalpoint / psi_transmission_totalinput

    power_tot = np.sum(np.abs(psi_tot[y_slice, ind_x]) ** 2)

    # Only print what you requested
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"The total power within the focal region is: {power_tot / 270:.6f}.")
    print(f"The binarization degree is: {degreebinary:.6f}.")

    return {
        "n_large": n_large,
        "psi_tot": psi_tot,
    }


def save_composite_figure(save_path, n_large, psi_tot, b_adj, x_metasurface, y_metasurface, kernel, dx):
    redblue = sns.diverging_palette(250, 10, n=256, as_cmap=True)

    x_microns = x_metasurface * 1e6 - 10 * dx * 1e6
    y_microns = y_metasurface * 1e6
    densed_b_adj = cv2.dilate(b_adj.todense(), kernel, iterations=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    cut_idx = 27

    mask_left = np.zeros_like(n_large)
    mask_right = np.zeros_like(n_large)
    mask_left[:, :cut_idx] = 1
    mask_right[:, cut_idx:] = 1

    im1 = ax.imshow(
        n_large[40:-40, 21:-10],
        extent=[
            x_microns[21:-10].min(),
            x_microns[21:-10].max(),
            y_microns[40:-40].min(),
            y_microns[40:-40].max(),
        ],
        origin="lower",
        cmap="gray",
        aspect="auto",
        alpha=mask_left[40:-40, 21:-10],
        vmin=1.0,
        vmax=2.4,
    )

    im2 = ax.imshow(
        np.abs(psi_tot[40:-40, 21:-10]) ** 2 / 30 + densed_b_adj[40:-40, 21:-10],
        extent=[
            x_microns[21:-10].min(),
            x_microns[21:-10].max(),
            y_microns[40:-40].min(),
            y_microns[40:-40].max(),
        ],
        origin="lower",
        cmap=redblue,
        aspect="auto",
        alpha=mask_right[40:-40, 21:-10],
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("x (micron)")
    ax.set_ylabel("y (micron)")
    ax.tick_params(labelsize=15)
    ax.axis("image")

    cbar2 = fig.colorbar(im2, ax=ax, label=r"$|\Psi_{sca}|^2$")
    cbar2.ax.tick_params(labelsize=15)
    cbar1 = fig.colorbar(im1, ax=ax, label="Refractive index")
    cbar1.ax.tick_params(labelsize=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=1500, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
model_path = "/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/design_splitter/metlens_design/results_new_adjoint_fourier/2025-07-21_09-02-19_size10width12_x80y80/"
result_folder_path = "/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/design_splitter/metlens_design/results_new_adjoint_fourier"
result_deep_path = os.path.join(result_folder_path, "2025-07-21_09-02-19_size10width12_x80y80")
result_save_path = os.path.join(model_path, "resufulcor")
os.makedirs(result_save_path, exist_ok=True)

model = build_model()
binarization = Binarization().to(device)

optimizer = optim.Adam(model.parameters(), lr=2e-4)
st_epoch, optimizer_load = load_checkpoint(
    model,
    optimizer,
    model_path,
    model_name="unet_model_epoch_8300",
)
model.eval()

epoch = 12100

outputs_evalu = np.load(os.path.join(result_deep_path, "output_resul8300.npy"))
ny_grid = 11
nx_full = 11

efficiencydeep, digreebinary, distances_deep, X_deep, Y_deep, flip_mask = process_full_grid_to_half(
    outputs_evalu,
    ny_grid=ny_grid,
    nx_full=nx_full,
)

values_deep80 = efficiencydeep.flatten()
x_deep80 = X_deep.flatten()
y_deep80 = Y_deep.flatten()
flip_mask_flat = flip_mask.flatten()

kernel = np.ones((5, 5), np.uint8)

nx = nx + 10
x_metasurface = ((np.arange(0.5, nx) - N_PML - 1 - (t_sub / dx)) * dx)

with torch.no_grad():
    for value, y_ratio, x_ratio, need_flip in zip(
        values_deep80,
        x_deep80,
        y_deep80,
        flip_mask_flat,
    ):
        end_x = nx - 1 - N_PML - 5
        y_middle = int(ny / 2)
        size_x = 80
        size_y = 80
        start_x = end_x - size_x
        start_y = y_middle - size_y / 2

        ind_x = int(x_ratio * size_x + start_x)
        ind_y = int(y_ratio * size_y + start_y)

        b_adj = sparse.csr_matrix((ny, nx))
        b_adj[ind_y, ind_x] = 1

        output, embedding = compute_field_for_point(
            model=model,
            binarization=binarization,
            x_ratio=x_ratio,
            y_ratio=y_ratio,
            epoch=epoch,
        )

        result = build_performance_case(
            output=output,
            rows=rows,
            cols=cols,
            n=n,
            nx=nx,
            ny=ny,
            N_PML=N_PML,
            k0dx=k0dx,
            dx=dx,
            x_metasurface=x_metasurface,
            ind_x=ind_x,
            ind_y=ind_y,
        )

        n_large = result["n_large"]
        psi_tot = result["psi_tot"]

        if need_flip:
            n_large_vis = np.fliplr(n_large)
            psi_tot_vis = np.fliplr(psi_tot)
            save_tag = "_flip"
        else:
            n_large_vis = n_large
            psi_tot_vis = psi_tot
            save_tag = ""

        save_path = os.path.join(result_save_path, f"{ind_x}_{ind_y}out{save_tag}.png")
        save_composite_figure(
            save_path=save_path,
            n_large=n_large_vis,
            psi_tot=psi_tot_vis,
            b_adj=b_adj,
            x_metasurface=x_metasurface,
            y_metasurface=y_metasurface,
            kernel=kernel,
            dx=dx,
        )



        
