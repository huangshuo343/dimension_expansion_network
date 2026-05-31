import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve


def fun_s(u):
    """
    Complex coordinate stretching profile for PML.
    """
    pi = np.pi
    power_kappa = 3
    kappa_max = 15
    power_sigma = 3
    lambda_over_dx = 40
    sigma_over_omega_max = 0.8 * (power_sigma + 1) * lambda_over_dx / (2 * pi)

    u = np.asarray(u)
    kappa = 1 + (kappa_max - 1) * u**power_kappa
    sigma_over_omega = sigma_over_omega_max * u**power_sigma

    return kappa + 1j * sigma_over_omega


def build_laplacian_1d_PML(N, N_PML_L, N_PML_R):
    """
    Builds dx^2 times the 1D Laplacian with PML on both sides.
    """
    pi = np.pi
    N = int(N)

    ddx_1 = sparse.spdiags([np.ones(N), -np.ones(N)], [0, -1], N + 1, N)
    ddx_2 = -ddx_1.transpose()

    if N_PML_L == 0 and N_PML_R == 0:
        A = ddx_2 @ ddx_1
        return A

    power_kappa = 3
    kappa_max = 15
    power_sigma = 3
    lambda_over_dx = 40
    sigma_over_omega_max = 0.8 * (power_sigma + 1) * lambda_over_dx / (2 * pi)

    s_half = np.array([1 + 0j for _ in range(N + 1)])
    s_int = np.array([1 + 0j for _ in range(N)])

    if N_PML_R > 0:
        s_half[(N - N_PML_R):(N + 1)] = fun_s(np.arange(0.5, N_PML_R + 1, 1) / (N_PML_R + 1))
        s_int[(N - N_PML_R):N] = fun_s(np.arange(1, N_PML_R + 1, 1) / (N_PML_R + 1))

    if N_PML_L > 0:
        s_half[N_PML_L::-1] = fun_s(np.arange(0.5, N_PML_L + 1, 1) / (N_PML_L + 1))
        s_int[N_PML_L - 1::-1] = fun_s(np.arange(1, N_PML_L + 1, 1) / (N_PML_L + 1))

    A = sparse.spdiags(1.0 / s_int, 0, N, N) @ ddx_2 @ sparse.spdiags(1.0 / s_half, 0, N + 1, N + 1) @ ddx_1
    return A


def build_laplacian_2d_PML(nx, ny, N_PML_L, N_PML_R, N_PML_B, N_PML_T):
    """
    Builds dx^2 times the 2D Laplacian with PML on all four sides.
    """
    A = sparse.kron(build_laplacian_1d_PML(nx, N_PML_L, N_PML_R), sparse.eye(ny)) + \
        sparse.kron(sparse.eye(nx), build_laplacian_1d_PML(ny, N_PML_B, N_PML_T))
    return A


def myfunc_Rod(n, rows, cols, n_design, N_PML, k0dx, dx, x_metasurface, n_bg, ind_x, ind_y, psi_point_in_air):
    """
    Forward simulation and gradient evaluation for the design region.
    """
    n_design_tiled = np.tile(n_design[:, None], (1, np.sum(cols)))
    n[np.ix_(rows, cols)] = n_design_tiled

    ny, nx = n.shape
    A = build_laplacian_2d_PML(nx, ny, N_PML, N_PML, N_PML, N_PML) + \
        sparse.diags((k0dx ** 2) * n.flatten("F") ** 2, 0, shape=(nx * ny, nx * ny))

    psi_in_2D = np.zeros((ny, nx), dtype=np.complex64)
    psi_in_2D[40:-40, N_PML + 1] = 1.0

    b_direct = psi_in_2D
    psi_sca = spsolve(A, b_direct.flatten("F")).reshape(ny, nx, order="F")
    psi_tot = psi_in_2D + psi_sca

    b_adj = -k0dx ** 2 * (n ** 2 - n_bg ** 2) * psi_point_in_air
    psi_sca_adj = spsolve(A, b_adj.flatten("F")).reshape(ny, nx, order="F")
    psi_tot_adj = psi_sca_adj + psi_point_in_air

    FoM = -np.abs(psi_tot[ind_y, ind_x]) ** 2

    start_rows = np.where(rows == 1)[0][0]
    end_rows = np.where(rows == 1)[0][-1] + 1
    start_cols = np.where(cols == 1)[0][0]
    end_cols = np.where(cols == 1)[0][-1] + 1

    gradient = -1 * (-4) * k0dx**2 * np.sum(
        np.real(
            n[start_rows:end_rows, start_cols:end_cols]
            * psi_tot[start_rows:end_rows, start_cols:end_cols]
            * psi_tot_adj[start_rows:end_rows, start_cols:end_cols]
        ),
        axis=1,
    )

    return FoM, gradient


# -----------------------------
# Constants and design region initialization
# -----------------------------
c = 3e8

lambda_ = 500e-9
D = 20 * lambda_
n_bg = 1
f = 10 * lambda_

NA = n_bg * np.sin(np.arctan(D / (2 * f)))

D_extra = 2 * lambda_
dx = lambda_ / 15
x_extra = 30 * dx

N_PML = 10

n_post = 2.4
n_sub = 1.5

t_post = 12 * dx
t_sub = 15 * dx

k0dx = 2 * np.pi / lambda_ * dx

D = round(D / dx) * dx + dx * (round(D / dx) % 2 == 0)
D_extra = round(D_extra / dx) * dx

W_tot = D + 2 * D_extra

nx = 2 * N_PML + np.ceil((t_sub + t_post + 2 * dx + f + x_extra) / dx).astype(int)
ny = int(round(W_tot / dx)) + 2 * N_PML

x_metasurface = (np.arange(0.5, nx) - N_PML - 1 - (t_sub / dx)) * dx
y_metasurface = (np.arange(0.5, ny) - (ny / 2)) * dx

n = np.ones((ny, nx))

X, Y = np.meshgrid(x_metasurface, y_metasurface, indexing="xy")

n[(X < 0) & (X >= -t_sub)] = n_sub

n_mid = 0.5 * (n_post + n_bg)

mask = (X <= t_post) & (X >= 0) & (np.abs(Y) <= D / 2)
n[mask] = n_mid

rows = np.any(mask, axis=1)
cols = np.any(mask, axis=0)

n_design = n[np.ix_(rows, cols)]


def aim_point_in_air(ind_x, ind_y, nx, ny, n_PML, k0dx):
    """
    Compute the point-source field in air.
    """
    n_point_source = np.ones((ny, nx))

    A = build_laplacian_2d_PML(nx, ny, N_PML, N_PML, N_PML, N_PML) + \
        sparse.diags((k0dx**2) * n_point_source.flatten("F"), 0, shape=(nx * ny, nx * ny))

    bb = sparse.csr_matrix((ny, nx))
    bb = bb.toarray()
    bb[ind_y, ind_x] = 1
    bb = sparse.csr_matrix(bb)

    psi_point_in_air = spsolve(A, bb.toarray().flatten("F"))
    psi_point_in_air = psi_point_in_air.reshape(ny, nx, order="F")
    return psi_point_in_air



