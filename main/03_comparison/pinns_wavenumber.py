
"""
Script: pinns_L2_vs_k.py

Description:
    Train PINNs for different wavenumbers k and compute
    relative L2 error vs k.
"""

#%% -------- IMPORTS --------
from datetime import datetime
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Paths
#current_dir = os.getcwd()
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')
os.chdir(current_dir)
sys.path.insert(0, utilities_dir)

# Custom imports
from analytical_solution_functions import sound_hard_circle_calc, mask_displacement
from pinns_solution_functions import (
    set_seed, generate_points, MLP, init_weights,
    train_adam_with_logs, train_lbfgs_with_logs,
    predict_displacement_pinns
)

set_seed(1)

output_dir = "figs_k"
os.makedirs(output_dir, exist_ok=True)

#%% -------- PARAMETERS --------

r_i = np.pi / 4
l_e = np.pi
side_length = 2 * l_e
n_grid = 501

n_Omega_P = 10_000
n_Gamma_I = 100
n_Gamma_E = 250

# Training
adam_lr = 1e-2
adam_iters = 1000
lbfgs_iters = 3000   # reduce for speed

hidden_layers_ = 3
hidden_units_  = 25

# Study range
k_values = np.linspace(1, 10, 10)
l2_errors = []

# Grid
Y, X = np.mgrid[-l_e:l_e:n_grid*1j, -l_e:l_e:n_grid*1j]
R_exact = np.sqrt(X**2 + Y**2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Activation
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

activation_function_ = Sine()

#%%


# -------- Training points --------
x_f, y_f, x_inner, y_inner, x_left, y_left, x_right, y_right, \
x_bottom, y_bottom, x_top, y_top = generate_points(
    n_Omega_P, side_length, r_i, n_Gamma_I, n_Gamma_E
) 

#%% ======================== LOOP OVER k ========================

for k in k_values:

    print(f"\nTraining PINN for k = {k:.2f}")

    # -------- Analytical solution --------
    u_inc_exact, u_scn_exact, u_exact = sound_hard_circle_calc(k, r_i, X, Y, n_terms=None)
    u_inc_exact = mask_displacement(R_exact, r_i, l_e, u_inc_exact)
    u_scn_exact = mask_displacement(R_exact, r_i, l_e, u_scn_exact)
    u_exact = mask_displacement(R_exact, r_i, l_e, u_exact)

    # -------- Model --------
    model = MLP(
        input_size=2,
        output_size=2,
        hidden_layers=hidden_layers_,
        hidden_units=hidden_units_,
        activation_function=activation_function_
    ).to(device)

    model.apply(init_weights)

    results = []
    iter_train = 0

    # -------- Adam --------
    iter_train = train_adam_with_logs(
        model,
        x_f, y_f,
        x_inner, y_inner,
        x_left, y_left,
        x_right, y_right,
        x_bottom, y_bottom,
        x_top, y_top,
        k,
        iter_train,
        results,
        adam_lr,
        num_iter=adam_iters,
        save_csv_path=None,
        save_csv_path_no_datetime=None,
        l_e=l_e,
        r_i=r_i,
        n_grid=n_grid,
        X=X,
        Y=Y,
        R_exact=R_exact,
        u_scn_exact=u_scn_exact,
        u_exact=u_exact
    )

    # -------- L-BFGS --------
    train_lbfgs_with_logs(
        model,
        x_f, y_f,
        x_inner, y_inner,
        x_left, y_left,
        x_right, y_right,
        x_bottom, y_bottom,
        x_top, y_top,
        k,
        iter_start=iter_train,
        results=results,
        lbfgs_lr=1.0,
        num_iter=lbfgs_iters,
        save_csv_path=None,
        save_csv_path_no_datetime=None,
        l_e=l_e,
        r_i=r_i,
        n_grid=n_grid,
        X=X,
        Y=Y,
        R_exact=R_exact,
        u_scn_exact=u_scn_exact,
        u_exact=u_exact
    )

    # -------- Prediction --------
    u_sc_amp_pinns, _, _, _ = predict_displacement_pinns(
        model, l_e, r_i, k, n_grid
    )

    # -------- L2 error --------
    u_exact_masked = np.copy(u_scn_exact)
    u_pred_masked  = np.copy(u_sc_amp_pinns)

    u_exact_masked[R_exact < r_i] = 0
    u_pred_masked[R_exact < r_i] = 0

    rel_L2 = np.linalg.norm(u_exact_masked.real - u_pred_masked.real, 2) / \
             np.linalg.norm(u_exact_masked.real, 2)

    print(f"k = {k:.2f}, PINNs L2 error = {rel_L2:.3e}")

    l2_errors.append(rel_L2)

    # ------------------ Error map ------------------
    error_map = np.abs(u_exact_masked.real - u_pred_masked.real)

    # ------------------ Masked fields for plotting ------------------
    u_pinns_plot = np.ma.masked_where(R_exact < r_i, u_sc_amp_pinns)
    u_exact_plot = np.ma.masked_where(R_exact < r_i, u_scn_exact.real)
    error_plot   = np.ma.masked_where(R_exact < r_i, error_map)

    # ------------------ Plot ------------------
    fig, axs = plt.subplots(1, 3, figsize=(8, 3))

    # --- PINNs Numerical solution ---
    im0 = axs[0].imshow(
        u_pinns_plot,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin='lower',
        vmin=np.min(u_pinns_plot), vmax=np.max(u_pinns_plot),
        cmap='twilight_shifted',
    )
    axs[0].set_title(f'PINNs (k={k:.2f})', fontsize=8)
    plt.colorbar(im0, ax=axs[0], shrink=0.6)

    # --- Exact solution ---
    im1 = axs[1].imshow(
        u_exact_plot,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin='lower',
        vmin=np.min(u_exact_plot), vmax=np.max(u_exact_plot),
        cmap='twilight_shifted',
    )
    axs[1].set_title(f'Exact (k={k:.2f})', fontsize=8)
    plt.colorbar(im1, ax=axs[1], shrink=0.6)

    # --- Error map ---
    im2 = axs[2].imshow(
        error_plot,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin='lower',
        vmin=0, vmax=np.max(error_plot),
        cmap='magma',
    )
    axs[2].set_title(f'Error - L2 = {rel_L2:.2e}', fontsize=8)
    plt.colorbar(im2, ax=axs[2], shrink=0.6)

    plt.tight_layout()

    # ------------------ Save figure ------------------
    filename = os.path.join(output_dir, f"pinns_k_{k:.2f}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.close(fig)
        


#%% -------- PLOT --------
 
os.makedirs("data", exist_ok=True)
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

np.save(f"data/pinns_L2_errors_wavenumbers_{date_str}.npy", np.array(l2_errors))

print("\nSaved L2 errors to:")
print(f"data/pinns_L2_errors_wavenumbers_{date_str}.npy")
 
#%% -------- Sort correctly by k value --------
 
image_files = sorted(
    glob.glob(os.path.join(output_dir, "pinns_k_*.png")),
    key=lambda x: float(os.path.basename(x).replace(".png", "").split("_")[-1])
)

n = len(image_files)

fig, axs = plt.subplots(n, 1, figsize=(6, 1.5*n))

if n == 1:
    axs = [axs]

for ax, img_path in zip(axs, image_files):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"PINNs_vertical_{date_str}.pdf"), dpi=300)
plt.close(fig) 