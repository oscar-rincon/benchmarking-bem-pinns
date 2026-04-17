
from datetime import datetime
import sys, os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimg
import pandas as pd
from matplotlib.lines import Line2D

# Paths
#current_dir = os.getcwd()
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')
sys.path.insert(0, utilities_dir)


# Paths
current_dir = os.getcwd()
utilities_dir = os.path.join(current_dir, '../../utilities')
sys.path.insert(0, utilities_dir)

# Custom imports
from analytical_solution_functions import sound_hard_circle_calc, mask_displacement

import importlib
import pinns_solution_functions

importlib.reload(pinns_solution_functions)
from pinns_solution_functions import (
    set_seed, generate_points, MLP, init_weights,
    train_adam_with_logs_adaptive, train_lbfgs_with_logs_adaptive,
    predict_displacement_pinns, mse_f_adaptive, mse_b_adaptive,
    split_adaptive_points, to_training_format,prepare_points_and_residuals,
    compute_rad_probability
)

# Seed
set_seed(42)

output_dir = "figs_sampling"
os.makedirs(output_dir, exist_ok=True)

#gaussian_kde
from scipy.stats import gaussian_kde

#%%

def plot_sampling_and_residual_control(
    x_f_vis, y_f_vis,
    x_inner_vis, y_inner_vis,
    x_left_vis, y_left_vis,
    x_right_vis, y_right_vis,
    x_bottom_vis, y_bottom_vis,
    x_top_vis, y_top_vis,        
    x_f, y_f,
    x_inner, y_inner,
    x_left, y_left,
    x_right, y_right,
    x_bottom, y_bottom,
    x_top, y_top,
    iter
):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(2.3, 1.9),
        gridspec_kw={'width_ratios': [1.0, 0.92]}  # right plot bigger
    )

    # ======================================================
    # LEFT: Residual map
    # ======================================================
    axes[0].scatter(
        x_f_vis.detach().cpu().numpy(),
        y_f_vis.detach().cpu().numpy(),
        s=1.5, rasterized=True,
        color="#AFAFAF", edgecolors='none'
    )

    axes[0].scatter(x_inner_vis.detach().cpu(), y_inner_vis.detach().cpu(), s=1.5, color='#1f77b4',edgecolors='none')
    axes[0].scatter(x_left_vis.detach().cpu(),  y_left_vis.detach().cpu(),  s=1.5, color='#1f77b4',edgecolors='none')
    axes[0].scatter(x_right_vis.detach().cpu(), y_right_vis.detach().cpu(), s=1.5, color="#1f77b4",edgecolors='none')
    axes[0].scatter(x_bottom_vis.detach().cpu(),y_bottom_vis.detach().cpu(),s=1.5, color='#1f77b4',edgecolors='none')
    axes[0].scatter(x_top_vis.detach().cpu(),   y_top_vis.detach().cpu(),   s=1.5, color='#1f77b4',edgecolors='none')

    axes[0].set_aspect('equal')
    axes[0].axis('off')
 
    # ======================================================
    # RIGHT: Sampling points (your original plot)
    # ======================================================

    x_all = torch.cat([
        x_f, x_inner, x_left, x_right, x_bottom, x_top
    ]).detach().cpu().numpy().flatten()

    y_all = torch.cat([
        y_f, y_inner, y_left, y_right, y_bottom, y_top
    ]).detach().cpu().numpy().flatten()

    xy = np.vstack([x_all, y_all])
    kde = gaussian_kde(xy)

    xmin, xmax = x_all.min(), x_all.max()
    ymin, ymax = y_all.min(), y_all.max()

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 200),
        np.linspace(ymin, ymax, 200)
    )

    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(grid_coords).reshape(xx.shape)

    r_i = np.pi / 4

    # Distance from center
    rr = np.sqrt(xx**2 + yy**2)

    # Mask inside the circle
    zz_masked = np.where(rr < r_i, np.nan, zz)

    # Plot
    im = axes[1].imshow(
        zz_masked,
        extent=[xmin, xmax, ymin, ymax],
        origin='lower',
        aspect='equal',
        cmap='magma',
        vmin=0, vmax=0.05#11#np.nanmax(zz_masked)
    )

    axes[1].axis('off')
 
    fig.text(
        -0.10, 0.50, "Uniform",   # move to left
        rotation=90,
        va='center',
        ha='left',
        fontsize=8
    ) 
    # ======================================================
    # Layout & save
    # ======================================================
    plt.tight_layout(pad=0.5, w_pad=1.0)
    plt.savefig(
        f"figures/residual_sampling_control_{iter+1}.svg",
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.01
    )
    plt.close()

#%%

r_i = np.pi / 4
l_e = np.pi
side_length = 2 * l_e
n_grid = 501
k = 3.0

# Base sampling
n_Omega_P = 10_000
n_Gamma_I = 100
n_Gamma_E = 250

# For adaptative
n_Omega_P_adaptive = 10_000 * 10
n_Gamma_I_adaptive = 100 * 10
n_Gamma_E_adaptive = 250 * 10

# For visualization
n_Omega_P_vis = 10_000 //10
n_Gamma_I_vis = 100 //5
n_Gamma_E_vis = 250 //10

# Training
adam_lr = 1e-2
adam_iters = 1_000
lbfgs_iters = 5_000

hidden_layers_ = 3
hidden_units_  = 25

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

x_f, y_f, x_inner, y_inner, x_left, y_left, x_right, y_right, \
x_bottom, y_bottom, x_top, y_top = generate_points(
    n_Omega_P, side_length, r_i, n_Gamma_I, n_Gamma_E
)

# Generate points for visualization
x_f_vis, y_f_vis, x_inner_vis, y_inner_vis, x_left_vis, y_left_vis, x_right_vis, y_right_vis, \
x_bottom_vis, y_bottom_vis, x_top_vis, y_top_vis = generate_points(
    n_Omega_P_vis, side_length, r_i, n_Gamma_I_vis, n_Gamma_E_vis
)

#%%

u_inc_exact, u_scn_exact, u_exact = sound_hard_circle_calc(k, r_i, X, Y, n_terms=None)
u_inc_exact = mask_displacement(R_exact, r_i, l_e, u_inc_exact)
u_scn_exact = mask_displacement(R_exact, r_i, l_e, u_scn_exact)
u_exact = mask_displacement(R_exact, r_i, l_e, u_exact)

#%%

model = MLP(
    input_size=2,
    output_size=2,
    hidden_layers=hidden_layers_,
    hidden_units=hidden_units_,
    activation_function=activation_function_
).to(device)

model.apply(init_weights)

#%%

it = 0
 

plot_sampling_and_residual_control(
    x_f_vis, y_f_vis,
    x_inner_vis, y_inner_vis,
    x_left_vis, y_left_vis,
    x_right_vis, y_right_vis,
    x_bottom_vis, y_bottom_vis,
    x_top_vis, y_top_vis,        
    x_f, y_f,
    x_inner, y_inner,
    x_left, y_left,
    x_right, y_right,
    x_bottom, y_bottom,
    x_top, y_top,
    it
    )

#%%

# # ================= RAD LOOP =================
n_RAD_iters = 3

results = []
iter_train = 0

# -------- ADAM --------
iter_train = train_adam_with_logs_adaptive(
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

for it in range(n_RAD_iters):


    print(f"\n===== RAD Iteration {it+1}/{n_RAD_iters} =====")

    # -------- L-BFGS --------
    iter_train, res_f, res_inner, res_left, res_right, res_bottom, res_top = train_lbfgs_with_logs_adaptive(
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
        save_csv_path_no_datetime=f'data/control_uniform_sampling.csv',
        l_e=l_e,
        r_i=r_i,
        n_grid=n_grid,
        X=X,
        Y=Y,
        R_exact=R_exact,
        u_scn_exact=u_scn_exact,
        u_exact=u_exact
    )


