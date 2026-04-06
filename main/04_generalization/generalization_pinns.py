
# ============================================================
"""
Script: generalization_pinns.py

Description:
    This script evaluates a trained Physics-Informed Neural Network (PINN) 
    for the acoustic scattering problem of a sound-hard circular obstacle. 
    The workflow includes:
        - Computing the analytical reference solution.
        - Loading the trained PINN model and predicting the scattered field.
        - Computing relative L2 errors between analytical and PINN solutions.
        - Generating error line profiles along a defined axis.
        - Plotting displacement fields and error distributions.

Inputs:
    - Wave number k, inner radius r_i, domain size l_se.
    - Trained PINN model (stored in ./models/).
    - Grid resolution for evaluation.

Outputs:
    - Error metrics saved to ./data/error_results.txt
    - Plots of PINN vs analytical displacements with error visualization.
    - Log file (TXT) with script name, timestamp, and execution time, saved in ./logs/
"""

#%% ======================== IMPORTS ========================
# Standard library imports
from datetime import datetime
import sys
import os
import time
import numpy as np
import torch
from torch import nn
from scipy.interpolate import griddata
import pandas as pd

#%% ======================== PATH SETUP ========================
# Set the current directory and utilities path
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the script's directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)

#%% ======================== FUNCTION IMPORTS ========================
from analytical_solution_functions import sound_hard_circle_calc, mask_displacement
from plotting_functions import plot_pinns_displacements_with_errorline
from pinns_solution_functions import set_seed, initialize_and_load_model, predict_displacement_pinns, process_displacement_pinns
set_seed(42)

#%% ======================== LOGGING SETUP ========================
# Record start time
start_time = time.time()

# Get script name without extension
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Define output folder (e.g., "logs" inside the current script directory)
output_folder = os.path.join(current_dir, "logs")
os.makedirs(output_folder, exist_ok=True)

#%% ======================== PARAMETERS ========================
r_i = np.pi / 4  # Inner radius
l_se = 10 * np.pi  # Outer semi-length
k = 3  # Wave number
n_grid = 2 * 501  # Number of grid points in x and y
r_exclude = np.pi / 4 # Radius of excluded circular region

#%% ======================== ANALYTICAL SOLUTION ========================
Y, X = np.mgrid[-l_se:l_se:n_grid*1j, -l_se:l_se:n_grid*1j]
R_exact = np.sqrt(X**2 + Y**2)
u_inc_exact, u_scn_exact, u_exact = sound_hard_circle_calc(k, r_i, X, Y, n_terms=None)

# Mask the displacement
u_inc_exact = mask_displacement(R_exact, r_i, l_se, u_inc_exact)
u_scn_exact = mask_displacement(R_exact, r_i, l_se, u_scn_exact)
u_exact = mask_displacement(R_exact, r_i, l_se, u_exact)

#%% ======================== MODEL SETUP ========================
n_Omega_P = 10_000
n_Gamma_I = 100
n_Gamma_E = 250
l_e = 10 * np.pi
k = 3.0
iter = 0
side_length = 2 * l_e

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)
activation_function_ = Sine()

model_path = 'models/3_layers_25_neurons_best.pt'
model = initialize_and_load_model(model_path, 3, 25, activation_function_)

#%% ======================== PINNs PREDICTION & PROCESSING ========================
u_sc_amp_pinns, u_sc_phase_pinns, u_amp_pinns, u_phase_pinns = predict_displacement_pinns(
    model, l_e, r_i, k, n_grid
)

u_sc_amp_pinns, u_sc_phase_pinns, u_amp_pinns, u_phase_pinns, diff_uscn_amp_pinns, diff_u_scn_phase_pinns = process_displacement_pinns(
    model, l_e, r_i, k, n_grid, X, Y, R_exact, u_scn_exact
)

#%% ======================== ERROR COMPUTATION ========================
R_grid = np.sqrt(X**2 + Y**2)
u_scn_exact_masked = np.copy(u_scn_exact)
u_scn_amp_masked = np.copy(u_sc_amp_pinns)
u_scn_exact_masked[R_grid < r_i] = 0
u_scn_amp_masked[R_grid < r_i] = 0

relative_error = np.linalg.norm(u_scn_exact_masked.real - u_scn_amp_masked.real, 2) / np.linalg.norm(u_scn_exact_masked.real, 2)
print(f"Relative L2 error: {relative_error:.2e}")

# Save result to txt with timestamp
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# File paths
file_no_date = os.path.join("data", "pinns_error_results.txt")
file_with_date = os.path.join("data", f"pinns_error_results_{date_str}.txt")

# Save result (no date, always overwritten)
with open(file_no_date, "w") as f:
    f.write("Relative L2 error computation\n")
    f.write("=============================\n")
    f.write(f"Relative L2 error: {relative_error:.6e}\n")

# Save result (with timestamp in filename, historical)
with open(file_with_date, "w") as f:
    f.write(f"Error results generated on {date_str}\n")
    f.write("=============================\n")
    f.write(f"Relative L2 error: {relative_error:.6e}\n")

print(f"Error results saved to '{file_no_date}' (latest)")
print(f"Error results also saved to '{file_with_date}' (historical)")

#%% ======================== ERROR LINE PROFILE ========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_line = np.linspace(np.pi, 2 * np.pi, 200)
y_line = np.zeros_like(x_line)

X_ten = torch.tensor(x_line).float().reshape(-1, 1).to(device)
Y_ten = torch.tensor(y_line).float().reshape(-1, 1).to(device)

model = initialize_and_load_model(model_path, 3, 25, Sine()).to(device)
domain_ten = torch.cat([X_ten, Y_ten], dim=1)

u_sc_pred = model(domain_ten)
u_sc_amp_pred = u_sc_pred[:, 0].detach().cpu().numpy().reshape(x_line.shape)
u_sc_phase_pred = u_sc_pred[:, 1].detach().cpu().numpy().reshape(x_line.shape)

u_inc_line, u_scn_exact_line, u_tot_exact_line = sound_hard_circle_calc(k, r_exclude, x_line, y_line)

error_line = np.abs(np.real(u_scn_exact_line) - u_sc_amp_pred)
rel_error_line = error_line / np.max(u_inc_line + u_sc_amp_pred)


#%% ======================== SQUARE RING ERRORS (PINNs) ========================

error_field = np.abs(np.real(u_scn_exact) - u_sc_amp_pinns)

n_levels = np.arange(1, 11)

error_map = np.zeros_like(error_field)
prev_bound = 0.0
avg_errors = []

for n_level in n_levels:
    current_bound = n_level * np.pi

    mask_outer = (np.abs(X) <= current_bound) & (np.abs(Y) <= current_bound)
    mask_inner = (np.abs(X) <= prev_bound) & (np.abs(Y) <= prev_bound)

    ring_mask = mask_outer & (~mask_inner)

    if np.any(ring_mask):
        values = error_field[ring_mask]

        if np.any(~np.isnan(values)):
            avg_error = np.nanmean(values)
        else:
            avg_error = np.nan

        avg_errors.append(avg_error)
        error_map[ring_mask] = avg_error

        #print(f"PINNs n={n_level}, Avg Error: {avg_error:.2e}")

    else:
        avg_errors.append(np.nan)

    prev_bound = current_bound

# Optional normalization (same as BEM)
error_map = error_map / np.max(np.abs(np.real(u_scn_exact)))
#%% ======================== PLOTTING ========================
 
 
plot_pinns_displacements_with_errorline(
    X, Y,
    u_sc_amp_pinns,
    np.real(u_inc_exact) + u_sc_amp_pinns,
    np.abs(np.real(u_scn_exact) - u_sc_amp_pinns),
    u_sc_phase_pinns,
    u_sc_phase_pinns + np.real(u_inc_exact),
    np.abs(np.imag(u_scn_exact) - u_sc_phase_pinns),
    x_line,
    rel_error_line,
    avg_errors
)

#%% Record runtime and save to .txt
end_time = time.time()
elapsed_time = end_time - start_time

# Build log text
log_text = f"Script: {script_name}\nExecution time (s): {elapsed_time:.2f}\n"

# Get current date and time
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define log filenames inside the logs folder
log_filename_with_date = os.path.join(output_folder, f"{script_name}_log_{date_str}.txt")
log_filename_no_date   = os.path.join(output_folder, f"{script_name}_log.txt")

# Write log file with date
with open(log_filename_with_date, "w") as f:
    f.write(log_text)

# Write log file without date
with open(log_filename_no_date, "w") as f:
    f.write(log_text)

print(f"Log saved to: {log_filename_with_date}")
print(f"Log also saved to: {log_filename_no_date}")
# %%