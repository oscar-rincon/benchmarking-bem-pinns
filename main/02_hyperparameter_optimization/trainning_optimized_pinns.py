
# ============================================================
"""
Script: trainning_optimized_pinns.py

Description:
    This script trains a Physics-Informed Neural Network (PINN) 
    to solve the scattering problem for a sound-hard circular obstacle. 
    It first runs Adam optimization, followed by L-BFGS refinement, 
    while logging training progress and metrics.

Inputs:
    - Problem parameters: wave number k, geometry (inner radius, domain size), 
      number of training points.
    - Neural network architecture (layers, neurons, activation).
    - Exact analytical displacement fields for error evaluation.

Outputs:
    - Training log (CSV) with losses, errors, and runtime, saved in ./data/
    - Log file (TXT) with script name and execution time, saved in ./logs/
"""
# ============================================================

#%%
from datetime import datetime

import sys
import os
import time
import random
 
 
# Set the current directory and utilities path
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the notebook's directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch 
import torch.nn as nn
import importlib
import analytical_solution_functions
import bem_solution_functions
import pinns_solution_functions
import plotting_functions

# Reload them each time this file runs
importlib.reload(analytical_solution_functions)
importlib.reload(bem_solution_functions)
importlib.reload(pinns_solution_functions)
importlib.reload(plotting_functions)

# Importar funciones personalizadas
from analytical_solution_functions import sound_hard_circle_calc, mask_displacement, calculate_relative_errors
from pinns_solution_functions import set_seed, generate_points, MLP, init_weights, train_adam_with_logs, train_lbfgs_with_logs, initialize_and_load_model, predict_displacement_pinns, process_displacement_pinns
set_seed(42)
 
#%% Start time measurement and folder setup

# Record start time
start_time = time.time()

# Get script name without extension
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Define base output folders
output_folder = os.path.join(os.path.dirname(__file__), "logs")
data_folder = os.path.join(os.path.dirname(__file__), "data")

# Create folders if they do not exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)
#%%

# Parámetros
r_i = np.pi / 4  # Radio interno
l_e = np.pi      # Radio externo
side_length = 2 * l_e
k = 3.0
n_Omega_P = 10_000
n_Gamma_I = 100
n_Gamma_E = 250
n_grid = 501

Y, X = np.mgrid[-l_e:l_e:n_grid*1j, -l_e:l_e:n_grid*1j]
R_exact = np.sqrt(X**2 + Y**2)
u_inc_exact, u_scn_exact, u_exact = sound_hard_circle_calc(k, r_i, X, Y, n_terms=None)
u_inc_exact = mask_displacement(R_exact, r_i, l_e, u_inc_exact)
u_scn_exact = mask_displacement(R_exact, r_i, l_e, u_scn_exact)
u_exact = mask_displacement(R_exact, r_i, l_e, u_exact)

results = []
iter_train = 0

adam_lr        = 1e-2
hidden_layers_ = 3
hidden_units_  = 25
 
adam_iters     = 1_000
lbfgs_iters    = 5_000

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)
activation_function_ = Sine()

x_f, y_f, x_inner, y_inner, x_left, y_left, x_right, y_right, x_bottom, y_bottom, x_top, y_top = generate_points(
    n_Omega_P, side_length, r_i, n_Gamma_I, n_Gamma_E
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = MLP(
    input_size=2,
    output_size=2,
    hidden_layers=hidden_layers_,
    hidden_units=hidden_units_,
    activation_function=activation_function_,
).to(device)

model.apply(init_weights)

# Ensure logs directory exists
os.makedirs("data", exist_ok=True)

# Build CSV filename
# Current date as YYYYMMDD
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

csv_name = f"data/training_log_{hidden_layers_}_layers_{hidden_units_}_neurons_{date_str}.csv"
csv_name_no_date = f"data/training_log_{hidden_layers_}_layers_{hidden_units_}_neurons.csv"

# --- Adam training with logs ---
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
    save_csv_path=csv_name,  # dynamic filename
    save_csv_path_no_datetime=csv_name_no_date,  # static filename
    l_e=l_e,
    r_i=r_i,
    n_grid=n_grid,
    X=X,
    Y=Y,
    R_exact=R_exact,
    u_scn_exact=u_scn_exact,
    u_exact=u_exact
)

# --- L-BFGS training with logs (continues iteration count) ---
iter_train = train_lbfgs_with_logs(
    model,
    x_f, y_f,
    x_inner, y_inner,
    x_left, y_left,
    x_right, y_right,
    x_bottom, y_bottom,
    x_top, y_top,
    k,
    iter_start=iter_train,   # continue from Adam
    results=results,
    lbfgs_lr=1.0,
    num_iter=lbfgs_iters,
    save_csv_path=csv_name,  # same CSV, continues appending
    save_csv_path_no_datetime=csv_name_no_date,  # also save without date
    l_e=l_e,
    r_i=r_i,
    n_grid=n_grid,
    X=X,
    Y=Y,
    R_exact=R_exact,
    u_scn_exact=u_scn_exact,
    u_exact=u_exact
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