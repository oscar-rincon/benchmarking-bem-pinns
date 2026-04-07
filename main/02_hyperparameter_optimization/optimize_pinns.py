#%%
"""
Script: Hyperparameter optimization of Physics-Informed Neural Networks (PINNs)

Description:
    This script performs hyperparameter optimization using Optuna for PINNs 
    applied to the Helmholtz equation with a circular scatterer. It compares
    different architectures (hidden layers, units, activations, learning rates),
    trains models using Adam + L-BFGS optimizers, and evaluates them against
    the analytical solution. The best configuration is stored, and runtime is logged.

Inputs:
    - Analytical solution functions (Bessel expansions for circular scatterer).
    - Utility functions for PINN training and evaluation.
    - Problem parameters (wave number k, geometry, number of training points).

Outputs:
    - Optuna study object stored in `data/study_<timestamp>.pkl`
    - Log file with execution time in `logs/<script_name>_log_<timestamp>.txt`
    - Console printout of best hyperparameters and error values.
"""
#%% Imports y configuración
from datetime import datetime
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import optuna
import pandas  
import joblib

# Set the current directory and utilities path
current_dir = os.getcwd()
utilities_dir = os.path.join(current_dir, '../../utilities')
os.chdir(current_dir)
sys.path.insert(0, utilities_dir)

# Importar funciones personalizadas
from analytical_solution_functions import sound_hard_circle_calc, mask_displacement, calculate_relative_errors
from pinns_solution_functions import set_seed, generate_points, MLP, init_weights, train_adam, train_lbfgs, initialize_and_load_model, predict_displacement_pinns, process_displacement_pinns
set_seed(42)

#%% Start time measurement
# Record start time
start_time = time.time()

# Get script name without extension
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Define output folder (e.g., "logs" inside the current script directory)
output_folder = os.path.join(os.path.dirname(__file__), "logs")

# Create folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Define output file path
output_file = os.path.join(output_folder, f"{script_name}_log.txt")


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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
 
#%%
# Activación personalizada
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

#%%
# Función objetivo para Optuna

def objective(trial):
    results = []
    iter_train = 0
    adam_lr        = trial.suggest_categorical("adam_lr", [1e-1, 1e-2, 1e-3])
    hidden_layers_ = trial.suggest_categorical("hidden_layers", [1, 2, 3])
    hidden_units_  = trial.suggest_categorical("hidden_units", [25, 50, 75])
    activation_str = trial.suggest_categorical("activation", ["Tanh", "Sigmoid", "Sine"])

    #adam_fraction = 0.5
    #total_iter    = 6_000
    adam_iters    = 1_000 #int(total_iter * adam_fraction)
    lbfgs_iters   = 5_000 #total_iter - adam_iters

    if activation_str == "Tanh":
        activation_function_ = nn.Tanh()
    elif activation_str == "ReLU":
        activation_function_ = nn.ReLU()
    elif activation_str == "Sigmoid":
        activation_function_ = nn.Sigmoid()
    elif activation_str == "Sine":
        activation_function_ = Sine()

    x_f, y_f, x_inner, y_inner, x_left, y_left, x_right, y_right, x_bottom, y_bottom, x_top, y_top = generate_points(
        n_Omega_P, side_length, r_i, n_Gamma_I, n_Gamma_E
    )

    model = MLP(
        input_size=2,
        output_size=2,
        hidden_layers=hidden_layers_,
        hidden_units=hidden_units_,
        activation_function=activation_function_,
    ).to(device)

    model.apply(init_weights)

    train_adam(
        model, x_f, y_f, x_inner, y_inner, x_left, y_left,
        x_right, y_right, x_bottom, y_bottom, x_top, y_top,
        k, iter_train, results, adam_lr, num_iter=adam_iters
    )

    train_lbfgs(
        model, x_f, y_f, x_inner, y_inner, x_left, y_left,
        x_right, y_right, x_bottom, y_bottom, x_top, y_top,
        k, iter_train, results, 1, num_iter=lbfgs_iters
    )

    u_sc_amp_pinns, u_sc_phase_pinns, u_amp_pinns, u_phase_pinns = predict_displacement_pinns(
        model, l_e, r_i, k, n_grid
    )

    u_sc_amp_pinns, u_sc_phase_pinns, u_amp_pinns, u_phase_pinns, diff_uscn_amp_pinns, diff_u_scn_phase_pinns = process_displacement_pinns(
        model, l_e, r_i, k, n_grid, X, Y, R_exact, u_scn_exact
    )

    rel_error_uscn_amp_pinns, rel_error_uscn_phase_pinns, max_diff_uscn_amp_pinns, min_diff_uscn_amp_pinns, max_diff_u_phase_pinns, min_diff_u_phase_pinns = calculate_relative_errors(
        u_scn_exact, u_exact, diff_uscn_amp_pinns,
        diff_u_scn_phase_pinns, R_exact, r_i
    )

    mean_rel_error_pinns = (rel_error_uscn_amp_pinns + rel_error_uscn_phase_pinns) / 2

    # print trial info at the end
    print(f"Trial {trial.number} finished | "
          f"adam_lr={adam_lr}, layers={hidden_layers_}, units={hidden_units_}, act={activation_str} "
          f"--> Mean Rel Error: {mean_rel_error_pinns:.6f}")

    return mean_rel_error_pinns

#%%
# Ejecutar Optuna
sampler = optuna.samplers.TPESampler(seed=42)  # Make the sampler behave in a deterministic way.

study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=50)
print("Best trial:")
best_trial = study.best_trial
print(f"  Value (mean error): {best_trial.value:.3e}")
print("  Params:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

#%%
# Creaate data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Get current date and time
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Save study with date in filename
joblib.dump(study, f"data/study_{date_str}.pkl")
joblib.dump(study, "data/study.pkl")  # also save without date for easy access

df = study.trials_dataframe()

# Guardar el DataFrame en un archivo CSV
df.to_csv("data/study.csv", index=False)

# Filter by the parameters of interest
df_filtered = df[(df["params_hidden_layers"] == 3) & (df["params_hidden_units"] == 25)]

# Sort by objective value (ascending = best first)
df_filtered_sorted = df_filtered.sort_values(by="value", ascending=True)

# Save the filtered and sorted DataFrame to a new CSV file
df_filtered_sorted.to_csv("data/study_filtered_sorted.csv", index=False)

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
