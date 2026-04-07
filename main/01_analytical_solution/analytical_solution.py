
#%%
"""
Script: analytical_solution.py

Description:
    This script computes the analytical solution of the 
    scattering problem for a sound-hard circular obstacle.
    It evaluates the incident, scattered, and total displacement 
    fields over a square domain, applies masking, and produces 
    corresponding plots for visualization.

Inputs:
    - Analytical solution functions (Bessel expansions).
    - Problem parameters: wave number k, geometry (inner radius, domain size), grid resolution.

Outputs:
    - Plots of the real and imaginary parts of the displacements.
    - Log file saved in ./logs/ with script name and runtime.
"""
# ==========================================================
#%% Script start and time measurement
import time
import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

#%% Start time measurement
start_time = time.time()

# Get script name without extension
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Define output folder (e.g., "logs" inside the current script directory)
output_folder = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(output_folder, exist_ok=True)

#%% Directory setup
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change working directory
os.chdir(current_dir)
sys.path.insert(0, utilities_dir)

# Import functions
from analytical_solution_functions import sound_hard_circle_calc, mask_displacement
from plotting_functions import plot_exact_displacement

#%% Parameters and grid
r_i = np.pi/4   # Inner radius
l_se = np.pi    # Outer semi-length
k = 7           # Wave number
n_grid = 501    # Number of grid points in x and y 

Y, X = np.mgrid[-l_se:l_se:n_grid*1j, -l_se:l_se:n_grid*1j]
R_exact = np.sqrt(X**2 + Y**2)

#%% Compute displacements
u_inc_exact, u_scn_exact, u_exact = sound_hard_circle_calc(k, r_i, X, Y, n_terms=None)

u_inc_exact = mask_displacement(R_exact, r_i, l_se, u_inc_exact)
u_scn_exact = mask_displacement(R_exact, r_i, l_se, u_scn_exact)
u_exact = mask_displacement(R_exact, r_i, l_se, u_exact)

#%% Plot
plot_exact_displacement(
    X, Y,
    np.real(u_inc_exact), np.real(u_scn_exact), np.real(u_exact),
    np.imag(u_inc_exact), np.imag(u_scn_exact), np.imag(u_exact)
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
