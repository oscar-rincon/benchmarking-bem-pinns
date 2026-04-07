# ============================================================
"""
Script: comparison_figure_compose.py

Description:
    This script composes multiple SVG figures (BEM error, PINNs error, 
    relative error vs. runtime, etc.) into a combined comparison figure. 
    It saves the output as both SVG and PDF formats.

Inputs:
    - figures/bem_error.svg
    - figures/pinns_error.svg
    - figures/rel_error_time.svg
    - figures/comparison_base.svg

Outputs:
    - figures/errors.svg
    - figures/06_accuracy_time_error_bem_pinns.svg
    - figures/06_accuracy_time_error_bem_pinns.pdf
    - Log file (TXT) with script name, start/end times, and execution 
      duration, saved in ./logs/ with timestamped filename
"""
# ============================================================

#%% Imports
# Standard library imports
import sys
import os
import time
from datetime import datetime
import cairosvg

# Third-party imports
from svgutils.compose import *

#%% Paths and setup
# Set the current directory and utilities path
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the script directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)

#%% Start time measurement
# Record start time
start_time = time.time()

# Get script name without extension
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Define log folder
output_folder = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(output_folder, exist_ok=True)

# Define figures folder
figures_folder = os.path.join(current_dir, "figures")
os.makedirs(figures_folder, exist_ok=True)

#%% Create intermediate figure (BEM + PINNs errors)
svg1 = SVG(os.path.join(figures_folder, "bem_error.svg")).scale(1.0)
svg2 = SVG(os.path.join(figures_folder, "pinns_error.svg")).scale(1.0)
svg3 = SVG(os.path.join(figures_folder, "bem_vs_pinns_L2_vs_k.svg")).scale(1.0)

Figure(
    631,  # total width
    480,  # height
    Panel(svg1),
    Panel(svg2).move(230, 0),
    Panel(svg3).move(0, 465)
).scale(1.3).save(os.path.join(figures_folder, "errors.svg"))

#%% Create final combined figure
svg1 = SVG(os.path.join(figures_folder, "rel_error_time.svg")).scale(1.0)
svg2 = SVG(os.path.join(figures_folder, "errors.svg")).scale(1.0)
svg3 = SVG(os.path.join(figures_folder, "bem_vs_pinns_L2_vs_k.svg")).scale(1.0)

Figure(
    617,  # total width
    480,  # height
    SVG(os.path.join(figures_folder, "comparison_base.svg")).scale(3.79),
    Panel(svg1).move(0, 0).scale(1.3),
    Panel(svg2).move(25, 175),
    Panel(svg3).move(6, 265).scale(1.3)
).save(os.path.join(figures_folder, "accuracy_time_error_bem_pinns.svg"))

# Convert to PDF
cairosvg.svg2pdf(
    url=os.path.join(figures_folder, "accuracy_time_error_bem_pinns.svg"),
    write_to=os.path.join(figures_folder, "accuracy_time_error_bem_pinns.pdf")
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
