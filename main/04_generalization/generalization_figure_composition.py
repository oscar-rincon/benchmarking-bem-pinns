

# ============================================================
"""
Script: generalization_figure_composition.py

Description:
    This script composes and exports figures for generalization analysis 
    by combining multiple SVGs. It creates a composite SVG figure, 
    then converts it into a PDF for publication-ready use. 
    Execution time and metadata are logged with timestamps.

Inputs:
    - Individual SVG figures located in ./figures/
        * generalization_bem.svg
        * generalization_pinns.svg

Outputs:
    - Composite SVG saved as ./figures/06_generalization.svg
    - PDF version saved as ./figures/06_generalization.pdf
    - Log file (TXT) with script name, timestamp, and execution time, saved in ./logs/
"""

#%% ======================== IMPORTS ========================
from datetime import datetime
import sys
import os
import time
import cairosvg
#%% ======================== PATH SETUP ========================
# Set the current directory and utilities path
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utilities')

# Change the working directory to the script's directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)

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



#%% ======================== SVG LOADING AND COMPOSITION ========================
from svgutils.compose import *

# Load the SVGs
svg1 = SVG("figures/generalization_bem.svg")
svg2 = SVG("figures/generalization_pinns.svg")

# Create a figure using known dimensions
Figure(
    658,  # total width (still no math if you avoid this by guessing)
    704,  # height (or just use one of them)
    SVG("figures/base.svg").scale(4.0),
    Panel(svg1).scale(1.30),
    Panel(svg2).move(255, 0).scale(1.30)
).save("figures/generalization.svg")

# Convert to PDF
cairosvg.svg2pdf(
    url="figures/generalization.svg", 
    write_to="figures/generalization.pdf"
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