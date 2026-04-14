
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import torch 
 
 
  

# Configuración de LaTeX para matplotlib
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "xelatex",        # change this if using xetex or lautex
    "text.usetex": False,                # use LaTeX to write all text
    "font.family": "sans-serif",
    # "font.serif": [],
    "font.sans-serif": ["DejaVu Sans"], # specify the sans-serif font
    "font.monospace": [],
    "axes.labelsize": 8,               # LaTeX default is 10pt font.
    "font.size": 0,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # "figure.figsize": (3.15, 2.17),     # default fig size of 0.9 textwidth
    "pgf.preamble": r'\usepackage{amsmath},\usepackage{amsthm},\usepackage{amssymb},\usepackage{mathspec},\renewcommand{\familydefault}{\sfdefault},\usepackage[italic]{mathastext}'
    }

mpl.rcParams.update(pgf_with_latex)

def plot_exact_displacement(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase):
    """
    Plot the amplitude and phase of the incident, scattered, and total displacement.

    Parameters:
    X (numpy.ndarray): X-coordinates of the grid.
    Y (numpy.ndarray): Y-coordinates of the grid.
    u_inc (numpy.ndarray): Incident displacement field.
    u_scn (numpy.ndarray): Scattered displacement field.
    u (numpy.ndarray): Total displacement field.
    """

    fig, axs = plt.subplots(2, 3, figsize=(6.5, 3.1))
    decimales = 1e+4  # Number of decimals for the color bar
    shrink = 0.36  # Shrink factor for the color bar

    # Amplitude of the incident wave
    c1 = axs[0, 0].pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb1 = fig.colorbar(c1, ax=axs[0, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"$u_{\rm{inc}}$")
    cb1.set_ticks([-1.5, 1.5])
    cb1.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 0].axis("off")
    axs[0, 0].set_aspect("equal")

    # Amplitude of the scattered wave
    c2 = axs[0, 1].pcolormesh(X, Y, u_scn_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb2 = fig.colorbar(c2, ax=axs[0, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb2.set_label(r"$u_{\rm{sct}}$")
    cb2.set_ticks([-1.5, 1.5])
    cb2.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 1].axis("off")
    axs[0, 1].set_aspect("equal")

    # Amplitude of the total wave
    c3 = axs[0, 2].pcolormesh(X, Y, u_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb3 = fig.colorbar(c3, ax=axs[0, 2], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb3.set_label(r"$u$")
    cb3.set_ticks([-1.5, 1.5])
    cb3.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 2].axis("off")
    axs[0, 2].set_aspect("equal")

    # Phase of the incident wave
    c4 = axs[1, 0].pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb4 = fig.colorbar(c4, ax=axs[1, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb4.set_label(r"$u_{\rm{inc}}$")
    cb4.set_ticks([-(np.pi),(np.pi)])
    cb4.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 0].axis("off")
    axs[1, 0].set_aspect("equal")

    # Phase of the scattered wave
    c5 = axs[1, 1].pcolormesh(X, Y, u_scn_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb5 = fig.colorbar(c5, ax=axs[1, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb5.set_label(r"$u_{\rm{sct}}$")
    cb5.set_ticks([-(np.pi),(np.pi)])
    cb5.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 1].axis("off")
    axs[1, 1].set_aspect("equal")

    # Phase of the total wave
    c6 = axs[1, 2].pcolormesh(X, Y, u_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb6 = fig.colorbar(c6, ax=axs[1, 2], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb6.set_label(r"$u$")
    cb6.set_ticks([-(np.pi),(np.pi)])
    cb6.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 2].axis("off")
    axs[1, 2].set_aspect("equal")

    # Add rotated labels "Amplitude" and "Phase"
    fig.text(0.05, 0.80, r'Exact - Amplitude', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')
    fig.text(0.05, 0.30, r'Exact - Phase', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')

    # Adjust space between rows (increase 'hspace' for more space between rows)
    plt.subplots_adjust(hspace=1.1)  # You can tweak this value (e.g., 0.5, 0.6) as needed

    # Tight layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("figures/displacement_exact.svg", dpi=300, bbox_inches='tight')
    plt.savefig("figures/displacement_exact.pdf", dpi=300, bbox_inches='tight')


def plot_bem_displacements(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase):
    """
    Plot the amplitude and phase of the incident, scattered, and total displacement.

    Parameters:
    X (numpy.ndarray): X-coordinates of the grid.
    Y (numpy.ndarray): Y-coordinates of the grid.
    u_inc (numpy.ndarray): Incident displacement field.
    u_scn (numpy.ndarray): Scattered displacement field.
    u (numpy.ndarray): Total displacement field.
    """

    # Square patch properties
    square_size = 2 * np.pi
    square_xy = (-square_size / 2, -square_size / 2)
    square_props = dict(edgecolor="gray", facecolor="none", lw=0.8)

    fig, axs = plt.subplots(2, 2, figsize=(4.5, 3.5))
    decimales = 1e+4  # Number of decimals for the color bar
    shrink = 0.4  # Shrink factor for the color bar

    # Amplitude of the incident wave
    c1 = axs[0, 0].pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb1 = fig.colorbar(c1, ax=axs[0, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"$u_{\rm{sct}}$")
    cb1.set_ticks([-1.5, 1.5])
    cb1.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[0, 0].axis("off")
    axs[0, 0].set_aspect("equal")

    # Amplitude of the total wave
    c3 = axs[0, 1].pcolormesh(X, Y, np.abs(u_amp)/np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
    cb3 = fig.colorbar(c3, ax=axs[0, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb3.set_label(r"|Error| / max($u$)")
    cb3.set_ticks([0, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())])
    cb3.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max()):.4f}'], fontsize=7)
    axs[0, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[0, 1].axis("off")
    axs[0, 1].set_aspect("equal")

    # Phase of the incident wave
    c4 = axs[1, 0].pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb4 = fig.colorbar(c4, ax=axs[1, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb4.set_label(r"$u_{\rm{sct}}$")
    cb4.set_ticks([-(np.pi),(np.pi)])
    cb4.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[1, 0].axis("off")
    axs[1, 0].set_aspect("equal")

    # Phase of the total wave
    c6 = axs[1, 1].pcolormesh(X, Y, np.abs(u_phase)/np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
    cb6 = fig.colorbar(c6, ax=axs[1, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb6.set_label(r"|Error| / max($u$)")
    cb6.set_ticks([0, np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())])
    cb6.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max()):.4f}'], fontsize=7)
    axs[1, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[1, 1].axis("off")
    axs[1, 1].set_aspect("equal")

    # Add rotated labels "Amplitude" and "Phase"
    fig.text(0.15, 0.80, r'BEM - Amplitude', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')
    fig.text(0.15, 0.30, r'BEM - Phase', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')

    # Adjust space between rows (increase 'hspace' for more space between rows)
    fig.subplots_adjust(wspace=-0.7)  # Reduce horizontal spacing, optionally adjust vertical spacing

    # Tight layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("figures/generalization_bem.svg", dpi=300, bbox_inches='tight')

    return fig, axs  # Return the figure and axes for further customization if needed

def plot_pinns_displacements(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase):
    """
    Plot the amplitude and phase of the incident, scattered, and total displacement.

    Parameters:
    X (numpy.ndarray): X-coordinates of the grid.
    Y (numpy.ndarray): Y-coordinates of the grid.
    u_inc (numpy.ndarray): Incident displacement field.
    u_scn (numpy.ndarray): Scattered displacement field.
    u (numpy.ndarray): Total displacement field.
    """

    # Square patch properties
    square_size = 2 * np.pi
    square_xy = (-square_size / 2, -square_size / 2)
    square_props = dict(edgecolor="gray", facecolor="none", lw=0.8)

    fig, axs = plt.subplots(2, 2, figsize=(4.5, 3.5))
    decimales = 1e+4  # Number of decimals for the color bar
    shrink = 0.4  # Shrink factor for the color bar

    # Amplitude of the incident wave
    c1 = axs[0, 0].pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb1 = fig.colorbar(c1, ax=axs[0, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"$u_{\rm{sct}}$")
    cb1.set_ticks([-1.5, 1.5])
    cb1.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=7)
    axs[0, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[0, 0].axis("off")
    axs[0, 0].set_aspect("equal")

    # Amplitude of the total wave
    c3 = axs[0, 1].pcolormesh(X, Y, np.abs(u_amp)/np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
    cb3 = fig.colorbar(c3, ax=axs[0, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb3.set_label(r"|Error| / max($u$)")
    cb3.set_ticks([0, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())])
    cb3.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max()):.4f}'], fontsize=7)
    axs[0, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[0, 1].axis("off")
    axs[0, 1].set_aspect("equal")

    # Phase of the incident wave
    c4 = axs[1, 0].pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb4 = fig.colorbar(c4, ax=axs[1, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb4.set_label(r"$u_{\rm{sct}}$")
    cb4.set_ticks([-(np.pi),(np.pi)])
    cb4.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=7)
    axs[1, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[1, 0].axis("off")
    axs[1, 0].set_aspect("equal")

    # Phase of the total wave
    c6 = axs[1, 1].pcolormesh(X, Y, np.abs(u_phase)/np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
    cb6 = fig.colorbar(c6, ax=axs[1, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb6.set_label(r"|Error| / max($u$)")
    cb6.set_ticks([0, np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())])
    cb6.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max()):.4f}'], fontsize=7)
    axs[1, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[1, 1].axis("off")
    axs[1, 1].set_aspect("equal")

    # Add rotated labels "Amplitude" and "Phase"
    fig.text(0.15, 0.80, r'PINNs - Amplitude', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')
    fig.text(0.15, 0.30, r'PINNs - Phase', fontsize=8, fontweight='regular', va='center', ha='center', rotation='vertical')

    # Adjust space between rows (increase 'hspace' for more space between rows)
    fig.subplots_adjust(wspace=-0.7)  # You can tweak this value (e.g., 0.5, 0.6) as needed

    # Tight layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("figures/generalization_pinns.svg", dpi=300, bbox_inches='tight')


def plot_bem_error(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase):
    """
    Plot only the scattered amplitude and phase as a row of two figures.

    Parameters:
    X, Y : 2D ndarrays - Grid coordinates.
    u_scn_amp : 2D ndarray - Amplitude of the scattered field.
    u_scn_phase : 2D ndarray - Phase of the scattered field.
    """
    fig, axs = plt.subplots(1, 2, figsize=(3.9, 1.7))
    shrink = 0.42 
  
    c1 = axs[0].pcolormesh(X, Y, u_amp/np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
    cb1 = fig.colorbar(c1, ax=axs[0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"|Error| / max($u$) ($\times 10^{-2}$)", fontsize=8)
    cb1.set_ticks([0, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())/2, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())])
    cb1.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())/2*100:.2f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())*100:.2f}'], fontsize=7)
    axs[0].set_title("Amplitude", fontsize=8, pad=6)  
    axs[0].axis("off")
    axs[0].set_aspect("equal")

     
    c2 = axs[1].pcolormesh(X, Y, u_phase/np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
    cb2 = fig.colorbar(c2, ax=axs[1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb2.set_label(r"|Error| / max($u$) ($\times 10^{-2}$)", fontsize=8)

    cb2.set_ticks([0,np.max(u_phase)/np.abs(u_scn_phase).max()/2, np.max(u_phase)/np.abs(u_scn_phase).max()])
    cb2.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())/2*100:.2f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())*100:.2f}'], fontsize=7)
 
    
    axs[1].set_title("Phase", fontsize=8, pad=6)  
    axs[1].axis("off")
    axs[1].set_aspect("equal")

    fig.text(0.13, 0.60, r'BEM', fontsize=8, va='center', ha='center', rotation='vertical')

    fig.subplots_adjust(wspace=-0.5)

    plt.tight_layout()
    plt.savefig("figures/bem_error.svg", dpi=150, bbox_inches='tight')
    #plt.show()

def plot_pinns_error(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase):
    """
    Plot only the scattered amplitude and phase as a row of two figures.

    Parameters:
    X, Y : 2D ndarrays - Grid coordinates.
    u_scn_amp : 2D ndarray - Amplitude of the scattered field.
    u_scn_phase : 2D ndarray - Phase of the scattered field.
    """
    fig, axs = plt.subplots(1, 2, figsize=(3.9, 1.7))
    shrink = 0.42   
  
    c1 = axs[0].pcolormesh(X, Y, u_amp/np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
    cb1 = fig.colorbar(c1, ax=axs[0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"|Error| / max($u$) ($\times 10^{-2}$)", fontsize=8)
    cb1.set_ticks([0, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())/2, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())])
    cb1.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())/2*100:.2f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())*100:.2f}'], fontsize=7)
    axs[0].set_title("Amplitude", fontsize=8, pad=6)  
    axs[0].axis("off")
    axs[0].set_aspect("equal")

     
    c2 = axs[1].pcolormesh(X, Y, u_phase/np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
    cb2 = fig.colorbar(c2, ax=axs[1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb2.set_label(r"|Error| / max($u$) ($\times 10^{-2}$)", fontsize=8)

    cb2.set_ticks([0, np.max(u_phase)/np.abs(u_scn_phase).max()/2, np.max(u_phase)/np.abs(u_scn_phase).max()])
    cb2.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())/2*100:.2f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())*100:.2f}'], fontsize=7)
    axs[1].set_title("Phase", fontsize=8, pad=6)  
    axs[1].axis("off")
    axs[1].set_aspect("equal")

    fig.text(0.13, 0.60, r'PINNs', fontsize=8, va='center', ha='center', rotation='vertical')

    fig.subplots_adjust(wspace=-0.5)

    plt.tight_layout()
    plt.savefig("figures/pinns_error.svg", dpi=150, bbox_inches='tight')
    #plt.show()

def format_func(value, tick_number):
    n = int(round(value / np.pi))
    if n == 0:
        return "0"
    elif n == 1:
        return r"$\pi$"
    else:
        return rf"${n}\pi$"

# def plot_bem_displacements_errors(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase, x_line, rel_error_line):
#     """
#     Plot the amplitude and phase of the incident, scattered, and total displacement, plus relative error along y = 0.
#     """
#     shrink = 0.8
#     square_size = 2 * np.pi
#     square_xy = (-square_size / 2, -square_size / 2)
#     square_props = dict(edgecolor="gray", facecolor="none", lw=0.8)

#     fig = plt.figure(figsize=(3.4, 5.5))
#     gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.6], hspace=0.4, wspace=0.05)

#     axs = np.empty((3, 2), dtype=object)

#     # Subplot 1: Amplitude of the incident wave
#     axs[0, 0] = fig.add_subplot(gs[0, 0])
#     c1 = axs[0, 0].pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
#     cb1 = fig.colorbar(c1, ax=axs[0, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
#     cb1.set_label(r"$u_{\rm{sct}}$", fontsize=8)
#     cb1.set_ticks([-1.5, 1.5])
#     cb1.set_ticklabels(['-1.5', '1.5'], fontsize=8)
#     axs[0, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
#     axs[0, 0].axis("off")
#     axs[0, 0].set_aspect("equal")


#     # Subplot 2: Amplitude of the total wave
#     axs[0, 1] = fig.add_subplot(gs[0, 1])
#     amp_ratio = np.abs(u_amp) / np.abs(u_scn_amp).max()
#     c2 = axs[0, 1].pcolormesh(X, Y, amp_ratio, cmap="magma", rasterized=True)
#     cb2 = fig.colorbar(c2, ax=axs[0, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
#     cb2.set_label(r"|Error| / max($u$)", fontsize=8)
#     cb2.set_ticks([0, np.max(amp_ratio)])
#     cb2.set_ticklabels([f'{0:.1f}', f'{np.max(amp_ratio):.4f}'], fontsize=8)
#     axs[0, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
#     axs[0, 1].axis("off")
#     axs[0, 1].set_aspect("equal")
#     # Add horizontal line from x = π to x = 10π at y = 0 (or center)
#     y_center = 0  # or e.g., y_center = Y.mean() or another value of interest
#     line1 = Line2D([np.pi, 10*np.pi], [y_center, y_center], color="#00a2ff", linewidth=1.0, linestyle='-')
#     axs[0, 1].add_line(line1)

#     # Subplot 3: Phase of the incident wave
#     axs[1, 0] = fig.add_subplot(gs[1, 0])
#     c3 = axs[1, 0].pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-np.pi, vmax=np.pi)
#     cb3 = fig.colorbar(c3, ax=axs[1, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
#     cb3.set_label(r"$u_{\rm{sct}}$", fontsize=8)
#     cb3.set_ticks([-np.pi, np.pi])
#     cb3.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=8)
#     axs[1, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))


#     axs[1, 0].axis("off")
#     axs[1, 0].set_aspect("equal")

#     # Subplot 4: Phase of the total wave
#     axs[1, 1] = fig.add_subplot(gs[1, 1])
#     phase_ratio = np.abs(u_phase) / np.abs(u_scn_phase).max()
#     c4 = axs[1, 1].pcolormesh(X, Y, phase_ratio, cmap="magma", rasterized=True)
#     cb4 = fig.colorbar(c4, ax=axs[1, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
#     cb4.set_label(r"|Error| / max($u$)", fontsize=8)
#     cb4.set_ticks([0, np.max(phase_ratio)])
#     cb4.set_ticklabels([f'{0:.1f}', f'{np.max(phase_ratio):.4f}'], fontsize=8)
#     axs[1, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
#     axs[1, 1].axis("off")
#     axs[1, 1].set_aspect("equal")


#     # Subplot 5: Relative error along y = 0 (full width)
#     ax_err = fig.add_subplot(gs[2, :])
#     ax_err.plot(x_line, rel_error_line, label='Relative error', color="#00a2ff", linewidth=1.0)
#     ax_err.axvline(x=np.pi, color="#acacac", linestyle='-', linewidth=1)
#     ax_err.axvline(x=2*np.pi, color="#acacac", linestyle='-', linewidth=1)  
#     ax_err.set_xlabel(r'$x$')
#     ax_err.set_ylabel(r"$|$Error$|$ / max($u$)")
#     ax_err.set_ylim(0, np.max(rel_error_line) * 1.1)
#     ax_err.set_yticks([0, np.max(amp_ratio)])
#     ax_err.set_yticklabels([f'{0:.1f}', f'{np.max(amp_ratio):.4f}'], fontsize=8)
#     ax_err.xaxis.set_major_locator(MultipleLocator(base=np.pi))
#     ax_err.xaxis.set_major_formatter(FuncFormatter(format_func))
#     ax_err.set_title('BEM - Amplitude', fontsize=8)
#     #ax_err.grid(True)

#     # Add rotated labels on the left
#     fig.text(0.08, 0.76, r'BEM - Amplitude', fontsize=8, va='center', ha='center', rotation='vertical')
#     fig.text(0.08, 0.46, r'BEM - Phase', fontsize=8, va='center', ha='center', rotation='vertical')

#     plt.tight_layout()
#     plt.savefig("figures/generalization_bem.svg", dpi=300, bbox_inches='tight')
#     #plt.show()

def plot_bem_displacements_errors(X, Y, u_inc_amp, u_scn_amp, u_amp, u_inc_phase, u_scn_phase, u_phase, x_line, rel_error_line, avg_errors):
    """
    Plot the amplitude and phase of the incident, scattered, and total displacement, plus relative error along y = 0.
    """
    shrink = 0.75
    square_size = 2 * np.pi
    square_xy = (-square_size / 2, -square_size / 2)
    square_props = dict(edgecolor="gray", facecolor="none", lw=0.8)

    fig = plt.figure(figsize=(3.6, 7.2))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 0.5, 0.5], hspace=0.5, wspace=0.05)
    # Update GridSpec to 4 rows: [amp, amp, phase, rel_error+zoom]
    # fig = plt.figure(figsize=(3.4, 6.5))
    # gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1.2], hspace=0.4, wspace=0.05)

    axs = np.empty((4, 2), dtype=object)

    # Subplot 1: Amplitude of the incident wave
    axs[0, 0] = fig.add_subplot(gs[0, 0])
    c1 = axs[0, 0].pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb1 = fig.colorbar(c1, ax=axs[0, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"$u_{\rm{sct}}$", fontsize=8)
    cb1.set_ticks([-1.5, 1.5])
    cb1.set_ticklabels(['-1.5', '1.5'], fontsize=8)
    axs[0, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[0, 0].axis("off")
    axs[0, 0].set_aspect("equal")

    # Subplot 2: Amplitude of the total wave
    axs[0, 1] = fig.add_subplot(gs[0, 1])
    amp_ratio = np.abs(u_amp) / np.abs(u_scn_amp).max()
    c2 = axs[0, 1].pcolormesh(X, Y, amp_ratio, cmap="magma", rasterized=True)
    cb2 = fig.colorbar(c2, ax=axs[0, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb2.set_label(r"|Error| / max($u$) ($\times 10^{-2}$)", fontsize=8)
    cb2.set_ticks([0, np.max(amp_ratio)/2, np.max(amp_ratio)])
    cb2.set_ticklabels([f'{0:.1f}', f'{np.max(amp_ratio)/2*100:.2f}', f'{np.max(amp_ratio)*100:.2f}'], fontsize=8)
    axs[0, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    # axs[0, 1].add_patch(Rectangle(
    # (-1.5 * square_size / 2, -1.5 * square_size / 2),  # new bottom-left
    # 1.5 * square_size,                                # new width
    # 1.5 * square_size,                                # new height
    # fill=False,
    # edgecolor='#00a2ff',
    # linestyle='-',
    # linewidth=1.0
    # ))
    axs[0, 1].axis("off")
    axs[0, 1].set_aspect("equal")
    # Add horizontal line from x = π to x = 10π at y = 0 (or center)
    y_center = 0  # or e.g., y_center = Y.mean() or another value of interest
    line1 = Line2D([np.pi, 10*np.pi], [y_center, y_center], color="#00a2ff", linewidth=1.0, linestyle='-')
    axs[0, 1].add_line(line1)
    

    # Subplot 3: Phase of the incident wave
    axs[1, 0] = fig.add_subplot(gs[1, 0])
    c3 = axs[1, 0].pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-np.pi, vmax=np.pi)
    cb3 = fig.colorbar(c3, ax=axs[1, 0], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb3.set_label(r"$u_{\rm{sct}}$", fontsize=8)
    cb3.set_ticks([-np.pi, np.pi])
    cb3.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=8)
    axs[1, 0].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[1, 0].axis("off")
    axs[1, 0].set_aspect("equal")

    # Subplot 4: Phase of the total wave
    axs[1, 1] = fig.add_subplot(gs[1, 1])
    phase_ratio = np.abs(u_phase) / np.abs(u_scn_phase).max()
    c4 = axs[1, 1].pcolormesh(X, Y, phase_ratio, cmap="magma", rasterized=True)
    cb4 = fig.colorbar(c4, ax=axs[1, 1], shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb4.set_label(r"|Error| / max($u$) ($\times 10^{-2}$)", fontsize=8)
    cb4.set_ticks([0, np.max(phase_ratio)/2, np.max(phase_ratio)])
    cb4.set_ticklabels([f'{0:.1f}', f'{np.max(phase_ratio)/2*100:.2f}', f'{np.max(phase_ratio)*100:.2f}'], fontsize=8)
    axs[1, 1].add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    axs[1, 1].axis("off")
    axs[1, 1].set_aspect("equal")

    def format_func(value, tick_number):
        N = int(np.round(value / np.pi))
        if N == 0:
            return "0"
        elif N == 1:
            return r"$\pi$"
        elif N == -1:
            return r"$-\pi$"
        else:
            return fr"${N}\pi$"
        
# ======================= BAR PLOT (NEW BOTTOM PANEL) =======================

    ax_bar = fig.add_subplot(gs[2, :])
    pos = ax_bar.get_position()
    ax_bar.set_position([pos.x0, pos.y0 - 0.01, pos.width, pos.height])

    n_levels = np.arange(1, len(avg_errors) + 1)

    # Bar positions at interval centers
    x_centers = (n_levels - 0.5) * np.pi

    bar_width = 0.8 * np.pi

    ax_bar.bar(
        x_centers,
        avg_errors,
        width=bar_width,
        color="#00a2ff",
        alpha=0.5,
    )

    # Interval labels
    interval_labels = [
        r"$\pi$" if n == 1 else rf"${n}\pi$"
        for n in n_levels
    ]

    ax_bar.set_xticks(x_centers)
    ax_bar.set_xticklabels(interval_labels, rotation=0, fontsize=8)
    ax_bar.set_ylim(0,1.0)
    #ax_bar.set_yticks([0, 0.04, 0.08])
    ax_bar.set_xlabel('Square ring interval', fontsize=8)
    ax_bar.set_ylabel('Average', fontsize=8)
    ax_bar.set_title('BEM - Amplitude', fontsize=8)

    #ax_bar.set_ylim(0, np.nanmax(avg_errors) * 1.2)

    #ax_bar.grid(True, axis='y', linestyle='--', alpha=0.5)

    # ======================= BARPLOT ZOOM =======================

    zm_bar = ax_bar.inset_axes([0.0, -1.4, 1.0, 0.6])

    # Replot bars inside zoom
    zm_bar.bar(
        x_centers,
        avg_errors,
        width=bar_width,
        color="#00a2ff",
        alpha=0.5,
        zorder=2
    )

    # Same zoom limits as line plot
    zm_bar.set_xlim(0, 10*np.pi)
    zm_bar.set_ylim(0, 0.1)
    zm_bar.xaxis.set_major_locator(MultipleLocator(base=np.pi))
    zm_bar.xaxis.set_major_formatter(FuncFormatter(format_func))

    zm_bar.set_yticks([0, 0.05, 0.10])
    zm_bar.set_yticklabels([0, 0.05, 0.10], fontsize=6)
    zm_bar.tick_params(axis="x", labelsize=6)
    zm_bar.set_xticks(x_centers)
    zm_bar.set_xticklabels(interval_labels, rotation=0, fontsize=6)
    
    zm_bar.set_xlabel(r'$x$', fontsize=6)
    zm_bar.set_ylabel('Average', fontsize=6)

    # Connect inset
    mark_inset(ax_bar, zm_bar, loc1=1, loc2=2, fc="none", ec="0.5", lw=1)

    # Subplot 5: Relative error along y = 0 (full width)
    ax_err = fig.add_subplot(gs[3, :])
    pos = ax_err.get_position()
    ax_err.set_position([pos.x0, pos.y0 - 0.16, pos.width, pos.height])

    ax_err.plot(x_line, rel_error_line, label='Relative error', color="#00a2ff", linewidth=1.0)
    #ax_err.axvline(x=np.pi, color="#acacac", linestyle='-', linewidth=1)
    #ax_err.axvline(x=2*np.pi, color="#acacac", linestyle='-', linewidth=1)  
    # Sombrear la región entre π y 2π
    ax_err.axvspan(np.pi, 2*np.pi, color='gray', alpha=0.2)  # alpha controla la transparencia

    ax_err.set_xlabel(r'$x$')
    ax_err.set_ylabel(r"$|$Error$|$ / max($u$)")
    ax_err.set_ylim(0, 1.0)
    # zm = ax_err.inset_axes([0.2, 0.5, 0.75, 0.45])
    # zm.plot(x_line, rel_error_line, color="#00a2ff")
    # zm.set_yticks([0, 0.03])
    # zm.set_yticklabels([0, 0.03], fontsize=7)

    # Zoomed inset
    zm = ax_err.inset_axes([0.0, -1.4, 1.0, 0.60])
    zm.plot(x_line, rel_error_line, color="#00a2ff", linewidth=1.0,zorder=2)

    zm.set_xlim(np.pi, 10*np.pi)
    zm.set_xlabel(r'$x$', fontsize=6)
    zm.set_ylabel(r"$|$Error$|$ / max($u$)", fontsize=6)
    zm.set_ylim(0, 0.04)   # adjust zoom y-limits
    #zm.axvline(x=np.pi, color="#acacac", linestyle='-', linewidth=1)
    #zm.axvline(x=2*np.pi, color="#acacac", linestyle='-', linewidth=1)   
    zm.axvspan(np.pi, 2*np.pi, color='gray', alpha=0.2)  # alpha controla la transparencia

    zm.xaxis.set_major_locator(MultipleLocator(2*np.pi))
    zm.xaxis.set_major_locator(MultipleLocator(base=np.pi))
    zm.xaxis.set_major_formatter(FuncFormatter(format_func))
    zm.set_yticks([0.0, 0.02, 0.04])
    zm.set_yticklabels([0, 0.02, 0.04], fontsize=6)
    zm.tick_params(axis="x", labelsize=6)

    # Connect inset to main plot
    # mark_inset(ax_err, zm, loc1=2, loc2=4, fc="none", ec="0.5")
    #zm = inset_axes(ax_err, [0.24, 0.5, 0.75, 0.45])
    #zm.plot(x_line, rel_error_line, color="#00a2ff", linewidth=1.0)
    # configure limits, ticks, etc. here
    mark_inset(ax_err, zm, loc1=1, loc2=2, fc="none", ec="0.5", lw=1)


    #ax_err.set_ylim(0, 10)#
    #ax_err.set_yticks([0, np.max(amp_ratio)])
    #ax_err.set_yticklabels([f'{0:.1f}', f'{np.max(amp_ratio):.4f}'], fontsize=8)
    ax_err.xaxis.set_major_locator(MultipleLocator(base=np.pi))
    ax_err.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax_err.set_title('BEM - Amplitude', fontsize=8)
    #ax_err.grid(True)


    

    # Add rotated labels on the left
    fig.text(0.08, 0.80, r'BEM - Amplitude', fontsize=8, va='center', ha='center', rotation='vertical')
    fig.text(0.08, 0.57, r'BEM - Phase', fontsize=8, va='center', ha='center', rotation='vertical')

    plt.tight_layout()
    plt.savefig("figures/generalization_bem.svg", dpi=300, bbox_inches='tight')
    #plt.show()

# def plot_pinns_displacements_with_errorline(X, Y, u_inc_amp, u_scn_amp, u_amp,
#                                             u_inc_phase, u_scn_phase, u_phase,
#                                             x_line, rel_error_line):
#     """
#     Combina gráficos 2D y gráfico de error relativo 1D usando GridSpec.
#     """

#     # Square patch properties
#     square_size = 2 * np.pi
#     square_xy = (-square_size / 2, -square_size / 2)
#     square_props = dict(edgecolor="gray", facecolor="none", lw=0.8)
#     shrink = 0.8
#     decimales = 1e+4

#     # Create figure and GridSpec layout
#     fig = plt.figure(figsize=(3.4, 5.5))
#     gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.6], hspace=0.4, wspace=0.05)

#     # Subplots for amplitude
#     ax0 = fig.add_subplot(gs[0, 0])
#     c1 = ax0.pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
#     cb1 = fig.colorbar(c1, ax=ax0, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
#     cb1.set_label(r"$u_{\rm{sct}}$", fontsize=8)
#     cb1.set_ticks([-1.5, 1.5])
#     cb1.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=8)
#     ax0.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
#     ax0.axis("off")
#     ax0.set_aspect("equal")

#     ax1 = fig.add_subplot(gs[0, 1])
#     c2 = ax1.pcolormesh(X, Y, np.abs(u_amp)/np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
#     cb2 = fig.colorbar(c2, ax=ax1, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
#     cb2.set_label(r"|Error| / max($u$)", fontsize=8)
#     cb2.set_ticks([0, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())])
#     cb2.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max()):.4f}'], fontsize=8)
#     ax1.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
#     ax1.axis("off")
#     ax1.set_aspect("equal")
#     y_center = 0  # or e.g., y_center = Y.mean() or another value of interest
#     line2 = Line2D([np.pi, 10*np.pi], [y_center, y_center], color="#00ff0d", linewidth=1.0, linestyle='-')
#     ax1.add_line(line2)

#     # Subplots for phase
#     ax2 = fig.add_subplot(gs[1, 0])
#     c3 = ax2.pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
#     cb3 = fig.colorbar(c3, ax=ax2, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
#     cb3.set_label(r"$u_{\rm{sct}}$", fontsize=8)
#     cb3.set_ticks([-(np.pi), (np.pi)])
#     cb3.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=8)
#     ax2.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
#     ax2.axis("off")
#     ax2.set_aspect("equal")

#     ax3 = fig.add_subplot(gs[1, 1])
#     c4 = ax3.pcolormesh(X, Y, np.abs(u_phase)/np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
#     cb4 = fig.colorbar(c4, ax=ax3, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
#     cb4.set_label(r"|Error| / max($u$)", fontsize=8)
#     cb4.set_ticks([0, np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())])
#     cb4.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max()):.4f}'], fontsize=8)
#     ax3.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
#     ax3.axis("off")
#     ax3.set_aspect("equal")

#     # Subplot 5: Relative error line plot
#     ax_err = fig.add_subplot(gs[2, :])
#     ax_err.axvline(x=np.pi, color="#acacac", linestyle='-', linewidth=1)
#     ax_err.axvline(x=2*np.pi, color="#acacac", linestyle='-', linewidth=1)    
#     ax_err.plot(x_line, rel_error_line, label='Relative error', color="#00ff0d")
#     # Agregar líneas verticales en pi y 2pi
#     ax_err.set_xlabel(r'$x$', fontsize=8)
#     ax_err.set_ylabel(r"$|$Error$|$ / max($u$)", fontsize=8)
#     ax_err.set_ylim(0, 10)
#     #ax_err.set_ylim(0, np.max(rel_error_line) * 1.1)
#     ax_err.xaxis.set_major_locator(MultipleLocator(base=np.pi))

#     def format_func(value, tick_number):
#         N = int(np.round(value / np.pi))
#         if N == 0:
#             return "0"
#         elif N == 1:
#             return r"$\pi$"
#         elif N == -1:
#             return r"$-\pi$"
#         else:
#             return fr"${N}\pi$"

#     ax_err.xaxis.set_major_formatter(FuncFormatter(format_func))
#     ax_err.set_title('PINNs - Amplitude', fontsize=8)

#     # Add rotated labels
#     fig.text(0.08, 0.76, r'PINNs - Amplitude', fontsize=8, va='center', ha='center', rotation='vertical')
#     fig.text(0.08, 0.46, r'PINNs - Phase', fontsize=8, va='center', ha='center', rotation='vertical')

#     # Save and show
#     plt.savefig("figures/generalization_pinns.svg", dpi=300, bbox_inches='tight')
#     #plt.show()


# def plot_pinns_displacements_with_errorline(X, Y, u_inc_amp, u_scn_amp, u_amp,
#                                             u_inc_phase, u_scn_phase, u_phase,
#                                             x_line, rel_error_line):
#     """
#     Combina gráficos 2D y gráfico de error relativo 1D usando GridSpec.
#     """

#     # Square patch properties
#     square_size = 2 * np.pi
#     square_xy = (-square_size / 2, -square_size / 2)
#     square_props = dict(edgecolor="gray", facecolor="none", lw=0.8)
#     shrink = 0.8
#     decimales = 1e+4

#     # Choose slice along y = 0
#     y_index = np.argmin(np.abs(Y[:, 0]))  # row closest to y=0

#     X_slice = X[y_index, :]
#     amp_slice = np.abs(u_amp[y_index, :]) / np.abs(u_scn_amp).max()

#     # Restrict to [pi, 10pi]
#     mask = (X_slice >= np.pi) & (X_slice <= 10*np.pi)
#     X_slice = X_slice[mask]
#     amp_slice = amp_slice[mask]

#     # Create figure and GridSpec layout
#     fig = plt.figure(figsize=(3.6, 7.5))
#     gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 0.5, 0.7], hspace=0.5, wspace=0.05)

#     # Subplots for amplitude
#     ax0 = fig.add_subplot(gs[0, 0])
#     c1 = ax0.pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
#     cb1 = fig.colorbar(c1, ax=ax0, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
#     cb1.set_label(r"$u_{\rm{sct}}$", fontsize=8)
#     cb1.set_ticks([-1.5, 1.5])
#     cb1.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=8)
#     ax0.set_xlim(-2*np.pi, 2*np.pi)
#     ax0.set_ylim(-2*np.pi, 2*np.pi)
#     ax0.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
#     ax0.axis("off")
#     ax0.set_aspect("equal")


#     ax1 = fig.add_subplot(gs[0, 1])
#     c2 = ax1.pcolormesh(X, Y, np.abs(u_amp)/np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
#     cb2 = fig.colorbar(c2, ax=ax1, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
#     cb2.set_label(r"|Error| / max($u$) ($\times 10^{-2}$)", fontsize=8)
#     cb2.set_ticks([0, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())/2, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())])
#     cb2.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())/2*100:.2f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())*100:.2f}'], fontsize=8)
#     ax1.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
#     # ax1.add_patch(Rectangle(
#     # (-1.5 * square_size / 2, -1.5 * square_size / 2),  # new bottom-left
#     # 1.5 * square_size,                                # new width
#     # 1.5 * square_size,                                # new height
#     # fill=False,
#     # edgecolor='#00ff0d',
#     # linestyle='-',
#     # linewidth=0.7
#     # ))    
#     ax1.axis("off")
#     ax1.set_aspect("equal")
#     ax1.set_xlim(-2*np.pi, 2*np.pi)
#     ax1.set_ylim(-2*np.pi, 2*np.pi)
#     y_center = 0  # or e.g., y_center = Y.mean() or another value of interest
#     line2 = Line2D([np.pi, 2*np.pi], [y_center, y_center], color="#00ff0d", linewidth=1.0, linestyle='-')
#     ax1.add_line(line2)

#     # Subplots for phase
#     ax2 = fig.add_subplot(gs[1, 0])
#     c3 = ax2.pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
#     cb3 = fig.colorbar(c3, ax=ax2, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
#     cb3.set_label(r"$u_{\rm{sct}}$", fontsize=8)
#     cb3.set_ticks([-(np.pi), (np.pi)])
#     cb3.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=8)  
#     ax2.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
#     ax2.axis("off")
#     ax2.set_aspect("equal")
#     ax2.set_xlim(-2*np.pi, 2*np.pi)
#     ax2.set_ylim(-2*np.pi, 2*np.pi)       

#     ax3 = fig.add_subplot(gs[1, 1])
#     c4 = ax3.pcolormesh(X, Y, np.abs(u_phase)/np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
#     cb4 = fig.colorbar(c4, ax=ax3, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
#     cb4.set_label(r"|Error| / max($u$) ($\times 10^{-2}$)", fontsize=8)
#     cb4.set_ticks([0, np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())/2, np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())])
#     cb4.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())/2*100:.2f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())*100:.2f}'], fontsize=8)
#     ax3.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
#     ax3.axis("off")
#     ax3.set_aspect("equal")
#     ax3.set_xlim(-2*np.pi, 2*np.pi)
#     ax3.set_ylim(-2*np.pi, 2*np.pi)

#     # Subplot 5: Relative error line plot
#     ax_err = fig.add_subplot(gs[2, :])
#     #ax_err.axvline(x=np.pi, color="#acacac", linestyle='-', linewidth=1)
#     #ax_err.axvline(x=2*np.pi, color="#acacac", linestyle='-', linewidth=1)  
#     ax_err.axvspan(np.pi, 2*np.pi, color='gray', alpha=0.2)   
#     ax_err.plot(X_slice,amp_slice, label='Relative error', color="#00ff0d")
#     #ax_err.plot(x_line, rel_error_line, label='Relative error', color="#00ff0d")
#     # Agregar líneas verticales en pi y 2pi
#     ax_err.set_xlabel(r'$x$', fontsize=8)
#     ax_err.set_ylabel(r"$|$Error$|$ / max($u$)", fontsize=8)
#     ax_err.set_ylim(0, 0.6)
#     #ax_err.set_ylim(0, np.max(rel_error_line) * 1.1)
#     #ax_err.set_xlim(0, 2*np.pi)
#     ax_err.xaxis.set_major_locator(MultipleLocator(base=np.pi))

#     def format_func(value, tick_number):
#         N = int(np.round(value / np.pi))
#         if N == 0:
#             return "0"
#         elif N == 1:
#             return r"$\pi$"
#         elif N == -1:
#             return r"$-\pi$"
#         else:
#             return fr"${N}\pi$"

#     ax_err.xaxis.set_major_formatter(FuncFormatter(format_func))
#     ax_err.set_title('PINNs - Amplitude', fontsize=8)

#     # Add rotated labels
#     fig.text(0.08, 0.79, r'PINNs - Amplitude', fontsize=8, va='center', ha='center', rotation='vertical')
#     fig.text(0.08, 0.48, r'PINNs - Phase', fontsize=8, va='center', ha='center', rotation='vertical')

#     # Save and show
#     plt.savefig("figures/generalization_pinns.svg", dpi=300, bbox_inches='tight')
#     #plt.show()


def plot_pinns_displacements_with_errorline(X, Y, u_inc_amp, u_scn_amp, u_amp,
                                            u_inc_phase, u_scn_phase, u_phase,
                                            x_line, rel_error_line, avg_errors):
    """
    Combina gráficos 2D y gráfico de error relativo 1D usando GridSpec.
    """

    # Square patch properties
    square_size = 2 * np.pi
    square_xy = (-square_size / 2, -square_size / 2)
    square_props = dict(edgecolor="gray", facecolor="none", lw=0.8)
    shrink = 0.8
    decimales = 1e+4

    # Choose slice along y = 0
    y_index = np.argmin(np.abs(Y[:, 0]))  # row closest to y=0

    X_slice = X[y_index, :]
    amp_slice = np.abs(u_amp[y_index, :]) / np.abs(u_scn_amp).max()

    # Restrict to [pi, 10pi]
    mask = (X_slice >= np.pi) & (X_slice <= 10*np.pi)
    X_slice = X_slice[mask]
    amp_slice = amp_slice[mask]

    # Create figure and GridSpec layout
    fig = plt.figure(figsize=(3.5, 7.5))
    #gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.5, wspace=0.05)
    gs = gridspec.GridSpec(4, 2, height_ratios=[0.88, 0.88, 0.5, 0.5], hspace=0.5, wspace=0.05)
    # Subplots for amplitude
    ax0 = fig.add_subplot(gs[0, 0])
    c1 = ax0.pcolormesh(X, Y, u_inc_amp, cmap="RdYlBu", rasterized=True, vmin=-1.5, vmax=1.5)
    cb1 = fig.colorbar(c1, ax=ax0, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb1.set_label(r"$u_{\rm{sct}}$", fontsize=8)
    cb1.set_ticks([-1.5, 1.5])
    cb1.set_ticklabels([f'{-1.5}', f'{1.5}'], fontsize=8)
    ax0.set_xlim(-2*np.pi, 2*np.pi)
    ax0.set_ylim(-2*np.pi, 2*np.pi)
    ax0.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    ax0.axis("off")
    ax0.set_aspect("equal")


    ax1 = fig.add_subplot(gs[0, 1])
    c2 = ax1.pcolormesh(X, Y, np.abs(u_amp)/np.abs(u_scn_amp).max(), cmap="magma", rasterized=True)
    cb2 = fig.colorbar(c2, ax=ax1, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb2.set_label(r"|Error| / max($u$) ($\times 10^{-2}$)", fontsize=8)
    cb2.set_ticks([0, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())/2, np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())])
    cb2.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())/2*100:.2f}', f'{np.max(np.abs(u_amp)/np.abs(u_scn_amp).max())*100:.2f}'], fontsize=8)
    ax1.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    # ax1.add_patch(Rectangle(
    # (-1.5 * square_size / 2, -1.5 * square_size / 2),  # new bottom-left
    # 1.5 * square_size,                                # new width
    # 1.5 * square_size,                                # new height
    # fill=False,
    # edgecolor='#00ff0d',
    # linestyle='-',
    # linewidth=0.7
    # ))    
    ax1.axis("off")
    ax1.set_aspect("equal")
    ax1.set_xlim(-2*np.pi, 2*np.pi)
    ax1.set_ylim(-2*np.pi, 2*np.pi)
    y_center = 0  # or e.g., y_center = Y.mean() or another value of interest
    line2 = Line2D([np.pi, 2*np.pi], [y_center, y_center], color="#00ff0d", linewidth=1.0, linestyle='-')
    ax1.add_line(line2)

    # Subplots for phase
    ax2 = fig.add_subplot(gs[1, 0])
    c3 = ax2.pcolormesh(X, Y, u_inc_phase, cmap="twilight_shifted", rasterized=True, vmin=-(np.pi), vmax=(np.pi))
    cb3 = fig.colorbar(c3, ax=ax2, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb3.set_label(r"$u_{\rm{sct}}$", fontsize=8)
    cb3.set_ticks([-(np.pi), (np.pi)])
    cb3.set_ticklabels([r'-$\pi$', r'$\pi$'], fontsize=8)  
    ax2.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    ax2.axis("off")
    ax2.set_aspect("equal")
    ax2.set_xlim(-2*np.pi, 2*np.pi)
    ax2.set_ylim(-2*np.pi, 2*np.pi)       

    ax3 = fig.add_subplot(gs[1, 1])
    c4 = ax3.pcolormesh(X, Y, np.abs(u_phase)/np.abs(u_scn_phase).max(), cmap="magma", rasterized=True)
    cb4 = fig.colorbar(c4, ax=ax3, shrink=shrink, orientation="horizontal", pad=0.07, format='%.4f')
    cb4.set_label(r"|Error| / max($u$) ($\times 10^{-2}$)", fontsize=8)
    cb4.set_ticks([0, np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())/2, np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())])
    cb4.set_ticklabels([f'{0:.1f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())/2*100:.2f}', f'{np.max(np.abs(u_phase)/np.abs(u_scn_phase).max())*100:.2f}'], fontsize=8)
    ax3.add_patch(Rectangle(square_xy, square_size, square_size, **square_props))
    ax3.axis("off")
    ax3.set_aspect("equal")
    ax3.set_xlim(-2*np.pi, 2*np.pi)
    ax3.set_ylim(-2*np.pi, 2*np.pi)


    # ======================= BAR PLOT (PINNs) =======================

    ax_bar = fig.add_subplot(gs[2, :])

    n_levels = np.arange(1, len(avg_errors) + 1)

    # Center of each square ring
    x_centers = (n_levels - 0.5) * np.pi
    bar_width = 0.8 * np.pi

    ax_bar.bar(
        x_centers,
        avg_errors,
        width=bar_width,
        color="#00ff0d",   # PINNs color (green)
        alpha=0.5,
    )

    # Labels like π, 2π, ..., 10π
    interval_labels = [
        r"$\pi$" if n == 1 else rf"${n}\pi$"
        for n in n_levels
    ]

    ax_bar.set_xticks(x_centers)
    ax_bar.set_xticklabels(interval_labels, fontsize=8)
    ax_bar.set_yticks([0,0.5,1.0])
    ax_bar.set_xlabel('Square ring interval', fontsize=8)
    ax_bar.set_ylabel('Average error', fontsize=8)

    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_title('PINNs - Amplitude', fontsize=8)

    # Subplot 5: Relative error line plot
    #ax_err = fig.add_subplot(gs[2, :])
    ax_err = fig.add_subplot(gs[3, :])

    pos = ax_err.get_position()
    ax_err.set_position([pos.x0, pos.y0 - 0.03, pos.width, pos.height])    
    #ax_err.axvline(x=np.pi, color="#acacac", linestyle='-', linewidth=1)
    #ax_err.axvline(x=2*np.pi, color="#acacac", linestyle='-', linewidth=1)  
    ax_err.axvspan(np.pi, 2*np.pi, color='gray', alpha=0.2)   
    ax_err.plot(X_slice,amp_slice, label='Relative error', color="#00ff0d")
    #ax_err.plot(x_line, rel_error_line, label='Relative error', color="#00ff0d")
    # Agregar líneas verticales en pi y 2pi
    ax_err.set_xlabel(r'$x$', fontsize=8)
    ax_err.set_ylabel(r"$|$Error$|$ / max($u$)", fontsize=8)
    ax_err.set_ylim(0, 0.6)
    #ax_err.set_ylim(0, np.max(rel_error_line) * 1.1)
    #ax_err.set_xlim(0, 2*np.pi)
    ax_err.xaxis.set_major_locator(MultipleLocator(base=np.pi))

    def format_func(value, tick_number):
        N = int(np.round(value / np.pi))
        if N == 0:
            return "0"
        elif N == 1:
            return r"$\pi$"
        elif N == -1:
            return r"$-\pi$"
        else:
            return fr"${N}\pi$"

    ax_err.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax_err.set_title('PINNs - Amplitude', fontsize=8)

    # Add rotated labels
    fig.text(0.08, 0.80, r'PINNs - Amplitude', fontsize=8, va='center', ha='center', rotation='vertical')
    fig.text(0.08, 0.57, r'PINNs - Phase', fontsize=8, va='center', ha='center', rotation='vertical')

    # Save and show
    plt.savefig("figures/generalization_pinns.svg", dpi=300, bbox_inches='tight')
    #plt.show()    