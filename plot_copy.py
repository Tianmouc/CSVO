from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from evo.tools import plot
from pathlib import Path
import plotly.graph_objects as go
import cv2
import shutil
import os
from scipy.interpolate import make_interp_spline
from dpvo.plot_utils import plot_trajectory, make_traj, best_plotmode

def plot_trajectory(pred_trajs, gt_traj=None, title="", filename="", align=True, correct_scale=True):
    """
    Plot multiple predicted trajectories and a ground truth trajectory on the same plot.

    Parameters:
    - pred_trajs: List of predicted trajectories.
    - gt_traj: Ground truth trajectory (optional).
    - title: Title of the plot.
    - filename: Filename to save the plot.
    - align: Whether to align the trajectories.
    - correct_scale: Whether to correct the scale when aligning.
    """
    # Convert predicted trajectories to the appropriate format
    pred_trajs.reverse()
    pred_trajs = [make_traj(pred_traj) for pred_traj in pred_trajs]
    
    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        for i in range(len(pred_trajs)):
            gt_traj, pred_trajs[i] = sync.associate_trajectories(gt_traj, pred_trajs[i])
            if align:
                pred_trajs[i].align(gt_traj, correct_scale=correct_scale)
    
    # Extract positions
    pred_xyzs = [pred_traj._positions_xyz for pred_traj in pred_trajs]
    gt_xyz = gt_traj._positions_xyz if gt_traj is not None else None

    # Create plot
    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8),dpi=900)
    plot_mode = best_plotmode(gt_traj if (gt_traj is not None) else pred_trajs[0])
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='datalim')
    # Plot ground truth trajectory if available
    

    # Plot predicted trajectories
    colors = ['blue','green','red', 'magenta', 'yellow']  # Add more colors if needed
    labels = ['RampVO', 'DPVO','CSVO']
    for i, pred_traj in enumerate(pred_trajs):
        plot.traj(ax, plot_mode, pred_traj, '-', colors[i % len(colors)], label=labels[i],alpha=0.5 if i != 2 else 1)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, '--', 'black', label='Ground Truth')
    for line in ax.lines:
        line.set_linewidth(3)
    ax.lines[-1].set_linewidth(1.5)
    # Add legend
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(fontsize=22, loc='lower right')
    # ax.invert_yaxis()
    # Save the plot
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    print(f"Saved {filename}")

    return pred_xyzs, gt_xyz
length = 2500
env = 'abandonedfactory'
condition = 'Hard'
scene = 'P011'
traj_ref = f'/data/zzx/DPVO_E2E/datasets/TartanAirNew/{env}/{condition}/{scene}/pose_left.txt'
traj_est1 = f'/data/zzx/DPVO_E2E/saved_trajectories/plana_aug_22_16w/TartanAir_{env}_{condition}_{scene}_Trial01.txt'
traj_est2 = f'/data/zzx/DPVO_E2E/saved_trajectories/dpvo_aug_overexpoture/TartanAir_{env}_{condition}_{scene}_Trial01.txt'
traj_est3 = f'/data/zzx/RampVO/trajectory_evaluation/full_data/trial_0/default/_data_zzx_DPVO_E2E_datasets_TartanAirNew_{env}_{condition}_{scene}/stamped_traj_estimate.txt'
# traj_ref = '/data/zzx/DPVO_E2E/datasets/TartanAirNew/endofworld/Easy/P009/pose_left.txt'
# traj_est1 = '/data/zzx/DPVO_E2E/saved_trajectories/plana_aug_22_16w/TartanAir_endofworld_Easy_P009_Trial01.txt'
# traj_est2 = '/data/zzx/DPVO_E2E/saved_trajectories/dpvo_aug_overexpoture/TartanAir_endofworld_Easy_P009_Trial01.txt'
# traj_est3 = '/data/zzx/RampVO/trajectory_evaluation/full_data/trial_0/default/P009/stamped_traj_estimate.txt'
PERM = [1,2,0, 4, 5, 3, 6]
traj_ref = np.loadtxt(traj_ref, delimiter=" ")[:length,:][::1, PERM].astype(np.float64)
# traj_est = np.loadtxt(traj_est, delimiter=" ")[:,1:][::1, PERM].astype(np.float64)
tstamps = np.array([i for i in range(len(traj_ref))])
print(len(traj_ref))
traj_paths = [traj_est1, traj_est2]

traj_ests = [(np.loadtxt(traj, delimiter=" ")[:length,1:][::1, PERM].astype(np.float64), tstamps) for traj in traj_paths]
PERM = [1, 2, 0, 4, 5, 3, 6]
traj_ests.append((np.loadtxt(traj_est3, delimiter=" ")[:length,1:][::1, PERM].astype(np.float64), tstamps))
for traj in traj_ests:
    # traj[0] = smooth_trajectory(traj[0])
    print(traj[0].shape)
plot_trajectory(traj_ests, (traj_ref, tstamps), 'output.pdf', align=True, correct_scale=True)