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
def create_html(traj1, traj2, file_path):
    # Extract x, y, z coordinates from the trajectories
    x1, y1, z1 = zip(*traj1)
    x2, y2, z2 = zip(*traj2)
    
    # Generate smooth trajectory points using spline interpolation
    t1 = np.linspace(0, len(traj1) - 1, len(traj1))
    t2 = np.linspace(0, len(traj2) - 1, len(traj2))
    smooth_t1 = np.linspace(0, len(traj1) - 1, 100)  # More points for smoother curve
    smooth_t2 = np.linspace(0, len(traj2) - 1, 100)
    
    spl_x1 = make_interp_spline(t1, x1, k=3)
    spl_y1 = make_interp_spline(t1, y1, k=3)
    spl_z1 = make_interp_spline(t1, z1, k=3)
    
    spl_x2 = make_interp_spline(t2, x2, k=3)
    spl_y2 = make_interp_spline(t2, y2, k=3)
    spl_z2 = make_interp_spline(t2, z2, k=3)
    
    smooth_x1 = spl_x1(smooth_t1)
    smooth_y1 = spl_y1(smooth_t1)
    smooth_z1 = spl_z1(smooth_t1)
    
    smooth_x2 = spl_x2(smooth_t2)
    smooth_y2 = spl_y2(smooth_t2)
    smooth_z2 = spl_z2(smooth_t2)
    
    # Create a 3D scatter plot with two lines
    fig = go.Figure(data=[
        go.Scatter3d(
            x=smooth_x1,
            y=smooth_y1,
            z=smooth_z1,
            mode='lines+markers',
            name='Trajectory 1 (Blue Solid Line)',
            line=dict(color='blue', width=2),
            marker=dict(size=1, color='blue')
        ),
        go.Scatter3d(
            x=smooth_x2,
            y=smooth_y2,
            z=smooth_z2,
            mode='lines+markers',
            name='Trajectory 2 (Gray Dashed Line)',
            line=dict(color='gray', width=2, dash='dash'),
            marker=dict(size=1, color='gray'),
        )
    ])

    # Add labels
    fig.update_layout(
        title='3D Trajectory with Two Lines',
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Z Coordinate'
        )
    )

    # Save the plot as an interactive HTML file
    fig.write_html(file_path)



def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, tuple):
        traj, tstamps = args
        return PoseTrajectory3D(positions_xyz=traj[:,:3], orientations_quat_wxyz=traj[:,3:], timestamps=tstamps)
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)

def best_plotmode(traj):
    _, i1, i2 = np.argsort(np.var(traj.positions_xyz, axis=0))
    plot_axes = "xyz"[i2] + "xyz"[i1]
    return getattr(plot.PlotMode, plot_axes)

def plot_trajectory(pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True):
    # print(pred_traj[0].shape)
    pred_traj = make_traj(pred_traj)
    
    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)
    pred_xyz = pred_traj._positions_xyz
    try:
        gt_xyz = gt_traj._positions_xyz
    except:
        gt_xyz = None
        pass
    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = best_plotmode(gt_traj if (gt_traj is not None) else pred_traj)
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, '--', 'gray')
    print(ax)
    print("______________________________________________________________________________________")
    print(plot_mode)
    print("______________________________________________________________________________________")
    print(pred_traj)
    print("______________________________________________________________________________________")
    plot.traj(ax, plot_mode, pred_traj, '-', 'blue')
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)
    print(f"Saved {filename}")
    return pred_xyz, gt_xyz

def save_trajectory_tum_format(traj, filename):
    traj = make_traj(traj)
    tostr = lambda a: ' '.join(map(str, a))
    with Path(filename).open('w') as f:
        for i in range(traj.num_poses):
            f.write(f"{traj.timestamps[i]} {tostr(traj.positions_xyz[i])} {tostr(traj.orientations_quat_wxyz[i][[1,2,3,0]])}\n")
    print(f"Saved {filename}")
def clear_directory(path):
    """
    清空指定路径下的所有文件和子文件夹。

    参数:
        path (str): 要清空的目标路径。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"路径 {path} 不存在！")
    
    if not os.path.isdir(path):
        raise ValueError(f"{path} 不是一个有效的文件夹！")
    
    # 遍历目标路径下的所有文件和子文件夹
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        
        # 如果是文件，删除文件
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"已删除文件: {item_path}")
        
        # 如果是文件夹，递归删除文件夹及其内容
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"已删除文件夹: {item_path}")
    
    print(f"路径 {path} 已被清空！")
def plot_trajectory_top_view(trajectory1, trajectory2, title="Top View of Trajectories", output_size=(640, 480)):
    """
    绘制两条轨迹的俯视图并返回一个指定大小的 np.array。

    参数:
        trajectory1 (np.array): 第一条轨迹，形状为 (n, 3)，其中 n 是轨迹点的数量。
        trajectory2 (np.array): 第二条轨迹，形状为 (n, 3)，其中 n 是轨迹点的数量。
        title (str): 图表标题，默认为 "Top View of Trajectories"。
        output_size (tuple): 输出图像的大小，默认为 (640, 480)。

    返回:
        np.array: 绘制的图像，形状为 (height, width, 3)，值域为 [0, 255]。
    """
    # 检查输入是否为 numpy 数组
    if not isinstance(trajectory1, np.ndarray) or not isinstance(trajectory2, np.ndarray):
        raise ValueError("轨迹数据必须是 numpy 数组")
    
    # 检查输入的形状是否正确
    if trajectory1.shape[1] != 3 or trajectory2.shape[1] != 3:
        raise ValueError("轨迹数据的形状必须为 (n, 3)")
    
    # 提取x和y坐标用于俯视图
    x1, y1 = trajectory1[:, 0], trajectory1[:, 1]
    x2, y2 = trajectory2[:, 0], trajectory2[:, 1]

    # 创建一个 Matplotlib 图像
    fig, ax = plt.subplots(figsize=(output_size[0] / 100, output_size[1] / 100))  # 设置图像大小
    ax.plot(x1, y1, label='Trajectory 1', marker='o', linestyle='-')
    ax.plot(x2, y2, label='Trajectory 2', marker='o', linestyle='-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')  # 保持比例

    # 将图像渲染到内存中的 numpy 数组
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(output_size[1], output_size[0], 3)  # 调整形状为 (height, width, 3)

    # 关闭图像
    plt.close(fig)

    return image
def make_video(image_folder, output_video):
    # 获取文件夹内所有文件的路径
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    # 按文件名排序
    images.sort()

    # 读取第一张图片以获取视频的大小
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 创建一个 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
    out = cv2.VideoWriter(output_video, fourcc, 24, (width, height))

    # 逐帧写入视频
    for img in images:
        img_path = os.path.join(image_folder, img)
        frame = cv2.imread(img_path)
        out.write(frame)

    # 释放 VideoWriter 对象
    out.release()
def create_video_from_traj_and_imgs(estimated_trajectory, actual_trajectory, images, output_video_path, align=True, correct_scale=True):
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30  # 每秒30帧
    frame_size = (640, 480)  # 可以根据需求调整输出视频分辨率
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    timestamps = estimated_trajectory[1]
    # 初始化轨迹
    # print(estimated_trajectory[0].shape)
    estimated_trajectory = make_traj(estimated_trajectory)
    actual_trajectory = make_traj(actual_trajectory)
    
    # 对轨迹进行同步和对齐（如果有提供实际轨迹）
    if actual_trajectory is not None:
        actual_trajectory, estimated_trajectory = sync.associate_trajectories(actual_trajectory, estimated_trajectory)
        
        if align:
            estimated_trajectory.align(actual_trajectory, correct_scale=correct_scale)

    # 获取轨迹的坐标
    pred_xyz = estimated_trajectory._positions_xyz
    # print(pred_xyz)
    # print(pred_xyz.shape)
    gt_xyz = actual_trajectory._positions_xyz
    # print(gt_xyz)
    # 创建视频绘图对象
    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = best_plotmode(actual_trajectory if actual_trajectory is not None else estimated_trajectory)
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title("Trajectory Video")

    # 设置绘制轨迹的标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # 调整坐标轴的显示范围


    print(len(images))
    for i in range(1,len(images)):
        img_path = images[i]
        # pose_est = estimated_trajectory[i]  # 估计轨迹
        # pose_actual = actual_trajectory[i]  # 实际轨迹

        # 读取图片并调整大小
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, frame_size)

        img_traj = plot_trajectory_top_view(pred_xyz[:i], gt_xyz[:i])

        # 清除之前的轨迹
        # ax.cla()
        # 绘制当前的轨迹：俯视图，X 和 Y 坐标
        # if actual_trajectory is not None:
        #     # ax.plot(gt_xyz[:i+1, 0], gt_xyz[:i+1, 1], '--', color='gray', label="Ground Truth")
        #     plot.traj(ax, plot_mode, actual_trajectory, '--', 'gray')
        
        # # ax.plot(pred_xyz[:i+1, 0], pred_xyz[:i+1, 1], '-', color='blue', label="Predicted")
        # plot.traj(ax, plot_mode, estimated_trajectory, '-', 'blue')

        # # 添加图例
        # # ax.legend()

        # # 从 2D 图生成图像
        # plt.draw()
        # plt.pause(0.001)  # 暂停以更新图像
        # trajectory_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # trajectory_img = trajectory_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # trajectory_img = cv2.resize(trajectory_img, (frame_size[0], 480))
        # 将轨迹图和图片合成
        combined_img = np.hstack((img_resized, img_traj))

        # 将合成图像写入视频
        if combined_img.shape[2] == 1:  # 如果只有一个通道（灰度）
            combined_img = cv2.cvtColor(combined_img, cv2.COLOR_GRAY2RGB)
        
        # 保存合成图像为 PNG 文件（可选）
        cv2.imwrite(f'demo_images/combined_image_{i}.png', combined_img)

        # 将合成图像写入视频
        out.write(combined_img)

    # 释放视频写入对象
    out.release()
    make_video('./demo_videos', output_video_path)
    print(f"Video saved to {output_video_path}")
    clear_directory('demo_images')
