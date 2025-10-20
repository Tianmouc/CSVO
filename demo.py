import cv2
import glob
import os
import datetime
import numpy as np
import os.path as osp
from pathlib import Path
import random
import cv2
import torch
from evo.tools import plot
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from csvo.csvo import CSVO
from torch.functional import F
from csvo.utils import Timer
from csvo.config import cfg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.signal import savgol_filter
from csvo.data_readers.tartan import test_split as val_split
from csvo.plot_utils import plot_trajectory, save_trajectory_tum_format, create_html, make_traj, best_plotmode

from tianmoucv.data import TianmoucDataReader

test_split = \
    ["MH%03d"%i for i in range(8)] + \
    ["ME%03d"%i for i in range(8)]

fx, fy, cx, cy = [707.8457,708.3163,389.9121,235.1899]
intrinsics = torch.as_tensor([fx, fy, cx, cy])
print('[demo.py] YOU MAY NEED TO UPDATE YOUR intrinsics METRIX')

frames = []
sds = []

DOWN_SAMPLE_STRIDE = 5


def smooth_trajectory(trajectory, window_size=11, poly_order=3):
    smoothed_trajectory = trajectory.copy()
    for i in range(3):  # 仅平滑 x, y, z 维度
        smoothed_trajectory[:, i] = savgol_filter(trajectory[:, i], window_size, poly_order)
    return smoothed_trajectory

    
def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)


def images_to_video(frame_list,name,size=(640,320),Flip=True):
    fps = 30        
    ftmax = 1
    ftmin = 0
    out = cv2.VideoWriter(name,0x7634706d , fps, size)
    for ft in frame_list:
        ft = (ft-ftmin)/(ftmax-ftmin)
        ft[ft>1]=1
        ft[ft<0]=0
        ft2 = (ft*255).astype(np.uint8)
        out.write(ft2)
    out.release()
    
@torch.no_grad()
def run(data_path, cfg, network, viz=False, ablation="RGB", sdEncoderPath="sdEncoder.pth", downsample=False, args=None):
    
    slam = CSVO(cfg, network, ht=320, wd=640, viz=viz, ablation=ablation, sdEncoderPath=sdEncoderPath, isTianmouc=True)

    dataset = TianmoucDataReader(data_path,uniformSampler=True,strict = True,print_info=True)
    i = 0
    device = torch.device('cuda')

    start_index = args.start_index
    end_index = args.end_index
    frame_list = []

    if start_index == -1:
        start_index = 0
    if end_index == -1:
        end_index = len(dataset)-1
    
    with Timer("SLAM", enabled=True):
        for index in range(start_index,end_index):

            sample = dataset[index] 
            if sample is None:
                continue 
                
            tsdiff = sample['tsdiff'].to(device)
            image = sample['F0_HDR'].permute(2,0,1).to(device)

            frame_list.append(sample['F0_HDR'].numpy()[:,:,[2,1,0]])
        
            if viz: 
                show_image(image, 1)
    
            for t in range(0,tsdiff.shape[1],DOWN_SAMPLE_STRIDE):
            
                if t > 0:
                    image = torch.zeros_like(image).to(device)
    
                sdl = tsdiff[0:1,t:t+1,...].unsqueeze(0)
                sdr = tsdiff[1:2,t:t+1,...].unsqueeze(0)
                    
                slam(t, image, sdr, sdl, intrinsics.to(device))
                
            if index%20 == 0:
                print('[evaluate_real_data.py]vo running..:',index,'/',len(dataset))

    for _ in range(12):
        slam.update()

    return slam.terminate(),frame_list


@torch.no_grad()
def run_tmdat(config, 
             net, 
             split="validation", 
             trials=1, 
             plot=False, 
             save=False, 
             data_path='./', 
             output_path='./',
             ablation="hybrid_rgb_sd", 
             sdEncoderPath="sdEncoder.pth", 
             window_size = 1,
             downsample=False,
             args=None):
    
    if config is None:
        config = cfg
        config.merge_from_file("config/default.yaml")

    scene_name = data_path.split('/')[-1]

    start_index = args.start_index
    end_index = args.end_index

    foldername = "_".join(net.split("/")[-2:])
    foldername += scene_name
    foldername += "_start_{}__end_{}".format(start_index, end_index)
    foldername = foldername.replace(".pth", "_")
    output_path_sample = os.path.join(output_path, foldername)
    Path(output_path_sample).mkdir(exist_ok=True)
        
    for j in range(trials):

        # run the slam system
        slam_result, frame_list= run(data_path, config, net, ablation=ablation, sdEncoderPath=sdEncoderPath, downsample=downsample, args=args)

        traj_est, tstamps  = slam_result

        if traj_est is None:
            print('[evaluate_real_data.py]:warning no data found3')
            continue

        PERM = [1, 2, 0, 4, 5, 3, 6] # ned -> xyz

        if plot:
            Path(os.path.join(output_path_sample, 'trajectory_plots')).mkdir(exist_ok=True)
            Path(os.path.join(output_path_sample, 'trajectory_htmls')).mkdir(exist_ok=True)
            images_to_video(frame_list,os.path.join(output_path_sample, 'rgb.mp4'),size=(640,320),Flip=True)
            pred_xyz, gt_xyz = plot_trajectory((traj_est, tstamps), None, f"Tianmouc {scene_name.replace('_', ' ')} Trial #{j+1}",
                                    os.path.join(output_path_sample, 'trajectory_plots',f"Tianmouc_{scene_name}_Trial{j+1:02d}.pdf"), align=True, correct_scale=True)
            create_html(pred_xyz, gt_xyz, os.path.join(output_path_sample, 'trajectory_htmls',f"Tianmouc_{scene_name}_Trial{j+1:02d}.html"))
        if save:
            Path(os.path.join(output_path_sample, 'saved_trajectories')).mkdir(exist_ok=True)
            save_trajectory_tum_format((traj_est, tstamps), os.path.join(output_path_sample, 'saved_trajectories',f"Tianmouc_{scene_name}_Trial{j+1:02d}.txt"))
        print(j,"done")

    return

def set_random_seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    import sys
    sys.setrecursionlimit(3000)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--id', type=int, default=-1)
    parser.add_argument('--weights', default="./ckpts/020000.pth")
    parser.add_argument('--sdEncoder', default="sdEncoder.pth")
    parser.add_argument('--ablation_name', default="async")
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--split', default="validation")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--downsample', action="store_true")
    parser.add_argument('--data_path', default='')
    parser.add_argument('--output_path', default='')
    parser.add_argument('--start_index', default=-1, type=int)
    parser.add_argument('--end_index', default=-1, type=int)
    parser.add_argument('--window_size', type=int, default=1)
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("Running with config...")
    print(cfg)
    set_random_seed_all()

    results = run_tmdat(cfg, args.weights, split=args.split,
                           trials=args.trials, plot=args.plot,
                           save=args.save_trajectory, 
                           data_path = args.data_path, 
                           output_path = args.output_path, 
                           ablation=args.ablation_name,
                           sdEncoderPath = args.sdEncoder,
                           window_size=args.window_size, 
                           downsample=args.downsample,
                           args=args)

