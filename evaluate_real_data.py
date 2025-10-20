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
# from Visualization.viz_sd import visualize_diff
from scipy.signal import savgol_filter
from csvo.data_readers.tartan import test_split as val_split
from csvo.plot_utils import plot_trajectory, save_trajectory_tum_format, create_html, make_traj, best_plotmode

test_split = \
    ["MH%03d"%i for i in range(8)] + \
    ["ME%03d"%i for i in range(8)]

fx, fy, cx, cy = [707.8457,708.3163,389.9121,235.1899]

frames = []
sds = []

DOWN_SAMPLE_STRIDE = 5


def smooth_trajectory(trajectory, window_size=11, poly_order=3):
    smoothed_trajectory = trajectory.copy()
    for i in range(3):  # 仅平滑 x, y, z 维度
        smoothed_trajectory[:, i] = savgol_filter(trajectory[:, i], window_size, poly_order)
    return smoothed_trajectory

def normalize_tensor(tensor):
    # 计算均值和标准差
    normalized_tensor = tensor/2
    
    return normalized_tensor
def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

def video_iterator(imagedir, ext=".png", preload=True, ablation='plana', downsample=False):
    
    
    imfiles = glob.glob(osp.join(imagedir, "*{}".format(ext)))
    data_list = []
    timeSurface0 = []
    timeSurface1 = []
    count = 0
    global frames
    frames = []

    print('[evaluate_real_data.py]imagedir:',imagedir)

    for imfile in sorted(imfiles):

        if ablation == 'sync':
            assert not downsample
            
        index = int(imfile.split("/")[-1].replace(".png",""))
        #prepare rgb
        if (downsample or ablation == 'dpvo') and index % DOWN_SAMPLE_STRIDE != 0:
                continue

        if ablation == 'sync':
            if int(imfile.split("/")[-1].replace(".png","")) % 25 != 0:
                frames.append(imfile)
                image = torch.from_numpy(cv2.imread(imfile)).permute(2,0,1)
                image[True] = 0
            else:
                frames.append(imfile)
                image = torch.from_numpy(cv2.imread(imfile)).permute(2,0,1)
        else:
            frames.append(imfile)
            image = torch.from_numpy(cv2.imread(imfile)).permute(2,0,1)
            
        image = torch.flip(image, dims=[1])

        
        if 'image_left_plana' in imfile:
            pathr = imfile.replace("image_left_plana", "SDR_frames_low_rate").replace(".png", '.npy')
            pathl = imfile.replace("image_left_plana", "SDL_frames_low_rate").replace(".png", '.npy')
        elif ablation in ['dpvo', 'async', 'sync']:
            pathr = imfile.replace("image_left_aligned_high", "SD_frames_aligned_high").replace(".png", '.npy')
        elif ablation == 'plana_td':
            pathr = imfile.replace("image_left_aligned_high", "TD_frames_aligned_high").replace(".png", '.npy')
        else:
            raise ValueError("Ablation type not permitted!")

        if ablation == 'plana_td':
            lx = torch.from_numpy(np.load(pathr)[:,:])
            lx = torch.flip(lx, dims=[1])
            lx *=5
            ly = lx.clone()
        else:
            lx = torch.from_numpy(np.load(pathr)[:,:,0])
            ly = torch.from_numpy(np.load(pathr)[:,:,1])
            #SDL,SDR翻转操作，五步
            lx = torch.flip(lx, dims=[1])
            ly = torch.flip(ly, dims=[1])
            temp = lx
            lx = ly
            ly = temp

        sdr=lx
        sdl=ly
        
        sds.append(sdr)
        if count == 0:
            print('[evaluate_real_data.py]sdl.shape:',sdl.shape)
        if sdr.shape[0] != 1:
            sdr = sdr.unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
            sdl = sdl.unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
            
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        data_list.append((image,sdr, sdl, intrinsics))
        count += 1

    print("dataset length:",len(data_list))
    if len(data_list)<25:
        print('[evaluate_real_data.py]:warning no data found')
        return None,None,None,None
        
    for (image,sdr, sdl, intrinsics) in data_list:
        yield image.cuda(), sdr.cuda(), sdl.cuda(), intrinsics.cuda()

@torch.no_grad()
def run(imagedir, cfg, network, viz=False, ablation="RGB", sdEncoderPath="sdEncoder.pth", downsample=False):
    slam = CSVO(cfg, network, ht=320, wd=640, viz=viz, ablation=ablation, sdEncoderPath=sdEncoderPath, isTianmouc=True)

    for t, (image,sdr, sdl, intrinsics) in enumerate(video_iterator(imagedir,ablation=ablation, downsample=downsample)):
        # print("Done")
        if intrinsics is None:
            print('[evaluate_real_data.py]:warning no data found')
            return
        if viz: 
            show_image(image, 1)
        if t % DOWN_SAMPLE_STRIDE != 0:
            image = torch.zeros_like(image)
        with Timer("SLAM", enabled=False):
            slam(t, image,sdr, sdl, intrinsics)
        if t%20 == 0:
            print('[evaluate_real_data.py]vo running..:',t)

    for _ in range(12):
        slam.update()

    return slam.terminate()


def ate(traj_ref, traj_est, timestamps):
    import evo
    import evo.main_ape as main_ape
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core.metrics import PoseRelation

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=timestamps)

    traj_ref = PoseTrajectory3D(
        positions_xyz=traj_ref[:,:3],
        orientations_quat_wxyz=traj_ref[:,3:],
        timestamps=timestamps)
    try:
        result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
    except:
        print("[CSVO/evaluate_real_data.py] ERROR: Alignment failed")
        return 1000
    return result.stats["rmse"]


@torch.no_grad()
def evaluate(config, 
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

    results = {}
    all_results = []
    
    scenes = [os.path.join(data_path, scene) for scene in os.listdir(data_path)]

    for i, scene in enumerate(scenes):
        scene_name = scene.replace("/", '')
        results[scene] = []

        start_index = args.start_index
        end_index = args.end_index

        #Path(output_path).mkdir(exist_ok=True)
        foldername = "_".join(net.split("/")[-2:])
        foldername += scene.split("/")[-1]
        foldername += "_start_{}__end_{}".format(start_index, end_index)
        foldername = foldername.replace(".pth", "_")
        output_path_sample = os.path.join(output_path, foldername)
        Path(output_path_sample).mkdir(exist_ok=True)
        
        for j in range(trials):

            traj_ref = osp.join(scene, "pose_left.txt")
            if ablation.lower() == 'sd_only':
                scene_path = os.path.join( scene, "SD_frames_aligned_high")
            if ablation.lower() == 'td_only':
                scene_path = os.path.join( scene, "TD_frames_aligned_high")
            elif ablation.lower() in['async','sync']:
                scene_path = os.path.join(scene, "image_left_aligned_high")
            else:
                raise ValueError("Ablation type not permitted! please setablation name in [sd_only, td_only, async, sync]")

            # run the slam system
            traj_est, tstamps = run(scene_path, config, net, ablation=ablation, sdEncoderPath=sdEncoderPath, downsample=downsample)

            PERM = [1, 2, 0, 4, 5, 3, 6] # ned -> xyz
            traj_ref = np.loadtxt(traj_ref, delimiter=" ")[:, PERM]
            traj_ref /= 100

            print(len(traj_ref),start_index,end_index)
            
            if not (start_index == -1 and end_index == -1):
                traj_ref = traj_ref[start_index:end_index]
            if ablation=='dpvo' or downsample:
                traj_ref = traj_ref[::DOWN_SAMPLE_STRIDE]

            print(len(traj_ref))

            minLen = min(len(traj_ref), len(traj_est))
            traj_ref = traj_ref[:minLen]
            traj_est = traj_est[:minLen]
            tstamps = tstamps[:minLen]
            ate_score = ate(traj_ref, traj_est, tstamps)
            all_results.append(ate_score)
            results[scene].append(ate_score)

            if plot:
                try:
                    scene_name = scene.split("/")[-1]
                    Path(os.path.join(output_path_sample, 'trajectory_plots')).mkdir(exist_ok=True)
                    Path(os.path.join(output_path_sample, 'trajectory_htmls')).mkdir(exist_ok=True)
                    pred_xyz, gt_xyz = plot_trajectory((traj_est, tstamps), (traj_ref, tstamps), f"Tianmouc {scene_name.replace('_', ' ')} Trial #{j+1} (ATE: {ate_score:.03f})",
                                    os.path.join(output_path_sample, 'trajectory_plots',f"Tianmouc_{scene_name}_Trial{j+1:02d}.pdf"), align=True, correct_scale=True)
                    create_html(pred_xyz, gt_xyz, os.path.join(output_path_sample, 'trajectory_htmls',f"Tianmouc_{scene_name}_Trial{j+1:02d}.html"))
                except:
                    print("[CSVO/evaluate_real_data.py] ERROR: Alignment failed")
            if save:
                Path(os.path.join(output_path_sample, 'saved_trajectories')).mkdir(exist_ok=True)
                save_trajectory_tum_format((traj_est, tstamps), os.path.join(output_path_sample, 'saved_trajectories',f"Tianmouc_{scene_name}_Trial{j+1:02d}.txt"))
            print(j,"done")
            # return

        print(scene, sorted(results[scene]))

    results_dict = dict([("Tartan/{}".format(k), np.median(v)) for (k, v) in results.items()])

    xs = []
    for scene in results:
        x = np.median(results[scene])
        xs.append(x)

    ates = list(all_results)
    results_dict["AUC"] = np.maximum(1 - np.array(ates), 0).mean()
    results_dict["AVG"] = np.mean(xs)

    return results_dict

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
    # torch.manual_seed(1234)
    if args.start_index != -1:
        start_index = args.start_index
        end_index = args.end_index


    if args.id >= 0:
        scene_path = os.path.join("datasets/mono", test_split[args.id])
        traj_est, tstamps = run(scene_path, cfg, args.weights, viz=args.viz)

        traj_ref = osp.join("datasets/mono", "mono_gt", test_split[args.id] + ".txt")
        traj_ref = np.loadtxt(traj_ref, delimiter=" ")[:,[1, 2, 0, 4, 5, 3, 6]]

        # do evaluation
        print(ate(traj_ref, traj_est, tstamps))

    else:
        results = evaluate(cfg, args.weights, split=args.split, trials=args.trials, plot=args.plot,
                           save=args.save_trajectory, data_path = args.data_path, output_path = args.output_path, 
                           ablation=args.ablation_name, sdEncoderPath = args.sdEncoder, window_size=args.window_size, downsample=args.downsample,args=args)
        for k in results:
            print(k, results[k])
