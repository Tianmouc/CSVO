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

STRIDE = 1
fx, fy, cx, cy = [707.8457,708.3163,389.9121,235.1899]

start_index = 1000
end_index = 1500
frames = []
sds = []


def smooth_trajectory(trajectory, window_size=11, poly_order=3):
    """
    对轨迹的前三个维度 (x, y, z) 进行平滑处理
    :param trajectory: (N, 7) 的 numpy 数组
    :param window_size: 滤波窗口大小（必须为奇数）
    :param poly_order: 拟合多项式阶数（通常小于 window_size）
    :return: 平滑后的轨迹 (N, 7)
    """
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
    print(imagedir)
    imfiles = glob.glob(osp.join(imagedir, "*{}".format(ext)))

    data_list = []
    timeSurface0 = []
    timeSurface1 = []
    count = 0
    global frames
    frames = []
    print(len(imfiles), start_index, end_index)
    for imfile in sorted(imfiles)[start_index:end_index][::STRIDE]:
        if ablation == 'dpvo' or downsample:
            if int(imfile.split("/")[-1].replace(".png","")) % 5 != 0:
                continue
        if downsample:
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
        count += 1

        if 'image_left_plana' in imfile:
            pathr = imfile.replace("image_left_plana", "SDR_frames_low_rate").replace(".png", '.npy')
            pathl = imfile.replace("image_left_plana", "SDL_frames_low_rate").replace(".png", '.npy')
        elif ablation in ['dpvo', 'plana']:
            # print(ablation)
            pathr = imfile.replace("image_left_aligned_high", "SD_frames_aligned_high").replace(".png", '.npy')
        elif ablation == 'plana_td':
            pathr = imfile.replace("image_left_aligned_high", "TD_frames_aligned_high").replace(".png", '.npy')
        else:
            raise ValueError("Ablation type not permitted!")
        if ablation != 'plana_td':
            lx = torch.from_numpy(np.load(pathr)[:,:,0])
            ly = torch.from_numpy(np.load(pathr)[:,:,1])
            lx /= 2
            ly /= 2
            lx = torch.flip(lx, dims=[1])
            ly = torch.flip(ly, dims=[1])
            temp = lx
            lx = ly
            ly = temp
        else:
            lx = torch.from_numpy(np.load(pathr)[:,:])
            ly = torch.from_numpy(np.load(pathr)[:,:])
            lx = torch.flip(lx, dims=[1])
            ly = torch.flip(ly, dims=[1])
            lx *=5
            ly *=5
        
        sdr=lx
        sdl=ly
        
        sds.append(sdr)
        if count == 0:
            print(sdl.shape)
            count  = 1
        if sdr.shape[0] != 1:
            sdr = sdr.unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
            sdl = sdl.unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        data_list.append((image,sdr, sdl, intrinsics))

    print("dataset length:",len(data_list))
    for (image,sdr, sdl, intrinsics) in data_list:
        yield image.cuda(), sdr.cuda(), sdl.cuda(), intrinsics.cuda()

@torch.no_grad()
def run(imagedir, cfg, network, viz=False, ablation="RGB", sdEncoderPath="sdEncoder.pth", downsample=False):
    slam = CSVO(cfg, network, ht=320, wd=640, viz=viz, ablation=ablation, sdEncoderPath=sdEncoderPath, isTianmouc=True)

    for t, (image,sdr, sdl, intrinsics) in enumerate(video_iterator(imagedir,ablation=ablation, downsample=downsample)):
        # print("Done")
        if t > 4000:
            break
        if viz: 
            show_image(image, 1)
        if t % 5 != 0:
            image = torch.zeros_like(image)
        with Timer("SLAM", enabled=False):
            slam(t, image,sdr, sdl, intrinsics)

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
        print("Alignment failed")
        return 1000
    return result.stats["rmse"]


@torch.no_grad()
def evaluate(config, net, split="validation", trials=1, plot=False, save=False, path='', ablation="RGB", sdEncoderPath="sdEncoder.pth", window_size = 1, downsample=False):
    path = "_".join(net.split("/")[-2:])
    if config is None:
        config = cfg
        config.merge_from_file("config/default.yaml")

    if not os.path.isdir("TartanAirResults"):
        os.mkdir("TartanAirResults")

    scenes = test_split if split=="test" else val_split

    results = {}
    all_results = []
    base_path = '/data/zzx/DPVO_E2E/datasets/TartanAirNew/tianmouc_splited_dataset_20250224/train'
    base_path_1 = '/data/zzx/DPVO_E2E/datasets/TartanAirNew/tianmouc_splited_dataset_20250114/test'

    scenes = [os.path.join(base_path_1, scene) for scene in os.listdir(base_path_1)]
    scenes = scenes[:3]

    for i, scene in enumerate(scenes):
        if not '21' in scene:
            continue
        scene_name = scene.replace("/", '')
        results[scene] = []
        path = "_".join(net.split("/")[-2:])
        path += scene.split("/")[-1]
        path += "_start_{}__end_{}".format(start_index, end_index)
        path = path.replace(".pth", "_")
        for j in range(trials):

            # estimated trajectory
            if split == 'test':
                scene_path = os.path.join("datasets/mono", scene)
                traj_ref = osp.join("datasets/mono", "mono_gt", scene + ".txt")
            
            elif split == 'validation':
                
                traj_ref = osp.join(scene, "pose_left.txt")
                if 'plana' in ablation.lower() or 'sd_only' in ablation.lower() or 'td_only' in ablation.lower():
                    scene_path = os.path.join( scene, "image_left_aligned_high")
                elif 'planb' in ablation.lower() or 'dpvo' in ablation.lower():
                    scene_path = os.path.join(scene, "image_left_aligned_high")
                else:
                    raise ValueError("Ablation type not permitted!")

            # run the slam system
            traj_est, tstamps = run(scene_path, config, net, ablation=ablation, sdEncoderPath=sdEncoderPath, downsample=downsample)
            PERM = [1, 2, 0, 4, 5, 3, 6] # ned -> xyz
            traj_ref = np.loadtxt(traj_ref, delimiter=" ")[::STRIDE, PERM]

            traj_ref /= 100
            if not (start_index == -1 and end_index == -1):

                traj_ref = traj_ref[start_index:end_index]
            if ablation=='dpvo' or downsample:
                traj_ref = traj_ref[::5]

            minLen = min(len(traj_ref), len(traj_est))
            traj_ref = traj_ref[:minLen]
            traj_est = traj_est[:minLen]
            tstamps = tstamps[:minLen]
            ate_score = ate(traj_ref, traj_est, tstamps)
            all_results.append(ate_score)
            results[scene].append(ate_score)

            
            if plot:
                scene_name = scene.split("/")[-1]
                Path("trajectory_plots"+'/'+path).mkdir(exist_ok=True)
                Path(f"trajectory_htmls/{path}").mkdir(exist_ok=True)
                pred_xyz, gt_xyz = plot_trajectory((traj_est, tstamps), (traj_ref, tstamps), f"Tianmouc {scene_name.replace('_', ' ')} Trial #{j+1} (ATE: {ate_score:.03f})",
                                f"trajectory_plots/{path}/Tianmouc_{scene_name}_Trial{j+1:02d}.pdf", align=True, correct_scale=True)
                create_html(pred_xyz, gt_xyz, f"trajectory_htmls/{path}/Tianmouc_{scene_name}_Trial{j+1:02d}.html")
            if save:
                Path("saved_trajectories"+'/'+path).mkdir(exist_ok=True)
                save_trajectory_tum_format((traj_est, tstamps), f"saved_trajectories/{path}/Tianmouc_{scene_name}_Trial{j+1:02d}.txt")
            print(j,"done")
            # return

        print(scene, sorted(results[scene]))

    results_dict = dict([("Tartan/{}".format(k), np.median(v)) for (k, v) in results.items()])

    # write output to file with timestamp
    # with open(osp.join("TartanAirResults", datetime.datetime.now().strftime('%m-%d-%I%p.txt')), "w") as f:
    #     f.write(','.join([str(x) for x in all_results]))

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
    parser.add_argument('--weights', default="dpvo.pth")
    parser.add_argument('--sdEncoder', default="sdEncoder.pth")
    parser.add_argument('--ablation_name', default="RGB")
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--split', default="validation")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--downsample', action="store_true")
    parser.add_argument('--path', default='')
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
        traj_ref = np.loadtxt(traj_ref, delimiter=" ")[::STRIDE,[1, 2, 0, 4, 5, 3, 6]]

        # do evaluation
        print(ate(traj_ref, traj_est, tstamps))

    else:
        results = evaluate(cfg, args.weights, split=args.split, trials=args.trials, plot=args.plot, save=args.save_trajectory, path=args.path, ablation=args.ablation_name, sdEncoderPath = args.sdEncoder, window_size=args.window_size, downsample=args.downsample)
        for k in results:
            print(k, results[k])
