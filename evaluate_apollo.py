import cv2
import glob
import os
import datetime
import numpy as np
import os.path as osp
from pathlib import Path

import torch
from csvo.csvo import CSVO
from csvo.utils import Timer
from csvo.config import cfg

from csvo.data_readers.tartan import test_split as val_split
from csvo.plot_utils import plot_trajectory, save_trajectory_tum_format

test_split = \
    ["MH%03d"%i for i in range(8)] + \
    ["ME%03d"%i for i in range(8)]

STRIDE = 1
fx, fy, cx, cy = [320, 320, 320, 240]


def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

def video_iterator(imagedir, ext=".png", preload=True):
    imfiles = glob.glob(osp.join(imagedir, "*{}".format(ext)))

    data_list = []
    timeSurface0 = []
    timeSurface1 = []
    count = 0
    for imfile in sorted(imfiles)[::STRIDE]:
        image = torch.from_numpy(cv2.imread(imfile)).permute(2,0,1)
        count += 1
        if 'image_left_plana' in imfile:
            pathr = imfile.replace("image_left_plana", "SDR_frames_low_rate").replace(".png", '.npy')
            pathl = imfile.replace("image_left_plana", "SDL_frames_low_rate").replace(".png", '.npy')
        else:
            pathr = imfile.replace("image_left", "SDR_frames_low_rate").replace(".png", '.npy')
            pathl = imfile.replace("image_left", "SDL_frames_low_rate").replace(".png", '.npy')
        sdr = torch.from_numpy(np.load(pathr))
        sdl = torch.from_numpy(np.load(pathl))
        if sdr.shape[0] != 1:
            sdr = sdr.unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
            sdl = sdl.unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        data_list.append((image,sdr, sdl, intrinsics))

    for (image,sdr, sdl, intrinsics) in data_list:
        yield image.cuda(), sdr.cuda(), sdl.cuda(), intrinsics.cuda()

@torch.no_grad()
def run(imagedir, cfg, network, viz=False, ablation="RGB", sdEncoderPath="sdEncoder.pth"):
    slam = CSVO(cfg, network, ht=480, wd=640, viz=viz, ablation=ablation, sdEncoderPath=sdEncoderPath)

    for t, (image,sdr, sdl, intrinsics) in enumerate(video_iterator(imagedir)):
        if t > 4000:
            break
        if viz: 
            show_image(image, 1)
        
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
    
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    return result.stats["rmse"]


@torch.no_grad()
def evaluate(config, net, split="validation", trials=1, plot=False, save=False, path='', ablation="RGB", sdEncoderPath="sdEncoder.pth"):

    if config is None:
        config = cfg
        config.merge_from_file("config/default.yaml")

    if not os.path.isdir("TartanAirResults"):
        os.mkdir("TartanAirResults")

    scenes = ['record1_segment0_54-320', "record3_segment0_81-208",'record4_segment0_96-252']
    results = {}
    all_results = []
    for i, scene in enumerate(scenes):
        scene_name = scene.replace("/", '')
        results[scene] = []
        for j in range(trials):
            if split == 'test':
                scene_path = os.path.join("datasets/mono", scene)
                traj_ref = osp.join("datasets/mono", "mono_gt", scene + ".txt")
            
            elif split == 'validation':
                
                traj_ref = osp.join("datasets/Apollo", scene, "pose_left.txt")
                if 'async' in ablation.lower() or 'sd_only' in ablation.lower():
                    scene_path = os.path.join("datasets/Apollo", scene, "image_left")
                elif 'sync' in ablation.lower() or 'dpvo' in ablation.lower():
                    scene_path = os.path.join("datasets/Apollo", scene, "image_left")
                else:
                    raise ValueError("Ablation type not permitted!")

            # run the slam system
            traj_est, tstamps = run(scene_path, config, net, ablation=ablation, sdEncoderPath=sdEncoderPath)
            PERM = [1, 2, 0, 4, 5, 3, 6] # ned -> xyz
            traj_ref = np.loadtxt(traj_ref, delimiter=" ")[::STRIDE, PERM]

            # do evaluation
            minLen = min(len(traj_ref), len(traj_est))
            traj_ref = traj_ref[:minLen]
            traj_est = traj_est[:minLen]
            tstamps = tstamps[:minLen]
            ate_score = ate(traj_ref, traj_est, tstamps)
            all_results.append(ate_score)
            results[scene].append(ate_score)

            if plot:
                scene_name = '_'.join(scene.split('/')).title()
                Path("trajectory_plots"+'/'+path).mkdir(exist_ok=True)
                plot_trajectory((traj_est, tstamps), (traj_ref, tstamps), f"Apollo {scene_name.replace('_', ' ')} Trial #{j+1} (ATE: {ate_score:.03f})",
                                f"trajectory_plots/{path}/Apollo{scene_name}_Trial{j+1:02d}.pdf", align=True, correct_scale=True)

            if save:
                Path("saved_trajectories"+'/'+path).mkdir(exist_ok=True)
                save_trajectory_tum_format((traj_est, tstamps), f"saved_trajectories/{path}/Apollo{scene_name}_Trial{j+1:02d}.txt")
            print(j,"done")

        print(scene, sorted(results[scene]))

    results_dict = dict([("Tartan/{}".format(k), np.median(v)) for (k, v) in results.items()])

    # write output to file with timestamp
    with open(osp.join("TartanAirResults", datetime.datetime.now().strftime('%m-%d-%I%p.txt')), "w") as f:
        f.write(','.join([str(x) for x in all_results]))

    xs = []
    for scene in results:
        x = np.median(results[scene])
        xs.append(x)

    ates = list(all_results)
    results_dict["AUC"] = np.maximum(1 - np.array(ates), 0).mean()
    results_dict["AVG"] = np.mean(xs)

    return results_dict


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    import sys
    sys.setrecursionlimit(3000)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--id', type=int, default=-1)
    parser.add_argument('--weights', default="csvo.pth")
    parser.add_argument('--sdEncoder', default="sdEncoder.pth")
    parser.add_argument('--ablation_name', default="RGB")
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--split', default="validation")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--path', default='')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("Running with config...")
    print(cfg)

    torch.manual_seed(1234)

    if args.id >= 0:
        scene_path = os.path.join("datasets/mono", test_split[args.id])
        traj_est, tstamps = run(scene_path, cfg, args.weights, viz=args.viz)

        traj_ref = osp.join("datasets/mono", "mono_gt", test_split[args.id] + ".txt")
        traj_ref = np.loadtxt(traj_ref, delimiter=" ")[::STRIDE,[1, 2, 0, 4, 5, 3, 6]]

        # do evaluation
        print(ate(traj_ref, traj_est, tstamps))

    else:
        results = evaluate(cfg, args.weights, split=args.split, trials=args.trials, plot=args.plot, save=args.save_trajectory, path=args.path, ablation=args.ablation_name, sdEncoderPath = args.sdEncoder)
        for k in results:
            print(k, results[k])
