import cv2
import os
import argparse
import numpy as np
from collections import OrderedDict
import random
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from csvo.data_readers.factory import dataset_factory
from csvo.lietorch import SE3
from csvo.logger import Logger
import torch.nn.functional as F
import time

from csvo.net import VONet
from temp.CSVO.evaluate_Tianmouc import evaluate as validate
import time

def save_checkpoint(epoch, model, optimizer, scheduler, total_steps, filename='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'total_steps': total_steps
    }, filename)


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def image2gray(image):
    image = image.mean(dim=0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def kabsch_umeyama(A, B):
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1)**2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)

    c = VarA / torch.trace(torch.diag(D))
    return c

def train(args):
    """ main training loop """

    # legacy ddp code
    rank = 0
    db = dataset_factory(['tartan'], datapath="datasets/TartanAirNew", n_frames=args.n_frames, ablation=args.ablation, use_pkl = args.use_pkl, training_type=args.training_type)
    train_loader = DataLoader(db, batch_size=1, shuffle=True, num_workers=1)

    net = VONet(ablation=args.ablation, use_pretrained_sd=True)
    if args.pretrained_model != None:
        state_dict = torch.load(args.pretrained_model)
        net.load_state_dict(state_dict)
    net.train()

    net.cuda()

    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        net.load_state_dict(new_state_dict, strict=False)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    if rank == 0:
        logger = Logger(args.name, scheduler)

    total_steps = 0
    i = 0
    count = 0
    report_gap = 10
    stime = time.time()
    loss1 = 0.0
    while total_steps < args.steps:
        for data_blob in train_loader:
            try:
                count += 1

                metainfo = data_blob[-1]
                images, poses, poses_teacher, disps, intrinsics, timeSurfaces0, timeSurfaces1, sdr, sdl = [x.to(rank).float() for x in data_blob[:-1]]

                tmdata_file_name,isTianmouc = metainfo
                optimizer.zero_grad()
                loss1 = 0.0
                so = False
                poses = SE3(poses).inv()
                poses_teacher = SE3(poses_teacher).inv()
                traj1 = net(images, poses, disps, intrinsics, M=1024, STEPS=18, structure_only=so, timeSurface0 = timeSurfaces0, timeSurface1 = timeSurfaces1, sdr = sdr, sdl = sdl)
                for i, (v, x, y, P1, P2, kl) in enumerate(traj1):
                    e = (x - y).norm(dim=-1)
                    e = e.reshape(-1, net.P**2)[(v > 0.5).reshape(-1)].min(dim=-1).values

                    N = P1.shape[1]
                    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
                    ii = ii.reshape(-1).cuda()
                    jj = jj.reshape(-1).cuda()

                    k = ii != jj
                    ii = ii[k]
                    jj = jj[k]

                    P1 = P1.inv()
                    P2 = P2.inv()

                    t1 = P1.matrix()[...,:3,3]
                    t2 = P2.matrix()[...,:3,3]
                    # print(t1[0], t2[0], i)
                    i += 1
                    # try:
                    s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)

                    P1 = P1.scale(s.view(1, 1))
                    # print(P1.data)
                    dP = P1[:,ii].inv() * P1[:,jj]
                    dG = P2[:,ii].inv() * P2[:,jj]
                    # print(dP.data)

                    e1 = (dP * dG.inv()).log()
                    tr = e1[...,0:3].norm(dim=-1)
                    ro = e1[...,3:6].norm(dim=-1)
                    # print(tr.mean(), ro.mean())
                    if not isTianmouc:
                        loss1 += args.flow_weight * e.mean()
                        if not so and i >= 2:
                            loss1 += args.pose_weight * ( tr.mean() + ro.mean() )
                    else:
                        if np.isnan(tr.mean().item()) or np.isnan(ro.mean().item()):
                            print("[train.py] tr or ro has nan value")
                            print(tmdata_file_name)
                            continue
                        if not so and i >= 2:
                            loss1 +=  tr.mean() + ro.mean()
                loss1 += kl


                loss1.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                total_steps += 1
                print("[train.py],total_steps:", total_steps)
                end_time=time.time()
                metrics = {
                    "loss": loss1.item(),
                    "kl": kl.item(),
                    "px1": (e < .25).float().mean().item(),
                    "ro": ro.float().mean().item(),
                    "tr": tr.float().mean().item(),
                    "r1": (ro < .001).float().mean().item(),
                    "r2": (ro < .01).float().mean().item(),
                    "t1": (tr < .001).float().mean().item(),
                    "t2": (tr < .01).float().mean().item(),
                }
                if rank == 0:
                    logger.push(metrics)
                if total_steps % 2000 == 0:
                    torch.cuda.empty_cache()
                    print("Model Saved")
                    if rank == 0:
                        os.makedirs('checkpoints/%s' % (args.name), exist_ok=True)
                        PATH = 'checkpoints/%s/%06d.pth' % (args.name, total_steps)
                        save_checkpoint(epoch=total_steps, model=net, optimizer=optimizer, scheduler=scheduler, total_steps=total_steps, filename=PATH)
                if rank == 0 and total_steps % 2000 == 0:
                    try:
                        validation_results = validate(None, PATH, ablation=args.ablation)
                        print(validation_results)
                        logger.write_dict(validation_results)
                    except Exception as ex:
                        torch.cuda.empty_cache()
                        print("The error is:", ex)
            except Exception as ex:
                print("The error is:", ex)
                torch.cuda.empty_cache()
                time.sleep(1)

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
    import sys
    sys.setrecursionlimit(3000)
    set_random_seed_all()
    torch.cuda.synchronize()
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='experiment', help='name of your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--steps', type=int, default=240000)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_frames', type=int, default=10)
    parser.add_argument('--pose_weight', type=float, default=10.0)
    parser.add_argument('--flow_weight', type=float, default=0.1)
    parser.add_argument("--ablation", default='RGB')
    parser.add_argument("--pretrained_model", default=None)
    parser.add_argument("--use_pkl", type=bool, default=False)
    parser.add_argument('--sde', type=bool, default=False)
    parser.add_argument('--use_pretrained', action='store_true', default=False)
    parser.add_argument('--training_type', type=str, default='None')
    args = parser.parse_args()
    print(args.use_pkl)
    train(args)
