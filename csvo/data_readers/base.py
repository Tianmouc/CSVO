import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import glob

import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp

from .augmentation import RGBDAugmentor, RGBSD_Augmentor, ExposureJitter
from .rgbd_utils import *
class RGBDDataset(data.Dataset):
    def __init__(self, name, datapath, n_frames=4, crop_size=[480,640], fmin=10.0, fmax=75.0, aug=True, sample=True):
        """ Base class for RGBD dataset """
        self.aug = None
        self.root = datapath
        self.name = name
        self.aug = aug
        self.sample = sample

        self.n_frames = n_frames
        self.fmin = fmin # exclude very easy examples
        self.fmax = fmax # exclude very hard examples
        
        self.aug = RGBDAugmentor(crop_size=crop_size)
        self.aug_rgb_sd = RGBSD_Augmentor(crop_size=crop_size)
        self.exposure_jitter = ExposureJitter(min_exposure=0.5, max_exposure=1.5)

        # building dataset is expensive, cache so only needs to be performed once
        cur_path = osp.dirname(osp.abspath(__file__))
        if not os.path.isdir(osp.join(cur_path, 'cache')):
            os.mkdir(osp.join(cur_path, 'cache'))

        self.scene_info = self.build_dataset()
        self._build_dataset_index()

    
        
    def _build_dataset_index(self):
        self.dataset_index = []
        for scene in self.scene_info.keys():
            print(scene)
            if not self.__class__.is_test_scene(scene):
                graph = self.scene_info[scene]['graph']
                print(len(graph), scene)
                for i in graph:
                    if i < len(graph) - 65:
                        self.dataset_index.append((scene, i))
            else:
                print("Reserving {} for validation".format(scene))

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        return np.load(depth_file)

    def build_dataset(self):
        pass   
    def build_frame_graph(self, poses, depths, intrinsics, f=16, max_flow=256):
        """ compute optical flow distance between all pairs of frames """
        def read_disp(fn):
            depth = self.__class__.depth_read(fn)[f//2::f, f//2::f]
            depth[depth < 0.01] = np.mean(depth)
            return 1.0 / depth

        poses = np.array(poses)
        intrinsics = np.array(intrinsics) / f
        disps = np.stack(list(map(read_disp, depths)), 0)
        d = f * compute_distance_matrix_flow(poses, disps, intrinsics)

        graph = {}
        for i in range(d.shape[0]):
            j, = np.where(d[i] < max_flow)
            graph[i] = (j, d[i,j])

        return graph
    def get_rgb(self, sd_path):
        sdFrame = ""
        for dic in self.map:
            if sd_path in dic["sdFrames"]:
                return dic["lastRGB"],dic["nextRGB"]
        print("SD does not exist!")
        raise("Timeout")
    

    def __len__(self):
        return len(self.dataset_index)

    def __imul__(self, x):
        self.dataset_index *= x
        return self
