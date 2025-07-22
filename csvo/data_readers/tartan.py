
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
import json
from ..lietorch import SE3
from .base import RGBDDataset
from .augmentation import RGBDAugmentor, RGBSD_Augmentor, ExposureJitter
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import time
def modify_list(lst):
    # 将第0个数修改为最接近的5的倍数
    lst[0] = round(lst[0] / 5) * 5

    # 修改第i个数为 (a[i] - a[0]) * 5 + a[0]
    for i in range(1, len(lst)):
        lst[i] = (lst[i] - lst[0]) * 5 + lst[0]

    return lst
def rotate_pose_90_degrees_z(poses):
    """
    将输入的 pose 信息沿 z 轴旋转 90 度。

    参数:
    poses (np.array): 输入的 pose 信息，shape 为 (n, 7)，其中 n 是 pose 的数量，
                      每一行包含 [x, y, z, qx, qy, qz, qw]。

    返回:
    np.array: 旋转后的 pose 信息，shape 为 (n, 7)。
    """
    # 提取位置和四元数部分
    positions = poses[:, :3]  # 前三列是位置 (x, y, z)
    quaternions = poses[:, 3:]  # 后四列是四元数 (qx, qy, qz, qw)

    # 定义 90 度绕 z 轴的旋转矩阵
    rotation_matrix = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])

    # 对位置部分进行旋转
    rotated_positions = np.dot(positions, rotation_matrix.T)

    # 对四元数部分进行旋转
    # 创建一个绕 z 轴旋转 90 度的四元数
    z_rotation_quat = R.from_euler('z', 90, degrees=True).as_quat()  # [qx, qy, qz, qw]
    z_rotation_quat = z_rotation_quat[[3, 0, 1, 2]]  # 转换为 [qw, qx, qy, qz] 格式

    # 将每个四元数与绕 z 轴旋转的四元数相乘
    rotated_quaternions = []
    for quat in quaternions:
        # 将四元数转换为 scipy 的 Rotation 对象
        r = R.from_quat(quat[[1, 2, 3, 0]])  # 输入格式为 [qx, qy, qz, qw]
        # 应用绕 z 轴的旋转
        r_rotated = r * R.from_quat(z_rotation_quat)
        # 转换回 [qx, qy, qz, qw] 格式
        rotated_quat = r_rotated.as_quat()[[3, 0, 1, 2]]  # 转换为 [qx, qy, qz, qw]
        rotated_quaternions.append(rotated_quat)

    rotated_quaternions = np.array(rotated_quaternions)

    # 合并旋转后的位置和四元数
    rotated_poses = np.hstack((rotated_positions, rotated_quaternions))

    return rotated_poses


# cur_path = osp.dirname(osp.abspath(__file__))
# test_split = osp.join(cur_path, 'tartan_test.txt')
# test_split = open(test_split).read().split()


# test_split = [
#     "abandonedfactory_night/abandonedfactory_night/Easy/P013",
#     "amusement/amusement/Easy/P008",
#     "seasidetown/seasidetown/Easy/P009",
#     "seasonsforest/seasonsforest/Easy/P011"
# ]
def normalize_tensor(tensor):
    # 计算均值和标准差
    normalized_tensor = tensor/2
    
    return normalized_tensor
def normalize_zero_one(tensor):
    # 计算均值和标准差
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    
    # 归一化张量
    normalized_tensor = (tensor - mean) / std
    
    return normalized_tensor
def SD2XY(sd_raw:torch.tensor) -> torch.tensor:
    '''
    input: [h,w,2]/[2,h,w]/[n,2,h,w]
    output: [h,2*w],[h,2*w] or [n,h,2*w],[n,h,2*w]
    坐标变换规则参照http://www.tianmouc.cn:40000/tianmoucv/introduction.html
    '''
    if len(sd_raw.shape) == 3:
        assert (sd_raw.shape[2]==2 or sd_raw.shape[0]==2)
        if sd_raw.shape[2] == 2:
            sd = sd_raw.permute(2,0,1).unsqueeze(0) #[h,w,c]->[1,c,h,w]
        else:
            sd = sd_raw.unsqueeze(0)
    else:
        assert (len(sd_raw.shape) == 4 and sd_raw.shape[1]==2)
        sd = sd_raw
        
    b,c,h,w = sd.shape
    sdul = sd[:,0:1,0::2,...]
    sdll = sd[:,0:1,1::2,...]
    sdur = sd[:,1:2,0::2,...]
    sdlr = sd[:,1:2,1::2,...]

    target_size = (h,w*2)
    sdul = F.interpolate(sdul, size=target_size, mode='bilinear', align_corners=False)
    sdll = F.interpolate(sdll, size=target_size, mode='bilinear', align_corners=False)
    sdur = F.interpolate(sdur, size=target_size, mode='bilinear', align_corners=False)
    sdlr = F.interpolate(sdlr, size=target_size, mode='bilinear', align_corners=False)

    sdx = ((sdul + sdll)/1.414 - (sdur + sdlr)/1.414)/2
    sdy = ((sdur - sdlr)/1.414 + (sdul - sdll)/1.414)/2

    if len(sd_raw.shape) == 3:
        return sdx.squeeze(0).squeeze(0), sdy.squeeze(0).squeeze(0)
    else:
        return sdx.squeeze(1), sdy.squeeze(1)

test_split = [
    "abandonedfactory/Easy/P011",
    "abandonedfactory/Hard/P011",
    "abandonedfactory_night/Easy/P013",
    "abandonedfactory_night/Hard/P014",
    "amusement/Easy/P008",
    "amusement/Hard/P007",
    "carwelding/Easy/P007",
    "endofworld/Easy/P009",
    "gascola/Easy/P008",
    "gascola/Hard/P009",
    "hospital/Easy/P036",
    "hospital/Hard/P049",
    "japanesealley/Easy/P007",
    "japanesealley/Hard/P005",
    "neighborhood/Easy/P021",
    "neighborhood/Hard/P017",
    "ocean/Easy/P013",
    "ocean/Hard/P009",
    "office2/Easy/P011",
    "office2/Hard/P010",
    "office/Hard/P007",
    "oldtown/Easy/P007",
    "oldtown/Hard/P008",
    "seasidetown/Easy/P009",
    "seasonsforest/Easy/P011",
    "seasonsforest/Hard/P006",
    "seasonsforest_winter/Easy/P009",
    "seasonsforest_winter/Hard/P018",
    "soulcity/Easy/P012",
    "soulcity/Hard/P009",
    "westerndesert/Easy/P013",
    "westerndesert/Hard/P007",
]

class TartanAir(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 100

    def __init__(self, mode='training', ablation = 'RGB',use_pkl=False, training_type=None, **kwargs):
        self.aug = RGBDAugmentor(crop_size=[480,640])
        self.aug_rgb_sd = RGBSD_Augmentor(crop_size=[480,640])
        self.exposure_jitter = ExposureJitter(min_exposure=0.5, max_exposure=1.5)
        self.mode = mode
        self.n_frames = 2
        self.ablation = ablation
        self.use_pkl = use_pkl
        self.training_type = training_type
        super(TartanAir, self).__init__(name='TartanAir', **kwargs)

    @staticmethod 
    def is_test_scene(scene):
        return any(x in scene for x in test_split)

    def build_dataset(self):
        import pickle
        if self.training_type == 'tianmouc' or self.training_type == 'tianmouc_org' or self.training_type == 'tianmouc_flip':
            pkl_path = '/data/zzx/DPVO_E2E/datasets/tianmouc_split_high.pickle'
        elif self.training_type == 'mix_flip' and self.ablation == 'plana':
            pkl_path = '/data/zzx/DPVO_E2E/datasets/mix_high_split.pickle'
        elif self.training_type == 'mix_flip' and self.ablation == 'plana_td':
            pkl_path = '/data/zzx/DPVO_E2E/datasets/mix_high_split_td.pickle'
        elif self.training_type == 'tianmouc_dpvo':
            pkl_path = '/data/zzx/DPVO_E2E/datasets/tianmouc_split_high_dpvo.pickle'
        elif self.training_type == 'mix':
            pkl_path = '/data/zzx/DPVO_E2E/datasets/mix_split_low.pickle'
        elif self.training_type == 'tartan':
            pkl_path = '/data/zzx/DPVO_E2E/datasets/tartan_new.pickle'
        elif self.training_type == 'tianmouc_new':
            pkl_path = '/data/zzx/DPVO_E2E/datasets/tianmouc_split_high_new.pickle'
        else:
            raise RuntimeError("training type not permitted")
        if self.use_pkl:
            print("Using pickle")
            return pickle.load(open(pkl_path, 'rb'))
        print(pkl_path)
        from tqdm import tqdm
        print("Building TartanAir dataset")
        scene_info = {} 
        scenes = glob.glob(osp.join(self.root, '*/*/*'))
        for scene in tqdm(sorted(scenes, reverse=True)):
            if self.ablation.lower() == 'planb' or self.ablation.lower() == 'plana' or self.ablation.lower() == 'dpvo':
                
                if not "tianmouc" in scene:
                    if 'tianmouc' in self.training_type:
                        continue
                    print(scene)
                    sdr = sorted(glob.glob(osp.join(scene, 'SDR_frames_low_rate/*.npy')))
                    sdl = sorted(glob.glob(osp.join(scene, 'SDL_frames_low_rate/*.npy')))
                    depths = sorted(glob.glob(osp.join(scene, 'depth_left/*.npy')))
                    intrinsics = [TartanAir.calib_read()] * len(sdr)
                    poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
                else:
                    if self.training_type == 'tartan':
                        continue
                    if 'splited_dataset' not in scene:
                        continue
                    if 'train' not in scene:
                        continue
                    if 'td' in scene:
                        continue
                    print(scene)
                    print("Building tianmouc dataset")
                    sdr = sorted(glob.glob(osp.join(scene, 'SD_frames_aligned_high/*.npy')))
                    sdl = sorted(glob.glob(osp.join(scene, 'SD_frames_aligned_high/*.npy')))
                    depths = ['/data/zzx/DPVO_E2E/datasets/TartanAirNew/abandonedfactory/Hard/P009/depth_left/000001_left_depth.npy' for _ in range(len(sdr))]
                    intrinsics = [np.array([707.8457,708.3163,389.9121,235.1899])] * len(sdr)
                    poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
                
                poses_teacher = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
            elif self.ablation.lower() == 'plana_td':
                if not "tianmouc" in scene:
                    if 'tianmouc' in self.training_type:
                        continue
                    # continue
                    print(scene)
                    sdr = sorted(glob.glob(osp.join(scene, 'TD_frames_low_rate/*.npy')))
                    sdl = sorted(glob.glob(osp.join(scene, 'TD_frames_low_rate/*.npy')))
                    depths = sorted(glob.glob(osp.join(scene, 'depth_left/*.npy')))
                    intrinsics = [TartanAir.calib_read()] * len(sdr)
                    poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
                else:
                    if self.training_type == 'tartan':
                        continue
                    if 'splited_dataset_2025' not in scene:
                        continue
                    if 'train' not in scene:
                        continue
                    print(scene)
                    print("Building tianmouc dataset")
                    sdr = sorted(glob.glob(osp.join(scene, 'TD_frames_aligned_high/*.npy')))
                    sdl = sorted(glob.glob(osp.join(scene, 'TD_frames_aligned_high/*.npy')))
                    depths = ['/data/zzx/DPVO_E2E/datasets/TartanAirNew/abandonedfactory/Hard/P009/depth_left/000001_left_depth.npy' for _ in range(len(sdr))]
                    intrinsics = [np.array([707.8457,708.3163,389.9121,235.1899])] * len(sdr)
                    poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
                
                poses_teacher = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')

            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            poses_teacher = poses_teacher[:, [1, 2, 0, 4, 5, 3, 6]]
            poses[:,:3] /= TartanAir.DEPTH_SCALE
            poses_teacher[:,:3] /= TartanAir.DEPTH_SCALE
            print('[tartan.py]',len(poses), len(depths), len(intrinsics), scene)
            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'sdr': sdr, 'sdl': sdl, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph, 'poses_teacher': poses_teacher}
        with open(pkl_path, 'wb') as file:
            pickle.dump(scene_info, file)
        return scene_info

    def process_file_path(self,file_path):
        # 提取文件名
        file_name = os.path.basename(file_path)
        
        # 提取六位数
        number_str = file_name.split('.')[0]
        number = int(number_str)
        
        # 计算10倍和11倍
        number_10x =int(number/10)
        number_11x = number_10x+1
        
        # 获取目录路径
        dir_path = os.path.dirname(file_path)
        
        # 构建新的文件路径
        file_path_10x = os.path.join(dir_path, f"{number_10x:06d}.png")
        file_path_11x = os.path.join(dir_path, f"{number_11x:06d}.png")
        
        return file_path_10x.replace("SD_frames", "image_left"), file_path_11x.replace("SD_frames", "image_left")
    def __getitem__(self, index):
        """ return training video """
        
        index = index % len(self.dataset_index)
        scene_id, ix = self.dataset_index[index]
        frame_graph = self.scene_info[scene_id]['graph']
        sdl_list = self.scene_info[scene_id]['sdl']

        tmdata_file_name = sdl_list[0]

        sdr_list = self.scene_info[scene_id]['sdr']
        depths_list = self.scene_info[scene_id]['depths']
        poses_list = self.scene_info[scene_id]['poses']
        poses_teacher_list = self.scene_info[scene_id]['poses_teacher']
        intrinsics_list = self.scene_info[scene_id]['intrinsics']
        if 'tianmouc' in sdl_list[0]:
            isTianmouc = True
        else:
            isTianmouc = False


        d = np.random.uniform(self.fmin, self.fmax)
        s = 1

        inds = [ ix ]
        while len(inds) < self.n_frames:
            # get other frames within flow threshold

            if self.sample:
                try:
                    k = (frame_graph[ix][1] > self.fmin) & (frame_graph[ix][1] < self.fmax)
                except:
                    print(frame_graph.keys())
                frames = frame_graph[ix][0][k]

                # prefer frames forward in time
                if np.count_nonzero(frames[frames > ix]):
                    ix = np.random.choice(frames[frames > ix])

                elif ix + 1 < len(sdr_list):
                    ix = ix + 1

                elif np.count_nonzero(frames):
                    ix = np.random.choice(frames)

            else:
                i = frame_graph[ix][0].copy()
                g = frame_graph[ix][1].copy()

                g[g > d] = -1
                if s > 0:
                    g[i <= ix] = -1
                else:
                    g[i >= ix] = -1

                if len(g) > 0 and np.max(g) > 0:
                    ix = i[np.argmax(g)]
                else:
                    if ix + s >= len(sdr_list) or ix + s < 0:
                        s *= -1

                    ix = ix + s
            
            inds += [ ix ]
        images, sdr, sdl, depths, poses, intrinsics, events, poses_teacher = [], [], [], [], [], [], [], []
        timeSurface0, timeSurface1 = [], []
        start_time = time.time()

        if self.ablation == 'dpvo' and self.training_type=='tianmouc_org':
            inds = modify_list(inds)
            current_max = max(inds)
    
            if current_max >= len(sdr_list):
                k = (current_max - len(sdr_list)) // 5 + 1
            else:
                k = 0  # 如果当前最大值已经小于 max_value，不需要调整
            
            # 将所有值减去 5 * k
            inds = [x - 5 * k for x in inds]
    

        for i in inds:
            if not isTianmouc:
                sdr.append(np.load(sdr_list[i]))
                sdl.append(np.load(sdl_list[i]))
                if self.ablation.lower() == 'planb' or self.ablation.lower() == 'sd_only' or self.ablation.lower() == 'dpvo':
                    interpolatedImage = self.__class__.image_read(sdr_list[i].replace("SDR_frames_low_rate", "image_left").replace(".npy", "_left.png"))
                elif self.ablation.lower() == 'plana':
                    interpolatedImage = self.__class__.image_read(sdr_list[i].replace("SDR_frames_low_rate", "image_left").replace(".npy", "_left.png"))
                elif self.ablation.lower() == 'plana_td':
                    interpolatedImage = self.__class__.image_read(sdr_list[i].replace("TD_frames_low_rate", "image_left").replace(".npy", "_left.png"))

                images.append(interpolatedImage)
            else:
                if self.training_type=='tianmouc_org':
                    
                    lx = torch.from_numpy(np.load(sdr_list[i])[:,:,0])
                    ly = torch.from_numpy(np.load(sdr_list[i])[:,:,1])
                    lx /= 2
                    ly /= 2
                elif self.training_type == 'tianmouc_flip' or self.training_type == 'mix_flip':
                    if self.ablation != 'plana_td':
                        matrix = np.load(sdr_list[i])
                        lx = torch.from_numpy(matrix[:,:,0])
                        ly = torch.from_numpy(matrix[:,:,1])
                    elif self.ablation == 'plana_td':
                        matrix = np.load(sdr_list[i])
                        lx = torch.from_numpy(matrix)
                        ly = torch.from_numpy(matrix)
                    lx *= 5
                    ly *= 5
                    lx = torch.flip(lx, dims=[1])
                    ly = torch.flip(ly, dims=[1])
                    if self.ablation != 'plana_td':
                        temp = lx
                        lx = ly
                        ly = temp
                else:
                    lx, ly = SD2XY(torch.stack([torch.from_numpy(np.load(sdr_list[i])[:,:,0]), torch.from_numpy(np.load(sdr_list[i])[:,:,1])], dim=2))
                    lx = torch.flip(lx, dims=[1])
                    ly = torch.flip(ly, dims=[1])
                    lx *=-1
                    end_time = time.time()
                    lx = normalize_tensor(lx)
                    ly = normalize_tensor(ly)
                
                sdr.append(lx)
                sdl.append(ly)

                if self.ablation.lower() == 'planb' or self.ablation.lower() == 'sd_only' or self.ablation.lower() == 'dpvo':
                    interpolatedImage = self.__class__.image_read(sdr_list[i].replace("SD_frames_aligned_high", "image_left_aligned_high").replace(".npy", ".png"))
                elif self.ablation.lower() == 'plana':
                    interpolatedImage = self.__class__.image_read(sdr_list[i].replace("SD_frames_aligned_high", "image_left_aligned_high").replace(".npy", ".png"))
                elif self.ablation.lower() == 'plana_td':
                    interpolatedImage = self.__class__.image_read(sdr_list[i].replace("TD_frames_aligned_high", "image_left_aligned_high").replace(".npy", ".png"))
                else:
                    raise RuntimeError("Ablation not permitted")
                if self.training_type == 'tianmouc_org':
                    images.append(interpolatedImage)
                elif self.training_type == 'tianmouc_flip':
                    images.append(np.flip(interpolatedImage, axis=1))
                elif self.training_type == 'mix_flip':
                    images.append(np.flip(interpolatedImage, axis=1))
            if isTianmouc:
                depths.append(self.__class__.depth_read(depths_list[0]))
            else:
                depths.append(self.__class__.depth_read(depths_list[i]))
            poses.append(poses_list[i])
            poses_teacher.append(poses_teacher_list[i])

            intrinsics.append(intrinsics_list[i])
            index = int(sdr_list[i].split("/")[-1].split(".")[0])

        images = np.stack(images).astype(np.float32)
        sdr = np.stack(sdr).astype(np.float32)
        sdl = np.stack(sdl).astype(np.float32)
        depths = np.stack(depths).astype(np.float32)
        poses = np.stack(poses).astype(np.float32)
        poses /= 5

        poses_teacher = np.stack(poses_teacher).astype(np.float32)
        intrinsics = np.stack(intrinsics).astype(np.float32)

        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)
        sdr = torch.from_numpy(sdr).float()
        sdr = sdr.unsqueeze(1)
        sdl = torch.from_numpy(sdl).float()
        sdl = sdl.unsqueeze(1)

        disps = torch.from_numpy(1.0 / (depths))
        poses = torch.from_numpy(poses)
        poses_teacher = torch.from_numpy(poses_teacher)
        intrinsics = torch.from_numpy(intrinsics)
        
        if self.aug:
            images,sdr, sdl, disps, intrinsics = \
                self.aug_rgb_sd(images,sdr, sdl, disps, intrinsics)
            images = self.exposure_jitter(images)

        s = .7 * torch.quantile(disps, .98)
        disps = disps / s
        
        poses[...,:3] *= s
        poses_teacher[...,:3] *= s

        metainfo = [tmdata_file_name,isTianmouc]

        end_time = time.time()
        elapsed_time = end_time - start_time
        return images, poses, poses_teacher, disps, intrinsics, torch.tensor(timeSurface0), torch.tensor(timeSurface1), sdr, sdl, metainfo
    
    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / TartanAir.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        return depth