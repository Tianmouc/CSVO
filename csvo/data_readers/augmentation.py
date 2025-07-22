import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F


class RGBDAugmentor:
    """ perform augmentation on RGB-D video """

    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.augcolor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2/3.14),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomInvert(p=0.1),
            transforms.ToTensor()])

        self.max_scale = 0.5

    def spatial_transform(self, images, depths, poses, intrinsics):
        """ cropping and resizing """
        ht, wd = images.shape[2:]

        max_scale = self.max_scale
        min_scale = np.log2(np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd)))

        scale = 1
        if np.random.rand() < 0.8:
            scale = 2 ** np.random.uniform(0.0, max_scale)

        intrinsics = scale * intrinsics

        ht1 = int(scale * ht)
        wd1 = int(scale * wd)

        depths = depths.unsqueeze(dim=1)

        images = F.interpolate(images, (ht1, wd1), mode='bicubic', align_corners=False)
        depths = F.interpolate(depths, (ht1, wd1), recompute_scale_factor=False)

        # always perform center crop (TODO: try non-center crops)
        y0 = (images.shape[2] - self.crop_size[0]) // 2
        x0 = (images.shape[3] - self.crop_size[1]) // 2

        intrinsics = intrinsics - torch.tensor([0.0, 0.0, x0, y0])
        images = images[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depths = depths[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        depths = depths.squeeze(dim=1)
        return images, poses, depths, intrinsics

    def color_transform(self, images):
        """ color jittering """
        num, ch, ht, wd = images.shape
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd*num)
        images = 255 * self.augcolor(images[[2,1,0]] / 255.0)
        return images[[2,1,0]].reshape(ch, ht, wd, num).permute(3,0,1,2).contiguous()

    def __call__(self, images, poses, depths, intrinsics):
        if np.random.rand() < 0.5:
            images = self.color_transform(images)

        return self.spatial_transform(images, depths, poses, intrinsics)

class RGBSD_Augmentor:
    """Perform augmentation on RGB and SD frames."""

    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.augcolor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2 / 3.14),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomInvert(p=0.1),
            transforms.ToTensor()
        ])
        self.max_scale = 0.5

    def spatial_transform(self, images, sdr_frames, sdl_frames, depths, intrinsics):
        """Cropping and resizing."""
        ht, wd = images.shape[2:]

        max_scale = self.max_scale
        min_scale = np.log2(np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd)))

        scale = 1
        if np.random.rand() < 0.8:
            scale = 2 ** np.random.uniform(0.0, max_scale)

        intrinsics = scale * intrinsics

        ht1 = int(scale * ht)
        wd1 = int(scale * wd)

        depths = depths.unsqueeze(dim=1)
        images = F.interpolate(images, (ht1, wd1), mode='bicubic', align_corners=False)
        sdr_frames = F.interpolate(sdr_frames, (ht1, wd1), mode='bicubic', align_corners=False)
        sdl_frames = F.interpolate(sdl_frames, (ht1, wd1), mode='bicubic', align_corners=False)
        depths = F.interpolate(depths, (ht1, wd1), recompute_scale_factor=False)

        # Always perform center crop
        y0 = (images.shape[2] - self.crop_size[0]) // 2
        x0 = (images.shape[3] - self.crop_size[1]) // 2

        intrinsics = intrinsics - torch.tensor([0.0, 0.0, x0, y0])
        images = images[:, :, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        sdr_frames = sdr_frames[:, :, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        sdl_frames = sdl_frames[:, :, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        depths = depths[:, :, y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        depths = depths.squeeze(dim=1)
        return images, sdr_frames, sdl_frames, depths, intrinsics

    def color_transform(self, images):
        """ color jittering """
        num, ch, ht, wd = images.shape
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd*num)
        images = 255 * self.augcolor(images[[2,1,0]] / 255.0)
        return images[[2,1,0]].reshape(ch, ht, wd, num).permute(3,0,1,2).contiguous()

    def __call__(self, images, sdr_frames, sdl_frames, depths, intrinsics):
        # Split SD frames into SDR and SDL
        # sdr_frames, sdl_frames = sd_frames.chunk(2, dim=1)

        # Apply color transform independently to SDR and SDL
        if np.random.rand() < 0.5:
            images = self.color_transform(images)

        # Perform spatial transform (applied to both SDR and SDL)
        images, sdr_frames, sdl_frames, depths, intrinsics = self.spatial_transform(
            images, sdr_frames, sdl_frames, depths, intrinsics
        )

        return images, sdr_frames, sdl_frames, depths, intrinsics
class ExposureJitter:
    def __init__(self, min_exposure=0.5, max_exposure=1.5):
        """
        初始化曝光抖动参数
        :param min_exposure: 最低曝光比例 (欠曝光，默认 0.5)
        :param max_exposure: 最高曝光比例 (过曝光，默认 1.5)
        """
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure

    def __call__(self, images):
        """
        对输入RGB图像进行曝光时间抖动
        :param images: (num, ch, ht, wd) 的张量，RGB格式
        :return: 曝光抖动后的图像
        """
        if images.ndim != 4 or images.size(1) != 3:
            raise ValueError("输入图像必须为 (num, 3, ht, wd) 的张量，且通道数为 3 (RGB)。")

        # 生成随机曝光比例，范围 [min_exposure, max_exposure]
        batch_size = images.size(0)
        exposure_factors = torch.empty(batch_size).uniform_(self.min_exposure, self.max_exposure)

        # 将曝光比例扩展到与图像维度匹配
        exposure_factors = exposure_factors.view(-1, 1, 1, 1)

        # 应用曝光时间抖动（clip 确保像素值在 [0, 255]）
        images = images * exposure_factors
        images = torch.clamp(images, 0, 255)

        return images
