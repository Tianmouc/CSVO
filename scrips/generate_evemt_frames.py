import cv2
import os

# 路径设置
image_directory = '/data_4T/zzx/DPVO/datasets/TartanAir/abandonedfactory_night/abandonedfactory_night/Easy/P001/image_left'  # 替换为您的图片文件夹路径
video_path = '/data_4T/zzx/v2e/output/abandoned_factory_night/dvs-video.avi'  # 替换为您的视频文件路径

# 统计文件目录中图片的个数
def count_images(directory):
    count = 0
    for file in os.listdir(directory):
        if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # 支持的图片格式
            count += 1
    return count

# 将视频分为x帧
def split_video(video_path, frames_count):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(1, total_frames // frames_count)

    # 创建输出目录
    output_directory = 'extracted_frames'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    frame_idx = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 如果没有帧了，就结束循环

        if frame_idx % skip_frames == 0:
            cv2.imwrite(f'{output_directory}/frame_{extracted_count:04d}.jpg', frame)
            extracted_count += 1
            if extracted_count >= frames_count:
                break  # 如果达到需要的帧数，就结束循环

        frame_idx += 1

    cap.release()

# 主逻辑
if __name__ == '__main__':
    num_images = count_images(image_directory)
    print(f'Found {num_images} images in directory.')

    if num_images > 0:
        split_video(video_path, num_images)
        print(f'Successfully extracted {num_images} frames.')
    else:
        print('No images found in the directory.')
