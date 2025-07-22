import os
import shutil

def copy_files(source_dir):
    # 遍历源目录
    for root, dirs, files in os.walk(source_dir):
        for dir_name in dirs:
            # 判断文件夹名是否是两位整数
            if len(dir_name) == 3 and dir_name.startswith("P"):
                # 获取两位整数
                short_name = dir_name[1:]
                # 构建对应的三位整数文件夹名
                long_name = "P" + "0" * (3 - len(short_name)) + short_name
                # 构建完整路径
                short_dir_path = os.path.join(root, dir_name)
                long_dir_path = os.path.join(root, long_name)
                # 如果三位整数文件夹不存在，则创建
                if not os.path.exists(long_dir_path):
                    os.makedirs(long_dir_path)
                # 复制文件
                for file_name in os.listdir(short_dir_path):
                    file_path = os.path.join(short_dir_path, file_name)
                    shutil.copytree(file_path, long_dir_path+"/SD_frames")

# 指定源目录
source_directory = "/data_4T/zzx/DPVO/datasets/TartanAir/seasidetown/seasidetown/Easy"
# 调用函数
copy_files(source_directory)
