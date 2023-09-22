import os
import shutil

# 源文件夹路径（A文件夹）
source_folder = "/home/cxf/cxm/yoloant/Yolo-Ant/dataset/label/trains"

# 目标文件夹路径（B文件夹）
destination_folder = "/home/cxf/cxm/yoloant/Yolo-Ant/dataset/labels/trains"

# 确保目标文件夹存在，如果不存在则创建它
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    source_file_path = os.path.join(source_folder, filename)

    # 检查文件名中是否包含下划线
    if "_" not in filename:
        destination_file_path = os.path.join(destination_folder, filename)

        # 将不带下划线的文件复制到目标文件夹中
        shutil.copy(source_file_path, destination_file_path)
        print(f"Copied: {filename} to {destination_file_path}")

print("Extraction complete.")
