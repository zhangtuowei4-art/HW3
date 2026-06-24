首先打开AUTODL连接SSH终端。
物体A：conda环境1
环境准备：
conda activate gs_env
cd /root/autodl-tmp/hw3_3dgs/gaussian-splatting
选择性：
# 清理可能的历史残留
rm -rf /root/autodl-tmp/hw3_3dgs/data/apple/dense
rm -rf /root/autodl-tmp/hw3_3dgs/data/apple/sparse
rm -f /root/autodl-tmp/hw3_3dgs/data/apple/database.db
特征提取：
QT_QPA_PLATFORM=offscreen colmap feature_extractor \
--database_path /root/autodl-tmp/hw3_3dgs/data/apple/database.db \
--image_path /root/autodl-tmp/hw3_3dgs/data/apple/images \
--ImageReader.single_camera 1 \
--ImageReader.camera_model PINHOLE \
--SiftExtraction.use_gpu 0
特征匹配：
QT_QPA_PLATFORM=offscreen colmap exhaustive_matcher \
--database_path /root/autodl-tmp/hw3_3dgs/data/apple/database.db \
--SiftMatching.use_gpu 0
稀疏重建：
mkdir -p /root/autodl-tmp/hw3_3dgs/data/apple/sparse
QT_QPA_PLATFORM=offscreen colmap mapper \
--database_path /root/autodl-tmp/hw3_3dgs/data/apple/database.db \
--image_path /root/autodl-tmp/hw3_3dgs/data/apple/images \
--output_path /root/autodl-tmp/hw3_3dgs/data/apple/sparse
去畸变：
QT_QPA_PLATFORM=offscreen colmap image_undistorter \
--image_path /root/autodl-tmp/hw3_3dgs/data/apple/images \
--input_path /root/autodl-tmp/hw3_3dgs/data/apple/sparse/0 \
--output_path /root/autodl-tmp/hw3_3dgs/data/apple/dense \
--output_type COLMAP
目录确认：ls /root/autodl-tmp/hw3_3dgs/data/apple/dense/sparse/0/下面必须有images.bin, cameras.bin, points3D.bin三个文件。
运行命令：
python train.py \
-s /root/autodl-tmp/hw3_3dgs/data/apple/dense \
-m /root/autodl-tmp/hw3_3dgs/output/apple \
-r 2 \
--eval \
--wandb_enable \
--test_iterations 1500 3000 4500 6000 7500 9000 10500 12000 13500 15000 16500 18000 19500 21000 22500 24000 25500 27000 28500 30000

物体B：base环境，不需要conda
cd /root/autodl-tmp/threestudio

python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 \
system.prompt_processor.prompt="a vintage ceramic watering can with a spout and handle, painted with colorful floral patterns, old paper label, rusty spots, highly detailed texture, 3d asset" \
system.prompt_processor.negative_prompt="plain, monochrome, solid color, symmetric, transparent, glass, clean, smooth" \
system.prompt_processor.pretrained_model_name_or_path="/root/autodl-tmp/sd-v1-5-complete" \
system.guidance.pretrained_model_name_or_path="/root/autodl-tmp/sd-v1-5-complete" \
trainer.max_steps=10000 \
trainer.val_check_interval=1000 \
system.loggers.wandb.enable=true \
system.loggers.wandb.project="HW3_ObjectB_WateringCan" \
system.loggers.wandb.name="textured_can_10k"
导出：
# 1. 创建新的目录
mkdir -p /root/autodl-tmp/hw3_3dgs/data/watering_can/images

# 2. 抽帧到新目录
ffmpeg -i /root/autodl-tmp/threestudio/outputs/dreamfusion-sd/export_textured_can@20260620-171054/save/it10000-test.mp4 -vf "fps=30" /root/autodl-tmp/hw3_3dgs/data/watering_can/images/frame_%04d.png

#裁剪
python -c "
import cv2, os, shutil
import numpy as np

input_dir = '/root/autodl-tmp/hw3_3dgs/data/watering_can/images/'
temp_dir = '/root/autodl-tmp/hw3_3dgs/data/watering_can/images_temp/'
os.makedirs(temp_dir, exist_ok=True)
count = 0

for img_name in sorted(os.listdir(input_dir)):
    if img_name.endswith('.png'):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # 1. 裁剪左侧 1/3
        cropped_img = img[:, :w//3]
        
        # 2. 将蓝色背景转为纯白底
        # 转换到 HSV 色彩空间
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
        # 定义蓝色的 HSV 范围 (大致是 H: 100-130, S: 50-255, V: 50-255)
        lower_blue = np.array([90, 40, 40])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # 把mask对应的区域设为白色
        cropped_img[blue_mask > 0] = [255, 255, 255]
        
        # 3. 顺便把残留的纯黑噪点也转成白色
        cropped_img[np.where((cropped_img <= [15, 15, 15]).all(axis=2))] = [255, 255, 255]
        
        cv2.imwrite(os.path.join(temp_dir, img_name), cropped_img)
        count += 1

shutil.rmtree(input_dir)
os.rename(temp_dir, input_dir)
print(f'处理完成，共 {count} 张，蓝底已成功转为白底！')
"

# 清理旧残留
rm -f /root/autodl-tmp/hw3_3dgs/data/watering_can/database.db
rm -rf /root/autodl-tmp/hw3_3dgs/data/watering_can/sparse
rm -rf /root/autodl-tmp/hw3_3dgs/data/watering_can/dense

conda activate gs_env

# 1. 特征提取
QT_QPA_PLATFORM=offscreen colmap feature_extractor \
--database_path /root/autodl-tmp/hw3_3dgs/data/watering_can/database.db \
--image_path /root/autodl-tmp/hw3_3dgs/data/watering_can/images \
--ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --SiftExtraction.use_gpu 0

# 2. 特征匹配
QT_QPA_PLATFORM=offscreen colmap exhaustive_matcher \
--database_path /root/autodl-tmp/hw3_3dgs/data/watering_can/database.db --SiftMatching.use_gpu 0

# 3. 稀疏重建
mkdir -p /root/autodl-tmp/hw3_3dgs/data/watering_can/sparse
QT_QPA_PLATFORM=offscreen colmap mapper \
--database_path /root/autodl-tmp/hw3_3dgs/data/watering_can/database.db \
--image_path /root/autodl-tmp/hw3_3dgs/data/watering_can/images \
--output_path /root/autodl-tmp/hw3_3dgs/data/watering_can/sparse

# 4. 去畸变
QT_QPA_PLATFORM=offscreen colmap image_undistorter \
--image_path /root/autodl-tmp/hw3_3dgs/data/watering_can/images \
--input_path /root/autodl-tmp/hw3_3dgs/data/watering_can/sparse/0 \
--output_path /root/autodl-tmp/hw3_3dgs/data/watering_can/dense --output_type COLMAP

conda activate gs_env
cd /root/autodl-tmp/hw3_3dgs/gaussian-splatting

python train.py \
-s /root/autodl-tmp/hw3_3dgs/data/watering_can/dense \
-m /root/autodl-tmp/hw3_3dgs/output/watering_can \
-r 2 \
--eval


物体C：conda环境2
图片预处理：用process.py和remove_bg.py两个文件将jpg文件转为png格式并抠掉原背景转为纯白背景。
训练：
# 1. 激活 conda 环境2
conda activate /root/autodl-tmp/conda_envs/hw3_zero123

# 2. 设置 HuggingFace 镜像（避免下载模型超时）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# 3. 进入工作目录
cd /root/autodl-tmp/threestudio

# 4. 开始训练（注意 project 和 name 已修改为 cup）
python launch.py \
--config configs/zero123.yaml \
--train \
--gpu 0 \
loggers.wandb.enable=true \
loggers.wandb.project="HW3_ObjectC_Peach" \
loggers.wandb.name="peach_10k" \
trainer.max_steps=5000
导出指令：
禁用zero123.yaml中的wandb
python launch.py \
--config configs/zero123.yaml \
--export \
--gpu 0 \
tag="export_peach" \
resume="$CKPT_PATH_C" \
system.exporter_type=mesh-exporter \
data.image_path="./load/images/peach.png" \
loggers.wandb.enable=false \
loggers.wandb.project="" \
loggers.wandb.name=""导出为mesh文件
将你的 model.obj, model.mtl, texture_kd.jpg 放在同一个文件夹下（mesh_c）
在同一目录下用下列代码采样为点云转化为高斯球：
import trimesh
import numpy as np
from plyfile import PlyData, PlyElement
import os

# === 参数配置 ===
input_obj_path = './mesh_c/model.obj'  # 替换为你的OBJ路径
output_ply_path = './object_C.ply'
num_samples = 50000  # 采样点数，5万通常足够单个小物体了，视模型大小可增减

# 1. 加载带纹理的 OBJ
print("正在加载 Mesh...")
mesh = trimesh.load(input_obj_path, force='mesh', process=False)

# 确保纹理颜色被正确解析
if hasattr(mesh.visual, 'to_color'):
    mesh.visual = mesh.visual.to_color()

# 2. 在表面采样点云
print(f"正在采样 {num_samples} 个点...")
points, face_index = trimesh.sample.sample_surface(mesh, num_samples)
points = np.array(points)

# 3. 获取采样点对应的颜色 (RGB 0-1 范围)
face_colors = mesh.visual.face_colors[face_index]
rgb = face_colors[:, :3].astype(np.float32) / 255.0

# 4. 转换为 3DGS 所需的属性
print("正在生成高斯球属性...")
# SH 0阶系数: 3DGS中 f_dc = (color - 0.5) / 0.282095
f_dc = (rgb - 0.5) / 0.282095

# Opacity (不透明度): 初始化为 logit(0.9) ≈ 2.197，使其几乎不透明
opacity = np.ones((num_samples, 1)) * 2.1972

# Scale (尺度): 根据模型包围盒大小，给一个较小的初始高斯球半径
scale_val = np.log(mesh.bounding_box.extents.max() / 200.0)  # 取对数，保证为负数，半径较小
scales = np.ones((num_samples, 3)) * scale_val

# Rotation (旋转): 初始化为单位四元数 (w=1, x=0, y=0, z=0)
rotations = np.zeros((num_samples, 4))
rotations[:, 0] = 1.0

# 5. 组装并保存为 PLY
print("正在保存 PLY 文件...")
elements = np.empty(num_samples, dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ('opacity', 'f4'),
    ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
    ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
])

elements['x'] = points[:, 0]
elements['y'] = points[:, 1]
elements['z'] = points[:, 2]
elements['f_dc_0'] = f_dc[:, 0]
elements['f_dc_1'] = f_dc[:, 1]
elements['f_dc_2'] = f_dc[:, 2]
elements['opacity'] = opacity[:, 0]
elements['scale_0'] = scales[:, 0]
elements['scale_1'] = scales[:, 1]
elements['scale_2'] = scales[:, 2]
elements['rot_0'] = rotations[:, 0]
elements['rot_1'] = rotations[:, 1]
elements['rot_2'] = rotations[:, 2]
elements['rot_3'] = rotations[:, 3]

el = PlyElement.describe(elements, 'vertex')
PlyData([el], text=False).write(output_ply_path)
print(f"成功！文件已保存至: {output_ply_path}")

背景counter:conda环境1
conda环境：conda activate gs_env
目录：cd /root/autodl-tmp/hw3_3dgs/gaussian-splatting
训练命令：
python train.py \
-s /root/autodl-tmp/hw3_3dgs/data/counter \
-m /root/autodl-tmp/hw3_3dgs/output/counter \
-r 2 \
--eval \
--wandb_enable \
--test_iterations 1500 3000 4500 6000 7500 9000 10500 12000 13500 15000 16500 18000 19500 21000 22500 24000 25500 27000 28500 30000
可视化见https://superspl.at/editor。

四个ply文件全部导出来之后在supersplat中手动融合，因为坐标难以目测不方便用代码融合，最终文件是A+B+C+counter.ply。
融合采用3DGS官方库里面的render.py脚本，反向渲染出train和test图片，将train的210张图片提出来，把 counter 训练时的 cfg_args 拷贝过来，模拟原相机路径，使用ffmpeg合成最终视频：
ffmpeg -framerate 30 -pattern_type glob -i '*.png' -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p -crf 18 /root/autodl-tmp/hw3_3dgs/final_roaming_video.mp4。
