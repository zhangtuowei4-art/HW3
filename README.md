 📌 3D Gaussian Splatting + ThreeStudio 实验流程说明

本项目包含三个物体（A / B / C）以及背景 counter 的完整重建流程，涵盖：

* COLMAP SfM 重建
* Gaussian Splatting 训练
* ThreeStudio 文生图/3D生成
* Zero123 mesh 转点云
* 最终多模型融合与渲染

---

# 🧭 1. 环境准备

首先打开 AUTODL 并连接 SSH 终端，环境配置见environment.txt

---

# 🧱 2. 物体 A（Gaussian Splatting + COLMAP）

## 📦 Conda 环境

```bash
conda activate gs_env
cd /root/autodl-tmp/hw3_3dgs/gaussian-splatting
```

---

## 🧹 清理旧数据

```bash
rm -rf /root/autodl-tmp/hw3_3dgs/data/apple/dense
rm -rf /root/autodl-tmp/hw3_3dgs/data/apple/sparse
rm -f /root/autodl-tmp/hw3_3dgs/data/apple/database.db
```

---

## 🔍 COLMAP 特征提取

```bash
QT_QPA_PLATFORM=offscreen colmap feature_extractor \
--database_path /root/autodl-tmp/hw3_3dgs/data/apple/database.db \
--image_path /root/autodl-tmp/hw3_3dgs/data/apple/images \
--ImageReader.single_camera 1 \
--ImageReader.camera_model PINHOLE \
--SiftExtraction.use_gpu 0
```

---

## 🔗 特征匹配

```bash
QT_QPA_PLATFORM=offscreen colmap exhaustive_matcher \
--database_path /root/autodl-tmp/hw3_3dgs/data/apple/database.db \
--SiftMatching.use_gpu 0
```

---

## 🏗 稀疏重建

```bash
mkdir -p /root/autodl-tmp/hw3_3dgs/data/apple/sparse

QT_QPA_PLATFORM=offscreen colmap mapper \
--database_path /root/autodl-tmp/hw3_3dgs/data/apple/database.db \
--image_path /root/autodl-tmp/hw3_3dgs/data/apple/images \
--output_path /root/autodl-tmp/hw3_3dgs/data/apple/sparse
```

---

## 🧼 去畸变（生成 Dense）

```bash
QT_QPA_PLATFORM=offscreen colmap image_undistorter \
--image_path /root/autodl-tmp/hw3_3dgs/data/apple/images \
--input_path /root/autodl-tmp/hw3_3dgs/data/apple/sparse/0 \
--output_path /root/autodl-tmp/hw3_3dgs/data/apple/dense \
--output_type COLMAP
```

---

## 📌 检查输出

确保存在：

```
images.bin
cameras.bin
points3D.bin
```

路径：

```
dense/sparse/0/
```

---

## 🚀 训练 Gaussian Splatting

```bash
python train.py \
-s /root/autodl-tmp/hw3_3dgs/data/apple/dense \
-m /root/autodl-tmp/hw3_3dgs/output/apple \
-r 2 \
--eval \
--wandb_enable \
--test_iterations 1500 3000 4500 6000 7500 9000 10500 12000 13500 15000 \
16500 18000 19500 21000 22500 24000 25500 27000 28500 30000
```

---

# 🧴 3. 物体 B（DreamFusion / ThreeStudio）

## 📦 环境（base）

```bash
cd /root/autodl-tmp/threestudio
```

---

## 🧠 训练

```bash
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
```

---

## 🎬 导出帧

```bash
mkdir -p /root/autodl-tmp/hw3_3dgs/data/watering_can/images

ffmpeg -i /root/autodl-tmp/threestudio/outputs/dreamfusion-sd/export_textured_can.mp4 \
-vf "fps=30" \
/root/autodl-tmp/hw3_3dgs/data/watering_can/images/frame_%04d.png
```

---

## 🎨 图像预处理（去蓝底 + 裁剪）

```python
import cv2, os, shutil
import numpy as np

input_dir = 'images/'
temp_dir = 'images_temp/'
os.makedirs(temp_dir, exist_ok=True)

for img_name in sorted(os.listdir(input_dir)):
    if img_name.endswith('.png'):
        img = cv2.imread(os.path.join(input_dir, img_name))
        h, w = img.shape[:2]

        # 裁剪左1/3
        cropped = img[:, :w//3]

        # 蓝底转白
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 40, 40])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        cropped[mask > 0] = [255, 255, 255]

        # 黑点转白
        cropped[np.where((cropped <= [15,15,15]).all(axis=2))] = [255,255,255]

        cv2.imwrite(os.path.join(temp_dir, img_name), cropped)

shutil.rmtree(input_dir)
os.rename(temp_dir, input_dir)
```

---

## 🧱 COLMAP + 训练流程（同 A）

```bash
conda activate gs_env
```

重复：

* feature_extractor
* matcher
* mapper
* image_undistorter

---

## 🚀 训练

```bash
python train.py \
-s /root/autodl-tmp/hw3_3dgs/data/watering_can/dense \
-m /root/autodl-tmp/hw3_3dgs/output/watering_can \
-r 2 \
--eval
```

---

# 🧊 4. 物体 C（Zero123 + Mesh → Gaussian）

## 📦 Conda 环境

```bash
conda activate /root/autodl-tmp/conda_envs/hw3_zero123
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
cd /root/autodl-tmp/threestudio
```

---

## 🚀 训练

```bash
python launch.py \
--config configs/zero123.yaml \
--train \
--gpu 0 \
loggers.wandb.enable=true \
loggers.wandb.project="HW3_ObjectC_Peach" \
loggers.wandb.name="peach_10k" \
trainer.max_steps=5000
```

---

## 📤 导出 Mesh

```bash
python launch.py \
--config configs/zero123.yaml \
--export \
--gpu 0 \
tag="export_peach" \
system.exporter_type=mesh-exporter \
data.image_path="./load/images/peach.png"
```

---

## 🧩 Mesh → 点云（Gaussian 初始化）

```python
import trimesh
import numpy as np
from plyfile import PlyData, PlyElement

mesh = trimesh.load('./mesh_c/model.obj', force='mesh', process=False)

points, face_idx = trimesh.sample.sample_surface(mesh, 50000)
rgb = mesh.visual.face_colors[face_idx][:,:3] / 255.0

f_dc = (rgb - 0.5) / 0.282095
opacity = np.ones((len(points),1))*2.197
scale = np.ones((len(points),3))*np.log(mesh.bounding_box.extents.max()/200)
rot = np.zeros((len(points),4)); rot[:,0]=1
```

（后续写入 PLY 同原文逻辑）

---

# 🏞 5. Background（counter）

```bash
conda activate gs_env
python train.py \
-s /root/autodl-tmp/hw3_3dgs/data/counter \
-m /root/autodl-tmp/hw3_3dgs/output/counter \
-r 2 \
--eval \
--wandb_enable
```

---

# 🧩 6. 最终融合流程

## 🧷 可视化工具

```
https://superspl.at/editor
```

手动融合：

```
A + B + C + counter → final.ply
```

---

## 🎥 最终渲染

使用 3DGS render.py：

```bash
python render.py
```

---

## 🎬 合成视频

```bash
ffmpeg -framerate 30 -pattern_type glob -i '*.png' \
-vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" \
-c:v libx264 -pix_fmt yuv420p -crf 18 \
final_roaming_video.mp4
```

---


