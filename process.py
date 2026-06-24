import os
from PIL import Image

# 1. 设置文件名
# ⚠️ 请在这里修改为你刚刚放入文件夹的真实照片文件名（包含 .jpg 或 .jpeg 后缀）
input_filename = "peach.jpg"  
output_filename = "peach_cut.png" # 输出为 PNG，方便下一步直接用于抠图

# 获取当前脚本所在的目录（即 MugProcess 文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_dir, input_filename)
output_path = os.path.join(current_dir, output_filename)

# 2. 检查文件是否存在
if not os.path.exists(input_path):
    print(f"❌ 错误：在文件夹中找不到文件 '{input_filename}'")
    print("请检查文件名是否拼写正确（包括 .jpg 或 .jpeg 后缀），注意大小写！")
else:
    print(f"✅ 找到文件，正在处理...")
    
    # 3. 打开图片
    img = Image.open(input_path)
    width, height = img.size
    print(f"📏 原始分辨率: {width} x {height}")

    # 4. 计算中心裁剪的正方形区域 (取宽高中的较小值，确保马克杯在画面正中心)
    size = min(width, height)
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    # 5. 执行裁剪
    img_cropped = img.crop((left, top, right, bottom))
    print(f"✂️ 裁剪为正方形后分辨率: {img_cropped.size[0]} x {img_cropped.size[1]}")

    # 6. 缩放至 512x512 (作业允许降低分辨率以省显存，这是 Zero123 的标准输入尺寸)
    # 使用 LANCZOS 算法可以保证缩放后马克杯边缘依然清晰
    img_resized = img_cropped.resize((512, 512), Image.Resampling.LANCZOS)

    # 7. 保存到当前文件夹
    img_resized.save(output_path)
    print(f"🎉 处理完成！已在当前文件夹生成: {output_filename}")