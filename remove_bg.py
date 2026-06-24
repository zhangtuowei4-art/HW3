import os
from rembg import remove

def process_images():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 支持的输入图片格式
    valid_exts = ['.png']
    
    # 遍历当前目录下的所有文件
    for filename in os.listdir(current_dir):
        if not any(filename.lower().endswith(ext) for ext in valid_exts):
            continue
            
        input_path = os.path.join(current_dir, filename)
        
        # 构造输出文件名，例如 original.png -> original_no_bg.png
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(current_dir, f"{name}_no_bg.png")
        
        print(f"正在处理: {filename} ...")
        
        try:
            # 1. 读取原图
            with open(input_path, 'rb') as f:
                input_data = f.read()
            
            # 2. 使用 rembg 抠图（输出原尺寸的RGBA格式图片）
            output_data = remove(input_data)
            
            # 3. 直接保存，不做任何裁剪或尺寸修改
            with open(output_path, 'wb') as f:
                f.write(output_data)
                
            print(f"✅ 已保存至: {output_path}")
            
        except Exception as e:
            print(f"❌ 处理 {filename} 时出错: {e}")

if __name__ == "__main__":
    print("开始批量抠图...")
    process_images()
    print("全部处理完成！")
