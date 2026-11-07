import os
import pickle
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from tqdm import tqdm

# ==================== 配置 ====================
# 修改为你的数据集根目录
DATA_ROOT = "./datasets/RAF-DB" 
# 输出文件保存位置（训练时代码会读取这个文件）
OUTPUT_PATH = "./data/identity_vectors.pkl"

# 7个情绪类别文件夹名（0-6）
EMOTIONS = ['0', '1', '2', '3', '4', '5', '6']

# 建议用 CPU 提取，防止占用显存
USE_GPU = False  
# =============================================

def main():
    # 初始化 InsightFace（首次运行会自动下载 buffalo_l 模型）
    if USE_GPU:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ctx_id = 0
    else:
        providers = ['CPUExecutionProvider']
        ctx_id = -1

    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    print(f"模型加载完成，使用设备: {'GPU' if USE_GPU else 'CPU'}")

    identity_dict = {}  # 存储格式: {"train/0/image_001.jpg": [512-dim array], ...}

    # 遍历 train 和 valid 文件夹
    for split in ['train', 'valid']:
        split_path = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_path):
            print(f"警告：{split_path} 不存在，跳过")
            continue

        for emotion in EMOTIONS:
            emo_dir = os.path.join(split_path, emotion)
            if not os.path.isdir(emo_dir):
                continue

            # 获取目录下所有图片
            img_list = [f for f in os.listdir(emo_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

            for img_name in tqdm(img_list, desc=f'Extracting {split}/{emotion}'):
                img_path = os.path.join(emo_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # 检测人脸并提取特征
                faces = app.get(img)

                if len(faces) == 0:
                    # 未检测到人脸，用全 0 向量填充
                    feat = np.zeros(512, dtype=np.float32)
                else:
                    # 取最大人脸（faces 通常按面积排序，faces[0] 即最大）
                    # normed_embedding 是已 L2 归一化的 512 维特征
                    feat = faces[0].normed_embedding

                # 用相对路径作为 key，保证训练时能精确匹配
                rel_path = os.path.join(split, emotion, img_name).replace('\\', '/')
                identity_dict[rel_path] = feat

    # 保存为 pickle 文件
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(identity_dict, f)

    print(f"\n提取完成！共处理 {len(identity_dict)} 张图片")
    print(f"身份向量已保存至: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
