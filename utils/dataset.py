import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class RAFDBDisentangleDataset(Dataset):
    def __init__(self, split='train', data_root="", identity_file="", image_size=224):
        self.split = split
        self.data_root = data_root
        self.identity_file = identity_file
        
        if os.path.exists(self.identity_file):
            with open(self.identity_file, 'rb') as f:
                self.identity_dict = pickle.load(f)
        else:
            print(f"Warning: {self.identity_file} not found. Using zero vectors.")
            self.identity_dict = {}

        self.samples = []
        EMOTIONS = ['0', '1', '2', '3', '4', '5', '6']
        split_path = os.path.join(self.data_root, split)

        for emotion_idx, emotion in enumerate(EMOTIONS):
            emo_dir = os.path.join(split_path, emotion)
            if not os.path.isdir(emo_dir): continue
            for img_name in os.listdir(emo_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(emo_dir, img_name)
                    rel_path = os.path.join(split, emotion, img_name)
                    self.samples.append((img_path, emotion_idx, rel_path))

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, rel_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        identity_vec = self.identity_dict.get(rel_path)
        if identity_vec is None:
            identity_vec = np.zeros(512, dtype=np.float32)
        identity_vec = torch.from_numpy(identity_vec).float()

        return image, label, identity_vec
