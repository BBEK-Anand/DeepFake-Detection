#import your nessessary libreries here
from torch.utils.data import Dataset

# from torch.utils.data import Dataset
# import os
# import librosa
# import torch
# import random
# 
# class BaseDataSet(Dataset):
#     def __init__(self, folder, sr=22050, crop_duration=0.47):
#         self.folder = folder
#         self.file_list = [f for f in os.listdir(folder)]
#         self.sr = sr
#         self.crop_duration = crop_duration
#         self.crop_size = int(self.crop_duration * self.sr)
# 
#     def __len__(self):
#         return len(self.file_list)
# 
#     def __getitem__(self, idx):
#         file_path = self.file_list[idx]
#         waveform, sr = librosa.load(os.path.join(self.folder, file_path))
#         waveform = torch.tensor(waveform).unsqueeze(0)
#         cropped_waveform = self.random_crop(waveform, self.crop_size)
# 
#         label = int(os.path.basename(file_path)[-5])
#         return cropped_waveform, label
# 
#     def random_crop(self, waveform, crop_size):
#         num_samples = waveform.size(1)
#         if num_samples <= crop_size:
#             padding = crop_size - num_samples
#             cropped_waveform = torch.nn.functional.pad(waveform, (0, padding))
#         else:
#             start = random.randint(0, num_samples - crop_size)
#             cropped_waveform = waveform[:, start:start + crop_size]
#         return cropped_waveform
 

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# this dataset class should take a path, which is we have fixed using set_default_config of parameter train_data_src and valid_data_src
# in our case real and fake images are in different folders, so the Dataset class should be made in such a way.
# also 
class DS01(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
                            transforms.Resize((128,128)),  # Resize images 
                            transforms.ToTensor()        # Convert images to tensor
                        ])
        self.image_paths = []
        self.labels = []
        
        # Map directories with 'real' or 'fake' in their name to corresponding labels
        self.class_to_idx = {'real': 1, 'fake': 0}  # 1 for real, 0 for fake
        
        # Traverse through subdirectories
        for subdir in os.listdir(root_dir):
            class_name = None
            
            # Identify if subdir is 'real' or 'fake' by checking the directory name
            if 'real' in subdir.lower():
                class_name = 'real'
            elif 'fake' in subdir.lower():
                class_name = 'fake'
            
            if class_name:
                class_dir = os.path.join(root_dir, subdir)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        # Only load .jpg files
                        if img_name.lower().endswith('.jpg'):
                            self.image_paths.append(img_path)
                            self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms if provided
        image = self.transform(image)
        
        return image, label

