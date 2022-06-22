import os
import cv2
from torch.utils.data import Dataset
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, img_dir, flow_dir, label_file, first_file, 
            horizontal_flip=None, transform=None):
        self.img_dir = img_dir
        self.flow_dir = flow_dir
        with open(label_file) as f:
            lines = f.readlines()
            self.labels = [float(line.strip()) for line in lines]
        self.first_file = first_file
        self.horizontal_flip = horizontal_flip
        self.transform = transform

    def __len__(self):
        return len(self.labels) - 1

    def __getitem__(self, idx):
        flow_path = os.path.join(self.flow_dir, '{}.jpg'.format(idx + self.first_file))
        flow = cv2.imread(flow_path)
        flow_height, flow_width, flow_channels = flow.shape

        image_path = os.path.join(self.img_dir, '{}.jpg'.format(idx + self.first_file))
        _image = cv2.imread(image_path)
        _image_resize = cv2.resize(_image, (flow_width, flow_height))
        combined_flow = 0.1 * _image_resize + flow

        if self.horizontal_flip:
            _flip = [0, 1]
            random_num = random.choice(_flip)
            if random_num == 1:
                combined_flow = cv2.flip(combined_flow, 1)

        combined_flow = cv2.normalize(combined_flow, None, alpha=-1, beta=1, 
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        combined_flow = cv2.resize(combined_flow, (0, 0), fx=0.5, fy=0.5)

        label = np.array([self.labels[idx]])
        if self.transform:
            combined_flow = self.transform(combined_flow)
        combined_flow = np.moveaxis(combined_flow, -1, 0)
        return combined_flow, label

