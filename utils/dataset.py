import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from collections import Counter
import copy

class SOPDataset(Dataset):
    def __init__(self, data_dir, mode, transform=None, samples_per_class=4, **kwargs):
        super().__init__(**kwargs)
        
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.samples_per_class = samples_per_class
        
        if mode not in ['train', 'train_split', 'val_split', 'test', 'query_split', 'db_split']:
            raise ValueError(f"Mode unrecognized {mode}")
        
        gt = pd.read_csv(os.path.join(self.data_dir, f'Ebay_{mode}.txt'), sep=' ')
        self.paths = gt['path'].apply(lambda x: os.path.join(self.data_dir, x)).tolist()
        
        # class_id 와 super_class_id가 1부터 시작되므로 1을 뺌 
        self.labels = (gt['class_id'] - 1).tolist()
        self.super_labels = (gt['super_class_id'] - 1).tolist()
        
        self.get_instance_dict()
        self.get_super_dict()

        self.avail_classes = sorted(list(self.instance_dict.keys()))
        self.is_init = True
        if self.mode in ['train', 'train_split', 'val_split']:
            if self.samples_per_class > 1:
                self.current_class = None
                self.classes_visited = []
                self.n_samples_drawn = 0

    @property
    def my_at_R(self):
        if not hasattr(self, '_at_R'):
            self._at_R = max(Counter(self.labels).values())
        return self._at_R
    
    def get_instance_dict(self):
        self.instance_dict = {cl: [] for cl in set(self.labels)}            # 각 클래스별 빈 리스트를 저장하는 딕셔너리 생성
        for idx, cl in enumerate(self.labels):
            self.instance_dict[cl].append(idx)                              # 각 class_id key에 대응되는 레이블 텍스트 파일의 인덱스를 리스트에 추가하여 저장   
        
    def get_super_dict(self):
        if hasattr(self, 'super_labels') and self.super_labels is not None:
            self.super_dict = {ct: {} for ct in set(self.super_labels)}
            for idx, cl, ct in zip(range(len(self.labels)), self.labels, self.super_labels):
                self.super_dict[ct].setdefault(cl, []).append(idx)
                    
    def __getitem__(self, idx):
        # Test 모드에서는 기존 인덱스 사용
        if self.mode not in ['train', 'train_split', 'val_split']:
            image_index = idx
        else:
            # Training 모드일 때, BaseTripletDataset 샘플링 로직 적용
            if self.samples_per_class == 1:
                # samples_per_class가 1일 경우엔 별도 로직 없이 직접 idx 사용
                image_index = idx
            else:
                # 클래스 단위로 sample
                if self.is_init:
                    # 첫 호출 시 현재 클래스 설정
                    self.current_class = self.avail_classes[idx % len(self.avail_classes)]
                    self.classes_visited = [self.current_class, self.current_class]
                    self.n_samples_drawn = 0
                    self.is_init = False

                if self.n_samples_drawn == self.samples_per_class:
                    # 현재 클래스에서 충분히 샘플했으니 다른 클래스로 넘어감
                    counter = copy.deepcopy(self.avail_classes)
                    # 바로 직전 및 그 전 클래스를 제외
                    for prev_class in self.classes_visited:
                        if prev_class in counter:
                            counter.remove(prev_class)

                    # 새로운 클래스 선택
                    self.current_class = counter[idx % len(counter)]
                    self.classes_visited = self.classes_visited[1:] + [self.current_class]
                    self.n_samples_drawn = 0

                # current_class에서 이미지를 고를 인덱스
                class_images = self.instance_dict[self.current_class]
                class_sample_idx = idx % len(class_images)
                image_index = class_images[class_sample_idx]
                self.n_samples_drawn += 1

        pth = self.paths[image_index]
        img = Image.open(pth).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[image_index]
        label = torch.tensor([label])
        out = {"image": img, "label": label, "path": pth}
        
        if hasattr(self, 'super_labels') and self.super_labels is not None:
            super_label = self.super_labels[image_index]
            super_label = torch.tensor([super_label])
            out['super_label'] = super_label
        
        return out
    
    def __len__(self):
        return len(self.paths)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(mode={self.mode}, len={len(self)})"
