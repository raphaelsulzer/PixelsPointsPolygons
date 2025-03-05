import os
from lidar_poly_dataloader import DefaultDataset

class TrainDataset(DefaultDataset):
    def __init__(self):
        super().__init__()
        self.ann_file = os.path.join(self.dataset_dir,"annotations_train.json")


