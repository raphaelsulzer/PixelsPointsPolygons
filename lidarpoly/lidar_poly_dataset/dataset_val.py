import os
from lidar_poly_dataloader import DefaultDataset

class ValDataset(DefaultDataset):
    def __init__(self,dataset_dir,**kwargs):
        super().__init__(dataset_dir,'val',**kwargs)


