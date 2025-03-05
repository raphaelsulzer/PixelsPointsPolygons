import os
import torch
import numpy as np

from lidar_poly_dataloader import DefaultDataset

class TrainDataset(DefaultDataset):
    def __init__(self,dataset_dir,**kwargs):
        super().__init__(dataset_dir,'train',**kwargs)
    
    

