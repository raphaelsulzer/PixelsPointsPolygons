from lidar_poly_dataset.dataset import DefaultDataset

class TrainDataset(DefaultDataset):
    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,'train',**kwargs)
    
    

