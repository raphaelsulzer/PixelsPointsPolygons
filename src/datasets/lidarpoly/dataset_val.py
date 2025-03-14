from lidar_poly_dataset.dataset import DefaultDataset

class ValDataset(DefaultDataset):
    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,'val',**kwargs)


