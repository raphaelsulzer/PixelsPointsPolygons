from lidar_poly_dataset.dataset import DefaultDataset

class TrainDataset(DefaultDataset):
    def __init__(self,dataset_dir,**kwargs):
        super().__init__(dataset_dir,'train',**kwargs)
    
    

