from .dataset_default import DefaultDataset

class TrainDataset(DefaultDataset):
    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,'train',**kwargs)
    
    

