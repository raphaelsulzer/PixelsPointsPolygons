from .dataset_default import DefaultDataset

class ValDataset(DefaultDataset):
    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,'val',**kwargs)


