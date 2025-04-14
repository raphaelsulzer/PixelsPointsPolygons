import os
from glob import glob

path = "/data/rsulzer"

for method in ["ffl","hisup","pix2poly"]:
    
    method_path = os.path.join(path, f"{method}_outputs","lidarpoly")
    
    for res in ["224","512"]:
        
        experiments = glob(os.path.join(method_path, res, "*"))
        
        for exp in experiments:
            
            exp_path = os.path.join(method_path, exp)
            
            check_file = os.path.join(exp_path, "checkpoints","validation_best.pth")
            check_outfile = os.path.join(exp_path, "checkpoints","best_val_loss.pth")
            if os.path.isfile(check_file):
                os.rename(check_file, check_outfile)
                
            # pred_file = os.path.join(exp_path, "predictions","validation_best.json")
            # pred_outfile = os.path.join(exp_path, "predictions","best_val_loss.json")
            
            # if os.path.isfile(pred_file):
            #     os.rename(pred_file, pred_outfile)
        
        