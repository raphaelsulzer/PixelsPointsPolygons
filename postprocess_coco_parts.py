import numpy as np
import json
from tqdm import tqdm
import shapely

inpath = [
    "runs/CYENS_CLUSTER_train_Pix2Poly_AUGSRUNS_inria_coco_224_negAug_run1_vit_small_patch8_224_dino_AffineRotaugs0.8_LinearWarmupLRS_1.0xVertexLoss_10.0xPermLoss__2xScoreNet_initialLR_0.0004_bs_24_Nv_192_Nbins224_500epochs/predictions_inria_coco_224_negAug_val_images_epoch_499.json"
]
out_path = f"runs/CYENS_CLUSTER_train_Pix2Poly_AUGSRUNS_inria_coco_224_negAug_run1_vit_small_patch8_224_dino_AffineRotaugs0.8_LinearWarmupLRS_1.0xVertexLoss_10.0xPermLoss__2xScoreNet_initialLR_0.0004_bs_24_Nv_192_Nbins224_500epochs/combined_predictions_inria_coco_224_negAug_val_images_epoch_499.json"


################################################################
def combine_polygons(inpath,outpath):
    combined = []
    part_lengths = []
    for i, part in enumerate(inpath):
        print(f"PART {i}")
        with open(part) as f:
            pred_part = json.loads(f.read())
        part_lengths.append(len(pred_part))
        for ins in tqdm(pred_part):
            assert len(ins['segmentation']) == 1
            segm = ins['segmentation'][0]
            segm = np.array(segm).reshape(-1, 2)
            # segm = np.flip(segm, axis=0)  # invert order of vertices cw -> ccw or ccw -> cw
            if segm.shape[0] > 2 and shapely.Polygon(segm).area > 50.:
                segm = np.concatenate((segm, segm[0, None]), axis=0)
                segm = segm.reshape(-1).tolist()
                ins['segmentation'] = [segm]
                combined.append(ins)

    with open(outpath, "w") as fp:
            fp.write(json.dumps(combined))
################################################################

