import numpy as np
from pycocotools import mask as cocomask


def get_pixel_mask_from_coco_seg(coco_anns, img_id, return_n_verts=False):
    
    img = coco_anns.loadImgs(img_id)[0]
    
    annotation_ids = coco_anns.getAnnIds(imgIds=img['id'])
    annotations = coco_anns.loadAnns(annotation_ids)
    
    pix_mask = np.zeros((img['height'], img['width']))
    
    n_verts = 0
    if not len(annotations):
        pix_mask = pix_mask.astype(np.bool_)
        if return_n_verts:
            return pix_mask, n_verts
        else:
            return pix_mask
    
    for annotation in annotations:
        segs = annotation['segmentation']
        
        rle = cocomask.frPyObjects(segs, img['height'], img['width'])
        m = cocomask.decode(rle)
        pix_mask += m.squeeze(-1)
        
        n_verts += len(segs[0]) // 2
        
    pix_mask = pix_mask != 0
    
    
    if return_n_verts:
        return pix_mask, n_verts
    else:
        return pix_mask