import cv2
import random
import numpy as np
import os
import copclib as copc
from .logger import make_logger, logging

from skimage import io
from sklearn.preprocessing import MinMaxScaler
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

class DefaultDataset(Dataset):
    def __init__(self, root, ann_file, use_lidar=False, use_images=True, augment=False, transform=None, logger=None, logging_level=logging.INFO):
        self.root = root
        self.lidar_root = self.root.replace('images', 'lidar')

        self.use_lidar = use_lidar
        self.use_images = use_images

        if logger is None:
            self.logger = make_logger('Dataset', logging_level)

        self.ann_file = os.path.abspath(ann_file)
        if not os.path.isfile(self.ann_file):
            raise FileNotFoundError(self.ann_file)

        self.coco = COCO(ann_file)
        images_id = self.coco.getImgIds()
        self.tiles = images_id.copy()
        self.num_samples = len(self.tiles)

        self.transform = transform
        self.augment = augment

    def augmentation(self, image, ann, points=None):

        reminder = ann['reminder']
        width = ann['width']
        height = ann['height']
        seg_mask = ann['mask']

        if reminder == 1:  # horizontal flip
            self.logger.debug('apply horizontal flip')
            if image is not None:
                image = image[:, ::-1, :]
            if points is not None:
                points[:, 0] = width - points[:, 0]
            ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
            ann['bbox'] = ann['bbox'][:, [2, 1, 0, 3]]
            ann['bbox'][:, 0] = width - ann['bbox'][:, 0]
            ann['bbox'][:, 2] = width - ann['bbox'][:, 2]
            seg_mask = np.fliplr(seg_mask)
        elif reminder == 2:  # vertical flip
            self.logger.debug('apply vertical flip')
            if image is not None:
                image = image[::-1, :, :]
            if points is not None:
                points[:, 1] = height - points[:, 1]
            ann['junctions'][:, 1] = height - ann['junctions'][:, 1]
            ann['bbox'] = ann['bbox'][:, [0, 3, 2, 1]]
            ann['bbox'][:, 1] = height - ann['bbox'][:, 1]
            ann['bbox'][:, 3] = height - ann['bbox'][:, 3]
            seg_mask = np.flipud(seg_mask)
        elif reminder == 3:  # horizontal and vertical flip
            self.logger.debug('apply horizontal and vertical flip')
            if image is not None:
                image = image[::-1, ::-1, :]
            if points is not None:
                points[:, 0] = width - points[:, 0]
                points[:, 1] = height - points[:, 1]
            seg_mask = np.fliplr(seg_mask)
            seg_mask = np.flipud(seg_mask)
            ann['junctions'][:, 0] = width - ann['junctions'][:, 0]
            ann['junctions'][:, 1] = height - ann['junctions'][:, 1]
            ann['bbox'] = ann['bbox'][:, [2, 3, 0, 1]]
            ann['bbox'][:, 0] = width - ann['bbox'][:, 0]
            ann['bbox'][:, 2] = width - ann['bbox'][:, 2]
            ann['bbox'][:, 1] = height - ann['bbox'][:, 1]
            ann['bbox'][:, 3] = height - ann['bbox'][:, 3]
        elif reminder == 4:  # rotate 90 degree
            self.logger.debug('apply 90 degree rotation')
            if points is not None:
                points[:, :2] = points[:, :2] - [width / 2, height / 2]
                R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
                points = np.dot(points, R.T)
                points[:, :2] = points[:, :2] + [width / 2, height / 2]
            rot_matrix = cv2.getRotationMatrix2D((int(width / 2), (height / 2)), 90, 1)
            if image is not None:
                image = cv2.warpAffine(image, rot_matrix, (width, height))
            seg_mask = cv2.warpAffine(seg_mask, rot_matrix, (width, height))
            ann['junctions'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['junctions']], dtype=np.float32)
            ann['bbox'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['bbox']], dtype=np.float32)
        elif reminder == 5:  # rotate 270 degree
            self.logger.debug('apply 270 degree rotation')
            if points is not None:
                points[:, :2] = points[:, :2] - [width / 2, height / 2]
                R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                points = np.dot(points, R.T)
                points[:, :2] = points[:, :2] + [width / 2, height / 2]
            rot_matrix = cv2.getRotationMatrix2D((int(width / 2), (height / 2)), 270, 1)
            if image is not None:
                image = cv2.warpAffine(image, rot_matrix, (width, height))
            seg_mask = cv2.warpAffine(seg_mask, rot_matrix, (width, height))
            ann['junctions'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['junctions']], dtype=np.float32)
            ann['bbox'] = np.asarray([affine_transform(p, rot_matrix) for p in ann['bbox']], dtype=np.float32)
        else:
            pass
        ann['mask'] = seg_mask

        return image, points, ann

    def load_lidar_points(self, lidar_file_name, img_info, z_scale=(0, 512)):

        if os.path.isfile(lidar_file_name):
            # Create a reader object
            reader = copc.FileReader(lidar_file_name)
            # Get the node metadata from the hierarchy
            node = reader.FindNode(copc.VoxelKey(0, 0, 0, 0))
            # Fetch the points of a node
            points = reader.GetPoints(node)

            points = np.stack([points.x, points.y, points.z], axis=1)
            points[:, :2] = (points[:, :2] - img_info['top_left']) / img_info.get('res_x', 0.25)
            points[:, 1] = img_info['height'] - points[:, 1]

            # scale z vals to [0,512]
            scaler = MinMaxScaler(feature_range=z_scale)  # Change range as needed
            points[:, -1] = scaler.fit_transform(points[:, -1].reshape(-1, 1)).squeeze()

        else:
            self.logger.debug(f'Lidar file {lidar_file_name} missing. Generating random point cloud.')

            n = 5
            # make a random point cloud
            x = np.random.uniform(0, img_info['width'], n)
            y = np.random.uniform(0, img_info['height'], n)
            z = np.random.uniform(z_scale[0], z_scale[1], n)
            points = np.vstack((x, y, z)).T

        return points

    def make_annotations(self, ann_coco, height, width):

        ann = {
            'junctions': [],
            'juncs_index': [],
            'juncs_tag': [],
            'edges_positive': [],
            'bbox': [],
            'width': width,
            'height': height,
        }

        pid = 0
        instance_id = 0
        seg_mask = np.zeros([width, height])
        for ann_per_ins in ann_coco:
            juncs, tags = [], []
            segmentations = ann_per_ins['segmentation']
            for i, segm in enumerate(segmentations):
                segm = np.array(segm).reshape(-1, 2)  # the shape of the segm is (N,2)
                segm[:, 0] = np.clip(segm[:, 0], 0, width - 1e-4)
                segm[:, 1] = np.clip(segm[:, 1], 0, height - 1e-4)
                points = segm[:-1]
                junc_tags = np.ones(points.shape[0])
                if i == 0:  # outline
                    poly = Polygon(points)
                    if poly.area > 0:
                        convex_point = np.array(poly.convex_hull.exterior.coords)
                        convex_index = [(p == convex_point).all(1).any() for p in points]
                        juncs.extend(points.tolist())
                        junc_tags[convex_index] = 2  # convex point label
                        tags.extend(junc_tags.tolist())
                        ann['bbox'].append(list(poly.bounds))
                        seg_mask += self.coco.annToMask(ann_per_ins)
                else:
                    juncs.extend(points.tolist())
                    tags.extend(junc_tags.tolist())
                    interior_contour = segm.reshape(-1, 1, 2)
                    cv2.drawContours(seg_mask, [np.int0(interior_contour)], -1, color=0, thickness=-1)

            idxs = np.arange(len(juncs))
            edges = np.stack((idxs, np.roll(idxs, 1))).transpose(1, 0) + pid

            ann['juncs_index'].extend([instance_id] * len(juncs))
            ann['junctions'].extend(juncs)
            ann['juncs_tag'].extend(tags)
            ann['edges_positive'].extend(edges.tolist())
            if len(juncs) > 0:
                instance_id += 1
                pid += len(juncs)

        seg_mask = np.clip(seg_mask, 0, 1)
        ann['mask'] = seg_mask

        for key, _type in (['junctions', np.float32],
                           ['edges_positive', np.longlong],
                           ['juncs_tag', np.longlong],
                           ['juncs_index', np.longlong],
                           ['bbox', np.float32],
                           ):
            ann[key] = np.array(ann[key], dtype=_type)

        if len(ann['junctions']) == 0:
            ann['mask'] = np.zeros((height, width), dtype=np.float64)
            ann['junctions'] = np.asarray([[0, 0]])
            ann['bbox'] = np.asarray([[0, 0, 0, 0]])
            ann['juncs_tag'] = np.asarray([0])
            ann['juncs_index'] = np.asarray([0])

        return ann

    def __getitem__(self, idx_):
        # basic information
        tile_ids = self.tiles[idx_]

        img_info = self.coco.loadImgs(ids=[tile_ids])[0]

        width = img_info['width']
        height = img_info['height']

        # load annotations
        ann_ids = self.coco.getAnnIds(imgIds=[tile_ids])
        ann_coco = self.coco.loadAnns(ids=ann_ids)

        # make annotations
        ann = self.make_annotations(ann_coco, height, width)

        # load image
        if self.use_images:
            filename = os.path.join(self.root, img_info['image_path'])
            if not os.path.isfile(filename):
                raise FileNotFoundError
            image = io.imread(filename).astype(float)[:, :, :3]
        else:
            image = None

        # load lidar
        if self.use_lidar:
            filename = os.path.join(self.root, img_info['lidar_path']).replace("/images/", "/lidar/")
            if not os.path.isfile(filename):
                raise FileNotFoundError
            points = self.load_lidar_points(filename, img_info)
        else:
            points = None

        # augmentation
        if self.augment and len(ann['junctions']):
            ann['reminder'] = random.randint(0,
                                             5)  # originally, b = 3, meaning only do flips for augmentation, no rotation
            image, points, ann = self.augmentation(image, ann, points)

        # self.debug_vis(image, points, ann)

        ann.update(img_info)

        if self.transform is not None:
            return self.transform(image, points, ann)

        return image, points, ann

    def __len__(self):
        return self.num_samples

    def debug_vis(self, image, point_cloud, ann):

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        # Example polygon data
        polygon_indices = ann['juncs_index']
        polygon_vertices = ann['junctions']

        # Get unique polygon IDs
        unique_polygons = np.unique(polygon_indices)

        # Assign a different color to each polygon
        colors = list(mcolors.TABLEAU_COLORS.values())

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image / 255.0)  # No cmap, assuming it's an RGB image

        # Normalize Z-values for colormap
        z_min, z_max = point_cloud[:, 2].min(), point_cloud[:, 2].max()
        norm = plt.Normalize(vmin=z_min, vmax=z_max)
        cmap = plt.cm.turbo  # 'turbo' colormap

        # Plot point cloud below polygons
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], c=cmap(norm(point_cloud[:, 2])), s=0.2, zorder=2)

        # Plot polygons
        for i, pid in enumerate(unique_polygons):
            # Get vertices belonging to this polygon
            mask = polygon_indices == pid
            poly = polygon_vertices[mask]

            # Draw polygon edges
            color = colors[i % len(colors)]  # Cycle through colors
            ax.plot(*zip(*np.vstack([poly, poly[0]])), color=color, linewidth=4)

            # Draw polygon vertices
            ax.scatter(poly[:, 0], poly[:, 1], color=color, zorder=3, s=10)

        plt.show()


def collate_fn(batches, use_lidar, use_images):
    if use_images and not use_lidar:
        return (default_collate([b[0] for b in batches]), None, [b[2] for b in batches])

    elif not use_images and use_lidar:
        pcds = []
        for batch in batches:
            pcds.append(batch[1][0, :, :])

        return (None, pcds, [b[2] for b in batches])

    elif use_images and use_lidar:
        pcds = []
        for batch in batches:
            pcds.append(batch[1][0, :, :])

        return (default_collate([b[0] for b in batches]),
                pcds,
                [b[2] for b in batches])
    else:
        raise NotImplementedError("You must either activate 'use_images' or 'use_lidar'")

