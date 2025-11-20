class GeoTile:
    def __init__(self, image=None, lidar=None, translation=None, image_resolution=0.25):
        self.image = image  # Tensor of shape (C, H, W)
        self.lidar = lidar  # Tensor of shape (N, 3)
        self.translation = translation  # Affine transform for georeferencing
        self.image_resolution = image_resolution
