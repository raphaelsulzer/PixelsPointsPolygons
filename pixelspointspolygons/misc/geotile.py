class GeoTile:
    def __init__(self, image=None, lidar=None, transform=None):
        self.image = image  # Tensor of shape (C, H, W)
        self.lidar = lidar  # Tensor of shape (N, 3)
        self.transform = transform
        # self.translation = translation  # Affine transform for georeferencing
        # self.image_resolution = image_resolution
