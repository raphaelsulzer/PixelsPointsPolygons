import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

def plot_polygons(polygon_vertices, image=None, point_cloud=None, pointsize=3, linewidth=2):

    # # Example polygon data
    # polygon_indices = ann['juncs_index']
    # polygon_vertices = ann['junctions']

    # Get unique polygon IDs
    # unique_polygons = np.unique(polygon_indices)

    # Assign a different color to each polygon
    colors = list(mcolors.TABLEAU_COLORS.values())

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

    if image is not None:
        ax.imshow(image)

    if point_cloud is not None:
        # Normalize Z-values for colormap
        z_min, z_max = point_cloud[:, 2].min(), point_cloud[:, 2].max()
        norm = plt.Normalize(vmin=z_min, vmax=z_max)
        cmap = plt.cm.turbo  # 'turbo' colormap

        # Plot point cloud below polygons
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], c=cmap(norm(point_cloud[:, 2])), s=0.2, zorder=2)

    # Plot polygons
    for i, poly in enumerate(polygon_vertices):
        # Get vertices belonging to this polygon
        # mask = polygon_indices == pid
        # poly = polygon_vertices[mask]
        # poly = np.vstack([poly, poly[0]])

        # Draw polygon edges
        color = colors[i % len(colors)]  # Cycle through colors
        ax.plot(*zip(*np.vstack([poly, poly[0]])), color=color, linewidth=linewidth)

        # Draw polygon vertices
        ax.scatter(poly[:, 0], poly[:, 1], color=color, zorder=3, s=pointsize)

    plt.show(block=False)


def plot_mask(image, alpha=1.0, ax=None, show_axis='off', show=False):
    
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=50)
    ax.axis(show_axis)

    # Plot the image
    ax.imshow((1 - image) * 255, cmap='gray', alpha=alpha)

    if show:
        plt.show(block=False)


def plot_image(image, ax=None, show_axis='off', show=False):
    
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=50)
    ax.axis(show_axis)

    # Plot the image
    ax.imshow(image)

    if show:
        plt.show(block=False)
    

def plot_corners(corner_image, ax=None, show_axis='off', show=False):
    
    if isinstance(corner_image, torch.Tensor):
        corner_image = corner_image.permute(1, 2, 0).cpu().numpy()
        
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        
    ax.axis(show_axis)

    # Get the coordinates of all 1 values
    ones_coords = np.argwhere(corner_image == 1)
    
    # Plot a red cross at each 1 value
    for y, x, _ in ones_coords:
        ax.plot(x,y, color='red', linewidth=25, marker='x')
        
        # Plot a red cross at each 1 value
    for x, y, _ in ones_coords:
        ax.plot(x,y, color='blue', linewidth=25, marker='x')

    if show:
        plt.show(block=False)
    
    
def plot_pix2poly(image_batch,mask_batch=None,corner_image_batch=None):
    
    fig, ax = plt.subplots(4,4,figsize=(8, 8), dpi=150)
    ax = ax.flatten()

    image_batch = image_batch.permute(0, 2, 3, 1).cpu().numpy()
    
    if mask_batch is not None:
        mask_batch = mask_batch.permute(0, 2, 3, 1).cpu().numpy()
    
    if corner_image_batch is not None:
        corner_image_batch = corner_image_batch.permute(0, 2, 3, 1).cpu().numpy()
    
    for i in range(image_batch.shape[0]):
        plot_image(image_batch[i], show=False, ax=ax[i])
        if mask_batch is not None:
            plot_mask(mask_batch[i], alpha=0.5 , show=False, ax=ax[i])
        if corner_image_batch is not None:
            plot_corners(corner_image_batch[i], show=False, ax=ax[i])
    
    plt.tight_layout()
    plt.show(block=True)
    

