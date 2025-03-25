import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch


def plot_point_cloud(point_cloud, ax=None, show=False, alpha=0.15):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=50)
    
    # Normalize Z-values for colormap
    z_min, z_max = point_cloud[:, 2].min(), point_cloud[:, 2].max()
    norm = plt.Normalize(vmin=z_min, vmax=z_max)
    cmap = plt.cm.turbo  # 'turbo' colormap

    # Plot point cloud below polygons
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], 
               c=cmap(norm(point_cloud[:, 2])), s=0.1, zorder=2,
               alpha=alpha)
    
    if show:
        plt.show(block=False)
    

def plot_polygons_pix2poly(polygon_vertices, ax=None, pointsize=3, linewidth=2, show=False,
                  polygon_format="xy"):

    # Assign a different color to each polygon
    colors = list(mcolors.TABLEAU_COLORS.values())

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)


    # Plot polygons
    for i, poly in enumerate(polygon_vertices):
        if polygon_format == "xy":
            pass
        elif polygon_format == "yx":
            poly = poly[:, [1, 0]]
        else:
            raise ValueError("polygon_format must be 'xy' or 'yx'")
        

        # Draw polygon edges
        color = colors[i % len(colors)]  # Cycle through colors
        ax.plot(*zip(*np.vstack([poly, poly[0]])), color=color, linewidth=linewidth)

        # Draw polygon vertices
        ax.scatter(poly[:, 0], poly[:, 1], color=color, zorder=3, s=pointsize)
        
    if show:
        plt.show(block=False)    


def plot_polygons_hisup(annotations, ax=None, pointsize=3, linewidth=2, show=False,
                  polygon_format="xy"):

    # Assign a different color to each polygon
    colors = list(mcolors.TABLEAU_COLORS.values())

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)


    if len(annotations['junctions']) < 3:
        return
    
    n_polys = np.unique(annotations["juncs_index"]).shape[0]
    
    
    # Plot polygons
    for i in range(n_polys):
        if polygon_format == "xy":
            pass
        elif polygon_format == "yx":
            poly = poly[:, [1, 0]]
        else:
            raise ValueError("polygon_format must be 'xy' or 'yx'")
        
        poly = annotations['junctions'][annotations['juncs_index']==i]

        # Draw polygon edges
        color = colors[i % len(colors)]  # Cycle through colors

        ax.plot(*zip(*np.vstack([poly, poly[0]])), color=color, linewidth=linewidth)

        convex_concave_color = []
        for j in range(len(poly)):
            if annotations['juncs_tag'][j] == 1:
                convex_concave_color.append('green')
            elif annotations['juncs_tag'][j] == 2:
                convex_concave_color.append('red')
            else:
                convex_concave_color.append('blue')
        # Draw polygon vertices
        ax.scatter(poly[:, 0], poly[:, 1], color=convex_concave_color, zorder=3, s=pointsize)
        
    if show:
        plt.show(block=False)    
        

def plot_mask(image, alpha=1.0, ax=None, show_axis='off', show=False):
    
    if image.ndim == 2:
        image = image[None, :, :]
    
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
        


def plot_point_activations(corner_image, ax=None, show_axis='off', show=False):
    
    if isinstance(corner_image, torch.Tensor):
        corner_image = corner_image.permute(1, 2, 0).cpu().numpy()
        
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        
    ax.axis(show_axis)

    # Get the coordinates of all 1 values
    ones_coords = np.argwhere(corner_image == 1)
    
    # Plot a red cross at each 1 value
    for y, x, _ in ones_coords:
        ax.plot(x,y, color='magenta', linewidth=25, marker='x')
        
    #     # Plot a red cross at each 1 value
    # for x, y, _ in ones_coords:
    #     ax.plot(x,y, color='blue', linewidth=25, marker='x')

    if show:
        plt.show(block=False)


def plot_hisup(image_batch=None, lidar_batch=None, annotations_batch=None, tile_names=None, polygon_format="xy"):
    batch_size = len(image_batch) if image_batch is not None else len(lidar_batch)
    
    n_rows = np.ceil(np.sqrt(batch_size)).astype(int)
    n_cols = n_rows
    
    fig, ax = plt.subplots(n_rows,n_cols,figsize=(int(n_cols*2), int(n_cols*2)), dpi=150)
    ax = ax.flatten()

    if image_batch is not None:
        image_batch = image_batch.permute(0, 2, 3, 1).cpu().numpy()
        
    if lidar_batch is not None:
        lidar_batch = list(torch.unbind(lidar_batch, dim=0))
    
    # if mask_batch is not None:
    #     mask_batch = mask_batch.permute(0, 2, 3, 1).cpu().numpy()
    
    # if corner_image_batch is not None:
    #     corner_image_batch = corner_image_batch.permute(0, 2, 3, 1).cpu().numpy()
    
    for i in range(batch_size):
        if image_batch is not None:
            plot_image(image_batch[i], show=False, ax=ax[i])
        if lidar_batch is not None:
            plot_point_cloud(lidar_batch[i], show=False, ax=ax[i])    
        if annotations_batch is not None:
            plot_mask(annotations_batch[i]["mask"], alpha=0.7 , show=False, ax=ax[i])
            plot_polygons_hisup(annotations_batch[i], show=False, ax=ax[i])              
            # plot_polygons(annotations_batch["junctions"][i], show=False, ax=ax[i], polygon_format=polygon_format)
        if tile_names is not None:
            ax[i].set_title(tile_names[i], fontsize=4)
    
    plt.tight_layout()
    plt.show(block=True)
    
    
def plot_pix2poly(image_batch=None,lidar_batch=None,tile_names=None,mask_batch=None,corner_image_batch=None,polygon_batch=None,polygon_format="xy"):
    
    batch_size = len(image_batch) if image_batch is not None else len(lidar_batch)
    
    n_rows = np.ceil(np.sqrt(batch_size)).astype(int)
    n_cols = n_rows
    
    fig, ax = plt.subplots(n_rows,n_cols,figsize=(int(n_cols*2), int(n_cols*2)), dpi=150)
    ax = ax.flatten()

    if image_batch is not None:
        image_batch = image_batch.permute(0, 2, 3, 1).cpu().numpy()
        
    if lidar_batch is not None:
        lidar_batch = list(torch.unbind(lidar_batch, dim=0))
    
    if mask_batch is not None:
        mask_batch = mask_batch.permute(0, 2, 3, 1).cpu().numpy()
    
    if corner_image_batch is not None:
        corner_image_batch = corner_image_batch.permute(0, 2, 3, 1).cpu().numpy()
    
    for i in range(batch_size):
        if image_batch is not None:
            plot_image(image_batch[i], show=False, ax=ax[i])
        if lidar_batch is not None:
            plot_point_cloud(lidar_batch[i], show=False, ax=ax[i])    
        if mask_batch is not None:
            plot_mask(mask_batch[i], alpha=0.7 , show=False, ax=ax[i])
        if corner_image_batch is not None:
            plot_point_activations(corner_image_batch[i], show=False, ax=ax[i])
        if polygon_batch is not None:              
            plot_polygons_pix2poly(polygon_batch[i], show=False, ax=ax[i], polygon_format=polygon_format)
        if tile_names is not None:
            ax[i].set_title(tile_names[i], fontsize=4)
    
    plt.tight_layout()
    plt.show(block=True)
    

