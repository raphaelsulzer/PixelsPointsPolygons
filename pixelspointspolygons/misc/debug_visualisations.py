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
        tags = annotations['juncs_tag'][annotations['juncs_index']==i]
        
        
        # Draw polygon edges
        color = colors[i % len(colors)]  # Cycle through colors

        ax.plot(*zip(*np.vstack([poly, poly[0]])), color=color, linewidth=linewidth)

        convex_concave_color = []
        for j in range(len(poly)):
            if tags[j] == 1:
                convex_concave_color.append('green')
            elif tags[j] == 2:
                convex_concave_color.append('red')
            else:
                convex_concave_color.append('blue')
        # Draw polygon vertices
        ax.scatter(poly[:, 0], poly[:, 1], color=convex_concave_color, zorder=3, s=pointsize)
        
    if show:
        plt.show(block=False)    
        

def plot_mask(image, color=[1,0,0,1], ax=None, show_axis='off', show=False):
    
    # if isinstance(image, torch.Tensor):
    #     if image.ndim == 2:
    #         image = image[None, :, :]
    #     image = image.permute(1, 2, 0).cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=50)
    ax.axis(show_axis)

    rgba_image = np.zeros((image.shape[0], image.shape[1], 4))
    rgba_image[image == 1] = color
    
    # Plot the image
    # ax.imshow((image) * 255, cmap='gray', alpha=alpha)
    ax.imshow(rgba_image)

    if show:
        plt.show(block=False)


def plot_crossfield_jet(image, mask=None, alpha = 0.7, ax=None, show_axis='off', show=False):
    
    # if isinstance(image, torch.Tensor):
    #     if image.ndim == 2:
    #         image = image[None, :, :]
    #     image = image.permute(1, 2, 0).cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=50)
    ax.axis(show_axis)
    
    if mask is not None:
        image = np.ma.masked_where(mask != 1, image[0,:,:])
    else:
        image = image[0,:,:]
    
    # Plot the image
    ax.imshow(image, cmap='jet', alpha=alpha)

    if show:
        plt.show(block=False)
        
def plot_crossfield(angles_rad, mask, ax=None, show_axis='off', show=False):

    angles_rad = angles_rad.squeeze()/2.0
    mask = (mask == 1).numpy()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=50)
    ax.axis(show_axis)
    
    # Compute arrow directions
    U = np.cos(angles_rad)  # X component
    V = np.sin(angles_rad)  # Y component


    
    # Generate correct X, Y grid matching (512, 512)
    Y, X = np.meshgrid(np.arange(angles_rad.shape[0]), np.arange(angles_rad.shape[1]), indexing='ij')

    # step = 50  # Adjust for desired density
    # X = X[::step, ::step]
    # Y = Y[::step, ::step]
    # U = U[::step, ::step]
    # V = V[::step, ::step]
    # mask = mask[::step, ::step]
    
    # Apply the mask
    X_masked, Y_masked = X[mask], Y[mask]
    U_masked, V_masked = U[mask], V[mask]
    
    step = 30
    X_masked = X_masked[::step]
    Y_masked = Y_masked[::step]
    U_masked = U_masked[::step]
    V_masked = V_masked[::step]

    length = 2.0
    # Create the plot
    ax.quiver(X_masked, Y_masked, U_masked, V_masked, color='blue', angles='xy', scale_units='xy', scale=None, width=0.05, headlength=40)

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
            plot_mask(mask_batch[i], alpha=0.03 , show=False, ax=ax[i])
        if corner_image_batch is not None:
            plot_point_activations(corner_image_batch[i], show=False, ax=ax[i])
        if polygon_batch is not None:              
            plot_polygons_pix2poly(polygon_batch[i], show=False, ax=ax[i], polygon_format=polygon_format)
        if tile_names is not None:
            ax[i].set_title(tile_names[i], fontsize=4)
    
    plt.tight_layout()
    plt.show(block=True)
    



def plot_ffl(batch):
    
    image_batch = batch.get("image",None)
    lidar_batch = batch.get("lidar",None)
    gt_polygons_image = batch.get("gt_polygons_image",None)
    building = gt_polygons_image[:,0,:,:]
    edges = gt_polygons_image[:,1,:,:]
    vertices = gt_polygons_image[:,2,:,:]
    gt_crossfield_angle = batch.get("gt_crossfield_angle",None)
    tile_names = batch.get("name",None)
    
    batch_size = len(image_batch) if image_batch is not None else len(lidar_batch)
    
    n_rows = np.ceil(np.sqrt(batch_size)).astype(int)
    n_cols = n_rows
    
    fig, ax = plt.subplots(n_rows,n_cols,figsize=(int(n_cols*3), int(n_cols*3)), dpi=150)
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
               
        # plot_mask(building[i], color=[1,0,0,0.1], show=False, ax=ax[i])
        # plot_mask(edges[i], color=[0,1,0,1], show=False, ax=ax[i])
        # plot_mask(vertices[i], color=[0,0,1,1], show=False, ax=ax[i])
        
        # plot_crossfield(gt_crossfield_angle[i], mask=edges[i], alpha=0.7, show=False, ax=ax[i])
        plot_crossfield(gt_crossfield_angle[i], mask=edges[i], show=False, ax=ax[i])
            # plot_polygons(annotations_batch["junctions"][i], show=False, ax=ax[i], polygon_format=polygon_format)
        # if tile_names is not None:
        #     ax[i].set_title(tile_names[i], fontsize=4)
    
    plt.tight_layout()
    plt.show(block=True)