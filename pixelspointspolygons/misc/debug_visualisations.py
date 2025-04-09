import torch

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as Patches

def plot_point_cloud(point_cloud, ax=None, show=False, alpha=0.15):
    
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()
    
    if point_cloud.ndim == 3:
        point_cloud = point_cloud.squeeze()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=50)
    
    # Normalize Z-values for colormap
    z_min, z_max = point_cloud[:, 2].min(), point_cloud[:, 2].max()
    norm = plt.Normalize(vmin=z_min, vmax=z_max)
    # cmap = plt.cm.turbo  # 'turbo' colormap
    cmap = plt.cm.grey  # 'turbo' colormap

    # Plot point cloud below polygons
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], 
               c=cmap(norm(point_cloud[:, 2])), s=0.1, zorder=2,
               alpha=alpha)
    
    if show:
        plt.show(block=False)


def plot_shapely_polygons(polygons, ax=None, color=[1,0,1,0.7], pointcolor=None, edgecolor=None, fillcolor=None, pointsize=3, linewidth=2, show=False):
    
    
    if pointcolor is None:
        pointcolor = color
    if edgecolor is None:
        edgecolor = color
    
    for poly in polygons:

        ax.add_patch(Patches.Polygon(poly.exterior.coords[:-1], fill=fillcolor, ec=edgecolor, linewidth=linewidth))
        juncs = np.array(poly.exterior.coords[:-1])
        ax.plot(juncs[:, 0], juncs[:, 1], color=pointcolor, marker='.', markersize=pointsize, linestyle='none')
        if len(poly.interiors) != 0:
            for inter in poly.interiors:
                ax.add_patch(Patches.Polygon(inter.coords[:-1], fill=False, ec=edgecolor, linewidth=linewidth))
                juncs = np.array(inter.coords[:-1])
                ax.plot(juncs[:, 0], juncs[:, 1], color=pointcolor, marker='.', markersize=pointsize, linestyle='none')
                
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
    
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:
            image = image[None, :, :]
        image = image.permute(1, 2, 0).detach().cpu().numpy()
        image = image.squeeze()

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
        


def plot_crossfield(crossfield, crossfield_stride=8, ax=None, show_axis='off', mask=None, alpha=0.8, width=1.8, add_scale=0.8, show=False):
    
    if isinstance(crossfield, torch.Tensor):
        if crossfield.ndim == 2:
            crossfield = crossfield[None, :, :]
        crossfield = crossfield.permute(1, 2, 0).detach().cpu().numpy()
        crossfield = crossfield.squeeze()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=50)
    ax.axis(show_axis)
    
    x = np.arange(0, crossfield.shape[1], crossfield_stride)
    y = np.arange(0, crossfield.shape[0], crossfield_stride)
    x, y = np.meshgrid(x, y)

    scale = add_scale * 1 / crossfield_stride
    
    u = np.cos(crossfield)
    v = np.sin(crossfield)
    
    u = u[::crossfield_stride, ::crossfield_stride]
    v = v[::crossfield_stride, ::crossfield_stride]
    
    quiveropts = dict(color=(0, 0, 1, alpha), headaxislength=0, headlength=0, pivot='middle', angles='xy', units='xy',
                      scale=scale, width=width, headwidth=1)
    ax.quiver(x, y, u, -v, **quiveropts)

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
    



def plot_ffl(batch,show=True):
    
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
    
    fig, ax = plt.subplots(n_rows,n_cols,figsize=(16,16), dpi=80)
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
    else:
        ax = [ax]
    
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
               
        plot_mask(building[i], color=[1,1,0,0.3], show=False, ax=ax[i])
        # plot_mask(edges[i], color=[0,1,0,1], show=False, ax=ax[i])
        # plot_mask(vertices[i], color=[0,0,1,1], show=False, ax=ax[i])
        
        # plot_crossfield(gt_crossfield_angle[i], mask=edges[i], alpha=0.7, show=False, ax=ax[i])
        plot_crossfield(gt_crossfield_angle[i], mask=edges[i], show=False, ax=ax[i])
            # plot_polygons(annotations_batch["junctions"][i], show=False, ax=ax[i], polygon_format=polygon_format)
        # if tile_names is not None:
        #     ax[i].set_title(tile_names[i], fontsize=4)
    
    plt.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if show:
        plt.show(block=True)
    
    return fig