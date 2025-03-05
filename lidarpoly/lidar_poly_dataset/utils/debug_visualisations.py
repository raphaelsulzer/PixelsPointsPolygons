import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_polygons(polygon_vertices, image=None, point_cloud=None, pointsize=3, linewidth=2):

    # # Example polygon data
    # polygon_indices = ann['juncs_index']
    # polygon_vertices = ann['junctions']

    # Get unique polygon IDs
    # unique_polygons = np.unique(polygon_indices)

    # Assign a different color to each polygon
    colors = list(mcolors.TABLEAU_COLORS.values())

    fig, ax = plt.subplots(dpi=200)

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

    plt.show()


def plot_mask(mask):

    fig, ax = plt.subplots()

    # Plot the image
    ax.imshow((1 - mask) * 255, cmap='gray')

    plt.show()