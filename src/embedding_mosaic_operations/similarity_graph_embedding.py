import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import networkx as nx


# Define function to reduce dimensionality of images using mini-batch PCA
def minibatch_pca(image_paths, n_components, batch_size):
    """Perform mini-batch PCA on a set of images."""
    # Calculate the actual batch sizes
    batch_sizes = [min(batch_size, len(image_paths) - i)
                   for i in range(0, len(image_paths), batch_size)]

    # Find minimum batch size to determine maximum possible components
    min_batch_size = min(batch_sizes)

    # Auto-adjust n_components if needed
    if n_components >= min_batch_size:
        adjusted_n_components = max(min_batch_size - 1, 2)  # Ensure at least 2 components
        print(f"Adjusting n_components from {n_components} to {adjusted_n_components} due to small batch size")
        n_components = adjusted_n_components

    # Initialize the incremental PCA
    ipca = IncrementalPCA(n_components=n_components)

    # First pass: partial_fit on all batches
    print("Fitting PCA model...")
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]

        # Load and flatten images in the current batch
        batch_data = []
        for img_path in batch_paths:
            img = Image.open(img_path)
            img_array = np.array(img).reshape(1, -1)  # Flatten the image
            batch_data.append(img_array)

        batch_data = np.vstack(batch_data)
        ipca.partial_fit(batch_data)

    # Second pass: transform all images
    print("Transforming images with PCA...")
    transformed_data = []
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]

        # Load and flatten images in the current batch
        batch_data = []
        for img_path in batch_paths:
            img = Image.open(img_path)
            img_array = np.array(img).reshape(1, -1)  # Flatten the image
            batch_data.append(img_array)

        batch_data = np.vstack(batch_data)
        batch_transformed = ipca.transform(batch_data)
        transformed_data.append(batch_transformed)

    # Combine all transformed batches
    transformed_data = np.vstack(transformed_data)
    return transformed_data


def create_similarity_graph(feature_vectors, k=5):
    """
    Create a similarity graph from feature vectors.
    Each node is connected to its k nearest neighbors.
    """
    # Calculate pairwise distances
    print("Calculating pairwise distances...")
    distances = pairwise_distances(feature_vectors, metric='euclidean')

    # Create a graph
    G = nx.Graph()

    # Add nodes
    for i in range(len(feature_vectors)):
        G.add_node(i, features=feature_vectors[i])

    # Connect each node to its k nearest neighbors
    print(f"Connecting each image to its {k} nearest neighbors...")
    for i in tqdm(range(len(distances))):
        # Get indices of k nearest neighbors (excluding self)
        indices = np.argsort(distances[i])[1:k + 1]

        # Add edges with weights based on similarity (inverse of distance)
        for j in indices:
            # Use inverse distance as weight (add small constant to avoid division by zero)
            weight = 1.0 / (distances[i][j] + 1e-5)
            G.add_edge(i, j, weight=weight)

    return G


def plot_similarity_connections(coords, G, ax, space_extension=1000, max_edges=1000):
    """Plot connections between similar images as lines."""
    # Scale coordinates
    scaled_coords = coords * space_extension

    # Get list of edges sorted by weight (highest first)
    edges = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
    edges.sort(key=lambda x: x[2], reverse=True)

    # Limit the number of edges to avoid overcrowding
    if len(edges) > max_edges:
        edges = edges[:max_edges]

    # Plot edges as lines with alpha proportional to weight
    print(f"Plotting {len(edges)} strongest connections...")
    max_weight = max(w for _, _, w in edges)
    min_weight = min(w for _, _, w in edges)
    weight_range = max_weight - min_weight if max_weight > min_weight else 1.0

    for u, v, weight in edges:
        # Normalize weight to [0.05, 0.3] for alpha (more subtle connections)
        alpha = 0.05 + 0.25 * ((weight - min_weight) / weight_range)
        x1, y1 = scaled_coords[u]
        x2, y2 = scaled_coords[v]
        ax.plot([x1, x2], [y1, y2], color='white', alpha=alpha, linewidth=0.3)

    return ax


def plot_images_in_2d(image_paths, coords, figsize=(30, 30), dpi=100,
                      background_color='black', space_extension=1000):
    """Plot images at their respective 2D coordinates with proper scaling."""
    # Create figure with black background
    fig, ax = plt.subplots(figsize=figsize, facecolor=background_color)
    ax.set_facecolor(background_color)

    # Scale coordinates by the space extension factor
    scaled_coords = coords * space_extension

    # Calculate plot boundaries
    x_min, x_max = scaled_coords[:, 0].min(), scaled_coords[:, 0].max()
    y_min, y_max = scaled_coords[:, 1].min(), scaled_coords[:, 1].max()

    # Calculate the total space dimensions
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Add padding (10%)
    padding_x = 0.1 * x_range
    padding_y = 0.1 * y_range
    ax.set_xlim(x_min - padding_x, x_max + padding_x)
    ax.set_ylim(y_min - padding_y, y_max + padding_y)

    # Determine appropriate image size (1% of the embedding space)
    # This ensures images are small relative to the total embedding space
    image_size = min(x_range, y_range) * 0.01

    # Plot each image at its coordinates
    print("Plotting images...")
    for img_path, (x, y) in tqdm(zip(image_paths, scaled_coords), total=len(image_paths)):
        try:
            # Load image
            img = Image.open(img_path)
            img_array = np.array(img)

            # Make black background transparent
            if img_array.shape[2] == 3:  # RGB image
                # Create mask for black pixels (more aggressive threshold)
                mask = (img_array[:, :, 0] < 20) & (img_array[:, :, 1] < 20) & (img_array[:, :, 2] < 20)

                # Convert to RGBA with transparency
                rgba = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
                rgba[:, :, :3] = img_array
                rgba[:, :, 3] = (~mask * 255).astype(np.uint8)  # Alpha channel
                img_array = rgba

            # Calculate image extent - using fixed size relative to embedding space
            half_size = image_size / 2
            extent = [x - half_size, x + half_size, y - half_size, y + half_size]

            # Plot the image
            ax.imshow(img_array, extent=extent, interpolation='nearest')

        except Exception as e:
            print(f"Error plotting {img_path}: {e}")

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


if __name__ == '__main__':
    photo_dir = Path('../../photos/SORTED_STARS')
    # Find all .jpg files at any depth in the photo_dir
    photo_files = list(photo_dir.glob('**/*.jpg'))
    print(f'Found {len(photo_files)} images')

    # Automatically adjust parameters based on the number of images
    total_images = len(photo_files)

    # Adjust batch size if needed
    batch_size = min(300, max(10, total_images // 5))

    # Set number of PCA components
    desired_npcs = 300
    npcs = min(desired_npcs, batch_size - 1, total_images - 1)

    print(f"Using batch_size={batch_size}, npcs={npcs}")

    # 1. Reduce dimensionality with mini-batch PCA
    pca_reduced_data = minibatch_pca(photo_files, npcs, batch_size)

    # 2. Create a similarity graph
    # Adjust k based on dataset size
    k = min(25, max(3, int(np.sqrt(total_images))))
    print(f"Using k={k} nearest neighbors for graph construction")
    similarity_graph = create_similarity_graph(pca_reduced_data, k=k)

    # 3. Run force-directed layout using NetworkX's spring_layout
    print("Running force-directed layout algorithm...")

    # Set the weight parameter to consider edge weights in the layout
    positions = nx.spring_layout(
        similarity_graph,
        weight='weight',  # Use edge weights
        k=1.0 / np.sqrt(total_images),  # Optimal distance between nodes
        iterations=500,  # More iterations for better convergence
        seed=42  # For reproducibility
    )

    # Convert dictionary positions to array
    pos_array = np.array([positions[i] for i in range(len(positions))])

    # 4. Normalize positions to [0, 1] range
    pos_min = pos_array.min(axis=0)
    pos_max = pos_array.max(axis=0)
    pos_normalized = (pos_array - pos_min) / (pos_max - pos_min)

    # 5. Center and scale coordinates to [-0.5, 0.5]
    embedded_coords = pos_normalized - 0.5

    # Calculate the range of embedded coordinates to determine space_extension
    x_min, x_max = embedded_coords[:, 0].min(), embedded_coords[:, 0].max()
    y_min, y_max = embedded_coords[:, 1].min(), embedded_coords[:, 1].max()
    coord_range = max(x_max - x_min, y_max - y_min)

    # Set space_extension to scale the embedding to a large space
    space_extension = 1000 / coord_range  # Normalize to roughly -500 to 500 range

    # 6. Plot images with connecting lines in one visualization
    figsize = min(30, max(10, total_images // 10))
    fig, ax = plt.subplots(figsize=(figsize, figsize), facecolor='black')
    ax.set_facecolor('black')

    # Calculate plot boundaries (needed for connection lines)
    scaled_coords = embedded_coords * space_extension
    x_min, x_max = scaled_coords[:, 0].min(), scaled_coords[:, 0].max()
    y_min, y_max = scaled_coords[:, 1].min(), scaled_coords[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Add padding (10%)
    padding_x = 0.1 * x_range
    padding_y = 0.1 * y_range
    ax.set_xlim(x_min - padding_x, x_max + padding_x)
    ax.set_ylim(y_min - padding_y, y_max + padding_y)

    # First plot connections in the background
    #max_edges = min(total_images * 5, 3000)  # Limit number of edges to avoid overcrowding
    #plot_similarity_connections(embedded_coords, similarity_graph, ax, space_extension, max_edges)

    # Determine appropriate image size
    image_size = min(x_range, y_range) * 0.01

    # Then plot the images on top (with no scatter points)
    print("Plotting images...")
    for img_path, (x, y) in tqdm(zip(photo_files, scaled_coords), total=len(photo_files)):
        try:
            # Load image
            img = Image.open(img_path)
            img_array = np.array(img)

            # Make black background transparent
            if img_array.shape[2] == 3:  # RGB image
                # Create mask for black pixels
                mask = (img_array[:, :, 0] < 20) & (img_array[:, :, 1] < 20) & (img_array[:, :, 2] < 20)

                # Convert to RGBA with transparency
                rgba = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
                rgba[:, :, :3] = img_array
                rgba[:, :, 3] = (~mask * 255).astype(np.uint8)  # Alpha channel
                img_array = rgba

            # Calculate image extent - using fixed size relative to embedding space
            half_size = image_size / 2
            extent = [x - half_size, x + half_size, y - half_size, y + half_size]

            # Plot the image
            ax.imshow(img_array, extent=extent, interpolation='nearest')

        except Exception as e:
            print(f"Error plotting {img_path}: {e}")

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the visualization
    output_path = Path('../../output')
    output_path.mkdir(exist_ok=True, parents=True)
    output_path = output_path / photo_dir.name
    output_path.mkdir(exist_ok=True, parents=True)

    plt.savefig(output_path / 'star_network_visualization.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'star_network_visualization_hires.png', dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Combined visualization saved to {output_path}")