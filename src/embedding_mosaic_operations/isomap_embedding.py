import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import Isomap
from tqdm import tqdm


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
    photo_dir = Path('../../photos/SORTED_STARS_CROPPED')
    # Find all .jpg files at any depth in the photo_dir
    photo_files = list(photo_dir.glob('**/*.jpg'))
    print(f'Found {len(photo_files)} images')

    # Automatically adjust parameters based on the number of images
    total_images = len(photo_files)

    # Adjust batch size if needed
    batch_size = min(50, max(10, total_images // 5))

    # Set number of PCA components
    desired_npcs = 40
    npcs = min(desired_npcs, batch_size - 1, total_images - 1)

    print(f"Using batch_size={batch_size}, npcs={npcs}")

    # 1. Reduce dimensionality with mini-batch PCA
    pca_reduced_data = minibatch_pca(photo_files, npcs, batch_size)

    # 2. Embed with Isomap
    print("Running Isomap embedding...")
    # Adjust n_neighbors based on data size - crucial parameter for Isomap
    # Too small: disconnected graph, too large: lose manifold structure
    n_neighbors = min(15, max(5, int(np.sqrt(total_images))))

    # Initialize and run Isomap
    isomap = Isomap(
        n_components=2,
        n_neighbors=n_neighbors,
        path_method='auto',  # 'auto' chooses between 'FW' (Floyd-Warshall) and 'D' (Dijkstra)
        n_jobs=-1  # Use all available cores
    )

    # Display info about Isomap parameters
    print(f"Using Isomap with n_neighbors={n_neighbors}")

    # For smaller datasets, we can use the exact algorithm; for larger ones, we might need to subsample
    if total_images > 5000:
        print("Large dataset detected. This may take a while...")

    embedded_coords = isomap.fit_transform(pca_reduced_data)

    # Calculate the range of embedded coordinates to determine space_extension
    x_min, x_max = embedded_coords[:, 0].min(), embedded_coords[:, 0].max()
    y_min, y_max = embedded_coords[:, 1].min(), embedded_coords[:, 1].max()
    coord_range = max(x_max - x_min, y_max - y_min)

    # Set space_extension to scale the embedding to a large space
    space_extension = 1000 / coord_range  # Normalize to roughly -500 to 500 range

    # 3. Plot images in 2D space
    figsize = min(30, max(10, total_images // 10))

    fig, ax = plot_images_in_2d(
        photo_files,
        embedded_coords,
        figsize=(figsize, figsize),
        background_color='black',
        space_extension=space_extension
    )

    # Save the visualization
    output_path = Path('../../output')
    output_path.mkdir(exist_ok=True, parents=True)
    output_path = output_path / photo_dir.name
    output_path.mkdir(exist_ok=True, parents=True)

    plt.savefig(output_path / 'isomap_embedding.png', dpi=300, bbox_inches='tight')
    # Also save a higher resolution version
    plt.savefig(output_path / 'isomap_embedding_hires.png', dpi=600, bbox_inches='tight')
    plt.close()

    # Save the embedding coordinates for potential later use
    np.save(output_path / 'isomap_coords.npy', embedded_coords)

    print(f"Visualization saved to {output_path / 'isomap_embedding.png'}")
    print(f"High-resolution version saved to {output_path / 'isomap_embedding_hires.png'}")
    print(f"Embedding coordinates saved to {output_path / 'isomap_coords.npy'}")