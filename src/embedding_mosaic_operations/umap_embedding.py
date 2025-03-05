import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.decomposition import IncrementalPCA
import umap
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
            if img_array.ndim == 3 and img_array.shape[2] == 3:  # RGB image check
                # Create mask for black pixels (with a threshold)
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
    batch_size = min(500, max(10, total_images // 5))

    # Set number of PCA components
    desired_npcs = 300
    npcs = min(desired_npcs, batch_size - 1, total_images - 1)

    print(f"Using batch_size={batch_size}, npcs={npcs}")

    # 1. Reduce dimensionality with mini-batch PCA
    pca_reduced_data = minibatch_pca(photo_files, npcs, batch_size)

    # 2. Embed with UMAP
    print("Running UMAP embedding...")

    # Adjust UMAP parameters based on dataset size
    n_neighbors = min(25, max(5, total_images // 20))  # Smaller for small datasets
    min_dist = 0.1  # Default value works well for most visualizations

    # For very small datasets, further adjust parameters
    if total_images < 50:
        n_neighbors = max(3, total_images // 5)
        min_dist = 0.05

    print(f"UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}")

    # Create and run UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='euclidean',
        random_state=42
    )

    embedded_coords = reducer.fit_transform(pca_reduced_data)

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

    # Save standard resolution
    plt.savefig(output_path / 'umap_embedding.png', dpi=300, bbox_inches='tight')

    # Also save a higher resolution version
    plt.savefig(output_path / 'umap_embedding_hires.png', dpi=600, bbox_inches='tight')

    print(f"Visualization saved to {output_path / 'umap_embedding.png'}")
    print(f"High-resolution version saved to {output_path / 'umap_embedding_hires.png'}")

    # Generate a few variations with different parameters
    variations = [
        {"min_dist": 0.01, "name": "tight_clusters"},    # Very tight clusters
        {"min_dist": 0.5,  "name": "spread_out"},        # More spread out points
        {"n_neighbors": max(3, n_neighbors // 2), "name": "local_focus"},   # More local focus
        {"n_neighbors": min(50, n_neighbors * 2), "name": "global_view"}    # More global structure
    ]

    # Only generate variations if we have enough images
    if total_images >= 30:
        print("Generating parameter variations for comparison...")

        for variation in variations:
            # Base UMAP parameters
            params = {
                "n_components": 2,
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "metric": 'euclidean',
                "random_state": 42
            }
            # Extract the 'name' and remove it from variation before updating params
            variation_name = variation["name"]

            # Update params but skip 'name'
            for k, v in variation.items():
                if k != "name":
                    params[k] = v

            # Create and run UMAP with updated parameters
            var_reducer = umap.UMAP(**params)
            var_coords = var_reducer.fit_transform(pca_reduced_data)

            # Adjust space extension for the variation
            var_x_min, var_x_max = var_coords[:, 0].min(), var_coords[:, 0].max()
            var_y_min, var_y_max = var_coords[:, 1].min(), var_coords[:, 1].max()
            var_range = max(var_x_max - var_x_min, var_y_max - var_y_min)
            var_space_extension = 1000 / var_range if var_range != 0 else 1.0

            # Plot the variation
            var_fig, var_ax = plot_images_in_2d(
                photo_files,
                var_coords,
                figsize=(figsize, figsize),
                background_color='black',
                space_extension=var_space_extension
            )

            # Save variation
            var_filename = f"umap_{variation_name}_embedding.png"
            plt.savefig(output_path / var_filename, dpi=300, bbox_inches='tight')
            plt.close(var_fig)

            print(f"Variation '{variation_name}' saved to {output_path / var_filename}")

    plt.close(fig)
    print("All visualizations complete!")
