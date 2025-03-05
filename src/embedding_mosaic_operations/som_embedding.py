import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
from minisom import MiniSom  # You may need to install this: pip install minisom


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


def create_som_mapping(data, grid_size=None):
    """Create a Self-Organizing Map from the input data."""
    # Determine grid size if not provided (roughly sqrt of the data points)
    if grid_size is None:
        side_length = int(np.sqrt(len(data) * 5))  # Multiply by 5 for a more spacious grid
        grid_size = (side_length, side_length)

    print(f"Creating SOM with grid size {grid_size}...")

    # Normalize the data
    data_normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Initialize and train the SOM
    som = MiniSom(
        grid_size[0], grid_size[1],
        data_normalized.shape[1],
        sigma=1.0,  # Initial neighborhood radius
        learning_rate=0.5,  # Initial learning rate
        random_seed=42
    )

    # Initialize weights with PCA
    som.pca_weights_init(data_normalized)

    # Train the SOM
    print("Training SOM...")
    som.train_batch(
        data_normalized,
        num_iteration=5000,  # More iterations for better organization
        verbose=True
    )

    # Get the best matching unit (BMU) for each data point
    print("Finding best matching units...")
    bmu_coords = np.array([som.winner(d) for d in tqdm(data_normalized)])

    return bmu_coords, grid_size, som


def plot_som_constellation(image_paths, bmu_coords, grid_size, figsize=(30, 30),
                           background_color='black', add_connections=True):
    """Create a constellation chart-like visualization using SOM coordinates."""
    # Create figure with black background
    fig, ax = plt.subplots(figsize=figsize, facecolor=background_color)
    ax.set_facecolor(background_color)

    # Set the axis limits with some padding
    padding = 1  # Add padding around the grid
    ax.set_xlim(-padding, grid_size[0] + padding)
    ax.set_ylim(-padding, grid_size[1] + padding)

    # Calculate appropriate image size (fraction of grid cell)
    image_size = 0.5  # Size of image as fraction of grid cell

    # Create a scatter plot of faint stars in the background for aesthetic effect
    if len(image_paths) > 50:  # Only add bg stars if we have enough images
        n_bg_stars = 1000
        bg_x = np.random.uniform(-padding, grid_size[0] + padding, n_bg_stars)
        bg_y = np.random.uniform(-padding, grid_size[1] + padding, n_bg_stars)
        bg_sizes = np.random.exponential(0.05, n_bg_stars)
        ax.scatter(bg_x, bg_y, s=bg_sizes, color='white', alpha=0.3)

    # Add grid connections for constellation effect
    if add_connections:
        print("Adding constellation connections...")
        # Find occupied cells in the grid
        occupied_cells = {}
        for i, (x, y) in enumerate(bmu_coords):
            if (x, y) not in occupied_cells:
                occupied_cells[(x, y)] = []
            occupied_cells[(x, y)].append(i)

        # Create connections between nearby occupied cells
        for (x1, y1), indices1 in occupied_cells.items():
            # Draw lines to neighboring cells (max distance of 2 cells)
            for (x2, y2), indices2 in occupied_cells.items():
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if 0 < dist <= 2:  # Only connect nearby cells, excluding self
                    ax.plot([x1, x2], [y1, y2], color='white', alpha=0.15,
                            linewidth=0.5, zorder=1)

    # Plot each image at its SOM coordinates
    print("Plotting images...")
    for img_path, (x, y) in tqdm(zip(image_paths, bmu_coords), total=len(image_paths)):
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

            # Calculate image extent based on grid cell size
            half_size = image_size / 2
            extent = [x - half_size, x + half_size, y - half_size, y + half_size]

            # Plot the image
            ax.imshow(img_array, extent=extent, interpolation='nearest', zorder=2)

            # Add a small white dot at the center of each star for constellation effect
            ax.scatter(x, y, s=2, color='white', alpha=0.8, zorder=3)

        except Exception as e:
            print(f"Error plotting {img_path}: {e}")

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Add grid lines for constellation chart effect
    if grid_size[0] * grid_size[1] < 1000:  # Only for smaller grids
        ax.grid(True, color='white', alpha=0.05, linestyle='-', linewidth=0.5)

    # Add a title with constellation-like name
    plt.title("Star Constellation Chart", color='white', fontsize=24, pad=20)

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

    # 2. Create SOM mapping
    # Calculate appropriate grid size based on number of images
    grid_side = max(10, int(np.sqrt(total_images * 3)))  # Make grid larger than sqrt(n) for spacing
    grid_size = (grid_side, grid_side)

    # Create SOM
    bmu_coords, final_grid_size, som = create_som_mapping(pca_reduced_data, grid_size)

    # 3. Plot images using SOM coordinates
    figsize = min(40, max(20, total_images // 8))

    fig, ax = plot_som_constellation(
        photo_files,
        bmu_coords,
        final_grid_size,
        figsize=(figsize, figsize),
        background_color='black',
        add_connections=True  # Add constellation-like connections
    )

    # Save the visualization
    output_path = Path('../../output')
    output_path.mkdir(exist_ok=True, parents=True)
    output_path = output_path / photo_dir.name
    output_path.mkdir(exist_ok=True, parents=True)

    plt.savefig(output_path / 'som_constellation.png', dpi=300, bbox_inches='tight')
    # Also save a higher resolution version
    plt.savefig(output_path / 'som_constellation_hires.png', dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to {output_path / 'som_constellation.png'}")
    print(f"High-resolution version saved to {output_path / 'som_constellation_hires.png'}")

    # Create a heatmap version showing the density of images per cell (optional)
    plt.figure(figsize=(15, 15), facecolor='black')
    plt.set_cmap('viridis')

    # Count images per cell
    activity_map = np.zeros((final_grid_size[0], final_grid_size[1]))
    for x, y in bmu_coords:
        activity_map[int(x), int(y)] += 1

    # Create heatmap
    plt.imshow(activity_map.T, origin='lower', interpolation='gaussian')
    plt.colorbar(label='Number of Stars')
    plt.title('SOM Cell Activation Density', color='white', fontsize=20)
    plt.tight_layout()

    # Save the heatmap
    plt.savefig(output_path / 'som_density.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

    print(f"Density map saved to {output_path / 'som_density.png'}")