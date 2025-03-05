from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist
import random
from multiprocessing import Pool, cpu_count
from functools import partial
import os
from concurrent.futures import ThreadPoolExecutor
import time

def compute_euclidean_distance_batch(scaled_patch_features, scaled_img_features, n_top=10):
    """
    Compute Euclidean distance between multiple patches and all image features in a batch.

    Parameters:
        scaled_patch_features (np.array): Scaled features of multiple patches
        scaled_img_features (np.array): Scaled features of all images in the dataset
        n_top (int): Number of top matches to return for each patch

    Returns:
        list: List of arrays containing the indices of the top N images for each patch
    """
    distances = cdist(scaled_patch_features, scaled_img_features, 'euclidean')
    top_n_indices = np.argsort(distances, axis=1)[:, :n_top]
    return top_n_indices

def process_patch_group(group_data, scaled_img_features, image_paths, patch_size, image_cache, n_top, feature_columns, sqrt_weights):
    """
    Process a group of patches in parallel.

    Parameters:
        group_data (pd.DataFrame): DataFrame containing patch features and positions
        scaled_img_features (np.array): Scaled features of all images in the dataset
        image_paths (list): List of image paths
        patch_size (tuple): Size of each patch (width, height)
        image_cache (dict): Cache of preloaded images
        n_top (int): Number of top matches to consider
        feature_columns (list): List of feature column names
        sqrt_weights (np.array): Square root of weights for scaling features

    Returns:
        list: List of (position, image) tuples for placement in the mosaic
    """
    results = []
    # Extract positions from 'patch_row' and 'patch_col'
    positions = [(int(row), int(col)) for row, col in zip(group_data['patch_row'], group_data['patch_col'])]
    # Select only the feature columns and scale them
    patch_features = group_data[feature_columns].values
    scaled_patch_features = patch_features * sqrt_weights
    # Compute distances and get top N indices
    top_n_indices_batch = compute_euclidean_distance_batch(scaled_patch_features, scaled_img_features, n_top=n_top)
    for idx, (row, col) in enumerate(positions):
        top_n_indices = top_n_indices_batch[idx]
        selected_index = random.choice(top_n_indices)
        selected_image_path = image_paths[selected_index]
        x = col * patch_size[0]
        y = row * patch_size[1]
        # Use cached image if available, otherwise load and resize
        if selected_image_path in image_cache:
            selected_image = image_cache[selected_image_path]
        else:
            selected_image = Image.open(selected_image_path).resize(patch_size)
            image_cache[selected_image_path] = selected_image
        results.append(((x, y), selected_image))
    return results

def load_and_resize_images(image_paths, patch_size, max_cache_size=200):
    """
    Preload and resize the most frequently used images.

    Parameters:
        image_paths (list): List of all image paths
        patch_size (tuple): Size to resize images to
        max_cache_size (int): Maximum number of images to cache

    Returns:
        dict: Dictionary mapping image paths to resized images
    """
    cache_paths = random.sample(image_paths, min(max_cache_size, len(image_paths)))
    image_cache = {}
    print(f"Preloading {len(cache_paths)} images...")
    def load_and_resize(path):
        try:
            return path, Image.open(path).resize(patch_size)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return path, None
    with ThreadPoolExecutor(max_workers=16) as executor:
        for path, img in executor.map(load_and_resize, cache_paths):
            if img is not None:
                image_cache[path] = img
    print(f"Preloaded {len(image_cache)} images.")
    return image_cache

def construct_mosaic_optimized(reference_features, image_features, image_paths, patch_size, output_path, weights, n_top):
    """
    Construct the mosaic image by replacing each patch with a selected image, using optimized processing.

    Parameters:
        reference_features (pd.DataFrame): Features of the reference image patches
        image_features (pd.DataFrame): Features of the dataset images
        image_paths (list): Paths to the dataset images
        patch_size (tuple): Size of each patch (width, height)
        output_path (str): Path to save the mosaic image
        weights (list): Weights for each feature
        n_top (int): Number of top matches to consider for each patch
    """
    start_time = time.time()
    # Calculate mosaic dimensions
    rows = reference_features['patch_row'].unique()
    cols = reference_features['patch_col'].unique()
    mosaic_width = len(cols) * patch_size[0]
    mosaic_height = len(rows) * patch_size[1]
    mosaic = Image.new('RGB', (mosaic_width, mosaic_height))

    # Define feature columns
    feature_columns = ['mean_r', 'mean_g', 'mean_b', 'var_r', 'var_g', 'var_b',
                       'entropy_r', 'entropy_g', 'entropy_b', 'centroid_x',
                       'centroid_y', 'orientation', 'eccentricity', 'edge_density']
    if len(weights) != len(feature_columns):
        raise ValueError(f"Weights list must have {len(feature_columns)} elements, matching the number of features.")

    # Compute scaled image features
    sqrt_weights = np.sqrt(weights)
    img_features_array = image_features[feature_columns].values
    scaled_img_features = img_features_array * sqrt_weights

    # Preload images into cache
    image_cache = load_and_resize_images(image_paths, patch_size)
    print("Starting mosaic construction...")

    # Set up multiprocessing
    num_workers = min(cpu_count(), 8)
    total_patches = len(reference_features)
    batch_size = max(1, total_patches // (num_workers * 2))

    # Group reference features into batches
    patch_groups = []
    for i in range(0, total_patches, batch_size):
        patch_groups.append(reference_features.iloc[i:i + batch_size])

    # Create partially applied function with fixed arguments
    process_func = partial(process_patch_group,
                           scaled_img_features=scaled_img_features,
                           image_paths=image_paths,
                           patch_size=patch_size,
                           image_cache=image_cache,
                           n_top=n_top,
                           feature_columns=feature_columns,
                           sqrt_weights=sqrt_weights)

    # Process patches in parallel
    with Pool(num_workers) as pool:
        all_results = pool.map(process_func, patch_groups)

    # Assemble the mosaic
    for batch_results in all_results:
        for (x, y), img in batch_results:
            mosaic.paste(img, (x, y))

    # Save the mosaic image
    mosaic.save(output_path)
    elapsed_time = time.time() - start_time
    print(f"Mosaic image saved to {output_path} (Time: {elapsed_time:.2f} seconds)")

def main():
    # Define paths and parameters
    path_image_features = Path("../../photos/SORTED_STARS/image_features.csv")
    image_features = pd.read_csv(path_image_features)
    path_reference_features = Path('../../reference_images/madreporite_patch_features.csv')
    reference_features = pd.read_csv(path_reference_features)
    image_paths = image_features['path'].tolist()
    patch_size = (50,50)
    n_top = 20
    output_path = Path('../../output')
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / path_reference_features.stem.split('_patch')[0]
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / (path_reference_features.name.split('.csv')[0] + '_mosaic.jpg')
    # leave for user to define weights
    # (mean_r, mean_g, mean_b, var_r, var_g, var_b, entropy_r, entropy_g, entropy_b,
    # centroid_x, centroid_y, orientation, eccentricity, edge_density)
    weights = [5.0, 5.0, 5.0, # color
               0,0,0, # color variance (seem to ruin it)
               .5,.5,.5, # entropy
               .5, .5, # centroid
               1, # orientation
               1, # eccentricity
               1, # edge density
               ]

    # Construct the mosaic
    construct_mosaic_optimized(reference_features, image_features, image_paths, patch_size, output_path, weights, n_top)

if __name__ == '__main__':
    main()