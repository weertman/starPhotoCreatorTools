from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndimage
from multiprocessing import Pool, cpu_count
import os
from functools import partial


# Function to compute color features: mean, variance, and entropy for each RGB channel
def compute_color_features(img_array):
    """
    Compute color statistics for an RGB image array.

    Parameters:
        img_array (numpy.ndarray): Input image array

    Returns:
        tuple: (mean_r, mean_g, mean_b, var_r, var_g, var_b, entropy_r, entropy_g, entropy_b)
    """
    # Handle both RGB and grayscale images
    if len(img_array.shape) == 2:  # Grayscale image
        r = g = b = img_array
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # RGB image
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    else:
        raise ValueError("Unsupported image format")

    # Compute mean and variance for each channel (vectorized)
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
    var_r, var_g, var_b = np.var(r), np.var(g), np.var(b)

    # Compute entropy for each channel using histograms (vectorized)
    def calc_entropy(channel):
        hist, _ = np.histogram(channel, bins=256, range=(0, 256), density=True)
        # Remove zero probabilities to avoid log(0)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    entropy_r = calc_entropy(r)
    entropy_g = calc_entropy(g)
    entropy_b = calc_entropy(b)

    return mean_r, mean_g, mean_b, var_r, var_g, var_b, entropy_r, entropy_g, entropy_b


# Function to compute shape features: centroid, orientation, eccentricity, edge density
def compute_shape_features(img_array):
    """
    Compute shape features from the grayscale version of an image array.

    Parameters:
        img_array (numpy.ndarray): Input image array

    Returns:
        tuple: (centroid_x, centroid_y, orientation, eccentricity, edge_density)
    """
    # Convert to grayscale if RGB
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Fast grayscale conversion using weighted average
        img_gray_array = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
    else:
        img_gray_array = img_array

    # Total intensity
    M00 = np.sum(img_gray_array)

    if M00 == 0:  # Handle case of zero intensity (e.g., black image)
        return 0, 0, 0, 0, 0

    # Create coordinate grids (y for rows, x for columns)
    y_indices, x_indices = np.indices(img_gray_array.shape)

    # Compute centroids (vectorized)
    M10 = np.sum(x_indices * img_gray_array)
    M01 = np.sum(y_indices * img_gray_array)
    centroid_x = M10 / M00
    centroid_y = M01 / M00

    # Compute central moments (vectorized)
    x_centered = x_indices - centroid_x
    y_centered = y_indices - centroid_y
    mu20 = np.sum(x_centered ** 2 * img_gray_array)
    mu02 = np.sum(y_centered ** 2 * img_gray_array)
    mu11 = np.sum(x_centered * y_centered * img_gray_array)

    # Compute orientation
    if mu20 - mu02 != 0:
        orientation = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    else:
        orientation = 0

    # Compute eccentricity using covariance matrix
    cov = np.array([[mu20, mu11], [mu11, mu02]]) / M00
    try:
        eigenvalues = np.linalg.eigvals(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Largest eigenvalue first
        if eigenvalues[0] > 0 and eigenvalues[1] >= 0:
            eccentricity = np.sqrt(1 - (eigenvalues[1] / eigenvalues[0]))
        else:
            eccentricity = 0
    except:
        eccentricity = 0

    # Compute edge density using Sobel filters (vectorized)
    grad_x = ndimage.sobel(img_gray_array, axis=1)
    grad_y = ndimage.sobel(img_gray_array, axis=0)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    threshold = 0.1 * np.max(grad_magnitude)
    edge_density = np.mean(grad_magnitude > threshold)

    return centroid_x, centroid_y, orientation, eccentricity, edge_density


# Function to process a single patch using pre-loaded image array
def process_patch(full_img_array, patch_info):
    """
    Process a patch from the pre-loaded image array and compute its features.

    Parameters:
        full_img_array (numpy.ndarray): Full image as numpy array
        patch_info (tuple): (i, j, left, upper, right, lower) patch coordinates

    Returns:
        dict: Dictionary containing the patch index and computed features
    """
    i, j, left, upper, right, lower = patch_info

    try:
        # Extract patch from the full image array
        patch_array = full_img_array[upper:lower, left:right]

        # Compute color features
        mean_r, mean_g, mean_b, var_r, var_g, var_b, entropy_r, entropy_g, entropy_b = compute_color_features(
            patch_array)

        # Compute shape features
        cx, cy, orient, ecc, ed = compute_shape_features(patch_array)

        return {
            'patch_row': i,
            'patch_col': j,
            'mean_r': mean_r,
            'mean_g': mean_g,
            'mean_b': mean_b,
            'var_r': var_r,
            'var_g': var_g,
            'var_b': var_b,
            'entropy_r': entropy_r,
            'entropy_g': entropy_g,
            'entropy_b': entropy_b,
            'centroid_x': cx,
            'centroid_y': cy,
            'orientation': orient,
            'eccentricity': ecc,
            'edge_density': ed
        }
    except Exception as e:
        print(f"Error processing patch ({i}, {j}): {e}")
        return {
            'patch_row': i,
            'patch_col': j,
            'mean_r': np.nan,
            'mean_g': np.nan,
            'mean_b': np.nan,
            'var_r': np.nan,
            'var_g': np.nan,
            'var_b': np.nan,
            'entropy_r': np.nan,
            'entropy_g': np.nan,
            'entropy_b': np.nan,
            'centroid_x': np.nan,
            'centroid_y': np.nan,
            'orientation': np.nan,
            'eccentricity': np.nan,
            'edge_density': np.nan
        }


def main():
    # Define the path to the reference image
    path_reference_image = Path('../../reference_images/JDVANCELOL.jpeg')

    # Load the reference image once and convert to array
    reference_img = Image.open(path_reference_image)
    img_array = np.array(reference_img)

    # Define patch size
    patch_size = (10,10)
    patch_width, patch_height = patch_size

    # Get image dimensions
    width, height = reference_img.size

    # Calculate number of full patches along width and height
    num_patches_x = width // patch_width
    num_patches_y = height // patch_height

    # Check if image size is a multiple of patch size
    if width % patch_width != 0 or height % patch_height != 0:
        print("Warning: Image size is not a multiple of patch size. Some pixels will be ignored.")

    total_patches = num_patches_x * num_patches_y
    print(f'Processing {total_patches} patches from {path_reference_image}')

    # Generate list of patch information
    patches_info = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            left = j * patch_width
            upper = i * patch_height
            right = left + patch_width
            lower = upper + patch_height
            patches_info.append((i, j, left, upper, right, lower))

    # Use partial function to pass the image array to all workers
    process_patch_with_img = partial(process_patch, img_array)

    # Determine optimal number of processes
    num_processes = min(cpu_count(), 8)  # Limit to 8 cores max to prevent overhead

    # Use multiprocessing Pool to compute features in parallel
    with Pool(num_processes) as pool:
        results = pool.map(process_patch_with_img, patches_info)

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Add the image path column to maintain compatibility with original output
    df['path'] = str(path_reference_image)

    # Define output file name based on the reference image
    output_csv = path_reference_image.parent / (path_reference_image.stem + '_patch_features.csv')

    # Save DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f'Processing complete. Results saved to {output_csv}')


if __name__ == '__main__':
    main()