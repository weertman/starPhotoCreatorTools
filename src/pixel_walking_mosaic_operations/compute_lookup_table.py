from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndimage
from multiprocessing import Pool


# Function to compute color features: mean, variance, and entropy for each RGB channel
def compute_color_features(img, ignore_black=False):
    """
    Compute color statistics for an RGB image.

    Parameters:
        img (PIL.Image): Input image
        ignore_black (bool): If True, ignore pixels with RGB values (0,0,0) in calculations

    Returns:
        tuple: (mean_r, mean_g, mean_b, var_r, var_g, var_b, entropy_r, entropy_g, entropy_b)
    """
    img_array = np.array(img)
    # Handle both RGB and grayscale images
    if len(img_array.shape) == 2:  # Grayscale image
        r = g = b = img_array
        if ignore_black:
            # Create mask for non-black pixels in grayscale
            mask = r > 0
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # RGB image
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]
        if ignore_black:
            # Create mask for non-black pixels (where any channel > 0)
            mask = (r > 0) | (g > 0) | (b > 0)
    else:
        raise ValueError("Unsupported image format")

    # Compute mean for each channel
    if ignore_black:
        # Use masked arrays to ignore black pixels
        if np.any(mask):  # Check if there are any non-black pixels
            mean_r = np.mean(r[mask])
            mean_g = np.mean(g[mask])
            mean_b = np.mean(b[mask])
            # Compute variance for each channel
            var_r = np.var(r[mask])
            var_g = np.var(g[mask])
            var_b = np.var(b[mask])
        else:  # If all pixels are black
            mean_r = mean_g = mean_b = 0
            var_r = var_g = var_b = 0
    else:
        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)
        # Compute variance for each channel
        var_r = np.var(r)
        var_g = np.var(g)
        var_b = np.var(b)

    # Compute entropy for each channel using histogram
    hist_r, _ = np.histogram(r, bins=256, range=(0, 256))
    hist_g, _ = np.histogram(g, bins=256, range=(0, 256))
    hist_b, _ = np.histogram(b, bins=256, range=(0, 256))

    # Normalize histograms to get probabilities
    prob_r = hist_r / np.sum(hist_r)
    prob_g = hist_g / np.sum(hist_g)
    prob_b = hist_b / np.sum(hist_b)

    # Compute entropy, handling zero probabilities (0 * log(0) = 0)
    entropy_r = -np.sum(prob_r[prob_r > 0] * np.log2(prob_r[prob_r > 0]))
    entropy_g = -np.sum(prob_g[prob_g > 0] * np.log2(prob_g[prob_g > 0]))
    entropy_b = -np.sum(prob_b[prob_b > 0] * np.log2(prob_b[prob_b > 0]))

    return mean_r, mean_g, mean_b, var_r, var_g, var_b, entropy_r, entropy_g, entropy_b


# Function to compute shape features: centroid, orientation, eccentricity, edge density
def compute_shape_features(img):
    """
    Compute shape features from the grayscale version of an image.

    Parameters:
        img (PIL.Image): Input image

    Returns:
        tuple: (centroid_x, centroid_y, orientation, eccentricity, edge_density)
    """
    # Convert to grayscale
    img_gray = img.convert('L')
    img_gray_array = np.array(img_gray)

    # Create coordinate grids (y for rows, x for columns)
    y, x = np.mgrid[:img_gray_array.shape[0], :img_gray_array.shape[1]]

    # Compute raw image moments
    M00 = np.sum(img_gray_array)  # Total intensity
    if M00 == 0:  # Handle case of zero intensity (e.g., black image)
        centroid_x = 0
        centroid_y = 0
        orientation = 0
        eccentricity = 0
    else:
        M10 = np.sum(x * img_gray_array)  # Sum of x * intensity
        M01 = np.sum(y * img_gray_array)  # Sum of y * intensity
        centroid_x = M10 / M00
        centroid_y = M01 / M00

        # Compute central moments
        mu20 = np.sum((x - centroid_x) ** 2 * img_gray_array)
        mu02 = np.sum((y - centroid_y) ** 2 * img_gray_array)
        mu11 = np.sum((x - centroid_x) * (y - centroid_y) * img_gray_array)

        # Compute orientation (angle in radians)
        if mu20 - mu02 != 0:
            orientation = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        else:
            orientation = 0  # Default if no variance difference

        # Compute eccentricity using covariance matrix
        cov = np.array([[mu20, mu11], [mu11, mu02]]) / M00
        eigenvalues = np.linalg.eigvals(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Largest eigenvalue first
        if eigenvalues[0] > 0 and eigenvalues[1] >= 0:
            eccentricity = np.sqrt(1 - (eigenvalues[1] / eigenvalues[0]))
        else:
            eccentricity = 0  # Default if computation fails

    # Compute edge density using Sobel filters
    grad_x = ndimage.sobel(img_gray_array, axis=1)
    grad_y = ndimage.sobel(img_gray_array, axis=0)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # Adaptive threshold: 10% of max gradient
    threshold = 0.1 * np.max(grad_magnitude)
    edge_density = np.mean(grad_magnitude > threshold)

    return centroid_x, centroid_y, orientation, eccentricity, edge_density


# Function to process a single image and compute all features
def process_image(path_image, ignore_black=False):
    """
    Process an image file and compute its color and shape features.

    Parameters:
        path_image (Path): Path to the image file
        ignore_black (bool): If True, ignore pixels with RGB values (0,0,0) in color calculations

    Returns:
        dict: Dictionary containing the image path and all computed features
    """
    try:
        # Open the image
        img = Image.open(path_image)

        # Compute color features with ignore_black option
        mean_r, mean_g, mean_b, var_r, var_g, var_b, entropy_r, entropy_g, entropy_b = compute_color_features(img,
                                                                                                              ignore_black=ignore_black)

        # Compute shape features
        cx, cy, orient, ecc, ed = compute_shape_features(img)

        # Return a dictionary of features
        return {
            'path': str(path_image),
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
        print(f"Error processing {path_image}: {e}")
        # Return NaN values if processing fails
        return {
            'path': str(path_image),
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


if __name__ == '__main__':
    # Define the directory containing the images
    photo_dir = Path('../../photos/SORTED_STARS')
    # Find all .jpg files recursively in the directory and subdirectories
    photo_files = list(photo_dir.glob('**/*.jpg'))
    print(f'Found {len(photo_files)} images')

    # Set the number of processes to 4
    num_processes = 4

    # Set whether to ignore black pixels in color calculations (True/False)
    ignore_black_pixels = True

    # Create a partial function with the ignore_black parameter
    from functools import partial

    process_func = partial(process_image, ignore_black=ignore_black_pixels)

    # Create a pool of 4 workers and process images in parallel
    with Pool(num_processes) as pool:
        results = pool.map(process_func, photo_files)

    # Convert the list of results into a DataFrame
    df = pd.DataFrame(results)

    path_df = photo_dir / 'image_features.csv'
    # Save the DataFrame to a CSV file
    df.to_csv(path_df, index=False)
    print("Processing complete. Results saved to 'image_features.csv'.")