from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndimage
from multiprocessing import Pool
from scipy.spatial.distance import cdist
import random

# Function to compute color features: mean, variance, and entropy for each RGB channel
def compute_color_features(img):
    """
    Compute color statistics for an RGB image.

    Parameters:
        img (PIL.Image): Input image

    Returns:
        tuple: (mean_r, mean_g, mean_b, var_r, var_g, var_b, entropy_r, entropy_g, entropy_b)
    """
    img_array = np.array(img)
    if len(img_array.shape) == 2:  # Grayscale image
        r = g = b = img_array
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # RGB image
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]
    else:
        raise ValueError("Unsupported image format")

    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)

    var_r = np.var(r)
    var_g = np.var(g)
    var_b = np.var(b)

    hist_r, _ = np.histogram(r, bins=256, range=(0, 256))
    hist_g, _ = np.histogram(g, bins=256, range=(0, 256))
    hist_b, _ = np.histogram(b, bins=256, range=(0, 256))

    prob_r = hist_r / np.sum(hist_r)
    prob_g = hist_g / np.sum(hist_g)
    prob_b = hist_b / np.sum(hist_b)

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
    img_gray = img.convert('L')
    img_gray_array = np.array(img_gray)
    y, x = np.mgrid[:img_gray_array.shape[0], :img_gray_array.shape[1]]

    M00 = np.sum(img_gray_array)
    if M00 == 0:
        centroid_x = 0
        centroid_y = 0
        orientation = 0
        eccentricity = 0
    else:
        M10 = np.sum(x * img_gray_array)
        M01 = np.sum(y * img_gray_array)
        centroid_x = M10 / M00
        centroid_y = M01 / M00

        mu20 = np.sum((x - centroid_x) ** 2 * img_gray_array)
        mu02 = np.sum((y - centroid_y) ** 2 * img_gray_array)
        mu11 = np.sum((x - centroid_x) * (y - centroid_y) * img_gray_array)

        orientation = 0.5 * np.arctan2(2 * mu11, mu20 - mu02) if mu20 - mu02 != 0 else 0

        cov = np.array([[mu20, mu11], [mu11, mu02]]) / M00
        eigenvalues = np.linalg.eigvals(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eccentricity = np.sqrt(1 - (eigenvalues[1] / eigenvalues[0])) if eigenvalues[0] > 0 and eigenvalues[1] >= 0 else 0

    grad_x = ndimage.sobel(img_gray_array, axis=1)
    grad_y = ndimage.sobel(img_gray_array, axis=0)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    threshold = 0.1 * np.max(grad_magnitude)
    edge_density = np.mean(grad_magnitude > threshold)

    return centroid_x, centroid_y, orientation, eccentricity, edge_density

# Function to process a single image and compute all features
def process_image(path_image):
    """
    Process an image file and compute its color and shape features.

    Parameters:
        path_image (Path): Path to the image file

    Returns:
        dict: Dictionary containing the image path and all computed features
    """
    try:
        img = Image.open(path_image)
        mean_r, mean_g, mean_b, var_r, var_g, var_b, entropy_r, entropy_g, entropy_b = compute_color_features(img)
        cx, cy, orient, ecc, ed = compute_shape_features(img)
        return {
            'path': str(path_image),
            'mean_r': mean_r, 'mean_g': mean_g, 'mean_b': mean_b,
            'var_r': var_r, 'var_g': var_g, 'var_b': var_b,
            'entropy_r': entropy_r, 'entropy_g': entropy_g, 'entropy_b': entropy_b,
            'centroid_x': cx, 'centroid_y': cy, 'orientation': orient,
            'eccentricity': ecc, 'edge_density': ed
        }
    except Exception as e:
        print(f"Error processing {path_image}: {e}")
        return {
            'path': str(path_image),
            'mean_r': np.nan, 'mean_g': np.nan, 'mean_b': np.nan,
            'var_r': np.nan, 'var_g': np.nan, 'var_b': np.nan,
            'entropy_r': np.nan, 'entropy_g': np.nan, 'entropy_b': np.nan,
            'centroid_x': np.nan, 'centroid_y': np.nan, 'orientation': np.nan,
            'eccentricity': np.nan, 'edge_density': np.nan
        }

# Function to process a single patch
def process_patch(args):
    """
    Process a patch from the reference image and compute its features.

    Parameters:
        args (tuple): (path_image, box, i, j) where box is (left, upper, right, lower)

    Returns:
        dict: Dictionary containing the patch index and computed features
    """
    path_image, box, i, j = args
    try:
        img = Image.open(path_image)
        patch = img.crop(box)
        mean_r, mean_g, mean_b, var_r, var_g, var_b, entropy_r, entropy_g, entropy_b = compute_color_features(patch)
        cx, cy, orient, ecc, ed = compute_shape_features(patch)
        return {
            'path': str(path_image),
            'patch_row': i, 'patch_col': j,
            'mean_r': mean_r, 'mean_g': mean_g, 'mean_b': mean_b,
            'var_r': var_r, 'var_g': var_g, 'var_b': var_b,
            'entropy_r': entropy_r, 'entropy_g': entropy_g, 'entropy_b': entropy_b,
            'centroid_x': cx, 'centroid_y': cy, 'orientation': orient,
            'eccentricity': ecc, 'edge_density': ed
        }
    except Exception as e:
        print(f"Error processing patch ({i}, {j}) from {path_image}: {e}")
        return {
            'path': str(path_image),
            'patch_row': i, 'patch_col': j,
            'mean_r': np.nan, 'mean_g': np.nan, 'mean_b': np.nan,
            'var_r': np.nan, 'var_g': np.nan, 'var_b': np.nan,
            'entropy_r': np.nan, 'entropy_g': np.nan, 'entropy_b': np.nan,
            'centroid_x': np.nan, 'centroid_y': np.nan, 'orientation': np.nan,
            'eccentricity': np.nan, 'edge_density': np.nan
        }

# Function to compute Euclidean distance
def compute_euclidean_distance(patch_features, image_features):
    """
    Compute Euclidean distance between a patch's features and all image features.

    Parameters:
        patch_features (np.array): Features of the patch
        image_features (np.array): Features of all images in the dataset

    Returns:
        np.array: Euclidean distances
    """
    return cdist([patch_features], image_features, 'euclidean')[0]

# Function to find top N similar images
def find_top_n_images(patch_features, image_features, N=10):
    """
    Find the indices of the N most similar images based on Euclidean distance.

    Parameters:
        patch_features (np.array): Features of the patch
        image_features (np.array): Features of all images in the dataset
        N (int): Number of top matches to return

    Returns:
        list: Indices of the top N images
    """
    distances = compute_euclidean_distance(patch_features, image_features)
    return np.argsort(distances)[:N].tolist()

# Function to select a random image from top N
def select_random_image(top_n_indices):
    """
    Randomly select one image from the top N indices.

    Parameters:
        top_n_indices (list): Indices of the top N images

    Returns:
        int: Index of the selected image
    """
    return random.choice(top_n_indices)

# Function to construct the mosaic
def construct_mosaic(reference_features, image_features, image_paths, patch_size, output_path, weights):
    """
    Construct the mosaic image by replacing each patch with a selected image, using weighted features.

    Parameters:
        reference_features (pd.DataFrame): Features of the reference image patches
        image_features (pd.DataFrame): Features of the dataset images
        image_paths (list): Paths to the dataset images
        patch_size (tuple): Size of each patch (width, height)
        output_path (Path): Path to save the mosaic image
        weights (list): Weights for each feature in feature_columns
    """
    rows = reference_features['patch_row'].unique()
    cols = reference_features['patch_col'].unique()
    mosaic_width = len(cols) * patch_size[0]
    mosaic_height = len(rows) * patch_size[1]
    mosaic = Image.new('RGB', (mosaic_width, mosaic_height))

    feature_columns = ['mean_r', 'mean_g', 'mean_b', 'var_r', 'var_g', 'var_b',
                       'entropy_r', 'entropy_g', 'entropy_b', 'centroid_x',
                       'centroid_y', 'orientation', 'eccentricity', 'edge_density']

    if len(weights) != len(feature_columns):
        raise ValueError(f"Weights list must have {len(feature_columns)} elements, got {len(weights)}.")

    ref_features_array = reference_features[feature_columns].values
    img_features_array = image_features[feature_columns].values
    sqrt_weights = np.sqrt(weights)
    scaled_ref_features = ref_features_array * sqrt_weights
    scaled_img_features = img_features_array * sqrt_weights

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            patch_idx = (reference_features['patch_row'] == row) & (reference_features['patch_col'] == col)
            scaled_patch_features = scaled_ref_features[patch_idx][0]
            top_n_indices = find_top_n_images(scaled_patch_features, scaled_img_features, N=10)
            selected_index = select_random_image(top_n_indices)
            selected_image = Image.open(image_paths[selected_index]).resize(patch_size)
            x = j * patch_size[0]
            y = i * patch_size[1]
            mosaic.paste(selected_image, (x, y))

    mosaic.save(output_path)
    print(f"Mosaic image saved to {output_path}")

if __name__ == '__main__':
    # Define paths
    photo_dir = Path('../../photos/SORTED_STARS')
    path_reference_image = Path('../../reference_images/star_on_clump_weight.jpg')
    output_dir = Path('../../output')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define patch size
    patch_size = (50, 50)
    num_processes = 4

    # Step 1: Process dataset images
    photo_files = list(photo_dir.glob('**/*.jpg'))
    print(f'Found {len(photo_files)} images')
    with Pool(num_processes) as pool:
        image_results = pool.map(process_image, photo_files)
    image_features = pd.DataFrame(image_results)
    path_image_features = photo_dir / 'image_features.csv'
    image_features.to_csv(path_image_features, index=False)
    print(f"Image features saved to {path_image_features}")

    # Step 2: Process reference image patches
    reference_img = Image.open(path_reference_image)
    width, height = reference_img.size
    patch_width, patch_height = patch_size
    num_patches_x = width // patch_width
    num_patches_y = height // patch_height
    if width % patch_width != 0 or height % patch_height != 0:
        print("Warning: Image size is not a multiple of patch size. Some pixels will be ignored.")
    total_patches = num_patches_x * num_patches_y
    print(f'Processing {total_patches} patches from {path_reference_image}')

    args_list = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            left = j * patch_width
            upper = i * patch_height
            right = left + patch_width
            lower = upper + patch_height
            box = (left, upper, right, lower)
            args_list.append((str(path_reference_image), box, i, j))

    with Pool(num_processes) as pool:
        patch_results = pool.map(process_patch, args_list)
    reference_features = pd.DataFrame(patch_results)
    path_reference_features = path_reference_image.parent / (path_reference_image.stem + '_patch_features.csv')
    reference_features.to_csv(path_reference_features, index=False)
    print(f"Reference patch features saved to {path_reference_features}")

    # Step 3: Construct the mosaic
    # Define weights for different features
    # Order matches feature_columns: mean_r, mean_g, mean_b, var_r, var_g, var_b,
    # entropy_r, entropy_g, entropy_b, centroid_x, centroid_y, orientation, eccentricity, edge_density
    image_paths = image_features['path'].tolist()
    weights = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
    output_mosaic_dir = output_dir / path_reference_image.stem
    output_mosaic_dir.mkdir(parents=True, exist_ok=True)
    output_mosaic_path = output_mosaic_dir / (path_reference_image.stem + '_mosaic.jpg')
    construct_mosaic(reference_features, image_features, image_paths, patch_size, output_mosaic_path, weights)