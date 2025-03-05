import logging
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
import sys
import numpy as np


def process_image(image_path, model, target_dim):
    """
    Process an image by:
    1. Detecting objects using the YOLO model
    2. Finding the largest object
    3. Masking the image to keep only the largest object
    4. Cropping to the largest object's bounding box
    5. Making the image square
    6. Resizing to target dimensions
    """
    try:
        # Load the image
        img = Image.open(image_path)
        img_array = np.array(img)

        # Get original image dimensions
        orig_h, orig_w = img_array.shape[:2]

        # Run detection with retina_masks=True to get masks in original image space
        results = model(img_array, verbose=False, retina_masks=True)

        # Check if any objects were detected with masks
        if not hasattr(results[0], 'masks') or results[0].masks is None or len(results[0].masks) == 0:
            logging.warning(f"No objects with masks detected in {image_path}")
            return None

        # Find the largest mask by area
        masks_data = results[0].masks.data
        areas = [mask.sum().item() for mask in masks_data]
        largest_mask_idx = np.argmax(areas)

        # Get the mask in the original image coordinates
        mask = masks_data[largest_mask_idx].cpu().numpy()

        # If the mask shape doesn't match the original image shape, resize it
        if mask.shape != (orig_h, orig_w):
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
            mask = np.array(mask_img.resize((orig_w, orig_h), Image.NEAREST)) > 0

        # Get the bounding box coordinates
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            bbox = results[0].boxes.xyxy[largest_mask_idx].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)

            # Scale the bbox if needed
            model_h, model_w = results[0].masks.shape[1:]
            if model_w != orig_w or model_h != orig_h:
                x1 = int(x1 * orig_w / model_w)
                y1 = int(y1 * orig_h / model_h)
                x2 = int(x2 * orig_w / model_w)
                y2 = int(y2 * orig_h / model_h)
        else:
            # Compute bounding box from mask if not available
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                logging.warning(f"Empty mask for {image_path}")
                return None
            y1, y2 = np.min(y_indices), np.max(y_indices)
            x1, x2 = np.min(x_indices), np.max(x_indices)

        # Ensure the bounding box is within the image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(orig_w, x2)
        y2 = min(orig_h, y2)

        # Apply the mask to the image
        masked_img = img_array.copy()
        for c in range(min(3, masked_img.shape[2])):  # For RGB channels
            masked_img[:, :, c] = masked_img[:, :, c] * mask

        # Crop to bounding box
        cropped_img = masked_img[y1:y2, x1:x2]

        # Make square by cropping the longer dimension
        h, w = cropped_img.shape[:2]
        if h > w:
            # Height is larger, crop height
            diff = h - w
            top = diff // 2
            bottom = top + w
            # Check bounds
            if bottom > h:
                bottom = h
                top = max(0, h - w)
            cropped_img = cropped_img[top:bottom, :]
        elif w > h:
            # Width is larger, crop width
            diff = w - h
            left = diff // 2
            right = left + h
            # Check bounds
            if right > w:
                right = w
                left = max(0, w - h)
            cropped_img = cropped_img[:, left:right]

        # Convert back to PIL Image and resize
        pil_img = Image.fromarray(cropped_img)
        resized_img = pil_img.resize(target_dim, Image.LANCZOS)

        return resized_img

    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        return None


if __name__ == '__main__':
    path_model = Path('../../yolo_models/yolo11l-seg_starSeg.pt')
    model = YOLO(path_model)

    src_root_dir = Path(
        '/Users/wlweert/Downloads/SORTED_STARS')
    ## find paths to images for various file types at all depths
    src_images = list(src_root_dir.glob('**/*.jpg')) + list(src_root_dir.glob('**/*.jpeg')) + list(
        src_root_dir.glob('**/*.png')) + list(src_root_dir.glob('**/*.bmp')) + list(src_root_dir.glob('**/*.JPG')) + list(
        src_root_dir.glob('**/*.JPEG')) + list(src_root_dir.glob('**/*.PNG')) + list(src_root_dir.glob('**/*.BMP'))
    print(f'Found {len(src_images)} images')

    dst_root_dir = Path('../../photos')
    dst_root_dir.mkdir(exist_ok=True)
    dst_root_dir = dst_root_dir / src_root_dir.name
    dst_root_dir.mkdir(exist_ok=True)

    target_dim = (150, 150)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    target_ext = '.jpg'

    # Loop over all images
    n = 0
    for image_path in tqdm(src_images, desc="Processing images"):
        try:
            # Get the category directory name
            category_name = image_path.parent.name

            # Create destination directory for this category
            dst_dir = dst_root_dir / category_name
            dst_dir.mkdir(exist_ok=True)

            # Create destination path for this image
            dst_path = dst_dir / (image_path.stem + '_' + str(n) + target_ext)

            # Process the image
            processed_img = process_image(image_path, model, target_dim)

            # Save the processed image if successful
            if processed_img is not None:
                processed_img.save(dst_path)
            else:
                logging.warning(f"Skipping {image_path} due to processing failure")

            n += 1

        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            continue

    print("Processing complete")