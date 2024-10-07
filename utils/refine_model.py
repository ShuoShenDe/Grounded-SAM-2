import glob
import shutil
from PIL import Image
import os
import numpy as np
from utils.config import Config
from utils.mask_dictionary_model import MaskDictionaryModel
from utils.common_utils import CommonUtils

def split_image_to_folders(image_path, output_dir, tile_size, overlap_ratio_wh=(0.1, 0.1)):
    """
    Splits an image into smaller tiles with a specified overlap ratio and saves each tile in a separate folder.

    Parameters:
    - image_path: str, path to the input image.
    - output_dir: str, base directory where the tiles will be saved.
    - tile_size: tuple, (width, height) of each tile.
    - overlap_ratio_wh: tuple, (width_overlap_ratio, height_overlap_ratio) as a percentage (0 to 1).

    Returns:
    None
    """
    # Open the image
    image = Image.open(image_path)
    img_width, img_height = image.size
    image_basename = os.path.basename(image_path).split('.')[0]
    # Calculate overlap in pixels
    overlap_width = int(tile_size[0] * overlap_ratio_wh[0])
    overlap_height = int(tile_size[1] * overlap_ratio_wh[1])

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate the number of tiles in each dimension
    x_steps = (img_width - overlap_width) // (tile_size[0] - overlap_width)
    y_steps = (img_height - overlap_height) // (tile_size[1] - overlap_height)
    result_dir = []
    # Loop through the image to create tiles
    for i in range(x_steps + 1):
        for j in range(y_steps + 1):
            left = i * (tile_size[0] - overlap_width)
            upper = j * (tile_size[1] - overlap_height)
            right = min(left + tile_size[0], img_width)
            lower = min(upper + tile_size[1], img_height)

            # Crop the image
            # print(left, upper, right, lower)
            tile = image.crop((left, upper, right, lower))

            # Create a subfolder for each tile
            tile_folder = os.path.join(output_dir, f"refined_data_{i}_{j}/raw_data")
            result_dir.append(tile_folder)
            if not os.path.exists(tile_folder):
                os.makedirs(tile_folder)
            
            # Save the tile
            tile_filename = f"{image_basename}_{i}_{j}.png"
            tile_path = os.path.join(tile_folder, tile_filename)
            tile.save(tile_path)

    return result_dir


def get_target_file_by_base_name(base_name, folder_path, data_type):
    pattern = os.path.join(folder_path, f"{base_name}_*.{data_type}")
    filename = glob.glob(pattern)
    if len(filename) == 1:
        return filename[0]
    else:
        raise ValueError(f"Expected 1 file matching pattern {pattern}, but found {len(filename)} files.")

def get_majority_class(mask):
    unique, counts = np.unique(mask, return_counts=True)
    max_index = np.argmax(counts)
    return int(unique[max_index]), counts[max_index]

def paste_tile_to_full_image(full_size_mask, mask_tile, left, upper):
    # print("full_size_mask", full_size_mask.shape, "mask_tile", mask_tile.shape)
    
    full_size_mask = full_size_mask.astype(np.uint8)
    mask_tile = mask_tile.astype(np.uint8)
    if len(mask_tile.shape) == 0:
        return full_size_mask
    full_size_mask = Image.fromarray(full_size_mask)
    mask_tile = Image.fromarray(mask_tile)
    
    full_size_mask.paste(mask_tile, (left, upper))
    return np.array(full_size_mask)

def merge_slices_from_folders(base_dir, output_dir, tile_size, overlap_ratio_wh=(0.1, 0.1), target_size=(1080,1920)):
    """
    Merges image slices from separate folders back into a single image based on filename format 'basename_i_j.png'.

    Parameters:
    - base_dir: str, base directory containing folders with image tiles.
    - output_image_path: str, path where the merged image will be saved.
    - tile_size: tuple, (width, height) of each tile.
    - overlap_ratio_wh: tuple, (width_overlap_ratio, height_overlap_ratio) as a percentage (0 to 1).

    Returns:
    None
    """
    # List all folders in the base directory
    tile_folders = sorted([folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))])
    overlap_width = int(tile_size[0] * overlap_ratio_wh[0])
    overlap_height = int(tile_size[1] * overlap_ratio_wh[1])
    
    base_names = [name.split('_')[0] for name in os.listdir(os.path.join(base_dir, tile_folders[0], "raw_data"))]
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    raw_data_dir = os.path.join(output_dir, "raw_data")
    result_dir = os.path.join(output_dir, "result")
    # Second pass to place tiles into the full image
    for base_name in base_names:
        merged_mask = np.zeros(target_size)
        merged_dict = MaskDictionaryModel(promote_type = Config.PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{base_name}.npy", mask_height = target_size[1], mask_width = target_size[0])
        base_class_number = 0

        for folder in tile_folders:
            full_size_mask = np.zeros(target_size)
            mask_data_folder = os.path.join(base_dir, folder, "mask_data")
            json_data_folder = os.path.join(base_dir, folder, "json_data")
            mask_data_path, mask_tile, json_data_path, tile_dict = CommonUtils.get_mask_and_json(mask_data_folder, json_data_folder , base_name)
            try:
                parts = os.path.basename(json_data_path).replace('.json', '').split('_')
                i, j = map(int, parts[-2:])  # Indices are the last two parts
            except ValueError:
                print(f"Skipping file {json_data_path} as it does not match expected naming convention.")
                continue
            # Calculate the position where the tile should be placed
            left = i * (tile_size[0] - overlap_width)
            upper = j * (tile_size[1] - overlap_height)

            # Paste the tile into the full image at the calculated position
            
            if left == 0 and upper == 0:
                merged_mask = paste_tile_to_full_image(merged_mask, mask_tile, left, upper)
                base_class_number = int(merged_mask.max()+1)
                merged_dict = tile_dict
            else:
                print("base_class_number", base_class_number)
                full_size_mask = paste_tile_to_full_image(full_size_mask, mask_tile, left, upper)
                unique_number = np.unique(mask_tile)
                for number in unique_number:
                    if number != 0:
                        single_obj_mask = (full_size_mask == number)
                        overlay_mask = merged_mask[single_obj_mask]
                        if overlay_mask.sum() <200:
                            target_class_number = number + base_class_number
                            merged_mask[single_obj_mask] = target_class_number
                            merged_dict.labels[int(target_class_number)] = tile_dict.labels[number]

                        else:
                            target_class_number, counts = get_majority_class(overlay_mask)
                            print("majority class", target_class_number, counts, number)
                            if target_class_number == 0 or counts < single_obj_mask.sum() * 0.5:
                                target_class_number = number + base_class_number
                                merged_mask[single_obj_mask] = target_class_number
                                merged_dict.labels[int(target_class_number)] = tile_dict.labels[number]
                            else:
                                merged_mask[single_obj_mask] = target_class_number
                                # merged_dict.labels[target_class_number] = tile_dict.labels[number]
                        print("target_class_number", target_class_number)     
                        merged_dict.labels[target_class_number].instance_id = int(target_class_number)
                        merged_dict.labels[target_class_number].x1 += left
                        merged_dict.labels[target_class_number].x2 += left
                        merged_dict.labels[target_class_number].y1 += upper
                        merged_dict.labels[target_class_number].y2 += upper

                base_class_number += unique_number.max()


        merged_mask_path = os.path.join(mask_data_dir, f"mask_{base_name}.npy")
        np.save(merged_mask_path, merged_mask)
        merged_dict.to_json(os.path.join(json_data_dir, f"mask_{base_name}.json"), save_labels_as_list=True)
        # Save the merged image 
        print(f"Merged image saved to {merged_mask_path}")
    CommonUtils.draw_masks_and_box_with_supervision(raw_data_dir, mask_data_dir, json_data_dir, result_dir)


def clip_images(image_path, output_dir, image_size, overlap_ratio):
    frame_names = [
            p for p in os.listdir(image_path)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
        ]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Removed existing directory {output_dir}")
    split_folder = []
    for frame_name in frame_names:
        split_folder = split_image_to_folders(os.path.join(image_path, frame_name), output_dir,image_size, overlap_ratio)    
    return split_folder

if __name__ == "__main__":
    # Split an image into tiles
    image_path = "/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front/"
    output_dir = "/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front/refined_raw_data"
    image_size = (800, 800)
    overlap_ratio = (0.5, 0.5) # column, row
    # clip_images(image_path, output_dir, image_size, overlap_ratio)
    merge_slices_from_folders(output_dir, image_path, image_size, overlap_ratio)

