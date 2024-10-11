import os
import time
import torch
import numpy as np
from post_process import PostProcess
from utils.config import Config
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.common_utils import CommonUtils
from utils.grounding_sam2_model import GroundingSAM2Model
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from utils.refine_model import split_image_to_folders, paste_tile_to_full_image, clip_images
import gc
from PIL import Image
from tqdm import tqdm

import json
import copy
import glob


def get_image_shape(image_path):
    images = os.listdir(image_path)
    image = Image.open(os.path.join(image_path, images[0]))
    print("image.size", image.size)
    print("image shape", np.array(image).shape[0:2])
    return image.size


def check_box_in_range(left, upper, right, lower, x1, y1, x2, y2):
    # Determine if the box is completely outside the range

    if x2 <= left or x1 >= right or y2 <= upper or y1 >= lower:
        return False  # The box is completely outside

    # Determine if the box is completely inside the range
    if x1 >= left and x2 <= right and y1 >= upper and y2 <= lower:
        # Return the relative position of the box within the range
        relative_position = np.array([x1 - left, y1 - upper, x2 - left, y2 - upper])
        return relative_position

    # If the box is partially inside, clip the box to keep only the inside part
    # Calculate the coordinates of the part that is inside the range
    # inside_x1 = max(left, x1)
    # inside_y1 = max(upper, y1)
    # inside_x2 = min(right, x2)
    # inside_y2 = min(lower, y2)
    
    # Return the clipped box relative to the range
    # relative_clipped_position = np.array([inside_x1 - left, inside_y1 - upper, inside_x2 - left, inside_y2 - upper])
    return False


def transfer_missing_objects(old_mask, new_mask):
    # Step 1: Identify unique values (objects) in each mask, ignoring the background (0)
    old_objects = set(np.unique(old_mask)) - {0}
    new_objects = set(np.unique(new_mask)) - {0}
    
    # Step 2: Identify missing objects in the new mask
    missing_objects = old_objects - new_objects
    
    # Step 3: Copy missing objects from the old mask to the new mask
    for obj in missing_objects:
        new_mask[old_mask == obj] = obj

    return new_mask

def get_the_largest_mask(masks):
    largest_mask = None
    largest_area = 0
    for mask in masks:
        area = np.sum(mask)
        if area > largest_area:
            largest_area = area
            largest_mask = mask
    return largest_mask

def refined_model_main(input_dir):
    # start_time = time.time()
    """
    Step 1: Environment settings and model initialization
    """
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda").__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model

    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cuda:1"
    print("device", device)


    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    print("load sam2 model done")
    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot

    # input_dir = "/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front/raw_data"
    final_json_dir = os.path.join(os.path.dirname(input_dir), "json_data")
    final_output_dir = os.path.join(os.path.dirname(input_dir), "result_refined")
    final_mask_data_dir = os.path.join(os.path.dirname(input_dir), "mask_data")
    CommonUtils.creat_dirs(final_mask_data_dir)
    mask_data = os.path.join(os.path.dirname(input_dir), "mask_data_origin")

    base_names = [name.split('.')[0] for name in os.listdir(input_dir)]

    refined_base_dir = os.path.join(os.path.dirname(input_dir), "refined_raw_data")

    tile_size = (800, 800)
    overlap_ratio = (0.5, 0.5)
    refined_raw_data_dirs = clip_images(input_dir, refined_base_dir, tile_size, overlap_ratio)
    # print("refined_raw_data_dirs", refined_raw_data_dirs)
    # refined_raw_data_dirs = ['/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front/refined_raw_data/refined_data_0_0/raw_data', '/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front/refined_raw_data/refined_data_0_1/raw_data', '/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front/refined_raw_data/refined_data_1_0/raw_data', '/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front/refined_raw_data/refined_data_1_1/raw_data', '/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front/refined_raw_data/refined_data_2_0/raw_data', '/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front/refined_raw_data/refined_data_2_1/raw_data', '/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front/refined_raw_data/refined_data_3_0/raw_data', '/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front/refined_raw_data/refined_data_3_1/raw_data']
    get_image_shape(input_dir)
    image_size = (1080, 1920) #  
    # print("image_size", image_size)
    for base_name in base_names:
        merge_mask = np.zeros(image_size)
        json_dict = MaskDictionaryModel().from_json(os.path.join(final_json_dir, f"mask_{base_name}.json"))
        old_mask_path = os.path.join(mask_data, f"mask_{base_name}.npy")
        old_mask = np.load(old_mask_path)
        old_unique = np.unique(old_mask)
        for refined_raw_data_dir in tqdm(refined_raw_data_dirs):
            output_dir = os.path.dirname(refined_raw_data_dir)
            image_path = glob.glob(os.path.join(refined_raw_data_dir, f"*{base_name}*"))[0]
            image = Image.open(image_path)
            image = np.array(image.convert("RGB"))
            predictor.set_image(image)

            try:
                parts = os.path.basename(image_path).replace('.png', '').split('_')
                i, j = map(int, parts[-2:])  # Indices are the last two parts
            except ValueError:
                print(f"Skipping file {image_path} as it does not match expected naming convention.")
                continue
            # Calculate the position where the tile should be placed
            overlap_width = int(tile_size[0] * overlap_ratio[0])
            overlap_height = int(tile_size[1] * overlap_ratio[1])
            left = i * (tile_size[0] - overlap_width)
            upper = j * (tile_size[1] - overlap_height)

            right = left + tile_size[0]
            lower = upper + tile_size[1]
            for obj_id, obj_item in json_dict.labels.items():
                x1, y1, x2, y2 = obj_item.x1, obj_item.y1, obj_item.x2, obj_item.y2
                relative_position = check_box_in_range(left, upper, right, lower, x1, y1, x2, y2)
                if relative_position is not False:
                    masks, scores, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=relative_position[None, :],
                        multimask_output=False,
                    )
                    # print(f"obj_id: {obj_id}, mask sum: {masks[0].sum()}")
                    # print("masks", masks.shape)
                    largest_area_mask = get_the_largest_mask(masks)
                    obj_mask = np.where(largest_area_mask == True, obj_item.instance_id ,0)
                    # print("obj_mask", obj_mask.shape, "obj sum", obj_mask.sum())
                    # print("merge_mask", merge_mask.shape, "merge sum", merge_mask.sum())
                    full_size_mask = np.zeros(image_size)
                    full_size_mask = paste_tile_to_full_image(full_size_mask, obj_mask, left, upper)
                    merge_mask = np.where(full_size_mask > 0, full_size_mask, merge_mask)

        """
        Complete Mask
        """

        merge_mask = transfer_missing_objects(old_mask, merge_mask)
        merge_mask = merge_mask.astype(np.uint16)
        merged_mask_path = os.path.join(final_mask_data_dir, f"mask_{base_name}.npy")
        np.save(merged_mask_path, merge_mask)
        # print(f"Save mask to {merged_mask_path}, mask sum is: {merge_mask.sum()}")



    CommonUtils.draw_masks_and_box_with_supervision(input_dir, final_mask_data_dir, final_json_dir, final_output_dir)


            

if __name__ == "__main__":
    input_dir = "/media/NAS/sd_nas_03/shuo/denso_data/20240930/20240828_080916_3/sms_right/raw_data"
    refined_model_main(input_dir)
    
