import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.grounding_sam2_model import GroundingSAM2Model
from utils.sam2_tracking_model import SAM2TrackingModel
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import time


start_time = time.time()

"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)
model_id = "IDEA-Research/grounding-dino-tiny"

# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = "car."
PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
objects_count = 0
# the number of frames to skip between each video prediction
step = 1 
# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`  
video_dir = "/media/NAS/sd_nas_01/shuo/denso_data/20240613_103919_9_short/raw_data"
# 'output_dir' is the directory to save the annotated frames
output_dir = "/media/NAS/sd_nas_01/shuo/denso_data/20240613_103919_9_short/"
#  'output_video_path' is the path to save the final video
output_video_path = "./outputs/output.mp4"



# create the output directory
mask_data_dir = os.path.join(output_dir, "mask_data")
json_data_dir = os.path.join(output_dir, "json_data")
result_dir = os.path.join(output_dir, "result")
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# setup the environment
CommonUtils.setup_environment(device)

# init video predictor state
sam2_tracking_model = SAM2TrackingModel(model_cfg, sam2_checkpoint, video_dir, device, mask_data_dir, json_data_dir, PROMPT_TYPE_FOR_VIDEO)

"""
Step 2: Start first tracking and generate masks for the first frame
"""

start_frame_idx = 0
grounding_sam2_model = GroundingSAM2Model(grounding_model_id = model_id, sam2_checkpoint = sam2_checkpoint, model_cfg = model_cfg, device = device)
img_path = os.path.join(video_dir, frame_names[start_frame_idx])
image = Image.open(img_path).convert("RGB")
image_base_name = frame_names[start_frame_idx].split(".")[0]
# segment first frame
masks, boxes, labels, scores = grounding_sam2_model.forward(image, text, box_threshold=0.25, text_threshold=0.25)
print(labels)
first_frame = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")
if first_frame.promote_type == "mask":
    first_frame.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(boxes), label_list=labels, score_list=scores)
else:
    raise NotImplementedError("SAM 2 video predictor only support mask prompts")

# Propagate the video predictor to get the segmentation results for each frame
if len(first_frame.labels) != 0:
    print("Object detected in the frame, start tracking")
    sam2_tracking_model.forward(first_frame, start_frame_idx, frame_names)
    """
    Step 5: save the tracking masks and json files
    """
else:
    print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
    first_frame.save_empty_mask_and_json(mask_data_dir, json_data_dir)

for start_frame_idx in range(1, len(frame_names), step):
    print("continue tracking for frame {}".format(start_frame_idx))
    img_path = os.path.join(video_dir, frame_names[start_frame_idx])
    image = Image.open(img_path).convert("RGB")
    image_base_name = frame_names[start_frame_idx].split(".")[0]
    masks, boxes, labels, scores = grounding_sam2_model.forward(image, text, box_threshold=0.25, text_threshold=0.25)
    print("new segment frame", len(masks))
    new_seg_mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

    if new_seg_mask_dict.promote_type == "mask":
        new_seg_mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(boxes), label_list=labels)
    else:
        raise NotImplementedError("SAM 2 video predictor only support mask prompts")
    
    # load previous saved masks
    saved_frame_mask, saved_frame_json = CommonUtils.read_mask_and_json(mask_data_dir, json_data_dir, image_base_name)
    # load previous tracked masks
    objects_count = new_seg_mask_dict.delete_duplicate_masks(saved_frame_mask, objects_count=objects_count)
    print("after delete new_frame len", len(new_seg_mask_dict.labels))
    if len(new_seg_mask_dict.labels)==0:
        print("No object detected in the frame, skip the frame {}".format(frame_names[start_frame_idx]))
        continue

    sam2_tracking_model.forward(new_seg_mask_dict, start_frame_idx, frame_names)

print("Total time taken: ", time.time() - start_time)
CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)

create_video_from_images(result_dir, output_video_path, frame_rate=30)