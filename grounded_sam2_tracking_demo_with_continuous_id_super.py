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
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionatyModel, ObjectInfo
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
# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`  
video_dir = "notebooks/videos/car"
# 'output_dir' is the directory to save the annotated frames
output_dir = "./outputs"
# 'output_video_path' is the path to save the final video
output_video_path = "./outputs/output.mp4"

PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
objects_count = 0


# create the output directory
CommonUtils.creat_dirs(output_dir)
mask_data_dir = os.path.join(output_dir, "mask_data")
json_data_dir = os.path.join(output_dir, "json_data")
result_dir = os.path.join(output_dir, "result")
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)
# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
CommonUtils.setup_environment(device)


# init video predictor state
video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
inference_state = video_predictor.init_state(video_path=video_dir, offload_video_to_cpu=True, async_loading_frames=True)

"""
Step 2: Start first tracking and generate masks for the first frame
"""

start_frame_idx = 0
grounding_sam2_model = GroundingSAM2Model(grounding_model_id = model_id, sam2_checkpoint = sam2_checkpoint, model_cfg = model_cfg, device = device)
img_path = os.path.join(video_dir, frame_names[start_frame_idx])
image = Image.open(img_path)
image_base_name = frame_names[start_frame_idx].split(".")[0]
# segment first frame
masks, boxes, labels, scores = grounding_sam2_model.forward(image, text, box_threshold=0.25, text_threshold=0.25)
print(labels)
first_frame = MaskDictionatyModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")
if first_frame.promote_type == "mask":
    first_frame.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(boxes), label_list=labels, score_list=scores)
else:
    raise NotImplementedError("SAM 2 video predictor only support mask prompts")

# Propagate the video predictor to get the segmentation results for each frame
if len(first_frame.labels) != 0:
    print("Object detected in the frame, start tracking")
    video_predictor.reset_state(inference_state)

    for object_id, object_info in first_frame.labels.items():
        frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state,
                start_frame_idx,
                object_id,
                object_info.mask,
            )
        
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, start_frame_idx=start_frame_idx):
        image_base_name = frame_names[out_frame_idx].split(".")[0]
        frame_masks = MaskDictionatyModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")
        
        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
            class_name, logit = first_frame.get_target_class_name_and_logit(out_obj_id)
            object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = class_name, logit=logit)
            object_info.update_box()
            frame_masks.labels[out_obj_id] = object_info
            image_base_name = frame_names[out_frame_idx].split(".")[0]
            frame_masks.mask_name = f"mask_{image_base_name}.npy"
            frame_masks.mask_height = out_mask.shape[-2]
            frame_masks.mask_width = out_mask.shape[-1]
        video_segments = {out_frame_idx: frame_masks}
        CommonUtils.merge_mask_and_json(video_segments, mask_data_dir, json_data_dir)

    """
    Step 5: save the tracking masks and json files
    """
    
    
else:
    print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
    first_frame.save_empty_mask_and_json(mask_data_dir, json_data_dir)

    

for start_frame_idx in range(1, len(frame_names)):
    print("continue tracking for frame {}".format(start_frame_idx))
    img_path = os.path.join(video_dir, frame_names[start_frame_idx])
    image = Image.open(img_path)
    image_base_name = frame_names[start_frame_idx].split(".")[0]
    masks, boxes, labels, scores = grounding_sam2_model.forward(image, text, box_threshold=0.25, text_threshold=0.25)
    print("new segment frame", len(masks))
    new_seg_mask_dict = MaskDictionatyModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

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
    
    video_predictor.reset_state(inference_state)
    for object_id, object_info in new_seg_mask_dict.labels.items():
        frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state,
                start_frame_idx,
                object_id,
                object_info.mask,
            )
    
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, start_frame_idx=start_frame_idx):
        image_base_name = frame_names[out_frame_idx].split(".")[0]
        json_file = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
        previous_predicted_frame = MaskDictionatyModel().from_json(json_file)

        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
            class_name, logit = new_seg_mask_dict.get_target_class_name_and_logit(out_obj_id)
            object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = class_name, logit = logit)
            object_info.update_box()
            previous_predicted_frame.labels[out_obj_id] = object_info
        video_segments= {out_frame_idx: previous_predicted_frame}
        CommonUtils.merge_mask_and_json(video_segments, mask_data_dir, json_data_dir)

print("Total time taken: ", time.time() - start_time)
CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)

create_video_from_images(result_dir, output_video_path, frame_rate=30)