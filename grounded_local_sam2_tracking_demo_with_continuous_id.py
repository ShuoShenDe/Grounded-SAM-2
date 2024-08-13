import os
import time
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from utils.config import Config
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionatyModel, ObjectInfo
from grounding_dino.groundingdino.util.vl_utils import create_positive_map_from_span
from grounding_dino.groundingdino.util.utils import get_phrases_from_posmap
import gc

import json
import copy

start_time = time.time()
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
grounded_checkpoint = "gdino_checkpoints/groundingdino_swinb_cogcoor.pth" 
config_file = "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py" 
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cuda:1"
print("device", device)

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device = device)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)


# init grounding dino model from huggingface
grounding_dino_model = CommonUtils.load_model(config_file, grounded_checkpoint, device=device)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
box_threshold=0.25
text_threshold=0.20
text_prompt = "car. van. truck. person. motorcycle. bicycle. pole. balusters. bannister. rail."
# car. van. bus. truck. person. motorcycle. bicycle. flagpole. pole. balusters. bannister. stile. rail.
video_dir = "/media/NAS/sd_nas_01/shuo/denso_data/test_trip/raw_data"
# 'output_dir' is the directory to save the annotated frames
output_dir = os.path.dirname(video_dir)
print("output_dir", output_dir)
# 'output_video_path' is the path to save the final video
output_video_path = os.path.join(output_dir, "output.mp4")
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
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
# frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
frame_names.sort()  
# init video predictor state
inference_state = video_predictor.init_state(video_path=video_dir, offload_video_to_cpu=True, async_loading_frames=True)
step = 15 # the step to sample frames for Grounding DINO predictor

sam2_masks = MaskDictionatyModel()
PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
objects_count = 0

"""
Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
"""
print("Total frames:", len(frame_names))
for start_frame_idx in range(0, len(frame_names), step):
# prompt grounding dino to get the box coordinates on specific frame
    print("start_frame_idx", start_frame_idx)
    img_path = os.path.join(video_dir, frame_names[start_frame_idx])
    # image = Image.open(img_path)
    image, image_transformed = CommonUtils.load_image(img_path)
    image_base_name = frame_names[start_frame_idx].split(".")[0]
    size = image.size
    H, W = size[1], size[0]
    mask_dict = MaskDictionatyModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy", mask_height = H, mask_width = W)

    # run Grounding DINO on the image
    boxes_filt, pred_phrases = CommonUtils.get_grounding_output(
            grounding_dino_model, image_transformed, text_prompt, box_threshold, text_threshold=text_threshold, device=device#  token_spans=token_spans
        )


    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    boxes_filt, pred_phrases = CommonUtils.remove_redundant_box(boxes_filt, pred_phrases)
    # boxes_filt, pred_phrases = CommonUtils.remove_nested_box(boxes_filt, pred_phrases)
    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process the detection results
    input_boxes = boxes_filt # results[0]["boxes"] # .cpu().numpy()
    # print("results[0]",results[0])
    OBJECTS = pred_phrases
    if input_boxes.shape[0] != 0:
        # mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir)
        
        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)
            scores = scores.squeeze(1)
            logits = logits.squeeze(1)

        """
        Step 3: Register each object's positive points to video predictor
        """

        # If you are using point prompts, we uniformly sample positive points based on the mask
        if mask_dict.promote_type == "mask":
            mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
        else:
            raise NotImplementedError("SAM 2 video predictor only support mask prompts")

            """
        Step 4: Propagate the video predictor to get the segmentation results for each frame
        """
        objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, objects_count=objects_count)
        print("objects_count", objects_count)
    else:
        print("No object detected in the frame, skip merge the frame merge {}".format(frame_names[start_frame_idx]))
        mask_dict = sam2_masks

    video_predictor.reset_state(inference_state)

    if len(mask_dict.labels) == 0:
        mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list = frame_names[start_frame_idx:start_frame_idx+step])
        print("No object detected in the frame, skip the frame {}".format(frame_names[start_frame_idx]))
        continue

    for object_id, object_info in mask_dict.labels.items():
        frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state,
                start_frame_idx,
                object_id,
                object_info.mask,
            )
    
    video_segments = {}  # output the following {step} frames tracking masks
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
        image_base_name = frame_names[out_frame_idx].split(".")[0]
        frame_masks = MaskDictionatyModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy", mask_height = H, mask_width = W)
        
        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
            if out_mask.sum().item() < Config.mask_smallest_threshold:
                continue
            class_name, logit = mask_dict.get_target_class_name_and_logit(out_obj_id)
            object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = class_name, logit=logit)
            object_info.update_box()
            if object_info.y2 > Config.removed_box_y2_threshold:
                continue
            frame_masks.labels[out_obj_id] = object_info

        video_segments[out_frame_idx] = frame_masks
        sam2_masks = copy.deepcopy(frame_masks)

    
    """
    Step 5: save the tracking masks and json files
    """
    print("video_segments length", len(video_segments))
    for frame_idx, frame_masks_info in video_segments.items():
        mask = frame_masks_info.labels
        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
        for obj_id, obj_info in mask.items():
            mask_img[obj_info.mask == True] = obj_id

        mask_img = mask_img.numpy().astype(np.uint16)
        np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

        json_data = frame_masks_info.to_dict()
        json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
        # print(json_data_path, frame_masks_info)
        with open(json_data_path, "w") as f:
            json.dump(json_data, f)
    
    # Call the garbage collector
    gc.collect()
    # Empty the PyTorch cache
    torch.cuda.empty_cache()
    # del image, image_transformed, scores, logits
    print("cleaning")


"""
Step 6: Draw the results and save the video
"""
# CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)

# create_video_from_images(result_dir, output_video_path, frame_rate=30)

# 592
print("Total time:", time.time() - start_time)