import glob
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
import gc
from PIL import Image

import json
import copy

from utils.refine_model import clip_images, split_image_to_folders

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
grounding_model_config = "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py" 
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cuda:1"
print("device", device)


grounding_dino_model = GroundingSAM2Model(grounding_model_config, grounded_checkpoint, sam2_model_cfg=model_cfg, sam2_checkpoint=sam2_checkpoint, device=device)
video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device = device)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
box_threshold = 0.15  #0.25
text_threshold = 0.15
print("box_threshold", box_threshold, "text_threshold", text_threshold)
text_prompt = "car. van. truck. person. motorcycle. bicycle. pole. balusters. bannister. rail."
# car. van. bus. truck. person. motorcycle. bicycle. flagpole. pole. balusters. bannister. stile. rail.
input_dir = "/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_front/raw_data"
refined_base_dir = os.path.join(os.path.dirname(input_dir), "refined_raw_data")
refined_raw_data_dirs = clip_images(input_dir, refined_base_dir, (800, 800), (0.5, 0.5))
print("refined_raw_data_dirs", refined_raw_data_dirs)

for refined_raw_data in refined_raw_data_dirs:
    
    # 'output_dir' is the directory to save the annotated frames
    output_dir = os.path.dirname(refined_raw_data)
    print("output_dir", output_dir)
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    result_dir = os.path.join(output_dir, "result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(refined_raw_data)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    # frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frame_names.sort()  
    # init video predictor state
    inference_state = video_predictor.init_state(video_path=refined_raw_data, offload_video_to_cpu=True, async_loading_frames=True, offload_state_to_cpu = True)
    step = 20 # the step to sample frames for Grounding DINO predictor

    sam2_masks = MaskDictionaryModel()
    PROMPT_TYPE_FOR_VIDEO = Config.PROMPT_TYPE_FOR_VIDEO # box, mask or point
    objects_count = 0
    first_appearance = {}



    """
    Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
    """
    print("Total frames:", len(frame_names))
    for start_frame_idx in range(0, len(frame_names), step):
    # prompt grounding dino to get the box coordinates on specific frame
        print("start_frame_idx", start_frame_idx)
        img_path = os.path.join(refined_raw_data, frame_names[start_frame_idx])
        image_opened = Image.open(img_path)
        image_base_name = CommonUtils.get_image_base_name(frame_names[start_frame_idx])
        size = image_opened.size
        H, W = size[1], size[0]
        mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy", mask_height = H, mask_width = W)

        masks, input_boxes, OBJECTS = grounding_dino_model.forward(img_path, text_prompt, box_threshold, text_threshold)
        """
            Step 3: Register each object's mask to video predictor
        """
        if input_boxes.shape[0] != 0:
            
            if mask_dict.promote_type == "mask":
                # mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
                mask_dict.add_new_frame_annotation(mask_list=masks, box_list=input_boxes, label_list=OBJECTS)
            else:
                raise NotImplementedError("SAM 2 video predictor only support mask prompts")

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
            image_base_name = CommonUtils.get_image_base_name(frame_names[out_frame_idx])
            frame_masks = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy", mask_height = H, mask_width = W)
            
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                if out_mask.sum().item() < Config.mask_smallest_threshold:
                    # if the mask is too small, we remove it, and frame_object_count will decrease by 1
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
        for frame_idx, frame_masks_info in video_segments.items():
            mask_dict = frame_masks_info.labels
            mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
            for obj_id, obj_info in mask_dict.items():
                obj_info.update_box()
                mask_img[obj_info.mask == True] = obj_id
                if obj_info.mask.sum().item() < Config.mask_smallest_threshold:
                    # print("mask too small", obj_id, obj_info.mask.sum().item())
                    continue
                if obj_id not in first_appearance:
                    first_appearance[obj_id] = frame_idx
            mask_img = mask_img.numpy().astype(np.uint16)
            np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)
            
            json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
            json_data = frame_masks_info.to_json(json_data_path)

        
        # Call the garbage collector
        gc.collect()
        # Empty the PyTorch cache
        torch.cuda.empty_cache()
        # del image, image_transformed, scores, logits
        print("cleaning")
    """
    Step 5: Run Reverse prediction
    """
    print(first_appearance)
    if len(first_appearance) == 0:
        continue
    frame_object_count = CommonUtils.get_frames_first_objs(first_appearance)

    print("try reverse tracking", frame_object_count)
    start_object_id = 0
    object_info_dict = {}
    for frame_idx, current_object_count in frame_object_count.items():
        print("reverse tracking frame", frame_idx, frame_names[frame_idx])
        if frame_idx != 0:
            video_predictor.reset_state(inference_state)
            image_base_name = CommonUtils.get_image_base_name(frame_names[frame_idx])
            mask_data_path, mask_array, json_data_path, json_data = CommonUtils.get_mask_and_json(mask_data_dir, json_data_dir, image_base_name)
            for object_id in frame_object_count[frame_idx]:
                print("reverse tracking object", object_id)
                if object_id in json_data.labels.keys():
                    object_info_dict[object_id] = json_data.labels[object_id]
                    video_predictor.add_new_mask(inference_state, frame_idx, object_id, mask_array == object_id)
        start_object_id = current_object_count
            
        
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step*2,  start_frame_idx=frame_idx, reverse=True):
            image_base_name = CommonUtils.get_image_base_name(frame_names[out_frame_idx])
            mask_data_path, mask_array, json_data_path, json_data = CommonUtils.get_mask_and_json(mask_data_dir, json_data_dir, image_base_name)
            # merge the reverse tracking masks with the original masks
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0).cpu()
                if out_mask.sum() < Config.mask_smallest_threshold:
                    continue
                object_info = object_info_dict[out_obj_id]
                object_info.mask = out_mask[0]
                object_info.update_box()
                json_data.labels[out_obj_id] = object_info
                origin_mask_area = mask_array[object_info.mask]
                unique_numbers, counts = np.unique(origin_mask_area, return_counts=True)
                target_obj_id = out_obj_id
                for unique_number, count in zip(unique_numbers, counts):
                    if unique_number == 0:
                        continue
                    if count/out_mask.sum() > Config.mask_overlay_threshold:
                        target_obj_id = unique_number
                mask_array[mask_array == target_obj_id] = 0
                mask_array[object_info.mask] = target_obj_id
                
            np.save(mask_data_path, mask_array)
            json_data.to_json(json_data_path, save_labels_as_list=True)


    # create_video_from_images(result_dir, output_video_path, frame_rate=30)
    # 592
    print("Total time:", time.time() - start_time)

    PostProcess.remove_area(mask_data_dir, json_data_dir, frame_names)
    PostProcess.unified_classes(json_data_dir)
    CommonUtils.draw_masks_and_box_with_supervision(refined_raw_data, mask_data_dir, json_data_dir, result_dir)