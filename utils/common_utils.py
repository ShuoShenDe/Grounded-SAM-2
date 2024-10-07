import os
import json
import shutil
import cv2
import numpy as np
import supervision as sv
import random
import torch
from tqdm import tqdm
import  glob

# Grounding DINO
from grounding_dino.groundingdino.models import build_model
from grounding_dino.groundingdino.util.slconfig import SLConfig
from grounding_dino.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from utils.config import Config
import scipy.ndimage as ndimage

from utils.mask_dictionary_model import MaskDictionaryModel

class CommonUtils:
    @staticmethod
    def creat_dirs(path):
        """
        Ensure the given path exists. If it does not exist, create it using os.makedirs.

        :param path: The directory path to check or create.
        """
        try: 
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"Path '{path}' did exist and has been removed.")
            os.makedirs(path, exist_ok=True)
            # print(f"Path '{path}' did not exist and has been created.")
        except Exception as e:
            print(f"An error occurred while creating the path: {e}")




    @staticmethod
    def draw_masks_and_box_with_supervision(raw_image_path, mask_path, json_path, output_path):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)
        raw_image_name_list.sort()
        for raw_image_name in tqdm(raw_image_name_list):
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
            # load mask
            mask_npy_path = os.path.join(mask_path, "mask_"+raw_image_name.split(".")[0]+".npy")
            mask = np.load(mask_npy_path)
            # color map
            unique_ids = np.unique(mask)
            
            # get each mask from unique mask file
            all_object_masks = []
    
            
            
            # load box information
            file_path = os.path.join(json_path, "mask_"+raw_image_name.split(".")[0]+".json")
            
            all_object_boxes = []
            all_object_ids = []
            all_class_names = []
            object_id_to_name = {}
            with open(file_path, "r") as file:
                json_data = json.load(file)
                if "labels" not in json_data:
                    continue
                for obj_item in json_data["labels"]:
                    # box id
                    instance_id = obj_item["instance_id"]
                    if instance_id not in unique_ids: # not a valid box
                        continue
                    # box coordinates
                    x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                    all_object_boxes.append([x1, y1, x2, y2])
                    # box name
                    class_name = obj_item["class_name"]
                    object_mask = (mask == instance_id)
                    all_object_masks.append(object_mask[None])
                    # build id list and id2name mapping
                    all_object_ids.append(instance_id)
                    all_class_names.append(class_name)
                    object_id_to_name[instance_id] = class_name
                    
            all_object_masks = np.concatenate(all_object_masks, axis=0)
            # Adjust object id and boxes to ascending order
            paired_id_and_box = zip(all_object_ids, all_object_boxes)
            sorted_pair = sorted(paired_id_and_box, key=lambda pair: pair[0])
            
            # Because we get the mask data as ascending order, so we also need to ascend box and ids
            all_object_ids = [pair[0] for pair in sorted_pair]
            all_object_boxes = [pair[1] for pair in sorted_pair]
            
            detections = sv.Detections(
                xyxy=np.array(all_object_boxes),
                mask=all_object_masks,
                class_id=np.array(all_object_ids, dtype=np.int32),
            )
            
            # custom label to show both id and class name
            labels = [
                f"{instance_id}: {class_name}" for instance_id, class_name in zip(all_object_ids, all_class_names)
            ]
            
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            
            output_image_path = os.path.join(output_path, raw_image_name)
            cv2.imwrite(output_image_path, annotated_frame)
            # print(f"Annotated image saved as {output_image_path}")


    @staticmethod
    def draw_masks_and_box(raw_image_path, mask_path, json_path, output_path):
        CommonUtils.creat_dirs(output_path)
        raw_image_name_list = os.listdir(raw_image_path)
        raw_image_name_list.sort()
        for raw_image_name in raw_image_name_list:
            image_path = os.path.join(raw_image_path, raw_image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError("Image file not found.")
            # load mask
            mask_npy_path = os.path.join(mask_path, "mask_"+raw_image_name.split(".")[0]+".npy")
            mask = np.load(mask_npy_path)
            # color map
            unique_ids = np.unique(mask)
            colors = {uid: CommonUtils.random_color() for uid in unique_ids}
            colors[0] = (0, 0, 0)  # background color

            # apply mask to image in RBG channels
            colored_mask = np.zeros_like(image)
            for uid in unique_ids:
                colored_mask[mask == uid] = colors[uid]
            alpha = 0.5  # 调整 alpha 值以改变透明度
            output_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)


            file_path = os.path.join(json_path, "mask_"+raw_image_name.split(".")[0]+".json")
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                # Draw bounding boxes and labels
                for obj_id, obj_item in json_data["labels"].items():
                    # Extract data from JSON
                    x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                    instance_id = obj_item["instance_id"]
                    class_name = obj_item["class_name"]

                    # Draw rectangle
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put text
                    label = f"{instance_id}: {class_name}"
                    cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Save the modified image
                output_image_path = os.path.join(output_path, raw_image_name)
                cv2.imwrite(output_image_path, output_image)

                # print(f"Annotated image saved as {output_image_path}")

    @staticmethod
    def random_color():
        """random color generator"""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    @staticmethod
    def load_model(model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _ = model.eval()
        return model


    @staticmethod
    def remove_nested_box(boxes, labels):
        remove_list = set()
        
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if CommonUtils.is_nested(boxes[i].numpy(), boxes[j].numpy()):  # Convert tensors to numpy arrays for comparison
                    # print(f"Box {i} is nested inside box {j}.")
                    if CommonUtils.box_area(boxes[i].numpy()) > CommonUtils.box_area(boxes[j].numpy()):
                        remove_list.add(j)
                    else:
                        remove_list.add(i)
        
        # Filter out boxes and labels that are in remove_list
        keep_indices = [i for i in range(len(boxes)) if i not in remove_list]
        
        filtered_boxes = boxes[keep_indices]  # Indexing the tensor directly
        filtered_labels = [labels[i] for i in keep_indices]
        # print(f"Removed {len(remove_list)} nested boxes.")

        return filtered_boxes, filtered_labels

    @staticmethod
    def is_nested(box1, box2):

        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)

        inter_area = CommonUtils.box_area([inter_x1, inter_y1, inter_x2, inter_y2])
        area1 = CommonUtils.box_area(box1)
        area2 = CommonUtils.box_area(box2)
        # find the min area
        min_area = min(area1, area2)
        
        if inter_area /min_area >= Config.boxes_nested_threshold or (inter_area /min_area >= 0.8 and min_area<8000):
            return True
        return False

    @staticmethod
    def box_area(box):
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)

    @staticmethod
    def remove_redundant_box(boxes, labels):
        remove_list = set()
        for i in range(len(boxes)):
            boxes_y2 = boxes[i][3]
            if boxes_y2 > Config.removed_box_y2_threshold:
                remove_list.add(i)
            # Filter out boxes and labels that are in remove_list
        keep_indices = [i for i in range(len(boxes)) if i not in remove_list]
        
        filtered_boxes = boxes[keep_indices]  # Indexing the tensor directly
        filtered_labels = [labels[i] for i in keep_indices]
        # print(f"Removed {len(remove_list)} redundant boxes.")

        return filtered_boxes, filtered_labels
    

    def remove_discrete_areas(mask):
        # Label the connected components using 4-connectivity
        labeled_mask, num_features = ndimage.label(mask)
        # print(f"Number of features: {num_features}")
        # Measure the size of each component
        component_sizes = ndimage.sum(mask, labeled_mask, range(1, num_features + 1))
        # print(f"Component sizes: {component_sizes}")
        # Create a mask to keep large components
        large_components_mask = np.zeros_like(mask)
        for i, size in enumerate(component_sizes, start=1):
            if size >= Config.decrete_areas_min_size:
                large_components_mask[labeled_mask == i] = mask[labeled_mask == i]
            # else:
            #     print(f"Removing small component with size {size}")

        # Optionally re-label the mask
        labeled_mask, num_features = ndimage.label(large_components_mask)
        # print(f"Number of features after removing small components: {num_features}")
        
        return large_components_mask, component_sizes
    

    def get_mask_and_json(mask_data_dir, json_data_dir, image_base_name):
        mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}*")
        # print(mask_data_path)
        mask_data_path = glob.glob(mask_data_path)[0]
        mask_array = np.load(mask_data_path)
        json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}*")
        json_data_path = glob.glob(json_data_path)[0]
        json_data = MaskDictionaryModel().from_json(json_data_path)
        
        return mask_data_path, mask_array, json_data_path, json_data

    def get_image_base_name(image_name):
        base_name = image_name.split(".")[0].replace("mask_", "")
        # print(base_name)
        return base_name
    
    def get_frames_first_objs(first_appearance):
        frame_object_count = {}
        for obj_id, frame_id in first_appearance.items():
            if frame_id not in frame_object_count:
                frame_object_count[frame_id] = []
            frame_object_count[frame_id].append(obj_id)
        print("frame_object_count",frame_object_count)
        if 0 in frame_object_count:
            frame_object_count.pop(0)
        return frame_object_count
    

