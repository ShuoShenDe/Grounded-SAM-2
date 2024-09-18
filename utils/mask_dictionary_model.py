import numpy as np
import json
import torch
import copy
import os
from dataclasses import dataclass, field
from utils.config import Config

@dataclass
class MaskDictionaryModel:
    mask_name:str = ""
    mask_height: int = 1080
    mask_width:int = 1920
    promote_type:str = "mask"
    labels:dict = field(default_factory=dict)

    def add_new_frame_annotation(self, mask_list, box_list, label_list, background_value = 0):
        mask_img = torch.zeros(mask_list.shape[-2:])
        anno_2d = {}
        removed_ids = self.filter_overlay_masks_ids(mask_list)
        filtered_mask_list = [mask_list[i] for i in range(len(mask_list)) if i not in removed_ids]
        filtered_box_list = [box_list[i] for i in range(len(box_list)) if i not in removed_ids]
        filtered_label_list = [label_list[i] for i in range(len(label_list)) if i not in removed_ids]
        for idx, (mask, box, label) in enumerate(zip(filtered_mask_list, filtered_box_list, filtered_label_list)):
            final_index = background_value + idx + 1

            if mask.shape[0] != mask_img.shape[0] or mask.shape[1] != mask_img.shape[1]:
                raise ValueError("The mask shape should be the same as the mask_img shape.")
            # mask = mask
            mask_img[mask == True] = final_index
            # print("label", label)
            label_name_group = label.split("(")[0]
            if " " in label_name_group:
                name = label_name_group.split(" ")[0]  # label 
            else:
                name = label_name_group
            logit = label.split("(")[1].replace(")", "")
            box = box # .numpy().tolist()
            true_count = mask.sum().item()
            if box[3] < 1070 and true_count > Config.mask_smallest_threshold:
                new_annotation = ObjectInfo(instance_id = final_index, mask = mask, class_name = name, x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3], logit = logit)
                anno_2d[final_index] = new_annotation

        # np.save(os.path.join(output_dir, output_file_name), mask_img.numpy().astype(np.uint16))
        self.mask_height = mask_img.shape[0]
        self.mask_width = mask_img.shape[1]
        self.labels = anno_2d

    def filter_overlay_masks_ids(self, mask_list):
        removed_ids = []
        for mask_id in range(len(mask_list)):
            mask_img = mask_list[mask_id]
            for mask_id_2 in range(mask_id+1, len(mask_list)):
                mask_img_2 = mask_list[mask_id_2]
                iou_on_smaller, smaller_mask = self.calculate_mask_overlay(mask_img, mask_img_2)
                if iou_on_smaller > Config.mask_overlay_threshold:
                    removed_ids.append(mask_id)
        print("filter_overlay_masks_return_keep_ids", removed_ids)
        return removed_ids


    def update_masks(self, tracking_annotation_dict, objects_count=0):
        tracking_annotation_dict.clean_zero_mask()
        updated_masks = copy.deepcopy(tracking_annotation_dict.labels)   # seg masks
        removed_keys = []
        
        for seg_id, seg_info in self.labels.items():  # tracking_masks
            flag = 0 
            new_object_copy = copy.deepcopy(seg_info)
            true_count = seg_info.mask.sum().item()
            if true_count < Config.mask_smallest_threshold:
                # removed_keys.append(seg_id)
                continue
            for track_obj_id, track_object in tracking_annotation_dict.labels.items():  # grounded_sam masks
                iou, smaller_mask = self.calculate_mask_overlay(track_object.mask.cpu().numpy(), seg_info.mask)  # tensor, numpy
                # print("iou", iou)
                if iou > Config.mask_overlay_threshold:
                    flag = track_object.instance_id
                    new_object_copy.instance_id = track_object.instance_id
                    new_object_copy.mask = smaller_mask # seg_info.mask
                    updated_masks[track_obj_id] = new_object_copy
                    break
                
            if not flag:
                objects_count += 1
                flag = objects_count
                new_object_copy.instance_id = objects_count
                updated_masks[objects_count] = new_object_copy

        updated_masks = {k: v for k, v in updated_masks.items() if k not in removed_keys}
        self.labels = updated_masks
        print("updated_masks.keys()", updated_masks.keys())
        return objects_count

    def get_target_class_name_and_logit(self, instance_id):
        return self.labels[instance_id].class_name, self.labels[instance_id].logit

    def get_target_logit(self, instance_id):
        return self.labels[instance_id].logit
    
    @staticmethod
    def calculate_iou(mask1, mask2):
        # Convert masks to float tensors for calculations
        mask1 = mask1.to(torch.float32)
        mask2 = mask2.to(torch.float32)
        
        # Calculate intersection and union
        intersection = (mask1 * mask2).sum()
        union = mask1.sum() + mask2.sum() - intersection
        
        # Calculate IoU
        iou = intersection / union
        return iou

    @staticmethod
    def calculate_mask_overlay(mask1, mask2):
        # Convert masks to float tensors for calculations
        mask1 = mask1 # .to(torch.float32)
        mask2 = mask2 # .to(torch.float32)
        
        # Calculate intersection and areas
        intersection = (mask1 * mask2).sum()
        area1 = mask1.sum()
        area2 = mask2.sum()
        
        # Find the smaller and larger mask areas
        min_area = min(area1, area2)
        
        # Determine which mask is larger
        smaller_mask = mask1 if area1 == min_area else mask2
        
        # Calculate the intersection ratio relative to the smaller mask
        intersection_ratio = intersection / min_area
        
        return intersection_ratio, smaller_mask

    def clean_zero_mask(self, area_threshold=Config.mask_smallest_threshold):
        removed_keys = []
        for key, value in self.labels.items():
            if value.mask.sum() < area_threshold:
                removed_keys.append(key)
        for key in removed_keys:
            self.labels.pop(key)

    def save_empty_mask_and_json(self, mask_data_dir, json_data_dir, image_name_list=None):
        mask_img = torch.zeros((self.mask_height, self.mask_width))
        if image_name_list:
            for image_base_name in image_name_list:
                image_base_name = image_base_name.replace(".png", ".npy")
                mask_name = "mask_"+image_base_name
                np.save(os.path.join(mask_data_dir, mask_name), mask_img.numpy().astype(np.uint16))

                json_data_path = os.path.join(json_data_dir, mask_name.replace(".npy", ".json"))
                print("save_empty_mask_and_json", json_data_path)
                self.to_json(json_data_path)
        else:
            np.save(os.path.join(mask_data_dir, self.mask_name), mask_img.numpy().astype(np.uint16))
            json_data_path = os.path.join(json_data_dir, self.mask_name.replace(".npy", ".json"))
            print("save_empty_mask_and_json", json_data_path)
            self.to_json(json_data_path)
    
    def to_json(self, json_file, save_labels_as_list=False):
        json_data = {
            "mask_name": self.mask_name,
            "mask_height": self.mask_height,
            "mask_width": self.mask_width,
            "promote_type": self.promote_type,
            "labels": {k: v.to_dict() for k, v in self.labels.items()}
        }
        if save_labels_as_list:
            json_data["labels"] = [value.to_dict() for value in self.labels.values()]

        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=4)
            
    def from_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
            self.mask_name = data["mask_name"]
            self.mask_height = data["mask_height"]
            self.mask_width = data["mask_width"]
            self.promote_type = data["promote_type"]
            if type(data["labels"]) == list:
                self.labels = {int(v["instance_id"]): ObjectInfo(**v) for v in data["labels"]}
            else: 
                self.labels = {int(k): ObjectInfo(**v) for k, v in data["labels"].items()}
        return self


@dataclass
class ObjectInfo:
    instance_id:int = 0
    mask: any = None
    class_name:str = ""
    x1:int = 0
    y1:int = 0
    x2:int = 0
    y2:int = 0
    logit:float = 0.0

    def get_mask(self):
        return self.mask
    
    def get_id(self):
        return self.instance_id

    def update_box(self):
        # 找到所有非零值的索引
        nonzero_indices = torch.nonzero(self.mask)
        
        # 如果没有非零值，返回一个空的边界框
        if nonzero_indices.size(0) == 0:
            # print("nonzero_indices", nonzero_indices)
            self.x1 = 0
            self.y1 = 0
            self.x2 = 0
            self.y2 = 0
            return [0,0,0,0]
        
        # 计算最小和最大索引
        y_min, x_min = torch.min(nonzero_indices, dim=0)[0]
        y_max, x_max = torch.max(nonzero_indices, dim=0)[0]
        
        # 创建边界框 [x_min, y_min, x_max, y_max]
        bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]        
        self.x1 = bbox[0]
        self.y1 = bbox[1]
        self.x2 = bbox[2]
        self.y2 = bbox[3]

    def update_box_np(self):
        # Find all non-zero indices
        nonzero_indices = np.argwhere(self.mask)
        
        # If there are no non-zero indices, return an empty bounding box
        if nonzero_indices.shape[0] == 0:
            # print("nonzero_indices", nonzero_indices)
            self.x1 = 0
            self.y1 = 0
            self.x2 = 0
            self.y2 = 0
            return [0,0,0,0]
        
        # Calculate the minimum and maximum indices
        y_min, x_min = np.min(nonzero_indices, axis=0)
        y_max, x_max = np.max(nonzero_indices, axis=0)
        
        # Create bounding box [x_min, y_min, x_max, y_max]
        bbox = [x_min, y_min, x_max, y_max]
        self.x1 = int(x_min)
        self.y1 = int(y_min)
        self.x2 = int(x_max)
        self.y3 = int(y_max)

        # Return the bounding box for completeness
        return bbox

    def to_dict(self):
        return {
            "instance_id": self.instance_id,
            "class_name": self.class_name,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "logit": self.logit
        }