import os
import json
import cv2
import numpy as np
from dataclasses import dataclass
import supervision as sv
import random
import torch
from PIL import Image
from tqdm import tqdm

# Grounding DINO
import grounding_dino.groundingdino.datasets.transforms as T
from grounding_dino.groundingdino.models import build_model
from grounding_dino.groundingdino.util.slconfig import SLConfig
from grounding_dino.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from utils.config import Config
# from config import Config

class CommonUtils:
    @staticmethod
    def creat_dirs(path):
        """
        Ensure the given path exists. If it does not exist, create it using os.makedirs.

        :param path: The directory path to check or create.
        """
        try: 
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"Path '{path}' did not exist and has been created.")
            else:
                print(f"Path '{path}' already exists.")
        except Exception as e:
            print(f"An error occurred while creating the path: {e}")

    @staticmethod
    def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None, device="cuda"):
        assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # filter output
        if token_spans is None:
            logits_filt = logits.cpu().clone()
            boxes_filt = boxes.cpu().clone()
            filt_mask = logits_filt.max(dim=1)[0] > box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

            # get phrase
            tokenlizer = model.tokenizer
            tokenized = tokenlizer(caption)
            # build pred
            pred_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
                if with_logits:
                    pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                else:
                    pred_phrases.append(pred_phrase)
        else:
            # given-phrase mode
            positive_maps = create_positive_map_from_span(
                model.tokenizer(caption),
                token_span=token_spans
            ).to(image.device) 

            logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
            all_logits = []
            all_phrases = []
            all_boxes = []
            for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
                # get phrase
                phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
                # get mask
                filt_mask = logit_phr > box_threshold
                # filt box
                all_boxes.append(boxes[filt_mask])
                # filt logits
                all_logits.append(logit_phr[filt_mask])
                if with_logits:
                    logit_phr_num = logit_phr[filt_mask]
                    all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
                else:
                    all_phrases.extend([phrase for _ in range(len(filt_mask))])
            boxes_filt = torch.cat(all_boxes, dim=0).cpu()
            pred_phrases = all_phrases


        return boxes_filt, pred_phrases


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
            for uid in unique_ids:
                if uid == 0: # skip background id
                    continue
                else:
                    object_mask = (mask == uid)
                    all_object_masks.append(object_mask[None])
            
            # get n masks: (n, h, w)
            if len(all_object_masks) == 0:
                continue
    
            all_object_masks = np.concatenate(all_object_masks, axis=0)
            
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
                for obj_id, obj_item in json_data["labels"].items():
                    # box id
                    instance_id = obj_item["instance_id"]
                    if instance_id not in unique_ids: # not a valid box
                        continue
                    # box coordinates
                    x1, y1, x2, y2 = obj_item["x1"], obj_item["y1"], obj_item["x2"], obj_item["y2"]
                    all_object_boxes.append([x1, y1, x2, y2])
                    # box name
                    class_name = obj_item["class_name"]
                    
                    # build id list and id2name mapping
                    all_object_ids.append(instance_id)
                    all_class_names.append(class_name)
                    object_id_to_name[instance_id] = class_name
            
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
    def load_image(image_path):
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

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

if __name__ == "__main__":
    # Test the CommonUtils class
    input_file = '/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_1/sms_front_shorter/json_data/mask_1718266721139816000.json'

    with open(input_file, 'r') as file:
        data = json.load(file)
        labels = data.get('labels', {})
        for key1, label in enumerate(labels.values()):
            instance_id = label['instance_id']
            class_name = label['class_name']
            box1 = [label['x1'], label['y1'], label['x2'], label['y2']]
            for key2, label in enumerate(labels.values()):
                if key1 == key2:
                    continue
                instance_id2 = label['instance_id']
                class_name2 = label['class_name']
                box2 = [label['x1'], label['y1'], label['x2'], label['y2']]
                if CommonUtils.is_nested(box1, box2):
                    print(f"Box {instance_id} is nested inside box {instance_id2}.")
                else:
                    print(f"Box {instance_id} is not nested inside box {instance_id2}.")