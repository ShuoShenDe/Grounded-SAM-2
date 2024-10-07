import os
import torch
import numpy as np
from grounding_dino.groundingdino.util.vl_utils import create_positive_map_from_span
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.utils import get_phrases_from_posmap
import grounding_dino.groundingdino.datasets.transforms as T
from PIL import Image

from utils.common_utils import CommonUtils 
import torch



class GroundingSAM2Model:
    def __init__(self, grounding_model_config, grounded_checkpoint, sam2_model_cfg :str, sam2_checkpoint, device:str="cuda"):
        self.grounding_model = CommonUtils.load_model(grounding_model_config, grounded_checkpoint, device=device)
        self.device = device
        self.grounding_model.to(device)
        self.sam2_image_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=self.device)
        self.image_predictor = SAM2ImagePredictor(self.sam2_image_model)
        

    def forward(self, img_path, text_prompt, box_threshold, text_threshold):
        # image = Image.open(img_path)
        image, image_transformed = self.load_image(img_path)
        size = image.size
        H, W = size[1], size[0]
        boxes_filt, pred_phrases = self.get_grounding_output(
                image_transformed, text_prompt, box_threshold, text_threshold=text_threshold#  token_spans=token_spans
            )


        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt, pred_phrases = CommonUtils.remove_redundant_box(boxes_filt, pred_phrases)
        # boxes_filt, pred_phrases = CommonUtils.remove_nested_box(boxes_filt, pred_phrases)
        # prompt SAM image predictor to get the mask for the object
        self.image_predictor.set_image(np.array(image.convert("RGB")))

        # process the detection results
        input_boxes = boxes_filt # results[0]["boxes"] # .cpu().numpy()
        # print("results[0]",results[0])
        OBJECTS = pred_phrases
        if input_boxes.shape[0] != 0:
            # mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir)
            
            # prompt SAM 2 image predictor to get the mask for the object
            masks, scores, logits = self.image_predictor.predict(
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
        else:
            masks = torch.zeros(0, H, W)
            scores = torch.zeros(0)
            logits = torch.zeros

        return masks, input_boxes, OBJECTS


    def forward_with_loop(self, img_path, text_prompts, thresholds):
        # image = Image.open(img_path)
        image, image_transformed = self.load_image(img_path)
        size = image.size
        H, W = size[1], size[0]
        final_masks = np.empty((0, H, W))
        final_boxes = torch.tensor([])
        final_pred_phrases = []
        for text_prompt, threshold in zip(text_prompts, thresholds):
            
            boxes_filt, pred_phrases = self.get_grounding_output(
                    image_transformed, text_prompt, threshold, text_threshold=threshold#  token_spans=token_spans
                )

	    
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]
            boxes_filt, pred_phrases = CommonUtils.remove_redundant_box(boxes_filt, pred_phrases)
            print("text_prompt", text_prompt, "threshold", threshold, boxes_filt.shape)
            # boxes_filt, pred_phrases = CommonUtils.remove_nested_box(boxes_filt, pred_phrases)
            # prompt SAM image predictor to get the mask for the object
            self.image_predictor.set_image(np.array(image.convert("RGB")))

            # process the detection results
            input_boxes = boxes_filt # results[0]["boxes"] # .cpu().numpy()
            # print("results[0]",results[0])
            OBJECTS = pred_phrases
            # print("input_boxes type,", type(input_boxes), "input_boxes shape", input_boxes.shape)
            # print("OBJECTS shape,", OBJECTS)
            

            if input_boxes.shape[0] != 0:
                
                # mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir)
                
                # prompt SAM 2 image predictor to get the mask for the object
                masks, scores, logits = self.image_predictor.predict(
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

                # print("masks type", type(masks), "masks shape", masks.shape)

            else:
                masks = torch.zeros(0, H, W)
                scores = torch.zeros(0)
                logits = torch.zeros(0)
                
            final_masks = np.concatenate([final_masks, masks], axis=0)
            final_boxes = torch.cat([final_boxes, input_boxes])
            final_pred_phrases.extend(OBJECTS)
        # print("final_pred_phrases", len(final_pred_phrases),final_pred_phrases)
        return final_masks, final_boxes, final_pred_phrases
    
        
    def get_grounding_output(self, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
        assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(image[None], captions=[caption])
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
            tokenlizer = self.grounding_model.tokenizer
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
                self.grounding_model.tokenizer(caption),
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
