import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 



class GroundingSAM2Model:
    
    def __init__(self, grounding_model_id, sam2_checkpoint ,model_cfg :str, device:str="cuda"):
        self.processor = AutoProcessor.from_pretrained(grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)
        self.device = device
        self.sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.image_predictor = SAM2ImagePredictor(self.sam2_image_model)
        

    def forward(self, image, text_prompt, box_threshold, text_threshold):
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )
        # prompt SAM image predictor to get the mask for the object
        self.image_predictor.set_image(np.array(image.convert("RGB")))
        # process the detection results
        boxes = results[0]["boxes"] # .cpu().numpy()
        labels = results[0]["labels"]
        scores = results[0]["scores"]

        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
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
        
        return masks, boxes, labels, scores