import os
from sam2.build_sam import build_sam2_video_predictor
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionatyModel, ObjectInfo


class SAM2TrackingModel:
    def __init__(self, model_cfg, sam2_checkpoint, video_dir, device, mask_data_dir = "mask_data", json_data_dir = "json_data", prompt_type="mask"):
        self.device = device
        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        self.inference_state = self.video_predictor.init_state(video_path=video_dir, offload_video_to_cpu=True, async_loading_frames=True)
        self.prompt_type = prompt_type
        self.mask_data_dir = mask_data_dir
        self.json_data_dir = json_data_dir
    
    def forward(self, frame_dict, start_frame_idx, frame_names):
        self.video_predictor.reset_state(self.inference_state)
        for object_id, object_info in frame_dict.labels.items():
            frame_idx, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                    self.inference_state,
                    start_frame_idx,
                    object_id,
                    object_info.mask,
                )
                
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.inference_state, start_frame_idx=start_frame_idx):
            image_base_name = frame_names[out_frame_idx].split(".")[0]
            json_file = os.path.join(self.json_data_dir, f"mask_{image_base_name}.json")
            if os.path.exists(json_file):
                tracking_frame_mask = MaskDictionatyModel().from_json(json_file)

                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                    class_name, logit = frame_dict.get_target_class_name_and_logit(out_obj_id)
                    object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = class_name, logit = logit)
                    object_info.update_box()
                    tracking_frame_mask.labels[out_obj_id] = object_info

            else:
                tracking_frame_mask = MaskDictionatyModel(promote_type = self.prompt_type, mask_name = f"mask_{image_base_name}.npy")
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                    class_name, logit = frame_dict.get_target_class_name_and_logit(out_obj_id)
                    object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = class_name, logit=logit)
                    object_info.update_box()
                    tracking_frame_mask.mask_height = out_mask.shape[-2]
                    tracking_frame_mask.mask_width = out_mask.shape[-1]
                    tracking_frame_mask.mask_name = f"mask_{image_base_name}.npy"
                    tracking_frame_mask.labels[out_obj_id] = object_info
                    image_base_name = frame_names[out_frame_idx].split(".")[0]
                
                video_segments = {out_frame_idx: tracking_frame_mask}
                CommonUtils.merge_mask_and_json(video_segments, self.mask_data_dir, self.json_data_dir)
