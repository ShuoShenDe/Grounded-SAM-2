
class Config:
    mask_smallest_threshold = 250 # 400
    boxes_nested_threshold = 0.95
    removed_box_y2_threshold = 1070
    mask_overlay_threshold = 0.75  # used in MaskDictionatyModel.update_masks
    decrete_areas_min_size = 250 # 250
    PROMPT_TYPE_FOR_VIDEO = "mask"