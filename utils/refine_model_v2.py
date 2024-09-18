import numpy as np
from PIL import Image
import os
from pathlib import Path
import json

def mask_to_box(mask:np.ndarray):
    # Find the contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Find the bounding box of the largest contour (assuming it's the object of interest)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        x1, y1, x2, y2 = x, y, x + w, y + h
        return (x1, y1, x2, y2)

def apply_nms_by_class(obj_xyxy:np.ndarray, obj_scores:np.ndarray, obj_cls_ids:np.ndarray, iou_threshold:float) -> list:
    
    keep_inds = []
    if obj_cls_ids is not None:
        
        all_cls_ids = list(set(obj_cls_ids.tolist()))
            
        # Apply nms to each class
        for cls_id in all_cls_ids:
            cur_ids = np.where(obj_cls_ids == cls_id)[0]
            
            if len(cur_ids) > 1:
                cur_xyxy = obj_xyxy[cur_ids]
                cur_scores = obj_scores[cur_ids]
                
                nms_idx = torchvision.ops.nms(torch.from_numpy(cur_xyxy.astype(np.float32)),
                                    torch.from_numpy(cur_scores.astype(np.float32)),
                                    iou_threshold).numpy().tolist()
                cur_ids = cur_ids[nms_idx]
            keep_inds.extend(cur_ids.tolist())
    else:
        nms_idx = torchvision.ops.nms(torch.from_numpy(obj_xyxy.astype(np.float32)),
                                    torch.from_numpy(obj_scores.astype(np.float32)),
                                    iou_threshold).numpy()
        keep_inds.extend(nms_idx.tolist())

    return keep_inds


def sample(img_path, crop_size= (600, 600)):
    '''
    crop images
    '''
    img = np.array(Image.open(img_path)) # h,w,c(RGB)
    
    bigimgsize2, bigimgsize1, ch_num = img.shape # h,w
    
    sub_img_list = []
    left_top_list = []
    
    infer_subimg_resolution = crop_size[0]
    subimg_interval = infer_subimg_resolution // 2
    sampled_points = []
    for offsetx in range(subimg_interval, bigimgsize1+1-subimg_interval, subimg_interval):
        for offsety in range(subimg_interval, bigimgsize2+1-subimg_interval, subimg_interval):
            sampled_points.append([offsetx, offsety])
    for point in sampled_points:
        x, y = point
        if x < subimg_interval:
            x = subimg_interval
        elif x > (bigimgsize1 - subimg_interval):
            x = (bigimgsize1 - subimg_interval)
        if y < subimg_interval:
            y = subimg_interval
        elif y > (bigimgsize2 - subimg_interval):
            y = (bigimgsize2 - subimg_interval)
        bbox_minx, bbox_miny, bbox_maxx, bbox_maxy = round(x - subimg_interval), round(y - subimg_interval), \
            round(x + subimg_interval - 1), round(y + subimg_interval - 1)
        np_subimg = img[bbox_miny: bbox_maxy+1, bbox_minx: bbox_maxx+1, :]
        
        subimg = np.zeros((crop_size[0], crop_size[1], ch_num), dtype= np.uint8)
        subimg[:bbox_maxy+1, :bbox_maxx+1,:] = np_subimg
         
        sub_img_list.append(Image.fromarray(subimg))
        left_top_list.append([bbox_minx, bbox_miny])
    
    return sub_img_list, left_top_list


# Apply nms to merge box and seg area which has the same class type and are highly overlapped
def calculate_bbox_iou(box1, box2):
    """
    Calculate the IoU (Intersection over Union) between two bounding boxes.
    
    Arguments:
    box1, box2 -- dictionaries with 'x1', 'y1', 'x2', 'y2' coordinates of the boxes.
    
    Returns:
    IoU -- float value of Intersection over Union
    """
    x1_inter = max(box1['x1'], box2['x1'])
    y1_inter = max(box1['y1'], box2['y1'])
    x2_inter = min(box1['x2'], box2['x2'])
    y2_inter = min(box1['y2'], box2['y2'])
    
    # Compute the area of the intersection rectangle
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Compute the area of both bounding boxes
    box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y2'])
    
    # Compute the IoU: intersection area over union area
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    iou = inter_area / union_area
    return iou


def check_if_same_type(class_name1, class_name2):
    flag = False
    if class_name1 == class_name2:
        flag = True
    elif class_name1 in ['car', 'van', 'truck'] and class_name2 in ['car', 'van', 'truck']:
        flag = True
    elif class_name1 in ['motorcycle', 'bicycle'] and class_name2 in ['motorcycle', 'bicycle']:
        flag = True
    return flag

def merge_boxes_and_masks(orig_anno_2d, 
                          cropped_anno_2d, 
                           cropped_binary_mask, 
                          iou_threshold={'car':0.3,
                                         'truck':0.3,
                                         'van':0.3,
                                         'person':0.1,
                                         'motorcycle':0.1,
                                         'bicycle':0.1,
                                         'pole':0.1,
                                         'balusters':0.5,
                                         'bannister':0.5,
                                         'rail':0.1}):
    """
    Merge boxes that overlap and combine their corresponding masks.
    
    Arguments:
    orig_anno_2d -- List of bounding box annotations of the whole image
    cropped_anno_2d -- List of bounding box annotations of the cropped image.
    cropped_binary_mask -- instance mask of cropped image
    iou_threshold -- Threshold above which bounding boxes will be merged.
    
    Returns:
    merged_boxes_masks -- List of merged boxes and combined masks.
    """
    merged_boxes_list = []
    merged_masks_list = []
    used_orig_box_inds = set()
    used_cropped_box_inds = set()  # To record those merged cropped boxes

    # Step 1: use the boxes detected on the whole image to merge those detected on cropped images
    for i, box1 in enumerate(orig_anno_2d):
        merged_box = box1.copy()
        combined_mask = np.zeros((cropped_binary_mask[0].shape))
        
        for j, box2 in enumerate(cropped_anno_2d):
            iou = calculate_bbox_iou(merged_box, box2)
            # If IoU is greater than threshold and class_name matches, merge the boxes and masks
            if check_if_same_type(box1['class_name'], box2['class_name']):
                if iou > iou_threshold[box1['class_name']]:
                    # Merge bounding box by taking the min and max coordinates
                    merged_box['x1'] = min(merged_box['x1'], box2['x1'])
                    merged_box['y1'] = min(merged_box['y1'], box2['y1'])
                    merged_box['x2'] = max(merged_box['x2'], box2['x2'])
                    merged_box['y2'] = max(merged_box['y2'], box2['y2'])
                    # Combine masks using logical OR
                    combined_mask = np.logical_or(combined_mask, cropped_binary_mask[j])
                    used_cropped_box_inds.add(j)
                    used_orig_box_inds.add(i)  # Mark the original box as used
                    
        # Add the merged box and its combined mask to the result
        merged_boxes_list.append({
            'x1': merged_box['x1'],
            'y1': merged_box['y1'],
            'x2': merged_box['x2'],
            'y2': merged_box['y2'],
            'class_name': merged_box['class_name'],
            'logit': merged_box['logit'],
        })
        merged_masks_list.append(combined_mask)
        
    #print('Step 1: ', len(merged_boxes_list))
    # Step 2: 
    ## Step 2: 
    if len(used_cropped_box_inds) < len(cropped_anno_2d):
        left_cropped_inds = set(range(len(cropped_anno_2d))) - used_cropped_box_inds
        left_cropped_box  = [cropped_anno_2d[j] for j in left_cropped_inds]
        left_cropped_mask = [cropped_binary_mask[j] for j in left_cropped_inds ]
        
        tmp_box = (merged_boxes_list + left_cropped_box).copy()
        tmp_mask = (merged_masks_list + left_cropped_mask).copy()
        
        # Apply nms to get rid of redandent box or mask
        obj_xyxy = np.array([[box[key] for key in ['x1', 'y1', 'x2', 'y2']] for box in tmp_box])
        obj_score = np.array([box['logit'] for box in tmp_box])
        obj_cls_ids = np.array([box['class_name'] for box in tmp_box])
        keep_ids = apply_nms_by_class(obj_xyxy, obj_score, obj_cls_ids, iou_threshold= 0.2)
        
        merged_boxes_list = [tmp_box[i] for i in keep_ids]
        merged_masks_list = [tmp_mask[i] for i in keep_ids]
        
    
    # reorder the instance id
    final_mask = np.zeros((merged_masks_list[0].shape))
    for i in range(len(merged_masks_list)):
        merged_boxes_list[i]['instance_id'] = i+1
        final_mask[merged_masks_list[i]] = i+1

    return merged_boxes_list, final_mask


class RefinedModel:

    def image_slicer(imput_path:str, output_path:str)->list:
        """
            input_path: str, e.g. 'data/trip/raw_data'
            output_path: str, e.g. 'data/trip/refined_raw_data'
            ourtput: image paths under output_path, e.g. ['data/trip/refined_raw_data/refined_data_0/raw_data', 'data/trip/refined_raw_data/refined_data_1/raw_data', 'data/trip/refined_raw_data/refined_data_2/raw_data', 'data/trip/refined_raw_data/refined_data_3/raw_data']
        """
        def setup_savedir(imput_path, output_path, i):
            raw_data_dirname = Path(imput_path).name
            tmp_save_dir = os.path.join(output_path, 'refined_data_'+str(i))
            if not os.path.exists(tmp_save_dir):
                os.mkdir(tmp_save_dir)
            save_dir = os.path.join(tmp_save_dir, raw_data_dirname)
            if not os.path.exists(tmp_save_dir):
                os.mkdir(save_dir)
            return save_dir
        
        img_lists = list(Path(imput_path).glob("**/*.png"))
        for img_path in img_lists:
            left_tops = [[0,0]]
            sub_img_paths = [str(img_path)]
            sub_img_list, left_top_list = sample(img_path =str(img_path), crop_size=(600,600))
            left_tops.extend(left_top_list)
            for j, img in enumerate(sub_img_list):
                save_dir = setup_savedir(imput_path, output_path, j)
                save_path = os.path.join(save_dir, str(j)+"_"+img_path.name)
                sub_img_paths.append(save_path)
                img.save((save_path))

            cropped_img_info = dict(sub_img_list = sub_img_paths, left_top_list = left_tops)
            with open(os.path.join(save_dir, img_path.name.split('.')[0]+'_croppedInfo.json'), 'w') as f:
                json.dump(cropped_img_info, f)
        
            

    def merge_refined_results(cropped_img_info_dir:str, orig_input_path:str, refine_input_path:str, output_path:str)->bool:
        """
            orig_input_path: data/trip/raw_data/
           refine_input_path: data/trip/refined_raw_data/. The predicted data will include mask, and json file: format example: /media/NAS/asus_nas/denso_data/20240613_103919_6/sms_front/mask_data, /media/NAS/asus_nas/denso_data/20240613_103919_6/sms_front/json_data
                        mask_data is an 1080*1920 numpy array, 0 is background, other numbers are objects' segmentation mask. json will save the classes and box infomation.
           output_path: data/trip/refined_raw_data/, the merged result will save in the data/trip/refined_raw_data/result folder 
        
        """
        def read_anno_mask(mask_path, json_path):
            anno_2d, obj_mask = [], []
            raw_mask = np.load(mask_path)
            with open(json_path, 'r') as f:
                raw_json = json.load(f)
            # convert raw_mask to binary
            for anno in raw_json['labels']:
                obj_mask.append(raw_mask == anno["instance_id"])
            anno_2d = raw_json['labels']
            return anno_2d, obj_mask

        def read_single_results(orig_input_path, refine_input_path, cropped_img_info):
            big_image_name = Path(cropped_img_info['sub_img_list'][0]).name.split('.')

            big_mask_path = Path(orig_input_path).glob("**/*"+ big_image_name +".npy")[0]
            big_json_path = Path(orig_input_path).glob("**/*"+ big_image_name +".json")[0]
            big_mask, big_anno = read_anno_mask(big_mask_path, big_json_path)

            cropped_mask, cropped_anno = [], []
            for cropped_img_path, left_top in zip(cropped_img_info['sub_img_list'][1:], cropped_img_info['left_top_list'][1:]):
                cropped_image_name = cropped_img_path.split('/')[-1].split('.')[0]
                mask_path = Path(refine_input_path).glob("**/*"+ cropped_image_name +".npy")[0]
                json_path = Path(refine_input_path).glob("**/*"+ cropped_image_name +".json")[0]
                anno_2d, obj_masks = read_anno_mask(mask_path, json_path)
                left_x, top_y = left_top
                # restore the sub coord to the big image coord
                for box_dict, obj_mask in zip(anno_2d, obj_masks):
                    box_dict['x1'] += left_x
                    box_dict['y1'] += top_y
                    box_dict['x2'] += left_x
                    box_dict['y2'] += top_y
                    cropped_anno.append(box_dict)
                    
                    sub_h, sub_w = obj_mask[0].shape
                    tmp = np.zeros((sub_h, sub_w), dtype= np.uint8)
                    tmp[:, top_y:top_y + sub_h, left_x: left_x + sub_w] = obj_mask
                    cropped_mask.append(tmp)

            return big_anno, cropped_mask, cropped_anno


        json_files = list(Path(cropped_img_info_dir).glob("**/*.json"))
        # todo: multiprocess
        for json_file in json_files:
            with open(str(json_file), 'r') as f:
                cropped_img_info = json.load(f)
            big_anno, cropped_mask, cropped_anno = read_single_results(orig_input_path, refine_input_path, cropped_img_info)
            merged_boxes_list, final_mask = merge_boxes_and_masks(big_anno, 
                                                                cropped_anno, 
                                                                cropped_mask, 
                                                                iou_threshold={'car':0.3,
                                                                            'truck':0.3,
                                                                            'van':0.3,
                                                                            'person':0.1,
                                                                            'motorcycle':0.1,
                                                                            'bicycle':0.1,
                                                                            'pole':0.1,
                                                                            'balusters':0.5,
                                                                            'bannister':0.5,
                                                                            'rail':0.1})
            
            # save the merged results
            big_image_name = Path(cropped_img_info['sub_img_list'][0]).name.split('.')
            h, w = final_mask.shape
            json_dict = { "mask_name": "mask_"+ big_image_name+".npy",
                        "mask_height": h,
                        "mask_width": w,
                        "promote_type": "mask",
                        "labels": merged_boxes_list }
            json_save_path = os.path.join(output_path, "mask_"+big_image_name+".json")
            with open(json_save_path, 'w') as f:
                json.dump(json_dict, f)
            mask_save_path = os.path.join(output_path, "mask_"+big_image_name+".npy")
            final_mask.tofile(mask_save_path)


if __name__=="__main__":
    """
    preprocessing the raw data
    """
    input_path = '/media/NAS/asus_nas/denso_data/20240613_103919_6/sms_front/raw_data'
    output_path = '/media/NAS/asus_nas/denso_data/20240613_103919_6/sms_front/refined_raw_data'
    RefinedModel.image_slicer(input_path, output_path)
    RefinedModel.merge_refined_results(output_path, output_path)