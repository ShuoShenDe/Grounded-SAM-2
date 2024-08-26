import json
import os
from collections import defaultdict, Counter
import shutil
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel
import numpy as np

class PostProcess:
    def remove_area(mask_data_dir, json_data_dir, frame_names):
        i = 0
        for frame_name in frame_names:
            print("remove_area:", i, frame_name)
            i += 1
            image_base_name = CommonUtils.get_image_base_name(frame_name)
            mask_data_path, mask_array, json_data_path, json_data = CommonUtils.get_mask_and_json(mask_data_dir, json_data_dir, image_base_name)
            removed_object_ids = []
            for object_id, object_info in json_data.labels.items():
                # 提取 mask_array 中属于 object_id 的区域
                # print(object_id)
                new_mask_array = np.where(mask_array == object_id, 1, 0)
                # 处理离散区域
                filtered_mask, _ = CommonUtils.remove_discrete_areas(new_mask_array)
                
                # 清空 mask_array 中 object_id 的区域
                mask_array[mask_array == object_id] = 0
                
                # 将 filtered_mask 中的 1 值区域重新标记为 object_id
                mask_array[filtered_mask == 1] = object_id

                # 更新 object_info
                object_info.mask = filtered_mask
                box = object_info.update_box_np()
                json_data.labels[object_id] = object_info
                if sum(box) == 0:
                    removed_object_ids.append(object_id)
            for object_id in removed_object_ids:
                json_data.labels.pop(object_id)
            # 确保更新后的 mask_array 被保存
            np.save(mask_data_path, mask_array)
            json_data.to_json(json_data_path, save_labels_as_list=True)


    def unified_classes(input_directory, output_directory = ""):
        instance_classes = defaultdict(list)
        
        if not output_directory:
            output_directory = input_directory.replace('json_data', 'json_data_new')
            CommonUtils.creat_dirs(output_directory)
        mask_data_dir = input_directory.replace('json_data', 'mask_data')
        video_dir = input_directory.replace('json_data', 'raw_data')

        if len(os.listdir(video_dir)) != len(os.listdir(input_directory)):
            mask_list = os.listdir(mask_data_dir)
            video_list = os.listdir(video_dir)
            video_list.sort()
            for index, name in enumerate(video_list):
                if "mask_"+name.replace(".png", ".npy") not in mask_list:
                    print(index, name)
            raise ValueError("The number of files in the input directory and mask_data_dir directory should be the same.")
        
        # if "result_post" in result_dir:
        #     CommonUtils.creat_dirs(result_dir)

        for filename in os.listdir(input_directory):
            if filename.endswith('.json'):
                filepath = os.path.join(input_directory, filename)
                frame_dict = MaskDictionaryModel().from_json(filepath)

                for key, label in frame_dict.labels.items():
                    instance_id = label.instance_id
                    class_name = label.class_name
                        # print(label['class_name'])
                    instance_classes[instance_id].append(class_name)
        # print(instance_classes)
        most_common_classes = {}
        for instance_id, classes in instance_classes.items():
            most_common_class = Counter(classes).most_common(1)[0][0]
            most_common_classes[instance_id] = most_common_class

        for filename in os.listdir(input_directory):
            if filename.endswith('.json'):
                input_filepath = os.path.join(input_directory, filename)
                output_filepath = os.path.join(output_directory, filename)
                
                frame_dict = MaskDictionaryModel().from_json(input_filepath)
                for key, label in frame_dict.labels.items():
                    instance_id = label.instance_id
                    label.class_name = most_common_classes[instance_id]

                frame_dict.to_json(output_filepath, save_labels_as_list=True)


        shutil.rmtree(input_directory)
        shutil.move(output_directory, input_directory)
        print("All files have been processed and saved to the new directory.")

        


if __name__ == '__main__':
    input_directory = '/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_5/sms_right/raw_data'
    json_data_dir = input_directory.replace('raw_data', 'json_data')
    mask_data_dir = input_directory.replace('raw_data', 'mask_data')
    result_dir = input_directory.replace('raw_data', 'result')
    frame_names = os.listdir(input_directory)
    # frame_names = ["1718268027293916000.png"]
    PostProcess.remove_area(mask_data_dir, json_data_dir, frame_names)
    
    PostProcess.unified_classes(json_data_dir)
    # CommonUtils.draw_masks_and_box_with_supervision(input_directory, mask_data_dir, json_data_dir, result_dir)
    