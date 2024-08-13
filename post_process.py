import json
import os
from collections import defaultdict, Counter
import shutil
from utils.common_utils import CommonUtils


def unified_classes(input_directory, output_directory = ""):
    instance_classes = defaultdict(list)
    
    if not output_directory:
        output_directory = input_directory.replace('json_data', 'json_data_new')
    mask_data_dir = input_directory.replace('json_data', 'mask_data')
    video_dir = input_directory.replace('json_data', 'raw_data')
    result_dir = input_directory.replace('json_data', 'result_post')

    if len(os.listdir(video_dir)) != len(os.listdir(input_directory)):
        print("len(os.listdir(video_dir))", len(os.listdir(video_dir)))
        print("len(os.listdir(input_directory))", len(os.listdir(input_directory)))
        mask_list = os.listdir(mask_data_dir)
        video_list = os.listdir(video_dir)
        video_list.sort()
        for index, name in enumerate(video_list):
            if "mask_"+name.replace(".png", ".npy") not in mask_list:
                print(index, name)
        raise ValueError("The number of files in the input directory and mask_data_dir directory should be the same.")
    CommonUtils.creat_dirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):
            filepath = os.path.join(input_directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                labels = data.get('labels', {})
                for key, label in labels.items():
                    instance_id = label['instance_id']
                    if label['class_name'] == 'person':
                        label['class_name'] = 'pedestrian'
                    class_name = label['class_name']
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
            
            with open(input_filepath, 'r') as file:
                data = json.load(file)
            
            # update class_name
            labels = data.get('labels', {})
            for key, label in labels.items():
                instance_id = label['instance_id']
                # update class_name
                label['class_name'] = most_common_classes[instance_id]
            
            # save to json
            with open(output_filepath, 'w') as file:
                json.dump(data, file, indent=4)

    shutil.rmtree(input_directory)
    shutil.move(output_directory, input_directory)
    print("All files have been processed and saved to the new directory.")

    # CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, input_directory, result_dir)


if __name__ == '__main__':
    input_directory = '/media/NAS/sd_nas_01/shuo/denso_data/20240613_101744_6/sms_right/json_data'
    unified_classes(input_directory)