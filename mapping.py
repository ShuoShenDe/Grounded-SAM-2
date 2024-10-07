import json
import os


# 定义映射规则
class_mapping = {
    "rail":"fence",
    "person":"pedestrian",
    "motorcycle":"motorbike",
    "pole":"pole",
    "bicycle":"bicycle",
    "car":"car",
    "van":"van",
    "truck":"truck"
}

default_class = "other"

input_directory = "/data/20240923/20240828_075313_1/sms_front/json_data_old"
output_directory = "/data/20240923/20240828_075313_1/sms_front/json_data"

# 如果输出目录不存在，创建输出目录
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


for filename in os.listdir(input_directory):
    
    if filename.endswith(".json"):
        input_path = os.path.join(input_directory, filename)
        print("input_path,", input_path)
        # 读取 JSON 文件
        with open(input_path, 'r') as file:
            data = json.load(file)
        
        # 遍历并转换 class_name
        new_labels_list = []
        for label in data.get('labels', []):
            print("label,", label)
            class_name = label.get('class_name')
            # 使用映射规则转换，如果没有匹配则使用默认类别
            label['class_name'] = class_mapping.get(class_name, default_class)
            new_labels_list.append(label)
            print("chaning class name from ", class_name, " to ", label['class_name'])
        data["labels"] = new_labels_list
        # 保存修改后的 JSON 文件到输出目录
        output_path = os.path.join(output_directory, filename)
        with open(output_path, 'w') as outfile:
            json.dump(data, outfile, indent=4)

        print(f"Processed {filename}")