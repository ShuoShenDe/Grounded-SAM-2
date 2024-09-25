import os
import numpy as np
from utils.common_utils import CommonUtils

# Define the directory where the .npy files are stored
directory = '/media/NAS/sd_nas_03/shuo/denso_data/20240916/20240827_165656_2/sms_rear/mask_data_float'  # Replace with the actual path
output_dir = os.path.join(os.path.dirname(directory), "mask_data")  # Output directory is the same as input directory
CommonUtils.creat_dirs(output_dir)

# Traverse through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.npy'):  # Check if it's a .npy file
        file_path = os.path.join(directory, filename)
        
        # Load the .npy file
        data = np.load(file_path)
        print(data.dtype)
        if filename.endswith('_uint16.npy'):
            print(f"Skipping {filename} as it is already in uint16 format.")
            continue
        # Check if the data type is uint16
        if data.dtype != np.uint16:
            # Convert to uint16 if necessary
            data = data.astype(np.uint16)
            print(f"Converted {filename} to uint16")
        
        # # Save the new or unchanged data back to the file or new file
        new_file_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}.npy')
        np.save(new_file_path, data)
        print(f"Saved {new_file_path}")
