```
source ~/Documents/airflow/tracking/bin/activate

export PYTHONPATH="${PYTHONPATH}:/media/NAS/sd_nas_01/shuo/tracking/Grounded-SAM-2"

python multi_grounded_local_sam2_tracking_demo_with_continuous_id.py -i /data/20240923/20240828_075313_1/sms_front/raw_data

```

# Transfer to rosbag

```
cd /media/NAS/sd_nas_03/shuo/denso_class_mapper
source ~/Documents/airflow/tracking/bin/activate

python3 denso_pre_class_mapper.py -i /media/NAS/sd_nas_03/shuo/denso_data/20240923/20240828_075313_1/sms_front -r
```


```
./ld_rosutil files2bag -i /media/NAS/sd_nas_03/shuo/denso_data/20240923/20240828_075313_1/sms_front -t json_data mask_data -f .json .npy -o /media/NAS/sd_nas_03/shuo/denso_data/20240923/20240828_075313_1/sms_front/pre_bag
```




