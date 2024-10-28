# Get version of CUDA and enable it for compilation if CUDA > 11.0
# This solves https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/53
# and https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/84
# when running in Docker
# Check if nvcc is installed
NVCC := $(shell which nvcc)
ifeq ($(NVCC),)
	# NVCC not found
	USE_CUDA := 0
	NVCC_VERSION := "not installed"
else
	NVCC_VERSION := $(shell nvcc --version | grep -oP 'release \K[0-9.]+')
	USE_CUDA := $(shell echo "$(NVCC_VERSION) > 11" | bc -l)
endif

# Add the list of supported ARCHs
ifeq ($(USE_CUDA), 1)
	TORCH_CUDA_ARCH_LIST := "7.0;7.5;8.0;8.6+PTX"
	BUILD_MESSAGE := "I will try to build the image with CUDA support"
else
	TORCH_CUDA_ARCH_LIST :=
	BUILD_MESSAGE := "CUDA $(NVCC_VERSION) is not supported"
endif


build-image:
	@echo $(BUILD_MESSAGE)
	docker build --build-arg USE_CUDA=$(USE_CUDA) \
	--build-arg TORCH_ARCH=$(TORCH_CUDA_ARCH_LIST) \
	-t susanshende/denso_prelabling:2.0 .
run:
	docker run --gpus all -it --rm --net=host --privileged \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /media/NAS/sd_nas_03/shuo/denso_data/:/media/NAS/sd_nas_03/shuo/denso_data/ \
	--name=grounded_sam2 \
	--ipc=host -it susanshende/denso_prelabling:2.0
	-i /media/NAS/sd_nas_03/shuo/denso_data/20241007/20240827_161112_1/sms_front/raw_data

# -v "${PWD}":/home/appuser/Grounded-SAM-2 \



# docker run --gpus all -it --rm --net=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -v /data/airflow_appdata_tmp/DN_PRELABEL_DN_3_20241015_15_52_05_scheduled_2022_01_01T00_00_00_denso_prelabeling_denso_prelabeling_sms_front:/data/airflow_appdata_tmp/DN_PRELABEL_DN_3_20241015_15_52_05_scheduled_2022_01_01T00_00_00_denso_prelabeling_denso_prelabeling_sms_front --name=denso_prelabling --ipc=host -it susanshende/denso_prelabling:2.0 -i /data/airflow_appdata_tmp/DN_PRELABEL_DN_3_20241015_15_52_05_scheduled_2022_01_01T00_00_00_denso_prelabeling_denso_prelabeling_sms_front/sms_front/raw_data
