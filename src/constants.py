import os

DATA_FOLDER = 'data'

TRAIN_VIDEO_DOWNLOAD_LINK = 'https://marschallenge.s3.ap-northeast-2.amazonaws.com/train.mp4'
TEST_VIDEO_DOWNLOAD_LINK = 'https://marschallenge.s3.ap-northeast-2.amazonaws.com/test.mp4'
TRAIN_LABEL_DOWNLOAD_LINK = 'https://marschallenge.s3.ap-northeast-2.amazonaws.com/train.txt'

VIDEO_FILE = os.path.join(DATA_FOLDER, 'train_video.mp4')
TEST_VIDEO_FILE = os.path.join(DATA_FOLDER, 'test_video.mp4')
LABEL_FILE = os.path.join(DATA_FOLDER, 'labels.txt')

TRAIN_IMAGE_FOLDER = os.path.join(DATA_FOLDER, 'train_images')
TRAIN_LABEL_FILE = os.path.join(DATA_FOLDER, 'train.txt')
TRAIN_FLOW_FOLDER = os.path.join(DATA_FOLDER, 'train_flows')

VAL_IMAGE_FOLDER = os.path.join(DATA_FOLDER, 'val_images')
VAL_LABEL_FILE = os.path.join(DATA_FOLDER, 'val.txt')
VAL_FLOW_FOLDER = os.path.join(DATA_FOLDER, 'val_flows')

TEST_IMAGE_FOLDER = os.path.join(DATA_FOLDER, 'test_images')
TEST_FLOW_FOLDER = os.path.join(DATA_FOLDER, 'test_flows')

GMA_CONFIG_FILE = 'optical_flow_config_ckpt/gma_8x2_50k_kitti2015_288x960.py'
GMA_CKPT_FILE = 'optical_flow_config_ckpt/gma_8x2_50k_kitti2015_288x960.pth'

RESULTS_FOLDER = 'results'

