import argparse
import os
import urllib.request
import cv2
import torch
from torch.multiprocessing import set_start_method
import multiprocessing
from mmflow.apis import init_model, inference_model
from mmflow.datasets import visualize_flow

import constants


def download_videos():
    os.makedirs(constants.DATA_FOLDER, exist_ok=True)
    if os.path.isfile(constants.VIDEO_FILE):
        print('Train video already existed, not downloading')
    else:
        print('Downloading train video ...')
        urllib.request.urlretrieve(constants.TRAIN_VIDEO_DOWNLOAD_LINK, constants.VIDEO_FILE)
    if os.path.isfile(constants.LABEL_FILE):
        print('Train label already existed, not downloading')
    else:
        print('Downloading train label ...')
        urllib.request.urlretrieve(constants.TRAIN_LABEL_DOWNLOAD_LINK, constants.LABEL_FILE)

    if os.path.isfile(constants.TEST_VIDEO_FILE):
        print('Test video already existed, not downloading')
    else:
        print('Downloading test video ...')
        urllib.request.urlretrieve(constants.TEST_VIDEO_DOWNLOAD_LINK, constants.TEST_VIDEO_FILE)


def split_labels(label_file, train_label_file, val_label_file, validation_ratio):
    if label_file:
        print('Splitting labels into train and validation partitions...')
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                labels.append(line.strip())
        num_frames = len(labels)
        val_num_frames = int(num_frames * validation_ratio)
        train_num_frames = num_frames - val_num_frames
        with open(train_label_file, 'w') as f:
            for label in labels[:train_num_frames]:
                f.write('{}\n'.format(label))
        with open(val_label_file, 'w') as f:
            for label in labels[train_num_frames:]:
                f.write('{}\n'.format(label))


def video_to_images(video_file, image_folder, val_image_folder=None,
                    label_file=None, train_label_file=None, val_label_file=None, validation_ratio=None):
    os.makedirs(image_folder, exist_ok=True)
    if val_image_folder:
        os.makedirs(val_image_folder, exist_ok=True)

    print('Converting video file: {} to frames'.format(video_file))
    video_reader = cv2.VideoCapture(video_file)
    num_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total number of frames in video : {}'.format(num_frames))

    if val_image_folder:
        val_num_frames = int(num_frames * validation_ratio)
        train_num_frames = num_frames - val_num_frames
        print('Number of train frames: {}, validation frames: {}'.format(train_num_frames, val_num_frames))
    else:
        train_num_frames = num_frames
        print('Number of test frames: {}'.format(train_num_frames))

    frame_count = 0
    while True:
        flag, frame = video_reader.read()
        if not flag:
            break
        if val_image_folder and frame_count == train_num_frames:
            print('Saving validation frames ...')
            frame_folder = val_image_folder

        if frame_count < train_num_frames:
            img_name = os.path.join(image_folder, '{}.jpg'.format(str(frame_count)))
        else:
            img_name = os.path.join(val_image_folder, '{}.jpg'.format(str(frame_count)))

        cv2.imwrite(img_name, frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print('{} frames read and saved'.format(frame_count))

    video_reader.release()


def images_to_flows_multiproc(image_folder, flow_folder, num_gpus, transform, dist=1):
    images_partitions = []
    all_images = os.listdir(image_folder)
    for gpu in range(num_gpus):
        if gpu < num_gpus - 1:
            images_partitions.append(all_images[int(gpu / num_gpus * len(all_images)):
                                                int((gpu + 1) / num_gpus * len(all_images))])
        else:
            images_partitions.append(all_images[int(gpu / num_gpus * len(all_images)):])

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    processes = []
    for gpu in range(num_gpus):
        images_list = images_partitions[gpu].copy()
        p = multiprocessing.Process(target=_images_to_flows_single,
                                    args=(image_folder, flow_folder, images_partitions[gpu], gpu, transform, dist))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def _images_to_flows_single(image_folder, flow_folder, images_list, gpu, transform, dist):
    print('[GPU {}] Convert images in {} to flows in {}'.format(gpu, image_folder, flow_folder))
    os.makedirs(flow_folder, exist_ok=True)
    model = init_model(constants.GMA_CONFIG_FILE, constants.GMA_CKPT_FILE, device='cuda:{}'.format(gpu))
    count = 0
    print('[GPU {}] total number of images to convert: {}'.format(gpu, len(images_list)))
    for filename in images_list:
        if transform:
            splits = filename.split('_')
            prevfilename = str(int(splits[0]) - dist) + '_' + '_'.join(splits[1:])
        else:
            splits = filename.split('.')
            prevfilename = str(int(splits[0]) - dist) + '.' + '.'.join(splits[1:])
        if not os.path.exists(os.path.join(image_folder, prevfilename)):
            continue
        filepath = os.path.join(image_folder, filename)
        prevfilepath = os.path.join(image_folder, prevfilename)
        result = inference_model(model, prevfilepath, filepath)
        outpath = os.path.join(flow_folder, filename)
        visualize_flow(result, save_file=outpath)
        count += 1
        if count % 100 == 0:
            print('[GPU {}] converted {} images to flows'.format(gpu, count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.3,
        help="Validation ratio",
    )
    args = parser.parse_args()

    download_videos()
    video_to_images(constants.VIDEO_FILE, constants.TRAIN_IMAGE_FOLDER, constants.VAL_IMAGE_FOLDER, 
            constants.LABEL_FILE, constants.TRAIN_LABEL_FILE, constants.VAL_LABEL_FILE, validation_ratio=args.validation_ratio)
    video_to_images(constants.TEST_VIDEO_FILE, constants.TEST_IMAGE_FOLDER)

    images_to_flows_multiproc(constants.TRAIN_IMAGE_FOLDER, constants.TRAIN_FLOW_FOLDER, torch.cuda.device_count(), False)
    images_to_flows_multiproc(constants.VAL_IMAGE_FOLDER, constants.VAL_FLOW_FOLDER, torch.cuda.device_count(), False)
    images_to_flows_multiproc(constants.TEST_IMAGE_FOLDER, constants.TEST_FLOW_FOLDER, torch.cuda.device_count(), False)
    split_labels(constants.LABEL_FILE, constants.TRAIN_LABEL_FILE, constants.VAL_LABEL_FILE, validation_ratio=args.validation_ratio)

