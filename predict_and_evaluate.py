import os
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

from model import SpeedCNN


def prediction_results_plot(preds, labels, run_folder):
    print('saving prediction plot')
    plot_file = os.path.join(run_folder,'val_predictions.png')
    frame_range = range(1, len(preds) + 1)
    plt.plot(frame_range, preds, label='predictions')
    plt.plot(frame_range, labels, label='labels')
    plt.xlabel('frame')
    plt.ylabel('vehicle speed')
    plt.title('Validation vehicle speed predictions')
    plt.legend()
    plt.savefig(plot_file)
    plt.clf()


def predict_and_evaluate(flow_folder, img_folder, run_folder, label_file=None):
    model_ = SpeedCNN()
    model_ = torch.nn.DataParallel(model_)
    ckpt_file = os.path.join(run_folder, 'best_loss.pt')
    model_.load_state_dict(torch.load(ckpt_file))
    print('loaded model checkpoint at {}'.format(ckpt_file))

    model_.to('cuda:0')
    model_.eval()

    if label_file:
        pred_file = os.path.join(run_folder, 'val_predictions.txt')
    else:
        pred_file = os.path.join(run_folder, 'test_predictions.txt')

    print('generating predictions using flow_folder {}, img_folder {}, and run_folder {}'.format(flow_folder, img_folder, run_folder))
    preds = []
    flow_list = os.listdir(flow_folder)
    flow_list.sort()
    count = 0
    for flow_name in flow_list:
        flow_path = os.path.join(flow_folder, flow_name)
        img_path = os.path.join(img_folder, flow_name)
        flow = cv2.imread(flow_path)
        img = cv2.imread(img_path)
        flow_height, flow_width, flow_channels = flow.shape
        img_resize = cv2.resize(img, (flow_width, flow_height))
        combined_flow = 0.1 * img_resize + flow
        
        combined_flow = cv2.normalize(combined_flow, None, alpha=-1, beta=1, 
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        combined_flow = cv2.resize(combined_flow, (0, 0), fx=0.5, fy=0.5)
        combined_flow = np.moveaxis(combined_flow, -1, 0)
        combined_flow = np.expand_dims(combined_flow, axis=0)
        combined_flow = torch.from_numpy(combined_flow)

        pred = model_(combined_flow).item()
        preds.append(pred)
        count += 1
        if count % 100 == 0:
            print('{} predictions generated'.format(count))

    with open(pred_file, 'w') as f:
        for i in range(len(preds)):
            f.write('file: {}, prediction: {}\n'.format(flow_list[i], preds[i]))

    if label_file:
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                labels.append(float(line.strip()))
        labels = labels[1:]
        print('evaluating prediction error with {} labels and {} predictions'.format(len(labels), len(preds)))

        mse = np.square(np.subtract(preds, labels)).mean()
        val_mse_file = os.path.join(run_folder, 'val_mse.txt')
        with open(val_mse_file, 'w') as f:
            f.write(val_mse_file + '\n')
        print('mean square error {} saved at {}'.format(mse, val_mse_file))

        prediction_results_plot(preds, labels, run_folder)

