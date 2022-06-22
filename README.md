# Mars Challenge

# Results summary
Best validation MSE using 30% validation set:

Please check report.pdf file for summary of my experiment process.

# Environment setup
Please use the provided Dockerfile to generate the necessary environment for model training and recreating the reported results. An example of necessary commands to create such an environment is provided below.

## Build Docker image
```
sudo docker build --network host --tag [image_name] .
```

## Create container from image
```
sudo docker run -d -it --network host --ipc host --gpus all -v [source_codes_path]:[source_codes_path] [image_name]:latest
```

## Run container
```
sudo docker exec -it [container_id] bash
cd [source_codes_path]
```

# Data preparation
Please run below command for data preparation. This command will download the Mars Challenge videos, convert the videos to frames and save them into train/val/test sets, convert the frames to optical flows and save them into train/val/test sets, and split labels into train/val sets.
```
python prepare_data.py
```

# Recreate reported results (no training)
Please run below command to recreate reported results. The results, which include speed predictions and MSE results saved in text files and plots, will be saved in the 'pretrained_ckpt' folder.
```
python pretrained_predict_and_evaluate.py
```

# Training, evaluation, and testing
The below command will automatically train model, generate predictions for val set and evaluate them, and generate predictions for test set. The results, which include checkpoint, speed predictions, and MSE results, can be found in the latest sub-folder in the 'results' folder.
```
python train_val_test.py
```
