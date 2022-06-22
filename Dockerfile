from nvcr.io/nvidia/pytorch:20.03-py3

RUN apt-get update
RUN apt-get install -y libgl1
RUN pip install torch==1.10.1 torchvision==0.11.2
RUN pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.1/index.html
RUN pip install tqdm opencv-python==4.5.5.64 wandb mmdet gdown

WORKDIR /usr/local/src
RUN git clone https://github.com/open-mmlab/mmflow.git
WORKDIR /usr/local/src/mmflow
RUN pip install -r requirements/build.txt
RUN pip install -v -e .

