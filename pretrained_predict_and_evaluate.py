import constants
from predict_and_evaluate import predict_and_evaluate


if __name__ == '__main__':
    ckpt_folder = 'pretrained_ckpt'
    predict_and_evaluate(constants.VAL_FLOW_FOLDER, constants.VAL_IMAGE_FOLDER, 
            ckpt_folder, label_file=constants.VAL_LABEL_FILE)
    predict_and_evaluate(constants.TEST_FLOW_FOLDER, constants.TEST_IMAGE_FOLDER,
            ckpt_folder)

