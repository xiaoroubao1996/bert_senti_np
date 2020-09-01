import os

file_path = os.path.dirname(__file__)

# model_dir = os.path.join(file_path, 'chinese_L-12_H-768_A-12/')
model_dir = os.path.join('../chinese_L-12_H-768_A-12/')
config_name = os.path.join(model_dir, 'bert_config.json')
ckpt_name = os.path.join(model_dir, 'bert_model.ckpt')
output_dir = os.path.join(file_path, 'tmp/result/')
vocab_file = os.path.join(model_dir, 'vocab.txt')
data_dir = os.path.join(model_dir, '../data/')


DATA_COLUMN = "review"
LABEL_COLUMN = "label"

MAX_SEQ_LENGTH = 128

# 标签list
label_list = ["happy", "angry", "disgust", "sad"]

# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

