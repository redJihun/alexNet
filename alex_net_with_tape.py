# For load files
import os

# Deep-learning framework
import tensorflow as tf
import tensorflow_addons as tfa     # For SGDW(SGD with weight-decay) optimizer

# Manipulate tensor, array, etc.
import numpy as np
import random
RANDOM_SEED = 610
tf.random.set_seed(RANDOM_SEED)


#######################################################################################################################
# Set hyper-parameters
# Use in training session
NUM_EPOCHS = 90
BATCH_SIZE = 128

# Use in SGDW optimizer
MOMENTUM = 0.9
LR_DECAY = 0.0005       # == weight_decay
LR_INIT = 0.01

# Dataset
IMAGE_DIM = 227
NUM_CLASSES = 5
IMAGENET_MEAN = np.array([104., 117., 124.], dtype=np.float)


#######################################################################################################################
# Data directory
INPUT_ROOT_DIR = './input'
TRAIN_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'train')
OUTPUT_ROOT_DIR = './output'
LOG_DIR = os.path.join(OUTPUT_ROOT_DIR, 'logs')
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'models')

# Make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


#######################################################################################################################
# Load data & Make Tensorflow-dataset
# 리스트 변수 선언 후, 모든 이미지 파일의 경로를 리리스트에 추가
# 이미지 데이터의 루트 디렉토리 경로를 매개변수로 받아 텐서플로우 dataset을 리턴해주는 함수
def load_dataset(path):
    # 이미지의 파일경로와 이미지의 라벨(이미지 파일이 들어있는 하위 디렉토리의 이름)을 추가해줄 리스트
    imagepaths, labels = list(), list()
    # 문자열 형태의 라벨을 정수형으로 추가하기 위해 0으로 초기화된 변수 선언, 이후 하위 디렉토리를 순회하면서 1씩 늘어남
    label = 0
    # 루트 디렉토리 경로로 접근, 하위 디렉토
    classes = sorted(os.walk(path).__next__()[1])