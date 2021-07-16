# For files
import os
import time

# Deep-learning framework
import tensorflow as tf
# import tensorflow_datasets as tfds
import tensorflow_addons as tfa

# Manipulate
import numpy as np
import random

# visualization
import cv2
from sklearn.preprocessing import minmax_scale

RANDOM_SEED = 602
# random.seed(RANDOM_SEED)
# tf.random.set_seed(RANDOM_SEED)

# Hyper-parameters
NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005         # == weight_decay
LR_INIT = 0.01
NUM_CLASSES = 6
# IMAGENET_MEAN = np.array([104., 117., 124.], dtype=np.float)

# Data directory
INPUT_ROOT_DIR = './input/task'
TEST_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'test')
OUTPUT_ROOT_DIR = './output/task'
LOG_DIR = os.path.join(OUTPUT_ROOT_DIR, 'tblogs')
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'train')

# Make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


########################################################################################################################
# 이미지 로드 및 전처리 + 데이터 증강 함수들

# After init variables, append imagefile's path and label(in number, origin is name of sub-directory).
# Set the path of root dir, and use os.walk(root_dir) for append all images in sub-dir.
def load_imagepaths(path):
    imagepaths, labels = list(), list()
    label = 0
    classes = os.walk(path).__next__()[1]
    for c in classes:
        c_dir = os.path.join(path, c)
        walk = os.walk(c_dir).__next__()
        # Add each image
        for sample in walk[2]:
            imagepaths.append(os.path.join(c_dir, sample))
            labels.append(label)
        # next directory
        label += 1

    return imagepaths, labels


# 256x256으로 이미지 다운샘플링
def resize_images(imgpaths, w, h):
    # Read images from disk & resize
    images = list()
    for img in imgpaths:
        try:
            image = cv2.imread(img)
            image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_AREA)
            images.append(image)
        except:
            image = tf.io.read_file(img)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize_with_crop_or_pad(image, target_height=w, target_width=h)
            images.append(image)
    return images


def minmax(images, min, max):
    scaled_images = list()
    for img in images:
        # R, G, B 채널을 각각 순회하며 계산된 값을 각 픽셀마다 가감
        scaled_img = np.array(img).copy()
        for idx in range(3):
            scaled_img[..., idx] = minmax_scale(img[..., idx], feature_range=(min, max))
        scaled_images.append(scaled_img)

    return scaled_images


# 증강된 데이터를 입력받아 셔플 후 TF 데이터셋으로 리턴
def make_dataset(images, labels):
    # Shuffle with seed can keep the data-label pair. Without shuffle, data have same label in range.
    foo = list(zip(images, labels))
    random.Random(RANDOM_SEED).shuffle(foo)
    images, labels = zip(*foo)

    # Convert to Tensor
    test_X, test_Y = tf.convert_to_tensor(images, dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.int32)

    return test_X, test_Y
########################################################################################################################


# @tf.function
def loss(name, x, y, param):
    # inputs = tf.constant(x, name='inputs')
    inputs = x

    # layer 1
    l1_convolve = tf.nn.conv2d(input=inputs, filters=param['w1'], strides=4, padding='VALID', name='l1_convolve')
    l1_bias = tf.reshape(tf.nn.bias_add(l1_convolve, param['b1']), tf.shape(l1_convolve), name='l1_bias')
    l1_relu = tf.nn.relu(l1_bias, name='l1_relu')
    l1_norm = tf.nn.lrn(input=l1_relu, depth_radius=5, alpha=10e-4, beta=0.75, bias=2.0, name='l1_norm')
    l1_pool = tf.nn.max_pool(input=l1_norm, ksize=3, strides=2, padding='VALID', name='l1_pool')

    # layer 2
    l2_convolve = tf.nn.conv2d(input=l1_pool, filters=param['w2'], strides=1, padding='SAME', name='l2_convolve')
    l2_bias = tf.reshape(tf.nn.bias_add(l2_convolve, param['b2']), tf.shape(l2_convolve), name='l2_bias')
    l2_relu = tf.nn.relu(l2_bias, name='l2_relu')
    l2_norm = tf.nn.lrn(input=l2_relu, depth_radius=5, alpha=10e-4, beta=0.75, bias=2.0, name='l2_norm')
    l2_pool = tf.nn.max_pool(input=l2_norm, ksize=3, strides=2, padding='VALID', name='l2_pool')

    # layer 3
    l3_convolve = tf.nn.conv2d(input=l2_pool, filters=param['w3'], strides=1, padding='SAME', name='l3_convolve')
    l3_bias = tf.reshape(tf.nn.bias_add(l3_convolve, param['b3']), tf.shape(l3_convolve), name='l3_bias')
    l3_relu = tf.nn.relu(l3_bias, name='l3_relu')

    # layer 4
    l4_convolve = tf.nn.conv2d(input=l3_relu, filters=param['w4'], strides=1, padding='SAME', name='l4_convolve')
    l4_bias = tf.reshape(tf.nn.bias_add(l4_convolve, param['b4']), tf.shape(l4_convolve), name='l4_bias')
    l4_relu = tf.nn.relu(l4_bias, name='l4_relu')

    # layer 5
    l5_convolve = tf.nn.conv2d(input=l4_relu, filters=param['w5'], strides=1, padding='SAME', name='l5_convolve')
    l5_bias = tf.reshape(tf.nn.bias_add(l5_convolve, param['b5']), tf.shape(l5_convolve), name='l5_bias')
    l5_relu = tf.nn.relu(l5_bias, name='l5_relu')
    l5_pool = tf.nn.max_pool(input=l5_relu, ksize=3, strides=2, padding='VALID', name='l5_pool')

    # layer 6
    l6_flattened = tf.reshape(l5_pool, [-1, tf.shape(param['w6'])[0]], name='l6_flattened')
    l6_fc = tf.nn.bias_add(tf.matmul(l6_flattened, param['w6']), param['b6'], name='l6_fc')
    l6_relu = tf.nn.relu(l6_fc, name='l6_relu')

    # layer 7
    l7_fc = tf.nn.bias_add(tf.matmul(l6_relu, param['w7']), param['b7'], name='l7_fc')
    l7_relu = tf.nn.relu(l7_fc, name='l7_relu')

    # layer 8
    logits = tf.nn.bias_add(tf.matmul(l7_relu, param['w8']), param['b8'], name='l8_fc')
    predict = tf.argmax(logits, 1).numpy()

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(loss)
    target = y
    accuracy = np.sum(predict == target) / len(target)

    print('model\t=\t{}\tloss={}\taccuracy={}'.format(name, loss.numpy(), accuracy))

    return loss, predict


def test(imgs_path=TEST_IMG_DIR, ckpts_path=OUTPUT_ROOT_DIR):
    # 사전에 정의한 load_imagepaths 함수의 매개변수로 이미지를 저장한 파일경로의 루트 디렉토리 지정
    filepaths, labels = load_imagepaths(imgs_path)

    images = resize_images(filepaths, 227, 227)

    images = minmax(images, -1.0, 1.0)

    test_X, test_Y = make_dataset(images, labels)

    # 클래스명 출력을 위해 디렉토리명 저장
    dirs = list()

    for dir in os.walk(TEST_IMG_DIR).__next__()[1]:
        dirs.append(dir)

    # 저장된 trained 모델(=trained parameters) 들을 불러온 후, test set 에서 loss 계산
    loaded_param = np.load(os.path.join(OUTPUT_ROOT_DIR, 'best_model.npz'), allow_pickle=True)
    loaded_param = {key: loaded_param[key].item() for key in loaded_param}
    _, prediction = loss(name='best_model', x=test_X, y=test_Y, param=loaded_param['arr_0'])

    test_X, test_Y = tf.data.Dataset.from_tensor_slices(test_X), tf.data.Dataset.from_tensor_slices(test_Y)
    accs = list()

    for x, y, pred in zip(list(test_X.as_numpy_iterator()), list(test_Y.as_numpy_iterator()), prediction):
        # minmax(x, 0, 255)
        # print(y,pred)
        # print('Target = {}\t Predict = {}\n'.format(dirs[y], dirs[pred]))
        # cv2.putText(x, dirs[pred], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2, cv2.LINE_AA)
        # cv2.imshow('test', np.array(x, dtype=np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        accs.append(1 if y == pred else 0)
    print('Test accuracy = {}'.format(sum(accs) / len(accs)))


test()

