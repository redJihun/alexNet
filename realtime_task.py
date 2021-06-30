# For files
import os

# Deep-learning framework
import tensorflow as tf

# Manipulate
import numpy as np

# visualization
import cv2

# Data directory
from sklearn.preprocessing import minmax_scale

INPUT_ROOT_DIR = './input/task'
TEST_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'test')
OUTPUT_ROOT_DIR = './output/task'
LOG_DIR = os.path.join(OUTPUT_ROOT_DIR, 'tblogs')
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'train')

# 256x256으로 이미지 다운샘플링
def resize_image(img):
    # Read images from disk & resize
    # print('start resizing image')
    # image = cv2.imread(img)
    image = cv2.resize(img, dsize=(227, 227), interpolation=cv2.INTER_AREA)
    # print('end resizing')
    return image


def prediction(x, param):
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
    l6_dropout = tf.nn.dropout(l6_relu, rate=0.5, name='l6_dropout')

    # layer 7
    l7_fc = tf.nn.bias_add(tf.matmul(l6_dropout, param['w7']), param['b7'], name='l7_fc')
    l7_relu = tf.nn.relu(l7_fc, name='l7_relu')
    l7_dropout = tf.nn.dropout(l7_relu, rate=0.5, name='l7_dropout')

    # layer 8
    logits = tf.nn.bias_add(tf.matmul(l7_dropout, param['w8']), param['b8'], name='l8_fc')
    predict = tf.argmax(tf.nn.softmax(logits, 1), 1).numpy()

    return predict


def load_param(ckpts_path=OUTPUT_ROOT_DIR):

    # 클래스명 출력을 위해 디렉토리명 저장
    dirs = list()

    for dir in os.walk(TEST_IMG_DIR).__next__()[1]:
        dirs.append(dir)

    # 저장된 trained 모델(=trained parameters) 들을 불러온 후, test set 에서 loss 계산
    loaded_param = np.load(os.path.join(ckpts_path, 'best_model.npz'), allow_pickle=True)
    loaded_param = {key: loaded_param[key].item() for key in loaded_param}

    return dirs, loaded_param


def test(image, loaded_param, dirs):
    img = resize_image(image)

    pred = prediction(tf.cast(tf.reshape(img, [-1, 227, 227, 3]), dtype=tf.float32), loaded_param['arr_0'])

    return dirs[pred[0]]

def minmax(images):
    # R, G, B 채널을 각각 순회하며 계산된 값을 각 픽셀마다 가감
    scaled_img = np.array(img).copy()
    for idx in range(3):
        scaled_img[..., idx] = minmax_scale(img[..., idx], feature_range=(-1, 1))

    return scaled_img

if __name__ == "__main__":
    camera = cv2.VideoCapture(0);
    classes, param = load_param()
    # f, img = camera.read();
    # pred = test(image=img, loaded_param=param, dirs=classes)
    # foo = [1,2,3,4,5]
    # pred = 0
    # bar = 0

    while cv2.waitKey(1) != ord('q'):
        f, img = camera.read();
        pred = test(image=img, loaded_param=param, dirs=classes)
        # new = bar
        # if pred != new:
            # cv2.destroyWindow(pred)
            # cv2.destroyWindow('{}'.format(foo[pred]))
            # pred = new
        cv2.putText(img, pred, (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Prediction', img);
        # cv2.imshow('{}'.format(foo[pred]), img)
        # bar += 1

    camera.release()
    cv2.destroyAllWindows()
