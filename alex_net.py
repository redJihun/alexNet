# For files
import os

# Deep-learning framework
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

# Hyperparameters
NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005         # == weight_decay
LR_INIT = 0.01            # == weight_init
IMAGE_DIM = 227
NUM_CLASSES = 1000

# Initialized the weights in each layer from a zero-mean Gaussian distribution(standard deviation=0.01)
# initialized bias with constant 1, 2/4/5 conv layers, fully-connected hidden layers
# else layers initialized with constant 0

# Data directory
INPUT_ROOT_DIR = '/home/alexnet/input'
TRAIN_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'train')
VAL_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'valid')
TEST_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'test')
OUTPUT_ROOT_DIR = '/home/alexnet/output'
LOG_DIR = os.path.join(OUTPUT_ROOT_DIR, 'tblogs')
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'models')

# Make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Make CNN model
# bias init 1/3/8 = 0, 2/4/5/6/7 = 1
def alexnet(X, parameters, p_keep):
    input_layer = keras.Input(shape=(227, 227, 3), batch_size=128)
    # l1conv = layers.Conv2D(filters=96, strides=(4, 4), kernel_size=(11, 11),
    #                        activation='relu', bias_initializer='zeros')(input_layer)
    l1a = tf.nn.relu(tf.nn.conv2d(input=X, filters=parameters['w1'], strides=4, padding='VALID'))
    l1lrn = tf.nn.local_response_normalization(input=l1a, depth_radius=5, bias=2, alpha=10e-4, beta=0.75)
    # l1max_pool = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(l1lrn)
    l1max_pool = tf.nn.max_pool(input=l1lrn, ksize=3, strides=2, padding='VALID')

    # l2conv = layers.Conv2D(filters=256, strides=(1, 1), kernel_size=(5, 5), padding='same',
    #                        activation='relu', bias_initializer='ones')(l1max_pool)
    l2a = tf.nn.relu(tf.nn.conv2d(input=l1max_pool, filters=parameters['w2'], strides=1, padding='VALID'))
    l2lrn = tf.nn.local_response_normalization(input=l2a, depth_radius=5, bias=2, alpha=10e-4, beta=0.75)
    # l2max_pool = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(l2lrn)
    l2max_pool = tf.nn.max_pool(input=l2lrn, ksize=3, strides=2, padding='VALID')

    # l3conv = layers.Conv2D(filters=384, strides=(1, 1), kernel_size=(3, 3), padding='same',
    #                        activation='relu', bias_initializer='zeros')(l2max_pool)
    l3a = tf.nn.relu(tf.nn.conv2d(input=l2max_pool, filters=parameters['w3'], strides=1, padding='SAME'))

    # l4conv = layers.Conv2D(filters=384, strides=(1, 1), kernel_size=(3, 3), padding='same',
    #                        activation='relu', bias_initializer='ones')(l3conv)
    l4a = tf.nn.relu(tf.nn.conv2d(input=l3a, filters=parameters['w4'], strides=1, padding='SAME'))

    # l5conv = layers.Conv2D(filters=256, strides=(1, 1), kernel_size=(3, 3), padding='same',
    #                        activation='relu', bias_initializer='ones')(l4conv)
    l5a = tf.nn.relu(tf.nn.conv2d(input=l4a, filters=parameters['w5'], strides=1, padding='SAME'))
    # l5max_pool = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(l5conv)
    l5max_pool = tf.nn.max_pool(input=l5a, ksize=3, strides=2, padding='VALID')

    # l6flatten = layers.Flatten()(l5max_pool)
    flattened = tf.reshape(l5max_pool, [-1, 6*6*256])
    # l6dropout = layers.Dropout(rate=0.5)(l6flatten)
    l6dropout = tf.nn.dropout(flattened, rate=p_keep)
    # l6fc = layers.Dense(4096, activation='relu', bias_initializer='ones')(l6dropout)
    l6fc = tf.nn.relu(tf.matmul(l6dropout, parameters['w6']))


    # l7dropout = layers.Dropout(rate=0.5)(l6fc)
    l7dropout = tf.nn.dropout(l6fc, rate=p_keep)
    # l7fc = layers.Dense(4096, activation='relu', bias_initializer='ones')(l7dropout)
    l7fc = tf.nn.relu(tf.matmul(l7dropout, parameters['w7']))

    # l8output = layers.Dense(1000, activation='softmax', bias_initializer='zeros')(l7fc)
    l8output = tf.nn.softmax(tf.matmul(l7fc, parameters['w8']))

    return l8output


# @todo Load dataset
# @todo image down sampling - 짧은 면 256픽셀 + 긴면 같은 비율로 줄임, 긴 면의 가운데 256픽셀 자름 -> 256x256 이미지
# @todo Data augmentation - crop, RGB(pca)

# Initialize variables
tf.random.set_seed(602)
parameters = {
    'w1': tf.Variable(tf.random.normal(shape=[11, 11, 3, 96], mean=0.0, stddev=0.01, dtype=tf.float32), name='w1'),
    'b1': tf.Variable(tf.zeros(shape=[96], name='b1')),
    'w2': tf.Variable(tf.random.normal(shape=[5, 5, 96, 256], mean=0.0, stddev=0.01, dtype=tf.float32), name='w2'),
    'b2': tf.Variable(tf.zeros(shape=[256], name='b2')),
    'w3': tf.Variable(tf.random.normal(shape=[3, 3, 256, 384], mean=0.0, stddev=0.01, dtype=tf.float32), name='w3'),
    'b3': tf.Variable(tf.zeros(shape=[384], name='b3')),
    'w4': tf.Variable(tf.random.normal(shape=[3, 3, 384, 384], mean=0.0, stddev=0.01, dtype=tf.float32), name='w4'),
    'b4': tf.Variable(tf.zeros(shape=[384], name='b4')),
    'w5': tf.Variable(tf.random.normal(shape=[3, 3, 384, 256], mean=0.0, stddev=0.01, dtype=tf.float32), name='w5'),
    'b5': tf.Variable(tf.zeros(shape=[256], name='b5')),
    'w6': tf.Variable(tf.random.normal(shape=[4096], mean=0.0, stddev=0.01, dtype=tf.float32), name='w6'),
    'b6': tf.Variable(tf.zeros(shape=[4096], name='b6')),
    'w7': tf.Variable(tf.random.normal(shape=[4096], mean=0.0, stddev=0.01, dtype=tf.float32), name='w7'),
    'b7': tf.Variable(tf.zeros(shape=[4096], name='b7')),
    'w8': tf.Variable(tf.random.normal(shape=[200], mean=0.0, stddev=0.01, dtype=tf.float32), name='w8'),
    'b8': tf.Variable(tf.zeros(shape=[200], name='b8')),
}

# @todo Do training
# Launch the session
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(NUM_EPOCHS):

# @todo Do validation check & model save
