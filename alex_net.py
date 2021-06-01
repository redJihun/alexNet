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
def alexnet():
    input_layer = keras.Input(shape=(227, 227, 3), batch_size=128)
    l1conv = layers.Conv2D(filters=96, strides=(4, 4), kernel_size=(11, 11),
                           activation='relu', bias_initializer='zeros')(input_layer)
    l1lrn = tf.nn.local_response_normalization(input=l1conv, depth_radius=5, bias=2, alpha=10e-4, beta=0.75)
    l1max_pool = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(l1lrn)

    l2conv = layers.Conv2D(filters=256, strides=(1, 1), kernel_size=(5, 5), padding='same',
                           activation='relu', bias_initializer='ones')(l1max_pool)
    l2lrn = tf.nn.local_response_normalization(input=l2conv, depth_radius=5, bias=2, alpha=10e-4, beta=0.75)
    l2max_pool = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(l2lrn)

    l3conv = layers.Conv2D(filters=384, strides=(1, 1), kernel_size=(3, 3), padding='same',
                           activation='relu', bias_initializer='zeros')(l2max_pool)

    l4conv = layers.Conv2D(filters=384, strides=(1, 1), kernel_size=(3, 3), padding='same',
                           activation='relu', bias_initializer='ones')(l3conv)

    l5conv = layers.Conv2D(filters=256, strides=(1, 1), kernel_size=(3, 3), padding='same',
                           activation='relu', bias_initializer='ones')(l4conv)
    l5max_pool = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(l5conv)

    l6flatten = layers.Flatten()(l5max_pool)
    l6dropout = layers.Dropout(rate=0.5)(l6flatten)
    l6fc = layers.Dense(4096, activation='relu', bias_initializer='ones')(l6dropout)

    l7dropout = layers.Dropout(rate=0.5)(l6fc)
    l7fc = layers.Dense(4096, activation='relu', bias_initializer='ones')(l7dropout)

    l8output = layers.Dense(1000, activation='softmax', bias_initializer='zeros')(l7fc)

    return l8output


# Load dataset
train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True
)
train_ds = keras.preprocessing.image_dataset_from_directory(
    directory=TRAIN_IMG_DIR,
    labels='inferred',
    color_mode='rgb',
    batch_size=128,
    image_size=(256, 256)
)

valid_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True
)
valid_ds = keras.preprocessing.image_dataset_from_directory

print(train_ds)
# Launch the session
# with tf.Session() as sess:
#     tf.initialize_all_variables().run()
#
#     for i in range(NUM_EPOCHS):
