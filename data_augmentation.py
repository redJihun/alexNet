# For files
import os
from PIL import Image

# Deep-learning framework
import cv2
import tensorflow as tf

# Manipulate
import numpy as np
import random

RANDOM_SEED = 602
IMAGENET_MEAN = np.array([8., 8., 8.], dtype=np.float)

# Data directory
INPUT_ROOT_DIR = './input/task'
TRAIN_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'train')
VALID_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'valid')
OUTPUT_ROOT_DIR = './output/task'


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

    # 32개 데이터씩 split 하므로 미리 원 데이터를 셔플해준 후에 증강 진행.
    foo = list(zip(imagepaths, labels))
    random.Random(RANDOM_SEED).shuffle(foo)
    imagepaths, labels = zip(*foo)

    return imagepaths, labels


# 256x256으로 이미지 다운샘플링
def resize_images(imgpaths):
    # Read images from disk & resize
    print('Resizing...')
    for path in imgpaths:
        image = cv2.imread(path)
        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path+'_resized.jpeg', image)
        os.remove(path)
    print('End resizing.')


# horizontal reflection
def flip_image(fpaths):
    print('Flipping...')
    for path in fpaths:
        flipped_image = cv2.imread(path)
        flipped_image = cv2.flip(flipped_image, 1)
        cv2.imwrite(path+'_flipped.jpeg', flipped_image)
    print('End flipping.')


# RGB jittering
def fancy_pca(img_paths, alpha_std=0.1):
    print('Jittering...')
    for path in img_paths:
        img = cv2.imread(path)
        orig_img = np.array(img, dtype=np.float).copy()

        # 이미지 픽셀값에서 이미지넷 평균 픽셀값을 빼줌(평균 픽셀값은 사전에 정의됨)
        img_rs = np.reshape(img, (-1, 3))
        img_centered = img_rs - IMAGENET_MEAN

        # 해당 이미지의 공분산 행렬 구함
        img_cov = np.cov(img_centered, rowvar=False)

        # 고유벡터, 고유값 구함
        eig_vals, eig_vecs = np.linalg.eigh(img_cov)
        sort_perm = eig_vals[::-1].argsort()
        eig_vals[::-1].sort()
        eig_vecs = eig_vecs[:, sort_perm]

        # 고유벡터 세개를 쌓아서 3x3 행렬로 만듦
        m1 = np.column_stack((eig_vecs))
        m2 = np.zeros((3, 1))

        # 랜덤값과 고유값을 곱함
        alpha = np.random.normal(0, alpha_std)
        m2[:, 0] = alpha * eig_vals

        # 3x3 고유벡터 행렬과 3x1 랜덤값*고유값 행렬을 곱해서 3x1 행렬을 얻음(RGB 채널에 가감해줄 값)
        add_vect = np.matrix(m1) * np.matrix(m2)

        # R, G, B 채널을 각각 순회하며 계산된 값을 각 픽셀마다 가감
        for idx in range(3):
            orig_img[..., idx] += add_vect[idx]

        pca_img = orig_img
        cv2.imwrite(path+'_jitter.jpeg', pca_img)
    print('End jittering')


# Image cropping
def crop_image(paths):
    print('Cropping...')
    for path in paths:
        img = cv2.imread(path)
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        # for x_idx in range(img.shape[0]-227):
        #     for y_idx in range(img.shape[1]-227):
        #         cropped_image = img[x_idx:x_idx+227, y_idx:y_idx+227]
        #         cv2.imwrite(path+'_({},{})cropped.jpeg'.format(x_idx, y_idx), cropped_image)
        # left-top
        cropped_image = img[0:227, 0:227]
        cv2.imwrite(path+'_LT_cropped.jpeg', cropped_image)
        # right-top
        cropped_image = img[img.shape[0]-227:, 0:227]
        cv2.imwrite(path+'_RT_cropped.jpeg', cropped_image)
        # center
        cropped_image = img[14:img.shape[0]-15, 14:img.shape[1]-15]
        cv2.imwrite(path+'_CT_cropped.jpeg', cropped_image)
        # left-bottom
        cropped_image = img[0:227, img.shape[1]-227:]
        cv2.imwrite(path+'_LB_cropped.jpeg', cropped_image)
        # right-bottom
        cropped_image = img[img.shape[0]-227:, img.shape[1]-227:]
        cv2.imwrite(path+'_RB_cropped.jpeg', cropped_image)

        os.remove(path)
    print('End Cropping')


if __name__ == "__main__":
    img_paths, _ = load_imagepaths(TRAIN_IMG_DIR)
    resize_images(img_paths)
    img_paths, _ = load_imagepaths(TRAIN_IMG_DIR)
    fancy_pca(img_paths)
    img_paths, _ = load_imagepaths(TRAIN_IMG_DIR)
    flip_image(img_paths)
    img_paths, _ = load_imagepaths(TRAIN_IMG_DIR)
    crop_image(img_paths)
