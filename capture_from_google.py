# # For load files
# import os
#
# # Deep-learning framework
# import tensorflow as tf
# import tensorflow_addons as tfa     # For SGDW(SGD with weight-decay) optimizer
#
# # Manipulate tensor, array, etc.
# import numpy as np
# import random
# RANDOM_SEED = 610
# tf.random.set_seed(RANDOM_SEED)
#
# import cv2
#
# from sklearn.preprocessing import minmax_scale
#
#
# #######################################################################################################################
# # Set hyper-parameters
# # Use in training session
# NUM_EPOCHS = 90
# BATCH_SIZE = 128
#
# # Use in SGDW optimizer
# MOMENTUM = 0.9
# LR_DECAY = 0.0005       # == weight_decay
# LR_INIT = 0.01
#
# # Dataset
# IMAGE_DIM = 227
# NUM_CLASSES = 5
# IMAGENET_MEAN = np.array([104., 117., 124.], dtype=np.float)
#
#
# #######################################################################################################################
# # Data directory
# INPUT_ROOT_DIR = './input/task'
# TRAIN_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'train')
# OUTPUT_ROOT_DIR = './output/task'
# LOG_DIR = os.path.join(OUTPUT_ROOT_DIR, 'logs')
# CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'models')
#
# # Make checkpoint path directory
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#
#
# #######################################################################################################################
#
# def load_imagepaths(path):
#     imagepaths, labels = list(), list()
#     label = 0
#     classes = os.walk(path).__next__()[1]
#     for c in classes:
#         c_dir = os.path.join(path, c)
#         walk = os.walk(c_dir).__next__()
#         # Add each image
#         for sample in walk[2]:
#             imagepaths.append(os.path.join(c_dir, sample))
#             labels.append(label)
#         # next directory
#         label += 1
#     return imagepaths, labels
#
#
# # 256x256으로 이미지 다운샘플링
# def resize_images(imgpath):
#     # Read images from disk & resize
#     # print('start resizing image')
#
#     img = tf.io.read_file(imgpath)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize_with_crop_or_pad(img, target_height=256, target_width=256)
#     # print('end resizing')
#     return img
#
#
# # RGB jittering
# def fancy_pca(image, label, alpha_std=0.1):
#     # print('Start Jittering')
#     # pca_image,pca_label = image.copy(),label.copy()
#
#     orig_img = np.array(image, dtype=np.float)
#
#     # 이미지 픽셀값에서 이미지넷 평균 픽셀값을 빼줌(평균 픽셀값은 사전에 정의됨)
#     img_rs = np.reshape(image, (-1, 3))
#     img_centered = img_rs - IMAGENET_MEAN
#
#     # 해당 이미지의 공분산 행렬 구함
#     img_cov = np.cov(img_centered, rowvar=False)
#
#     # 고유벡터, 고유값 구함
#     eig_vals, eig_vecs = np.linalg.eigh(img_cov)
#     sort_perm = eig_vals[::-1].argsort()
#     eig_vals[::-1].sort()
#     eig_vecs = eig_vecs[:, sort_perm]
#
#     # 고유벡터 세개를 쌓아서 3x3 행렬로 만듦
#     m1 = np.column_stack((eig_vecs))
#     m2 = np.zeros((3, 1))
#
#     # 랜덤값과 고유값을 곱함
#     alpha = np.random.normal(0, alpha_std)
#     m2[:, 0] = alpha * eig_vals
#
#     # 3x3 고유벡터 행렬과 3x1 랜덤값*고유값 행렬을 곱해서 3x1 행렬을 얻음(RGB 채널에 가감해줄 값)
#     add_vect = np.matrix(m1) * np.matrix(m2)
#
#     # R, G, B 채널을 각각 순회하며 계산된 값을 각 픽셀마다 가감
#     for idx in range(3):
#         orig_img[..., idx] += add_vect[idx]
#         minmax_scale(orig_img[..., idx], feature_range=(0., 255.), copy=False)
#
#     # 0~255(rgb픽셀값) 범위로 값 재설정
#     # pca_img = orig_img
#
#     # print('End jittering')
#     return orig_img, label
#
#
# # horizontal reflection
# def flip_image(image, label):
#     # print('Start flipping')
#
#     flipped_image = tf.image.flip_left_right(image)
#     # print('End flipping')
#     return flipped_image, label
#
#
# # Image cropping
# def crop_image(img, label):
#     # print('Start cropping')
#     cropped_images, cropped_labels = list(), list()
#     # # left-top
#     cropped_img = tf.image.crop_to_bounding_box(img, 0, 0, 227, 227)
#     cropped_images.append(cropped_img)
#     cropped_labels.append(label)
#     # # right-top
#     cropped_img = tf.image.crop_to_bounding_box(img, np.shape(img)[0]-227, 0, 227, 227)
#     cropped_images.append(cropped_img)
#     cropped_labels.append(label)
#     # center
#     cropped_img = tf.image.crop_to_bounding_box(img, int((np.shape(img)[0]-227)/2-1), int((np.shape(img)[0]-227)/2-1), 227, 227)
#     cropped_images.append(cropped_img)
#     cropped_labels.append(label)
#     # # left-bottom
#     cropped_img = tf.image.crop_to_bounding_box(img, 0, np.shape(img)[0]-227, 227, 227)
#     cropped_images.append(cropped_img)
#     cropped_labels.append(label)
#     # # right-bottom
#     cropped_img = tf.image.crop_to_bounding_box(img, np.shape(img)[0]-228, np.shape(img)[1]-228, 227, 227)
#     cropped_images.append(cropped_img)
#     cropped_labels.append(label)
#     # print('End cropping')
#     return cropped_images, cropped_labels
#
# img_paths, img_labels = load_imagepaths(TRAIN_IMG_DIR)
# # print(tf.image.decode_jpeg(tf.io.read_file(img_paths[0])).numpy())
# # cv2.imshow('test', tf.image.decode_jpeg(tf.io.read_file(img_paths[0])).numpy() )
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# resize_img = resize_images(img_paths[0])
# # print(np.array(resize_img))
# # cv2.imshow('test', np.array(resize_img))
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# img, lbl = flip_image(resize_img, img_labels[0])
# # print(img)
# # cv2.imshow('test', img.numpy())
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# imgs, lbls = crop_image(resize_img, img_labels[0])
# # print(img)
#
# for i, l in zip(imgs, lbls):
#     cv2.imshow('test', i.numpy())
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# # img, lbl = fancy_pca(img, lbl)
# # print((img))
# # cv2.imshow('test', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# import time
# import urllib.request
# import os
# a=input("검색할 키워드를 입력하세요 : ")
# b=int(input("개수 : "))
# driver = webdriver.Chrome('./chromedriver')
# driver.get('http://www.google.co.kr/imghp?hl=ko')
# elem = driver.find_element_by_name("q")
# elem.send_keys(a)
# elem.send_keys(Keys.RETURN)
# images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
# count = 0
# for image in images:
#     try:
#         image.click()
#         time.sleep(2)
#         imgUrl = driver.find_element_by_xpath("/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img").get_attribute("src")
#         urllib.request.urlretrieve(imgUrl, "input/images" + a + str(count) + ".jpeg")
#         count += 1
#         if count == b:
#             break
#     except:
#         pass
# driver.close()

from selenium import webdriver
from bs4 import BeautifulSoup as soups


def search_selenium(search_name, search_path, search_limit):
    search_list = search_name.split(',')
    for name in search_list:
        search_url = "https://www.google.com/search?q=" + str(name) + "&hl=ko&tbm=isch"

        browser = webdriver.Chrome('./chromedriver')
        browser.get(search_url)

        image_count = len(browser.find_elements_by_tag_name("img"))

        print("로드된 이미지 개수 : ", image_count)

        browser.implicitly_wait(2)

        for i in range(search_limit):
            try:
                image = browser.find_elements_by_tag_name("img")[i]
                image.screenshot("/home/hong/PycharmProjects/pythonProject/alexNet/input/images/" + search_path + name + str(i) + ".png")
            except:
                continue

        browser.close()


if __name__ == "__main__":
    search_name = input("검색하고 싶은 키워드 : ")
    search_limit = int(input("원하는 이미지 수집 개수 : "))
    search_path = "tumbler/"
    # search_maybe(search_name, search_limit, search_path)
    search_selenium(search_name, search_path, search_limit)