#! /usr/bin/env python
'''
生成图片大小为256*256
'''

import json
import os
import sys
sys.path.append("..")
import numpy as np
import cv2
import config1 as config
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import DataLoader

def get_labels(file_name=config.MARK_FILE_JSON) -> dict:
    '''
    参数是 video_handler 输出的json文件, 其中记录了文件名以及标记
    函数主要是根据file_name和config.KEY_LABEL需要的标记, 解析处理文件
    并返回一个字典

    并且要求config.KEY_LABEL中全部组件都是齐全的
    '''
    with open(file_name, 'r') as json_data:
        ret = {}
        for line in json_data.readlines():
            line = json.loads(line)
            if not line:
                continue
            tmp = {}
            for label in config.KEY_LABEL:
                if label in line:
                    tmp[label] = line[label]
                else:
                    tmp[label] = []
                    break
            if not tmp[label]:
                continue
            ret[line['file_name']] = tmp
    return ret

def get_files(labels):
    '''
    根据get_labels过滤后得到的信息生成文件名列表
    '''
    return labels.keys()

def _trans_points(frame:'numpy.array', points:'list') -> 'list':
    '''
    将百分比表示的位置转换像素成表示的位置
    '''
    ret = []
    if len(frame.shape) == 3:
        rows, cols, _ = frame.shape
    else:
        rows, cols = frame.shape
    for point in points:
        tmp_ret = []
        tmp_ret.append(int(int(point[0]) * cols / 100))
        tmp_ret.append(int(int(point[1]) * rows / 100))
        ret.append(tmp_ret)
    return ret

def _gen_black_img(img):
    '''
    根据输入的img的大小, 生成完全一致大小的黑色mask底片
    '''
    if len(img.shape) == 3:
        rows, cols, _ = img.shape
    else:
        rows, cols = img.shape
    return np.zeros((rows, cols), np.uint8)


def gen_mask(img_name, marks, show_flag=False, channels=5):
    ret = None
    img = cv2.imread(img_name)
    if img is None:
        print('66666666666666666666666666666666'+img_name)
        return
    img = cv2.resize(img, (256, 256))
    shape = img.shape
    for mark in marks.values():
        mark = _trans_points(img, mark)
        tmp_img = cv2.fillPoly(_gen_black_img(img), np.array([mark], np.int32), (255))
        if ret is None:
            ret = tmp_img
        else:
            ret = np.vstack((ret, tmp_img))
    ret = ret.reshape(-1, shape[0], shape[1])
    if channels == 6:
        tmp_ret = None
        for img in ret:
            if tmp_ret is None:
                tmp_ret = img
            else:
                tmp_ret = tmp_ret | img
        tmp_ret = ~tmp_ret
        ret = ret.reshape(-1, shape[1])
        ret = np.vstack((ret, tmp_ret))
        ret = ret.reshape(-1, shape[0], shape[1])

    ret = np.rollaxis(ret, 0, 3)
    '''
    print('6666666666666666666666666')
    print(ret.shape)
    '''
    if show_flag:
        for img in ret:
            cv2.imshow('ret', img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                sys.exit(0)
    return ret



def gen_all_mask(labels):
    '''
    根据label里面指定的文件名以及轨迹数据产生相应的mask
    '''
    for img_name, marks in labels.items():
        img_name = os.path.join(config.OUTPUT_DIR, img_name)
        gen_mask(img_name, marks, True)



# 这个类是用来产生掩码的
class Ultrasound_mask_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, transform=None, channels=5, last=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imgs = list(get_files(get_labels()))
        print(len(self.imgs))
        self.root_dir = '../../expand_data'
        self.transform = transform
        self.marks = get_labels()
        self.channels = channels
        self.last = last

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.imgs[idx])
        image = gen_mask(img_name, self.marks[self.imgs[idx]], channels=self.channels)
        # image = Image.open(img_name).convert('L')
        # sample = {'image': image}

        if self.transform:
            sample = self.transform(image)
            if self.last:
                return sample[-1].reshape(1, 256, 256)
            return sample
        
        if self.last:
            return image[-1].reshape(1, 256,256)
        return image






def roberts(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Roberts算子
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    # 转转成uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Roberts

def prewitt(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Prewitt算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int) 
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    # 转转成uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Prewitt



def sobel(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Prewitt算子
    # kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    # kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    # x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    # y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)
    # 转成uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Sobel

def laplacian(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)
    dst = cv2.convertScaleAbs(dst)
    return dst




# 这个类是用来产生原始图像和掩码图像的
class Ultrasound_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, transform=None, channels=5, last=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imgs = list(get_files(get_labels()))
        print(len(self.imgs))
        self.root_dir = '../../expand_data'
        self.transform = transform
        self.marks = get_labels()
        self.channels = channels
        self.last = last

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.imgs[idx])
        image_raw = cv2.imread(img_name)
        image_raw_size = image_raw
        if image_raw is None:
            print('66666666666666666666666666666666'+img_name)
            return
        image_raw = cv2.resize(image_raw, (256, 256))
        roberts_img = roberts(image_raw)
        prewitt_img = prewitt(image_raw)
        sobel_img = sobel(image_raw)
        laplacian_img = laplacian(image_raw)
        image = gen_mask(img_name, self.marks[self.imgs[idx]], channels=self.channels)
        #image_raw = np.rollaxis(image_raw, 0, 2)
        #image_raw = np.transpose(image_raw,(2, 0, 1)) 
        
        # image = Image.open(img_name).convert('L')
        # sample = {'image': image}

        if self.transform:
            sample = self.transform(image)
            image_raw = self.transform(image_raw)
            image_raw_size = self.transform(image_raw_size)
            roberts_img = self.transform(roberts_img)
            prewitt_img = self.transform(prewitt_img)
            sobel_img = self.transform(sobel_img)
            laplacian_img = self.transform(laplacian_img)
            if self.last:
                return (sample[-1].reshape(1, 256, 256), image_raw, image_raw_size, image_raw, roberts_img, prewitt_img, sobel_img, laplacian_img)
            return (sample, image_raw, image_raw_size, image_raw, roberts_img, prewitt_img, sobel_img, laplacian_img)
        
        if self.last:
            return (image[-1].reshape(1, 256,256), image_raw, image_raw_size, image_raw, roberts_img, prewitt_img, sobel_img, laplacian_img)
        return (image, image_raw, image_raw_size, image_raw, roberts_img, prewitt_img, sobel_img, laplacian_img)


if __name__ == '__main__':
    # Configure data loader
    dataset = Ultrasound_Dataset(transform=transforms.Compose([
        #transforms.Resize((192, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,], std=[0.5,])
    ]), channels=6, last = True)

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                             shuffle=True, num_workers=0)
    from tqdm import tqdm
    for i, (imgs_mask, imgs_raw) in tqdm(enumerate(dataloader)):
        print(imgs_raw.data.shape)
        print(imgs_mask.data.shape)
        save_image(imgs_raw.data, "256_raw/%d_raw.png" % i, nrow=8, normalize=True)
        save_image(imgs_mask.data, "256_raw/%d_mask.png" % i, nrow=8, normalize=True)
