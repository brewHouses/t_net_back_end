#! /usr/bin/env python
'''
生成图片大小为256*256
'''

import json
import os
import sys
import numpy as np
import cv2
import config1 as config
import os

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


def gen_mask(img_name, marks, img, show_flag=False, channels=5):
    ret = None
    '''
    img = cv2.imread(img_name)
    if img is None:
        print('66666666666666666666666666666666'+img_name)
        return
    '''
    #img = cv2.resize(img, (256, 256))
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


colors = [
[255,0,0],
[0,255,0],
[0,0,255],
[255,255,0],
[255,0,255],
]
def get_ground_truth(raw_img, file_name, label):
    mask_img = gen_mask(file_name, label, raw_img, channels=5)
    if mask_img is None:
        return None
    for i in range(5) :
            img = mask_img[:, :, i]
            raw_img[img>0] = raw_img[img>0]*0.6 + np.array(colors[i])*0.4
    return raw_img



if __name__ == '__main__':
    labels = get_labels()
    file_name="common_f21584-3341734.avi.ogv_0_0.jpg"
    img = get_ground_truth(cv2.imread(file_name), file_name, labels[file_name])
    cv2.imwrite("a.png", img)
