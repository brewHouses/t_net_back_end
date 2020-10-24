import requests,json,cv2
import numpy as np



def getMask(url, filename):
    data = {'filename':filename}   #dumps：将python对象解码为json数据
    response = requests.post(url,data)

    if response.status_code == 200:
        img = np.frombuffer(response.content, np.uint8) # 转成8位无符号整型
        return cv2.imdecode(img, cv2.IMREAD_COLOR) # 解码
    return None


def pa(img1, img2):
    pix_err = np.where(img1 != img2)
    return 1-len(pix_err[0])/(img1.shape[0]*img1.shape[1])

def mpa(img1, img2):
    assert(img1.shape[0]*5 == img1.shape[1])
    sum = 0
    for i in range(5):
        pix_err = np.where(img1[:, img1.shape[0]*i:img1.shape[0]*(i+1)] != img2[:, img1.shape[0]*i:img1.shape[0]*(i+1)])
        sum += len(pix_err[0])/(img1.shape[0]*img1.shape[1])
    return 1-sum/5

def iou(img1, img2):
    jiao = np.where((img1==255)&(img2==255))
    bing = np.where((img1==255)|(img2==255))
    return len(jiao[0])/len(bing[0])


def miou(img1, img2):
    assert(img1.shape[0]*5 == img1.shape[1])
    sum = 0
    for i in range(5):
        jiao = np.where((img1[:, img1.shape[0]*i:img1.shape[0]*(i+1)]==255)&(img2[:, img1.shape[0]*i:img1.shape[0]*(i+1)]==255))
        bing = np.where((img1[:, img1.shape[0]*i:img1.shape[0]*(i+1)]==255)|(img2[:, img1.shape[0]*i:img1.shape[0]*(i+1)]==255))
        sum += len(jiao[0])/len(bing[0])
    return sum/5

def cm(img):
    for i in range(5):
        if len(np.where(img[:, img.shape[0]*i:img.shape[0]*(i+1)]==255)[0]) == 0:
            return False
        return True





if __name__ == '__main__':
    mask_img = getMask('http://10.2.3.195:32804/ground_truth_bin', 'common_233254-2656255-AP.avi.ogv_0_1.jpg')
    print(mask_img.shape)