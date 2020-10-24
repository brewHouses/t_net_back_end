import cv2
import numpy as np
import io
import argparse, os
from keras_segmentation.models.segnet import segnet
from keras_segmentation.predict import predict_http
import os
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from flask import Flask, request, send_file, make_response
import time
import lib

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
sess = tf.Session()
graph = tf.get_default_graph()

trained_weights = "/workspace/checkpoints.493"
set_session(sess)
model = segnet(n_classes=6 ,  input_height=256, input_width=256  )
model.load_weights(trained_weights)

colors = [
    [0, 0, 255],
    [255,0,0],
    [0,255,0],
    [0,0,255],
    [255,255,0],
    [255,0,255],
]

colors[1],colors[4],colors[5] = colors[5],colors[1],colors[4]

# 实例化app
app = Flask(import_name=__name__)

# 通过methods设置POST请求
@app.route('/predict', methods=["POST"])
def predict():

    # 接收post请求上传的文件
    input_file = request.files.get('file')
    filename = input_file.filename

    if input_file is None:
        return "未上传文件"

    # 从指定的内存缓存中读数据, 并转码成图像格式
    input_file = input_file.read()
    img = cv2.imdecode(np.frombuffer(input_file, np.uint8), cv2.IMREAD_COLOR)

    # 对输入数据进行处理
    img_ret = None
    with graph.as_default():
        set_session(sess)
        start = time.time()
        img_ret, predict_mask_img = predict_http(model=model, inp=img, overlay_img=True, colors=colors,)
        end = time.time()


    mask_img = lib.getMask('http://10.2.3.195:32804/ground_truth_bin', filename)
    if (not mask_img is None and predict_mask_img.shape != (mask_img.shape)):
        predict_mask_img = cv2.resize(predict_mask_img, (mask_img.shape[1], mask_img.shape[0]))
    if not mask_img is None:
        pa = lib.pa(mask_img, predict_mask_img)
        mpa = lib.mpa(mask_img, predict_mask_img)
        iou = lib.iou(mask_img, predict_mask_img)
        miou = lib.miou(mask_img, predict_mask_img)
    cm = lib.cm(predict_mask_img)
    # 将图片格式转码成数据流, 放到内存缓存中
    # img_encode = cv2.imencode('.jpg', img_ret)[1]
    img_encode = cv2.imencode('.jpg', img_ret)[1]
    data_encode = np.array(img_encode)
    str_encode = data_encode.tostring()
    resp = make_response(str_encode)
    #设置response的headers对象
    resp.headers['Content-Type'] = 'image/jpeg'
    if not mask_img is None:
        resp.headers['pa'] = pa
        resp.headers['mpa'] = mpa
        resp.headers['iou'] = iou
        resp.headers['miou'] = miou
    resp.headers['cm'] = cm
    resp.headers['time'] = end-start
    return resp


if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=True)