import cv2
import numpy as np
import io
from flask import Flask, request, send_file
import argparse, os
from keras_segmentation.models.unet import unet
from keras_segmentation.predict import predict_http
import os
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
sess = tf.Session()
graph = tf.get_default_graph()

trained_weights = "/workspace/checkpoints.999"
set_session(sess)
model = unet(n_classes=6 ,  input_height=256, input_width=256  )
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

    if input_file is None:
        return "未上传文件"

    # 从指定的内存缓存中读数据, 并转码成图像格式
    input_file = input_file.read()
    img = cv2.imdecode(np.frombuffer(input_file, np.uint8), cv2.IMREAD_COLOR)

    # 对输入数据进行处理
    img_ret = None
    with graph.as_default():
        set_session(sess)
        img_ret = predict_http(model=model, inp=img, overlay_img=True, colors=colors,)

    # 将图片格式转码成数据流, 放到内存缓存中
    img_encode = cv2.imencode('.jpg', img_ret)[1]
    data_encode = np.array(img_encode)
    str_encode = data_encode.tostring()
    return str_encode
    
    '''
    # 这个是用来base64编码, 用在浏览器里
    return send_file(
        io.BytesIO(img),
        mimetype='image/png',
        as_attachment=True,
        attachment_filename='result.jpg'
    )
    '''

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=True)

