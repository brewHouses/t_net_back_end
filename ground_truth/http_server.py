import cv2
import numpy as np
import io
from flask import Flask, request, send_file
import gen_mask_data

labels = gen_mask_data.get_labels()

# 实例化app
app = Flask(import_name=__name__)

# 通过methods设置POST请求
@app.route('/ground_truth', methods=["POST"])
def ground_truth():

    # 接收post请求上传的文件
    input_file = request.files.get('file')

    if input_file is None:
        return "未上传文件"

    # 从指定的内存缓存中读数据, 并转码成图像格式
    img = input_file.read()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    # 对输入数据进行处理
    img_ret = None
    img_ret = gen_mask_data.get_ground_truth(img, input_file.filename, labels[input_file.filename])
    if img_ret is None:
        return "The file {} has no label data".format(input_file.filename)

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


