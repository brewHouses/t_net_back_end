FROM wangwk/seg:v5
LABEL maintainer "wangwk@buaa.edu.cn"
#RUN pip3 install flask -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY predict.py /usr/local/lib/python3.6/dist-packages/keras_segmentation
RUN pip3 install requests -i https://pypi.tuna.tsinghua.edu.cn/simple
ENV PROJECT_DIR=/workspace/  
WORKDIR $PROJECT_DIR
COPY * $PROJECT_DIR  
ENV APP_PORT=5000
EXPOSE $APP_PORT
CMD ["python", "http_server.py"]  
