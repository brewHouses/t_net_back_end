FROM wangwk/t_net:v3
LABEL maintainer "wangwk@buaa.edu.cn"
RUN pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple
ENV PROJECT_DIR=/workspace/  
WORKDIR $PROJECT_DIR
COPY * $PROJECT_DIR  
ENV APP_PORT=5000
EXPOSE $APP_PORT
CMD ["python", "http_server.py"]  
