# Base Images 
## 从天池基础镜像构建(from的base img 根据自己的需要更换，建议使用天池open list镜像链接：https://tianchi.aliyun.com/forum/postDetail?postId=67720) 
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.6-cuda10.1-py3

##安装python依赖包 
RUN apt-get -y install gcc
RUN apt install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-dev
RUN apt-get install -y libgl1-mesa-dev

# RUN pip install logging
RUN pip install numpy
RUN pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install Cython -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install argparse -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install cython-bbox  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install numba  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install scipy  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install matplotlib  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install lap  -i https://pypi.tuna.tsinghua.edu.cn/simple 
RUN pip install motmetrics  -i https://pypi.tuna.tsinghua.edu.cn/simple 

## 把当前文件夹里的文件构建到镜像的根目录下,并设置为默认工作目录 
ADD . / 
WORKDIR / 
## 镜像启动后统一执行 sh run.sh 
CMD ["sh", "run.sh"]