# Installation
Here we provide a tutorial about how to generate a result file to submit base on [Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT) with the pretrained weight

**NOTE**

For training details, please refer to the origin [README.md](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/README.md)


# Test on the PANDA VIDEO dataset
1. Download the pretrained weight from [here](https://pan.baidu.com/s/1Ifgn0Y_JZE65_qSrQM2l-Q?_at_=1617862978210) and save it in the folder **weights**.
2. Update the root setting in `track_panda.py`.
3. To generate a result file, you can simply run the following code.
  ```
  python track_panda.py
  ```


# Acknowledgement
The tutorial is based on [Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT), many thanks to their wonderful work!

