import os
PATH = os.path.dirname(os.path.abspath(__file__))

# 需要读取的excel文件的位置
XLS_FILE = os.path.join(PATH, '../../mark.xls')
# '/Users/wangweikai/master/landmark_data/mark.xls'
# 原始的视频文件的位置
VIDEO_DIR = os.path.join(PATH, '../../VideoData20190715')
# '/Users/wangweikai/master/landmark_data/VideoData20190715'
# 截取的被标记视频的帧的存放位置
OUTPUT_DIR = os.path.join(PATH, '../../expand_data')
# '/Users/wangweikai/master/landmark_data/data'
# 截取的被标记视频的帧的对应标记数据
MARK_FILE_JSON = os.path.join(PATH, '/workspace/test.json')
# '/Users/wangweikai/master/landmark_data/data_landmark.json'
# encoder使用到的所有数据存放的dir, 就是截取视频的每一帧
ENCODER_OUTPUT_DIR = os.path.join(PATH, '../../encoder_data')
# '/Users/wangweikai/master/landmark_data/encoder_data'
# encoder输入图像的大小
ENCODER_IMG_SIZE = (128, 96)
# 是否在视频截取的时候进行resize
RESIZE_FLAG = False
# 是否在显示的时候resize到ENCODER_IMG_SIZE
DISPLAY_RESIZE = False
# 暂时存储图片的路径
TMP_IMG_DIR = os.path.join(PATH, '../../tmp_img')
# '/Users/wangweikai/master/landmark_data/tmp_img'
# mask的文件夹
MASK_DIR = os.path.join(PATH, '../../mask')
# '/Users/wangweikai/master/landmark_data/tmp_img'
# 是否保存图片
WRITE_IMG = False
# 是否保存JSON数据
# WRITE_JSON = True

MODE = {'DISPLAY':True}
MODE['GENERATE'] = False if MODE['DISPLAY'] else True


SHOW_LIST = [
    #"XG" ,     # 胸骨
    #"JZ" ,     # 脊柱
    "DA" ,     # 降主动脉
    #"LgZ1" ,   # 真肋骨1
    #"LgZ2" ,   # 真肋骨2
    #"LgJ1" ,   # 假肋骨1
    #"LgJ2" ,   # 假肋骨2
    #"XJWM" ,   # 心肌外膜
    #"FJMZ" ,   # 肺静脉左
    #"FJM" ,    # 肺静脉右
    #"SZ" ,     # 房间隔\r（原发）
    #"fjgJF" ,  # 房间隔\r（继发）
    #"sjg" ,    # 室间隔
    "LA" ,     # 左房\rLA
    "LV" ,     # 左室\rLV
    #"EJBQY" ,  # 二尖瓣\r前叶
    #"EJBHY" ,  # 二尖瓣\r后叶
    "RA" ,     # 右房\rRA
    "RV" ,     # 右室\rRV
    #"SJBGY" ,  # 三尖瓣\r隔叶
    #"SJBQY" ,  # 三尖瓣\r前叶
    #"RYKBM" ,  # 卵圆孔\r瓣膜
    #"RYKKK" ,  # 卵圆孔\r开口
  ]

KEY_LABEL = [
    #"XG" ,     # 胸骨
    #"JZ" ,     # 脊柱
    "DA" ,     # 降主动脉
    #"LgZ1" ,   # 真肋骨1
    #"LgZ2" ,   # 真肋骨2
    #"LgJ1" ,   # 假肋骨1
    #"LgJ2" ,   # 假肋骨2
    #"XJWM" ,   # 心肌外膜
    #"FJMZ" ,   # 肺静脉左
    #"FJM" ,    # 肺静脉右
    #"SZ" ,     # 房间隔\r（原发）
    #"fjgJF" ,  # 房间隔\r（继发）
    #"sjg" ,    # 室间隔
    "LA" ,     # 左房\rLA
    "LV" ,     # 左室\rLV
    #"EJBQY" ,  # 二尖瓣\r前叶
    #"EJBHY" ,  # 二尖瓣\r后叶
    "RA" ,     # 右房\rRA
    "RV" ,     # 右室\rRV
    #"SJBGY" ,  # 三尖瓣\r隔叶
    #"SJBQY" ,  # 三尖瓣\r前叶
    #"RYKBM" ,  # 卵圆孔\r瓣膜
    #"RYKKK" ,  # 卵圆孔\r开口
  ]
