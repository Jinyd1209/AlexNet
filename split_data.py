"""
@Author: yidon jin
@Email: m18818261998@163.com
@FileName: split_data.py
@DateTime：2021/12/17 20:40
@SoftWare: PyCharm
"""


import os
import random
from shutil import copy,rmtree   # os模块的补充，主要针对文件的拷贝，删除，移动，压缩和解压操作


def make_dir(file_path:str):  # 冒号为类型建议符，表明希望传入的参数类型
    if os.path.exists(file_path):
        rmtree(file_path)  # 若文件存在则先删除原文件再创建
    os.makedirs(file_path)


def main():
    random.seed(42)
    split_rate = 0.1

    # 得到训练文件的路径
    cwd = os.getcwd()
    data_root = os.path.join(cwd,"flower_data")
    flower_path = os.path.join(data_root,"flower_photos")
    assert os.path.exists(flower_path),"path {} not exit".format(flower_path)

    # 得到训练文件的标签
    flower_class = [cla for cla in os.listdir(flower_path)
                    if os.path.isdir(os.path.join(flower_path,cla))]

    # 建立保存分开后的训练集的文件夹
    train_root = os.path.join(data_root,"train")
    make_dir(train_root)
    # 对应类别建立文件夹
    for cla in flower_class:
        make_dir(os.path.join(train_root,cla))

    # 建立保存分开后的验证集的文件夹
    val_root = os.path.join(data_root,"val")
    make_dir(val_root)
    for cla in flower_class:
        make_dir(os.path.join(val_root,cla))

    # 将数据集随机分开
    for cla in flower_class:
        cla_path = os.path.join(flower_path,cla)
        images = os.listdir(cla_path)
        num = len(images)
        # 得到验证集随机取样的列表
        val_sample = random.sample(images,k=int(num*split_rate))
        for index,image in enumerate(images):
            if image in val_sample:
                image_path = os.path.join(cla_path,image)
                new_path = os.path.join(val_root,cla)
                copy(image_path,new_path)
            else:
                image_path = os.path.join(cla_path,image)
                new_path = os.path.join(train_root,cla)
                copy(image_path,new_path)

            print("\r[{}] processing [{}/{}]".format(cla,index+1,num),end="")
        print()

    print("processing done!")


if __name__ == '__main__':
    main()

