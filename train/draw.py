import os
import argparse
import random
from tqdm import tqdm
import shutil

train_dir = r'yx1\train'
val_dir = r'yx1\val'
"""
for spker in os.listdir(train_dir):
    train_list = os.listdir(os.path.join(train_dir, spker, "audio"))
    val_list = []
    for i in range(10):
        # 生成随机数，作为索引，随机选择一个文件
        _random = random.randint(0, len(train_list) - 1)
        val_list.append(train_list[_random])
        train_list.pop(_random)
        print(_random, len(train_list))
    for i in val_list:
        print(i)
        os.makedirs(os.path.join(val_dir, spker, "audio"), exist_ok=True)
        shutil.move(os.path.join(train_dir, spker, "audio", i), os.path.join(val_dir, spker, "audio", i))
        os.makedirs(os.path.join(val_dir, spker, "f0"), exist_ok=True)
        shutil.move(os.path.join(train_dir, spker, "f0", i + '.npy'), os.path.join(val_dir, spker, "f0", i + '.npy'))
        if int(spker) == 1:
            os.makedirs(os.path.join(val_dir, spker, "music"), exist_ok=True)
            shutil.move(os.path.join(train_dir, spker, "music", i), os.path.join(val_dir, spker, "music", i))
"""
