# -*- coding:utf-8 -*-
import requests
import os
import json
import numpy as np
import pickle
import gevent
import time
from retrying import retry
from PIL import Image
from gevent import monkey

monkey.patch_all()
src_path = 'D:\\python code\\snh\\image\\src\\'
grey_path = 'D:\\python code\\snh\\image\\grey\\'
data_path = 'D:\\python code\\snh\\data\\'


def time_cal(func):
    def wrapper(*args, **kw):
        print(''.join(['start ', func.__name__]))
        start = time.clock()
        func(*args, **kw)
        end = time.clock()
        print(''.join(['Total exec time: ', '%.2f' % (end-start), ' seconds']))
    return wrapper


@retry
def write_img(url, headers, index):
    try:
        img = requests.get(url, headers=headers, timeout=5)
        with open(''.join([src_path, str(index), '.png']), 'wb') as f:
            f.write(img.content)
    except requests.ConnectTimeout as e:
        print(''.join(['Timeout: ', e]))
    except requests.ConnectionError as e:
        print(''.join(['Image', str(index), ': ', e]))


@time_cal
def get_src_img(limit):
    if isinstance(limit, int):
        url = r'http://user.snh48.com/authcode/code.php'
        headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                 "Chrome/56.0.2924.87 Safari/537.36"}
        count = len(os.listdir(src_path)) + 1
        for index in range(count, limit+1):
            g = gevent.spawn(write_img, url, headers, index)
            g.join()
    else:
        raise TypeError
    print(''.join(['Download ', str(limit), ' ocr images']))


def grey_img(filename):
    with Image.open(''.join([src_path, filename])).convert('1') as img:
        width, height = img.size
        for y in range(height):
            for x in range(width):
                if not (2 < x < width - 2 and 2 < y < height - 2):
                    img.putpixel((x, y), 255)
                elif (not img.getpixel((x, y))) and img.getpixel((x - 1, y)) and img.getpixel((x + 1, y)) \
                        and img.getpixel((x, y - 1)) and img.getpixel((x, y + 1)):
                    img.putpixel((x, y), 255)
        img.save(''.join([grey_path, filename]))


@time_cal
def get_grey_img():
    for filename in os.listdir(src_path):
        g = gevent.spawn(grey_img, filename)
        g.join()
    print('%d images grey down' % len(os.listdir(src_path)))


@time_cal
def get_sample_img():
    length = len(os.listdir(grey_path))
    index = 0
    # with Image.open(''.join([grey_path, '1.png'])) as sam:
    #     width, height = sam.size
    #     ocr_data = np.empty((length, width*height))
    #     ocr_label = np.empty(length)
    ocr_data = []
    ocr_label = []
    with open(''.join([data_path, 'data.txt']), 'r') as f:
        data = json.load(f)
    for f in os.listdir(grey_path):
        with Image.open(''.join([grey_path, f])).convert('1') as img:
            num = 0
            width, height = img.size
            for x in range(3, width-7):
                for y in range(3, height-9):
                    piece = img.crop((x, y, x+8, y+10))
                    for i in range(10):
                        data_set = data[str(i)]
                        flag = True
                        for m, n in data_set:
                            if piece.getpixel((m, n)):
                                flag = False
                                break
                        if flag:
                            num = num*10 + i
                            break

            img_ndarray = np.asarray(img, dtype='float64')
            # ocr_data[index] = np.ndarray.flatten(img_ndarray)
            ocr_data.append(np.ndarray.flatten(img_ndarray))
            if num > 9999:
                num = num // 10
            buf = str(num)
            while len(buf) < 4:
                buf = ''.join(['0', buf])
            ocr_label.append(np.array([int(k) for k in buf]))

    print('******get image data done******')
    with open(''.join([data_path, 'ocr_test.pkl']), 'wb') as outfile:
        pickle.dump([[ocr_data[0:int(length*0.9)], ocr_label[0:int(length*0.9)]],
                     [ocr_data[int(length*0.9)+1:int(length*0.95)], ocr_label[int(length*0.9)+1:int(length*0.95)]],
                     [ocr_data[int(length*0.95):length], ocr_label[int(length*0.95):length]]],
                    outfile)


if __name__ == '__main__':
    # get_src_img(40000)
    # get_grey_img()
    get_sample_img()
