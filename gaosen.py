# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *
from numpy import*
import random
#读取图片并转为数组
im = array(Image.open('./1.jpg'))
#设定高斯函数的偏移
means = 0
#设定高斯函数的标准差
sigma = 50
#r通道
r = im[:,:,0].flatten()
#g通道
g = im[:,:,1].flatten()
#b通道
b = im[:,:,2].flatten()
#计算新的像素值
for i in range(im.shape[0]*im.shape[1]): 
    pr = int(r[i]) + random.gauss(0,sigma) 
    pg = int(g[i]) + random.gauss(0,sigma) 
    pb = int(b[i]) + random.gauss(0,sigma) 
    if(pr < 0):
        pr = 0
    if(pr > 255):
        pr = 255 
    if(pg < 0): 
        pg = 0 
    if(pg > 255): 
        pg = 255 
    if(pb < 0): 
        pb = 0 
    if(pb > 255): 
        pb = 255    
    r[i] = pr
    g[i] = pg
    b[i] = pb
im[:,:,0] = r.reshape([im.shape[0],im.shape[1]])
im[:,:,1] = g.reshape([im.shape[0],im.shape[1]])
im[:,:,2] = b.reshape([im.shape[0],im.shape[1]])
#显示图像
imshow(im)
show()         
 