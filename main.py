import numpy as np
from scipy import signal
import cv2,colorsys
from matplotlib import pyplot as plt
import math


class template():
	def __init__(self, size, c, w):
		self.size = size
		self.c = c
		self.w = w


x = template(2,1,10)
print x.size, x.c, x.w
# r: clockwise
# l: counter clockwise
#
# return
#   (+) : clockwise
#   (-) : counter clockwise
def func(c,h):
    if c > h:
        r = c-h
        #l = 360-r
    else:
        r = c+360-h
    l = 360-r
    if(r<=l): return r
    else: return -l


w = 10
size = 31
window = signal.gaussian(size, std=0.5*w)
interval = 360/(size-1)
print window, interval
center = 0
img = cv2.imread('01.jpg')
#bgdModel = np.zeros((1,65),np.float64)
#fgdModel = np.zeros((1,65),np.float64)

#rect = (480,800,720,260)
#cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
 
#mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
   
#img = img*mask2[:,:,np.newaxis]
height, width = img.shape[:2]
img  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print img[0][0][0]
for i in range(height):
    for j in range(width):
        hue = img[i][j][0]*2
        
        #print img[i][j][0],func(center,img[i][j][0]),img[i][j][0],window[func(center,img[i][j][0])]
        #print img[i][j][0],func(center,img[i][j][0]),int(0.5*w*(window[func(center,img[i][j][0])/6])+center),
        
        hue = int(center+0.5*w*(window[ int((func(center,hue)+180)/interval) ]))
        #print hue,center
        if hue >= 360:
            hue = hue -360
        elif hue < 0 :
            hue = hue + 360
        img[i][j][0] = hue/2
img  = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

plt.imshow(img),plt.colorbar(),plt.show()

