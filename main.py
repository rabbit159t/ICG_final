import numpy as np
from scipy import signal
import cv2,colorsys
from matplotlib import pyplot as plt
import math

'''
size = 1: i V T N-type
size = 2: L I Y X-type
'''
class template():
    def __init__(self, size, w):
        self.size = size
        self.w = w
    def info(self):
        print self.size, self.w
t = []
t.append(template(1, [5]))
t.append(template(1, [45]))
t.append(template(1, [90]))

'''
r: clockwise
l: counter clockwise

return
    (+) : clockwise
    (-) : counter clockwise
'''
def func(c,h):
    if c > h:
        r = c-h
    else:
        r = c+360-h
    l = 360-r
    if(r<=l): return r
    else: return -l

#========TODO: choose template=====================
w = t[1].w[0]
size = 31
window = signal.gaussian(size, std=0.5*w)
interval = 360/(size-1)
print window, interval

img = cv2.imread('01.jpg')
height, width = img.shape[:2]
img  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#bgdModel = np.zeros((1,65),np.float64)
#fgdModel = np.zeros((1,65),np.float64)

#rect = (480,800,720,260)
#cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
 
#mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
   
#img = img*mask2[:,:,np.newaxis]
total=0
for i in range(height):
    for j in range(width):
        total+=img[i][j][0]
center = total/height/width
print center

print img[0][0][0]
for i in range(height):
    for j in range(width):
        hue = img[i][j][0]<<1
        
        #print img[i][j][0],func(center,img[i][j][0]),img[i][j][0],window[func(center,img[i][j][0])]
        #print img[i][j][0],func(center,img[i][j][0]),int(0.5*w*(window[func(center,img[i][j][0])/6])+center),
        
        hue = int(center+0.5*w*(window[ int((func(center,hue)+180)/interval) ]))
        #print hue,center
        if hue >= 360:
            hue = hue -360
        elif hue < 0 :
            hue = hue + 360
        img[i][j][0] = hue>>1
img  = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

plt.imshow(img),plt.colorbar(),plt.show()

