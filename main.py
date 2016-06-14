import numpy as np
from scipy import signal
from scipy import optimize
import cv2,colorsys
from matplotlib import pyplot as plt
import math

'''
size = 1: i V T N-type
size = 2: L I Y X-type
'''
class template():
    def __init__(self, w ,d):
        self.w = w
        self.d = d
    def info(self):
        print self.w, self.d
t = []
t.append(template([5],-1))
t.append(template([45],-1))
t.append(template([5,30],90))
t.append(template([5,5],180))
t.append(template([90],-1))
t.append(template([45,5],180))
t.append(template([45,45],180))

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


img = cv2.imread('05.jpg')
height, width = img.shape[:2]
img  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def templateAngle(x,t):
    templateCenter = x
    while templateCenter<0: templateCenter += 360
    while templateCenter>=360: templateCenter -= 360

    #if templateCenter < 0 :
    #    templateCenter += 360
    #elif templateCenter >= 360:
    #    templateCenter -= 360
    if t.d != -1:
        center2 = templateCenter + t.d
        if center2 < 0 :
            center2 += 360
        elif center2 >= 360:
            center2 -= 360
  
    total = 0
    for i in range(height):
        for j in range(width):
            hue = img[i][j][0]<<1
            turn = abs(func(templateCenter,hue))
            turn2 = -1
            if t.d != -1:
                turn2 = abs(func(center2,hue))   
            if turn < t.w[0]:
                total += 0
            elif turn2 >=0 and turn2 <t.w[1]:
                total += 0
            else:
                if t.d != -1:
                    minDistance = min(turn-t.w[0],turn2-t.w[1])
                    total += minDistance*img[i][j][1]
                else :
                    total += (turn-t.w[0])*img[i][j][1]
    return total


'''   
total=0
for i in range(height):
    for j in range(width):
        total+=img[i][j][0]
center = total/height/width*2
'''

cList = []
valueList = []
for i in t:
    center = optimize.brent(templateAngle,(i,))
    cList.append(center)
    valueList.append(templateAngle(center,i))
    
center = cList[valueList.index(min(valueList))]   
while center<0: center+=360
while center>=360: center-=360


templateIndex = valueList.index(min(valueList))
w = t[templateIndex].w[0]
template = t[templateIndex]
size = 31
window = signal.gaussian(size, std=w>>1)
interval = 360/(size-1)
print window, interval

if template.d != -1:
    center2 =center + template.d
    if center2 < 0 : center2  += 360
    elif center2 >= 360: center2  -= 360

#print img[0][0][0]
for i in range(height):
    for j in range(width):
        hue = img[i][j][0]<<1
        turn = func(center, hue)
        
        #turn2 = -1
        if template.d != -1:
            turn2 = func(center2,hue)   
        if abs(turn) < template.w[0]:
            continue#pass
        elif template.d != -1 and abs(turn2) < template.w[1]:
            continue#pass
        else:
            if template.d != -1:
                if abs(turn)-template.w[0] > abs(turn2)-template.w[1]:
                    hue = int(center2 + 0.5*template.w[1]*(1 - window[ int((turn2+180)/interval) ]))
                else:
                    hue = int(center + 0.5*template.w[0]*(1 - window[ int((turn+180)/interval) ]))
            else :
                hue = int(center + 0.5*template.w[0]*(1 - window[ int((turn+180)/interval) ]))
    
        #if abs(turn) < w: continue
        #if turn >= 0: 
        #    hue = int(center + 0.5*w*(1 - window[ int((turn+180)/interval) ]))
        #else:
        #    hue = int(center - 0.5*w*(1 - window[ int((turn+180)/interval) ]))
        #print hue,center
        if hue >= 360:
            hue = hue -360
        elif hue < 0 :
            hue = hue + 360
        img[i][j][0] = hue>>1

print 'test: ', hue,center
print cList
print valueList
img  = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
plt.imshow(img),plt.colorbar(),plt.show()

