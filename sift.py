'''
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.
This sample shows interactive image segmentation using grabcut algorithm.
USAGE:
    python grabcut.py <filename>
README FIRST:
    Two windows will show up, one for input and one for output.
    At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.
Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground
Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
'''
# Python 2/3 compatibility
from __future__ import print_function

import templateGUI
import numpy as np
import cv2
import sys

#import numpy as np
from scipy import signal
from scipy import optimize
import cv2#,colorsys
from matplotlib import pyplot as plt
import math

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_BG = {'color' : RED, 'val' : 2}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}

# setting up flags
rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness

def onmouse(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

    # Draw Rectangle
    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            rect_or_mask = 0

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
        rect_or_mask = 0
        print(" Now press the key 'n' a few times until no further change \n")

    # draw touchup curves
    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print("first draw rectangle \n")
        else:
            drawing = True
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

class template():
    def __init__(self, w ,d):
        self.w = w
        self.d = d
    def info(self):
        print (self.w, self.d)

'''
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

def templateAngle(x,t):
    center1 = x
    while center1<0: center1 += 360
    while center1>=360: center1 -= 360

    if t.d != -1:
        center2 = center1 + t.d
        if center2 < 0 :
            center2 += 360
        elif center2 >= 360:
            center2 -= 360
  
    total = 0
    for i in range(height):
        for j in range(width):
            if mask3[i][j] == 255: continue
            hue = img3[i][j][0]<<1
            turn = abs(func(center1,hue))
            turn2 = -1
            if t.d != -1:
                turn2 = abs(func(center2,hue))   
            if turn < t.w[0]:
                total += 0
            elif turn2 >=0 and turn2 <t.w[1]:
                total += 0
            else:
                if t.d != -1:
                    minDistance = min(turn-t.w[0], turn2-t.w[1])
                    total += minDistance*img3[i][j][1]
                else :
                    total += (turn-t.w[0])*img3[i][j][1]
    return total

if __name__ == '__main__':
    # Loading images
    if len(sys.argv) == 2: filename = sys.argv[1] # for drawing purposes
    else:
        print("No input image given, so loading default image\n")
        print("Correct Usage: python grabcut.py <filename> \n")
        filename = '03.jpg'
    
    # init all template
    templateSelect = [0,0,0,0,0,0,0]
    t = []
    t.append(template([5],-1))
    t.append(template([45],-1))
    t.append(template([5,30],90))
    t.append(template([5,5],180))
    t.append(template([90],-1))
    t.append(template([45,5],180))
    t.append(template([45,45],180))

    img = cv2.imread(filename)
    height, width = img.shape[:2]
    img2 = img.copy()                               # a copy of original image
    img3 = img2.copy()
    mask = np.zeros(img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
    output = np.zeros(img.shape, np.uint8)           # output image to be shown

    # input and output windows
    cv2.namedWindow('output')
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', onmouse)
    cv2.moveWindow('input', img.shape[1]+110,90)

    print(" Instructions: \n")
    print(" Draw a rectangle around the object using right mouse button \n")

    while(1):
        cv2.imshow('output',output)
        cv2.imshow('input',img)
        k = 0xFF & cv2.waitKey(1)

        # key bindings
        if k == 27:         # esc to exit
            break
        elif k == ord('0'): # BG drawing
            print(" mark background regions with left mouse button \n")
            value = DRAW_BG
        elif k == ord('1'): # FG drawing
            print(" mark foreground regions with left mouse button \n")
            value = DRAW_FG
        elif k == ord('2'): # PR_BG drawing
            value = DRAW_PR_BG
        elif k == ord('3'): # PR_FG drawing
            value = DRAW_PR_FG
        elif k == ord('t'): # PR_FG drawing
            templateGUI.TypeSelect(templateSelect)
        elif k == ord('s'): # save image
            bar = np.zeros((img.shape[0],5,3),np.uint8)
            res = np.hstack((img2,bar,img,bar,output))
            cv2.imwrite('grabcut_output.png',res)
            print(" Result saved as image \n")
        elif k == ord('r'): # reset everything
            print("resetting \n")
            rect = (0,0,1,1)
            drawing = False
            rectangle = False
            rect_or_mask = 100
            rect_over = False
            value = DRAW_FG
            img = img2.copy()
            img3 = img2.copy()
            mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
            output = np.zeros(img.shape,np.uint8)           # output image to be shown
        elif k == ord('n'): # segment the image
            print(""" For finer touchups, mark foreground and background after pressing keys 0-3
            and again press 'n' \n""")
            if (rect_or_mask == 0):         # grabcut with rect
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
                rect_or_mask = 1
            elif rect_or_mask == 1:         # grabcut with mask
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
        elif k == ord('c'): #color harmony
            print("caculating color harmony")
            cList = []
            valueList = []

            mask3 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
            img3 = img2.copy()
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
            for i in range(len(t)):
                if templateSelect[i] == 1:
                    center = optimize.brent(templateAngle,(t[i],))
                    cList.append(center)
                    valueList.append(templateAngle(center,t[i]))

            templateIndex = valueList.index(min(valueList))
            center = cList[templateIndex]
            while center<0: center+=360
            while center>=360: center-=360
            #templateIndex = 2
            template = t[templateIndex]

            w = template.w[0]
            size = 31
            window = signal.gaussian(size, std=w>>1)
            interval = 360/(size-1)
            print (window, interval)

            if template.d != -1:
                center2 =center + template.d
                if center2 < 0 : center2 += 360
                elif center2 >= 360: center2 -= 360
            a =0
            shift =0
            for i in range(height):
                for j in range(width):
                    if mask3[i][j] == 0: continue
                    hue = img3[i][j][0]<<1
                    turn = func(center, hue)

                    if template.d != -1: 
                        turn2 = func(center2,hue)
                    if abs(turn) < template.w[0]: continue
                    elif template.d != -1 and abs(turn2) < template.w[1]: continue
                    else:
                        a +=   1
                        if template.d != -1:
                            if abs(turn)-template.w[0] > abs(turn2)-template.w[1]:
                                if turn2<0: shift-=1
                                else: shift+=1
                            else:
                                if turn<0: shift-=1
                                else: shift+=1
                        else:
                            if turn<0: shift-=1
                            else: shift+=1
            shift=5*shift/a
            print(shift)

            for i in range(height):
                for j in range(width):
                    if mask3[i][j] == 0: continue
                    hue = img3[i][j][0]<<1
                    turn = func(center, hue)

                    if template.d != -1: 
                        turn2 = func(center2,hue)
                    if abs(turn) < template.w[0]: continue
                    elif template.d != -1 and abs(turn2) < template.w[1]: continue
                    else:
                        hue = (img3[i][j][0]<<1+shift)%360
                        turn = func(center, hue)
                        if template.d != -1:
                            turn2 = func(center2,hue)
                            if abs(turn)-template.w[0] > abs(turn2)-template.w[1]:
                                hue = int(center2 + 0.5*template.w[1]*(1 - window[ int((turn2+180)/interval) ]))
                            else:
                                hue = int(center + 0.5*template.w[0]*(1 - window[ int((turn+180)/interval) ]))
                        else :
                            hue = int(center + 0.5*template.w[0]*(1 - window[ int((turn+180)/interval) ]))
                    if hue >= 360: hue = hue -360
                    elif hue < 0 : hue = hue + 360
                    img3[i][j][0] = hue>>1
            
            img4 = img3.copy()
            img4 = cv2.cvtColor(img4, cv2.COLOR_HSV2BGR)
            for i in range(height):
                for j in range(width):
                    if mask3[i][j] == 0: 
                        img4[i][j][0] = img2[i][j][0]
                        img4[i][j][1] = img2[i][j][1]
                        img4[i][j][2] = img2[i][j][2]
            img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
            print("end color harmony")
            print ('test: ', center)
            print (cList)
            print (valueList)
            
            #img3 = cv2.bitwise_and(img3, img3,mask=mask2)
            
            
            plt.imshow(img4),plt.colorbar(),plt.show()
        mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
        #print(np.where(mask2==255))
        # if mask[i] == 1: mask2[i] = 255
        # if mask[i] == 3: mask2[i] = 0
        output = cv2.bitwise_and(img2,img2,mask=mask2)
    cv2.destroyAllWindows()