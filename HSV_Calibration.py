
import cv2
import numpy as np
def trackChaned(x):
  pass
 
 
cv2.namedWindow('Color Track Bar')
Yhh='YMax'
Yhl='YMin'
Crhh='CrMax'
Crhl='CrMin'
Cbhh='CbMax'
Cbhl='CbMin'

wnd = 'Colorbars'
cv2.createTrackbar("YMax", "Color Track Bar",0,255,trackChaned)
cv2.createTrackbar("YMin", "Color Track Bar",0,255,trackChaned)
cv2.createTrackbar("CrMax", "Color Track Bar",0,255,trackChaned)
cv2.createTrackbar("CrMin", "Color Track Bar",0,255,trackChaned)
cv2.createTrackbar("CbMax", "Color Track Bar",0,255,trackChaned)
cv2.createTrackbar("CbMin", "Color Track Bar",0,255,trackChaned)

img = cv2.imread('./sample/ROI/888_Scn000_2020_05_26-09_43_56(0)_503_OK.bmp')
img_roi=img[img.shape[0]-200:img.shape[0],50:img.shape[1]-80]

# img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
YCrCb_img = cv2.cvtColor(img_roi, cv2.COLOR_BGR2YCR_CB)

while(True):
    Yhul=cv2.getTrackbarPos("YMax", "Color Track Bar")
    Yhuh=cv2.getTrackbarPos("YMin", "Color Track Bar")
    Crhul=cv2.getTrackbarPos("CrMax", "Color Track Bar")
    Crhuh=cv2.getTrackbarPos("CrMin", "Color Track Bar")
    Cbhul=cv2.getTrackbarPos("CbMax", "Color Track Bar")
    Cbhuh=cv2.getTrackbarPos("CbMin", "Color Track Bar")

    print('Yhul:',Yhul,'Yhuh:',Yhuh,'Crhul:',Crhul,'Crhuh:',Crhuh,'Cbhul:',Cbhul,'Cbhuh:',Cbhuh)

    lower_green = np.array([Yhul, Crhul, Cbhul])
    upper_green = np.array([Yhuh, Crhuh, Cbhuh])
    mask = cv2.inRange(YCrCb_img, lower_green, upper_green)
    rev_mask=255-mask
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(rev_mask,kernel,iterations = 1)


    cv2.imshow('img',erosion)
    cv2.imshow('ROI',img_roi)
    

    if cv2.waitKey(1) == 27:

        break

cv2.destroyAllWindows()