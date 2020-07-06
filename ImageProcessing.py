import cv2
import numpy as np
from matplotlib import pyplot as plt
from config import text_setting
import imutils
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist

class Image:
    def __init__(self):
        self.methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        Set=text_setting()
        self.template_folder=Set.template_folder
        self.dataset_folder=Set.test_dataset_folder



    def translate(self,image, x, y):
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))              
        return shifted

    def Template_Matching(self,img_name,template_name,UseHalfPic):
        # print('===',self.dataset_folder+img_name)

        img = cv2.imread(self.dataset_folder+img_name,0) 
        img_rgb= cv2.imread(self.dataset_folder+img_name) 
        if UseHalfPic:
            img=img[:,int(img.shape[1]/2):img.shape[1]].copy() 
            img_rgb=img_rgb[:,int(img_rgb.shape[1]/2):img_rgb.shape[1]].copy()
        # cv2.imshow('img1',img_rgb)
        # cv2.waitKey(0)

 
        template = cv2.imread(self.template_folder+template_name,0)
        template_rgb=cv2.imread(self.template_folder+template_name)
        # cv2.imshow('template_rgb',template_rgb)
        # cv2.waitKey(0)
        # cv2.imshow('template',template)
        # cv2.waitKey(0)
        # template=template[:,int(template.shape[1]/2)-template_start_pixel:template.shape[1]].copy() 
        # template_rgb=template_rgb[:,int(template_rgb.shape[1]/2)-template_start_pixel:template_rgb.shape[1]].copy() 
        # img2 = img.copy()
        w, h = template.shape[::-1]


        x_list=list()


        # # Apply template Matching
        res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if(template_name=='template2.JPG'):
            top_left = (max_loc[0],max_loc[1])
            top_right=(max_loc[0] + w, max_loc[1])
            bottom_right = (max_loc[0] + w, max_loc[1] + h)
        elif(template_name=='template3.JPG'):
            top_left = (max_loc[0],max_loc[1]-50)
            top_right=(max_loc[0] + w, max_loc[1]-50)
            bottom_right = (max_loc[0] + w, max_loc[1] + h)
        elif(template_name=='template7.JPG'):
            top_left = (max_loc[0],max_loc[1])
            top_right=(max_loc[0] + w, max_loc[1])
            bottom_right = (max_loc[0] + w, max_loc[1] + h)
        elif(template_name=='template8.JPG'):
            top_left = (max_loc[0],max_loc[1])
            top_right=(max_loc[0] + w, max_loc[1])
            bottom_right = (max_loc[0] + w, max_loc[1] + h)

        # cv2.rectangle(img_rgb,top_left, bottom_right, 255, 2)
        # cv2.imshow('img',img_rgb)
        # cv2.waitKey(0)
        return  top_left,top_right,bottom_right 
            
    def bi_demo(self,image):      #雙邊濾波
        dst = cv2.bilateralFilter(image, 0, 100, 5)
        # cv2.imshow("bi_demo", dst)
        return dst

    def shift_demo(self,image,sigmaColor):   #均值遷移
        # sigmaColor,sigmaSpace
        # dst = cv2.pyrMeanShiftFiltering(image, 10, 50)
        dst = cv2.pyrMeanShiftFiltering(image, sigmaColor, 50)
        # cv2.imshow("shift_demo", dst)
        return dst

    def balance(self,image):

        a = 2
        O = image * float(a)
        O[O > 255] = 255
        O = np.round(O)
        O = O.astype(np.uint8)
        return O


    def detect_color(self,color,image):                          
        if color=='white':
                lower = np.array([0,0,221])
                upper = np.array([180,30,255])
        elif color=='red':
                lower = np.array([156,43,46])
                upper= np.array([180,255,255])

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower ,upper)
        res = cv2.bitwise_and(image,image, mask= mask)

        return res

    def mean_shift_blur_filter(self,image):   #均值遷移

        dst = cv2.pyrMeanShiftFiltering(image, 50, 50)
        # cv2.imshow("shift_demo", dst)
        return dst

    def otsu_canny(self,image, lowrate=0.1):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Otsu's thresholding
        ret, _ = cv2.threshold(image, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
        edged = cv2.Canny(image, threshold1=(ret * lowrate), threshold2=ret)

        # return the edged image
        return edged

    def midpoint(self,ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
 

    def draw(self,pic_box,orig,pass_threshold):
        for box in pic_box:
            (tl, tr, br, bl) = box
            (top_mid_X, top_mid_Y) = self.midpoint(tl, tr)
            (bottom_mid_X, bottom_mid_Y) = self.midpoint(bl, br)
            (left_mid_X, left_mid_Y) =  self.midpoint(tl, bl)
            (right_mid_X, right_mid_Y) =  self.midpoint(tr, br)
            dH = dist.euclidean((top_mid_X, top_mid_Y), (bottom_mid_X, bottom_mid_Y))
            dW = dist.euclidean((left_mid_X, left_mid_Y), (right_mid_X, right_mid_Y))
            if(dW>pass_threshold):
                cv2.putText(orig,str(int(dW)),(box[0][0],box[0][1]),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
            else:
                cv2.putText(orig,str(int(dW)),(box[0][0],box[0][1]),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)

            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 1, (0, 0, 255), -1)


    def contour(self,canny_img,orig_img,pass_threshold):
            cnts = cv2.findContours(canny_img.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if(len(cnts)!=0):
                (cnts, _) = contours.sort_contours(cnts)

                pic_box=list()
                for c in cnts:
                    if cv2.contourArea(c) < 50:
                        continue
                    orig = orig_img.copy()
                    x,y,w,h = cv2.boundingRect(c)
                    box = cv2.minAreaRect(c)
                    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                    box = np.array(box, dtype="int")

                    box = perspective.order_points(box)
                    pic_box.append(box)
                    
                self.draw(pic_box,orig,pass_threshold)
                return orig
            else:
                return orig_img

    def addImage(self,original_img, mask_img):
        alpha = 0.5
        beta = 1-alpha
        gamma = 0
        img_add = cv2.addWeighted(original_img, alpha, mask_img, beta, gamma)
        return img_add

    def clearNoise(self,original_img,mask_img):
        mask_gray = cv2.GaussianBlur(mask_img, (7, 7), 0)
        mask_edged = cv2.Canny(mask_gray, 50, 100)
        mask_edged = cv2.dilate(mask_edged, None, iterations=3)

        cnts = cv2.findContours(mask_edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if(cnts):
            (cnts, _) = contours.sort_contours(cnts)
        
            box_max=list()
            cntsflag=False
            counter=0
            max_img=None
            orig = mask_img.copy()
            for c in cnts:
                if cv2.contourArea(c) < 10:
                    continue
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                (tl, tr, br, bl) = box
                (top_mid_X, top_mid_Y) = self.midpoint(tl, tr)
                (bottom_mid_X, bottom_mid_Y) = self.midpoint(bl, br)
                (left_mid_X, left_mid_Y) = self.midpoint(tl, bl)
                (right_mid_X, right_mid_Y) = self.midpoint(tr, br)
                dH = dist.euclidean((top_mid_X, top_mid_Y), (bottom_mid_X, bottom_mid_Y))
                dW = dist.euclidean((left_mid_X, left_mid_Y), (right_mid_X, right_mid_Y))
                if(dW*dH)<150:
                    orig[int(tl[1]):int(bl[1]),int(tl[0]):int(tr[0])]=0
            return orig
        else:
            return original_img

