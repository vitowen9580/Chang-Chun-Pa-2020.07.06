import sys
import numpy as np
import cv2
import imutils
from imutils import perspective
from imutils import contours
from PIL import *
import os
from config import text_setting,white_line_setting,white_and_red_line_setting
from PIL import Image
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from ImageProcessing import Image
from scipy.spatial import distance as dist
from PIL import ImageGrab 
import pytesseract 
from model import resnet50
from skimage import morphology,measure,color
import scipy.ndimage as ndi
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tkinter import font


pytesseract.pytesseract.tesseract_cmd = 'c:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0

#包裝袋正面白線瑕疵檢測
def detect_white_line():

    Setting=white_line_setting()
    Processing=Image()

    for _, _, fileNames in os.walk(Setting.folder):
        for name in fileNames:
            print('====',name,'===')
            original_img=cv2.imread(Setting.folder+name)
            h,w,_=original_img.shape

            #選取線條部分
            ROI_img=original_img[200:h,:]
            #白平衡
            balance_img=Processing.balance(ROI_img)
            #利用HSV選取白色線條
            line_img=Processing.detect_color('white',balance_img)

            #形態學
            erode_img = cv2.erode(line_img, None, iterations=1)

            gray_img =cv2.cvtColor(erode_img, cv2.COLOR_BGR2GRAY)
            canny = Processing.otsu_canny(gray_img)

            #邊緣強化
            canny_dilate = cv2.dilate(canny, None, iterations=2)

            #計算每個線條W,H
            Bounding_box_img=Processing.contour(canny_img=canny_dilate,orig_img=ROI_img,pass_threshold=Setting.pass_threshold)

            ROI_img=np.hstack((ROI_img,Bounding_box_img))
            cv2.imshow('original',ROI_img)
            cv2.waitKey(0)

#包裝袋反面紅白線瑕疵檢測
def detect_white_red_line():
    Setting=white_and_red_line_setting()
    Processing=Image()


    for  _, _, fileNames in os.walk(Setting.folder):
        for name in fileNames:
            print('=====',name,'=====')

            original_img=cv2.imread(Setting.folder+name)
            h,w,_=original_img.shape

            #選取線條部分
            ROI_img=original_img[50:200,:]

            #利用HSV選取紅色線條
            red_line_img=Processing.detect_color('red',ROI_img)  

            #Noise Filter 均值遷移
            mean_shift_blur_red_img=Processing.mean_shift_blur_filter(red_line_img)
            gray =cv2.cvtColor(mean_shift_blur_red_img, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(gray, 30, 150)

            #邊緣強化
            canny_dilate = cv2.dilate(canny, None, iterations=1)

            #計算每個線條W,H
            Red_Bounding_box_img=Processing.contour(canny_img=canny_dilate,orig_img=red_line_img,pass_threshold=Setting.pass_threshold)

            #利用HSV選取白色線條
            white_line_img=Processing.detect_color('white',ROI_img)  

            #Noise Filter 均值遷移
            mean_shift_blur_white_img=Processing.mean_shift_blur_filter(white_line_img)

            #白線與紅線合併
            add_img=Processing.addImage(mean_shift_blur_white_img,mean_shift_blur_red_img)

            ROI_img=np.hstack((ROI_img,add_img))
            ROI_img=np.hstack((ROI_img,Red_Bounding_box_img))

            cv2.imshow('img',ROI_img)
            cv2.waitKey(0)

#包裝袋文字瑕疵檢測
def Detect_Text():
    Setting=text_setting()
    Processing=Image()
    _Image=Image()
    Set=text_setting()
    _resnet50=resnet50()

    #統計accuracy
    correct_count=0
    not_correct_count=0


    total_img_count=0
    img_list=list()
    #計算多少張圖片
    for _, _, fileNames in os.walk(Setting.test_dataset_folder):
        for f in fileNames:
            img_list.append(f)
    total_img_count=len(img_list)
    print('total image count:',total_img_count)


    done_count=0
    for _, dirPath, fileNames in os.walk(Setting.test_dataset_folder):
        for f in fileNames:
            done_count=done_count+1
            print('{}/{}'.format(done_count,total_img_count))
            img_path=os.path.join(Setting.test_dataset_folder, f)
            img_rgb= cv2.imread(img_path) 

            #取的圖片名子
            img_name=img_path.split('test/')[1]

            #判斷包裝袋文字為底部或上方
            pred_class=_resnet50.test(img_path)
            correct=False

            # 0:上方,1:底部
            if(pred_class=='0'):
                #文字ROI
                (x_right,y_bottom),_,_  =_Image.Template_Matching(img_name,'template2.JPG',False)
                _,(x_left,y_top),_  =_Image.Template_Matching(img_name,'template3.JPG',False)
                x_right=x_right+100
                x_left=x_left-10
                img_roi=img_rgb[y_top:y_bottom,x_left:x_right]

                #若有框到文字
                if(img_roi.shape[0]*img_roi.shape[1]>0):

                    #透過HSV提取文字部分，先使用hsv_calibration.py決定lower及upper範圍
                    hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
                    for V_value in range(-10,10,1):
                        lower = np.array([0, 0, 0])
                        upper = np.array([180, 255, 100+V_value])
                        mask = cv2.inRange(hsv, lower, upper)

                        #去除雜訊
                        filter_img=Processing.clearNoise(img_roi,mask)

                        rev_mask=255-filter_img

                        kernel = np.ones((3,3),np.uint8)
                        erosion = cv2.erode(rev_mask,kernel,iterations =1)

                        #OCR
                        text = pytesseract.image_to_string(erosion)

                        #去除'\n'及空白char
                        text=text.strip('\n')
                        text_str=''
                        _list=list()
                        for char in text:
                            if(char!='\n')and(char!=' '):
                                _list.append(char)
                                text_str=text_str+char


                        if(text_str==Set.class_anwser):
                            correct=True
                            break
                # print('class:',pred_class,'img_name:',img_name,'pred:',text_str,'anwser:',Set.class_anwser,'correct:',correct)


            elif(pred_class=='1'):
                #直接框選ROI位置
                img_roi=img_rgb[img_rgb.shape[0]-200:img_rgb.shape[0]-50,50:img_rgb.shape[1]-50]
                #若有框到文字
                if(img_roi.shape[0]*img_roi.shape[1]>0):

                    #透過YCbCr提取文字部分，先使用hsv_calibration.py決定lower及upper範圍
                    hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2YCR_CB)
                    for CB_value in range(-10,10,1):

                        lower = np.array([0, 0, 0])
                        upper= np.array([146, 255, 255])
                        mask = cv2.inRange(hsv, lower, upper+CB_value)

                        #去除雜訊
                        filter_img=Processing.clearNoise(img_roi,mask)

                        rev_mask=255-filter_img

                        kernel = np.ones((5,5),np.uint8)
                        erosion = cv2.erode(rev_mask,kernel,iterations = 1)
 
                        #OCR
                        text = pytesseract.image_to_string(erosion)

                        #去除'\n'及空白char
                        text=text.strip('\n')
                        text_str=''
                        for char in text:
                            if(char!='\n') and (char!=' '):
                                
                                text_str=text_str+char

                        if(text_str==Set.class_anwser):
                            correct=True
                            break



                
            if(correct==False):
                print('class:',pred_class,'img_name:',img_name,'pred:',text_str,'anwser:',Set.class_anwser,'correct:',correct)

                not_correct_count=not_correct_count+1
                cv2.imwrite(Set.error_dataset_folder+img_name,img_rgb)
            else:
                correct_count=correct_count+1


        print('correct:',correct_count,'error:',not_correct_count,'accuracy:',correct_count/(correct_count+not_correct_count))

def main():
    # detect_white_line()
    # detect_white_red_line()
    Detect_Text()
        
if __name__ == "__main__":
    main()

