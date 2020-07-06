from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import numpy as np
import os
# 從參數讀取圖檔路徑
# files = sys.argv[1:]
class resnet50:
    def __init__(self):
        self.net = load_model('resnet50_0625.h5')
        self.cls_list = ['0', '1']

        pass
    def test(self,img_path):
        img = image.load_img(img_path, target_size=(224, 224))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        pred = self.net.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        


        max_confidience=-1
        pred_class=''
        for i in top_inds:
            if(pred[i]>max_confidience):
                max_confidience=pred[i]
                pred_class=self.cls_list[i]
        return str(pred_class)



# class KNNModel:
#     def __init__(self):
#        pass
    
#     def Recognize(self,img):
#         MIN_CONTOUR_AREA = 100
#         RESIZED_IMAGE_WIDTH = 20
#         RESIZED_IMAGE_HEIGHT = 30
        
#         # print(rootpath)
#         allContoursWithData = []                # declare empty lists,
#         validContoursWithData = []              # we will fill these shortly
#         # try:
#         npaClassifications = np.loadtxt(_Config.classifications, np.float32)                  # read in training classifications
#         # except:
#         #     print ("error, unable to open classifications.txt, exiting program\n")
#         #     os.system("pause")
#         #     return
#         # end try

#         # try:
#         npaFlattenedImages = np.loadtxt(_Config.flattened_images, np.float32)                 # read in training images
#         # except:
#         #     print ("error, unable to open flattened_images.txt, exiting program\n")
#         #     os.system("pause")
#         #     return
#         # end try


#         npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

#         kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object

#         kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

#         # img=cv2.resize(img,(60,60))

#         imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       # get grayscale image
#         # imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur
        

#                                                         # filter image from grayscale to black and white
#         imgThresh = cv2.adaptiveThreshold(imgGray,                           # input image
#                                       255,                                  # make pixels that pass the threshold full white
#                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
#                                       cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
#                                       11,                                   # size of a pixel neighborhood used to calculate threshold value
#                                       2)                                    # constant subtracted from the mean or weighted mean
        
        
        
        
#         # kernel = np.ones((2,2),np.uint8)
#         # dil = cv2.dilate(imgThresh,kernel,iterations =1)
#         # cv2.imwrite('dil.jpg', dil)

#         # cv2.imshow("dil", dil)
#         # cv2.waitKey(0)
                              
        
        
#         imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

#         kernel = np.ones((2,2),np.uint8)
#         dil = cv2.dilate(imgThreshCopy,kernel,iterations =1)

#         npaContours, npaHierarchy = cv2.findContours(dil,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
#                                                  cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
#                                                  cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points
  
#         clone=img.copy()    
#         for npaContour in npaContours:                             # for each contour

#             box = cv2.minAreaRect(npaContour)
#             box = np.int0(cv2.boxPoints (box))
#             (x, y, w, h) = cv2.boundingRect(npaContour)
#             # drawR=cv2.drawContours(clone, [box], -1, (0, 0, 255), 2)
#             # cv2.imshow('drawR',drawR)
#             # cv2.waitKey(0)
#             # x0=box[0][0]
#             # y0=box[0][1]
#             # x1=box[1][0]
#             # y1=box[1][1]
#             # theta=(y0-y1)/(x0-x1)
#             # degree=math.atan(theta)*180/3.14
#             # print("degree=",degree)
#             # print(w,h)
#             if (w*h>1500):
#                 contourWithData = ContourWithData()                                             # instantiate a contour with data object
#                 contourWithData.npaContour = npaContour                                         # assign contour to contour with data
#                 contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
#                 contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
#                 contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
#                 allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
#     # end for

#         for contourWithData in allContoursWithData:                 # for all contours
#             if contourWithData.checkIfContourIsValid():             # check if valid
#                 validContoursWithData.append(contourWithData)       # if so, append to valid contour list
#         # end if
#     # end for

#         validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

#         strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

#         for contourWithData in validContoursWithData:            # for each contour
#                                                 # draw a green rect around the current char
#             # cv2.rectangle(img,                                        # draw rectangle on original testing image
#             #           (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
#             #           (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
#             #           (0, 255, 0),              # green
#             #           2)                        # thickness

#             imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
#                            contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

#             imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

#             npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

#             npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

#             retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest

#             strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

#             strFinalString = strFinalString + strCurrentChar            # append current char to full string
#     # end for

#         # print ( "result=",strFinalString + "\n")                  # show the full string
#         text=strFinalString
#         # cv2.imshow("img", img)      # show input image with green boxes drawn around found digits
#         # cv2.waitKey(0)                                          # wait for user key press

#         # cv2.destroyAllWindows()             # remove windows from memory

#         return text