from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import os

# get frontal face is inbuilt
detector = dlib.get_frontal_face_detector()

# shape predictor we have to download locally
# insert your own path here, where you have downloaded this file locally
predictor = dlib.shape_predictor('/mnt/d/shape_predictor_68_face_landmarks.dat') 

# test import
img1_path = './images/good1.jpeg'

# test path check
img1 = cv2.imread(img1_path)

def processImage(image, folder_dir):
    image = cv2.imread(os.path.join(folder_dir, image))
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect the faces
    rects = detector(gray)
    
    # go through the face bounding boxes
    for rect in rects:
        # extract the coordinates of the bounding box
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()
        print(x1, y1, x2, y2)

        shape = predictor(gray, rect)
        
        #lip edge coordinates
        leftLipX = shape.part(49).x
        rightLipX = shape.part(55).x
        topLipY = shape.part(52).y
        LowerLipY = shape.part(58).y
        
        #lip pout ratio
        lipWidth = rightLipX - leftLipX
        lipHeight = topLipY - LowerLipY
        ratioLip = lipWidth / lipHeight
        print('ratio lip', ratioLip)
        
        #left eye coordinates
        topLeftY = shape.part(37).y
        topRightY = shape.part(38).y
        lowLeftY = shape.part(41).y
        lowRightY = shape.part(40).y
        
        #left eye ratio
        leftDifL = topLeftY - lowLeftY
        rightDifL = topRightY - lowRightY
        avgLeftEye = (leftDifL + rightDifL)/2
        
        #right eye coordinates
        topLeftYR = shape.part(37).y
        topRightYR = shape.part(38).y
        lowLeftYR = shape.part(41).y
        lowRightYR = shape.part(40).y
        
        #left eye ratio
        leftDifR = topLeftYR - lowLeftYR
        rightDifR = topRightYR - lowRightYR
        avgRightEye = (leftDifR + rightDifR)/2
        
        #size difference of eyes
        ratioEye = (avgLeftEye - avgRightEye)/avgLeftEye + avgRightEye
        print('ratio eyes', ratioEye )
        
        #coordinates of face
        leftCheekX= shape.part(4).x
        rightCheekX = shape.part(12).x
        topNoseY = shape.part(27).y
        lowChinY = shape.part(8).y
        
        #calculating ratio of face height
        cheekWidth = leftCheekX - rightCheekX
        faceHeight = topNoseY - lowChinY
        ratioFace = cheekWidth - faceHeight
        print('ratio face', ratioFace)
        
        return(ratioLip, ratioEye, ratioFace)
