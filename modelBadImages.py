from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os

model_name = "VGG-Face"
model = DeepFace.build_model(model_name)
dataset_dir = "./badImages"

def setcount():
    global count
    count = 0
    
setcount() 
   
def upcount():
    global count
    count += 1
    
def vgg_filter(img, folder_dir):
    check = 0
    print (img, 'params')
    for dataImg in os.listdir(dataset_dir):  
        dataImg_path = os.path.join(dataset_dir, dataImg)
        print("Filename : ",img) 
        img_path = os.path.join(folder_dir, img)
        try:
            result = DeepFace.verify(dataImg_path, img_path)
        except Exception:
            upcount()
            continue
        print(result['distance'])
        if (result['distance']<0.25): # checking if match threshold is > 0.75
            check += 1
    print(count)
    if(check>=3): # threshold has to be crossed for atleast 3/5 images
        return 1>0
    else:
        return 0>1  
    
#importing the images
img1_path = './images/bad1.jpeg'
img2_path = './images/bad2.jpeg'

#confirming the path of images
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# creating an object to analyze facial features

obj = DeepFace.analyze(img_path = "./images/good1.jpeg", actions = ['age', 'gender', 'race', 'emotion'])
print(obj["age"]," years old ",obj["dominant_race"]," ",obj["dominant_emotion"]," ", obj["gender"])
