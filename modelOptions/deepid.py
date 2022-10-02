from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# test importing image
img1_path = './images/good1.jpeg'
img2_path = './images/good2.jpeg'

# confirming the path of images
img1 = cv2.imread(img1_path)

plt.imshow(img1[:, :, ::-1 ]) # setting value as -1 to maintain saturation
plt.show()

# calling the model
model_name = 'DeepID'
# creating a function named resp to store the result
resp = DeepFace.verify(img1_path = img1_path , img2_path = img2_path, model_name = model_name)
print(resp) #generating our result 

# keep in mind iska threshold bohot low hai