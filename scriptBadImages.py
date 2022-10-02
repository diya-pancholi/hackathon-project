# import the modules
import os
from os import listdir
import modelBadImages as model
import shutil
import faceLandmarkBad as fl2

folder_dir = "/home/diyapancholi/photos/photos" # make folder for generalized uses maybe
image_list = []
destination_dir = "/mnt/d/finalTestBad" # set your own path
for images in os.listdir(folder_dir):
    # check if the image ends with jpg
    if (images.endswith(".jpg")):
        ready = model.vgg_filter(images, folder_dir)
        if (ready):
            try:
                ratios = fl2.processImage(images, folder_dir)
                print(ratios)
                # if(ratios[0]>-1.75 and ratios[1]>-11 and ratios[2]>2):
                # set appropriate ratios for bad images
                image_list.append(images)
            except:
               print("error while running face_landmark")
            
print(image_list)
for images in image_list:
    try:
        shutil.copy(os.path.join(folder_dir, images), destination_dir)
        print("File copied successfully.")
    except:
        print("Error occurred while copying file.")