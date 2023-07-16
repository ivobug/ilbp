# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:15:45 2023

@author: Ivan
"""


import os
import imghdr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
#from skimage import feature

PATH='Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000'



def find_images(dirpath):
    """Recursively detect image files contained in a folder.

    Parameters
    ----------
    dirpath : string
        Path name of the folder that contains the images.

    Returns
    -------
    imgfiles : list
        Full path names of all the image files in `dirpath` (and its
        subfolders).

    Notes
    -----
    The complete list of image types being detected can be found at:
    https://docs.python.org/3/library/imghdr.html
    
    """
    imgfiles = [os.path.join(root, filename)
                for root, dirs, files in os.walk(dirpath)
                for filename in files
                if imghdr.what(os.path.join(root, filename))]
    
    return imgfiles


def get_class(filename):
    """Extract the class label from the path of a Kather dataset sample.
    
    Parameters
    ----------
    filename : string
        Filename (including path) of a texture sample 
        from Kather dataset.

    Returns
    -------
    class_name : string
        Class name to which the texture sample belongs.
    
    """
    folder, _ = os.path.split(filename)
    _, class_name = os.path.split(folder)
    
    return class_name


def get_pixel(img, center, x, y):
      
    new_value = 0
      
    try:
        # If local neighbourhood pixel 
        # value is greater than or equal
        # to center pixel values then 
        # set it to 1
        if img[x][y] >= center:
            new_value = 1
              
    except:
        # Exception is required when 
        # neighbourhood value of a center
        # pixel value is null i.e. values
        # present at boundaries.
        pass
      
    return new_value


def threshold_apply(pixel_to_analyze_value, threshold_pixel_value, threshold):   
    if pixel_to_analyze_value - (threshold_pixel_value * threshold) >= 0:
        return 1
    else:
        return 0
   
# Function for calculating LBP
def lbp_calculated_pixel(img, x, y, threshold=1):
   
    center = img[x][y]
   
    val_ar = []
      
    # top_left
    val_ar.append(get_pixel(img, center, x-1, y-1))
      
    # top
    val_ar.append(get_pixel(img, center, x-1, y))
      
    # top_right
    val_ar.append(get_pixel(img, center, x-1, y + 1))
      
    # right
    val_ar.append(get_pixel(img, center, x, y + 1))
      
    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
      
    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))
      
    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y-1))
      
    # left
    val_ar.append(get_pixel(img, center, x, y-1))
    
    max_val = np.max(val_ar)
    for i in range(len(val_ar)):
        val_ar[i] = threshold_apply(val_ar[i],max_val, threshold)
       
       
    # Now, we need to convert binary
    # values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
   
    val = 0
      
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
          
    return val



path = 'Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/01_TUMOR/1A11_CRC-Prim-HE-07_022.tif_Row_601_Col_151.tif'
img_bgr = cv2.imread(path, 1)
   
height, width, _ = img_bgr.shape
   
# We need to convert RGB image 
# into gray one because gray 
# image has one channel only.
img_gray = cv2.cvtColor(img_bgr,
                        cv2.COLOR_BGR2GRAY)
   
# Create a numpy array as 
# the same height and width 
# of RGB image
img_lbp = np.zeros((height, width),
                   np.uint8)
   
for i in range(0, height):
    for j in range(0, width):
        img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j, 0.1)
  
plt.imshow(img_bgr)
plt.show()
   
plt.imshow(img_lbp, cmap ="gray")
plt.show()


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
		# store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    def describe(self, image, threshold):
		# compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        height, width = image.shape
        img_lbp = np.zeros((height, width),
                           np.uint8)
    
        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = lbp_calculated_pixel(image, i, j, threshold)
        (hist, _) = np.histogram(img_lbp.ravel(),
                bins=np.arange(0, 11),
                
                range=(0, 8 +2))
        
        return hist.astype("float") / hist.sum()
		# return the histogram of Local Binary Patterns
        return hist
        
        
    
    
desc = LocalBinaryPatterns(24, 8)

data = []
labels = []


def load_and_describe_images(color=False,threshold=0.9):
    data.clear()
    labels.clear()
    if color==False:
        # loop over the training images
        for imagePath in find_images(PATH):
        	# load the image, convert it to grayscale, and describe it
        	image = cv2.imread(imagePath)
        	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        	hist = desc.describe(gray,threshold)
        	# extract the label from the image path, then update the
        	# label and data lists 

        	labels.append(get_class(imagePath))
        	data.append(hist)

    elif color:     
        for imagePath in find_images(PATH):
            image = cv2.imread(imagePath)
            image_features = []
            b,g,r = cv2.split(image)
            separated_colors = np.array([b,g,r])
            for color_channel in range(len(separated_colors)):
                img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                #hist, _ = np.histogram(image[:, :, color_channel], bins=256, range=(0, 256))
                hist = desc.describe(img_gray, threshold)
                image_features.extend(hist)
            data.append(image_features)
            labels.append(get_class(imagePath))
            

load_and_describe_images(False, 0.9)

#Discriminative LBP 
###### Takes t0o long #######
#np.concatenate((load_and_describe_images, load_and_describe_images(False, 0.9)))
    
#Separate train and test data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

#Create model and train
nearest_neighbour_classifier = KNeighborsClassifier(7)
nearest_neighbour_classifier.fit(X_train, y_train)
predicted_textures = nearest_neighbour_classifier.predict(X_test)





acc = accuracy_score(predicted_textures,y_test)
print("Accuracy:",acc)






