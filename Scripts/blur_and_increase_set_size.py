#!/usr/bin/env python
# coding: utf-8

# In[5]:


#load the modules
from imutils import paths
import random
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


# In[6]:


#function for motion blur
#adapted from Joshi, P. (2015). OpenCV with Python by example. Packt Publishing Ltd.
def blurpic(img, size):
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    img = cv2.filter2D(img, -1, kernel_motion_blur)
    return img


# In[7]:


#function to resize picture
def resizing(img, x, y, n,m):
    #x and y as the width and height of the image
    #n and m define the interval by which the image will be downsized
    t=round(random.uniform(n, m),2)
    img = cv2.resize(img, (int(x/t), int(y/t))) 
    return img


# In[8]:


#function for gaussian noise transformation
def gnoise(img, mean, sd):
    img = img+ np.random.normal(mean,sd, img.shape)
    img = np.clip(img, 0, 255)
    return img


# In[9]:


#function for gaussian blur transformation
def gblur(img, size):
    img=cv2.GaussianBlur(img,(size,size),0)
    return img


# In[14]:


#load a test image to check the different transformations
#plot after resizing to 224x224 (the image size used for training the VGG19)
"""img =cv2.imread('E:\Swift Project Images\2796.Chimney Swift\2796.Chimney_Swift_0_-2_106_449_557_-_e_0376.jpg')
fig=plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(cv2.resize(img, (224,224)), cv2.COLOR_BGR2RGB))
plt.title("original")
plt.show()"""


# In[7]:

"""
#plot the different transformations for comparison
fig=plt.figure(figsize=(10, 10))
columns = 2
rows = 2
x=img.shape[1]
y=img.shape[0]
images_list=[]
images_list.append(blurpic(img, 15))
images_list.append(gblur(img, 7))
images_list.append(resizing(img, x, y ,1.5,2.5))

transformations_titles=[ "motion_blur", "gaussian_blur", "resizing"]
p=0
for i in range(1, columns*rows):
    fig.add_subplot(rows, columns, i)
    plt.title(transformations_titles[p])
    plt.imshow(cv2.cvtColor(cv2.resize(images_list[p], (224,224)), cv2.COLOR_BGR2RGB))
    p=p+1

#extra step to plot the gnoise transformation
#which only properly plots after converting to .astype(np.uint8)
fig.add_subplot(rows, columns, 4)
imgnoise=gnoise(img,15,25).astype(np.uint8)
plt.title("gaussian_noise")
plt.imshow(cv2.cvtColor(cv2.resize(imgnoise, (224,224)), cv2.COLOR_BGR2RGB))

plt.show()"""


# In[ ]:


#directory containing the original dataset
dataset="/media/adhi/Warzone & other Big Games/Swift Project Images/madness"
#list all images in the original dataset
imagePaths = sorted(list(paths.list_images(dataset)))

#loop through all images to apply the transformations
for imagePath in imagePaths:
    #define the output folder and add the name of the image file
    directory_output="/media/adhi/Warzone & other Big Games/Swift Project Images/ting/dove2/"+imagePath.split(os.path.sep)[-1][:-4]
    #load the image
    img =cv2.imread(imagePath)
    #apply the motion blur transformation
    imgb=blurpic(img, 15)
    #save the image
    cv2.imwrite(str(directory_output+"b"+".jpg"), imgb)
    
    #apply the resizing  transformation
    x=img.shape[1]
    y=img.shape[0]
    imgr=resizing(img,x, y, 2,3.5)
    #save the image
    cv2.imwrite(str(directory_output+"r"+".jpg"), imgr)
    
    #apply the gaussian noise transformation
    imgg=gnoise(img,15,25)
    #save the image
    cv2.imwrite(str(directory_output+"g"+".jpg"), imgg)
    #apply the gaussian blur transformation
    imggb=gblur(img, 7)
    #save the image
    cv2.imwrite(str(directory_output+"gb"+".jpg"), imggb)
    
    #in order to apply a mixture of transformations to the same image
    #generate two random numbers that will define which transformation
    #will be applied
    transformations=random.sample(set([1,2,3,4]), 2)
    
    for t in range(0,1):
        if transformations[t]==1:
            img=blurpic(img, 15)
        if transformations[t]==2:
            img=gnoise(img,15,25)
        if transformations[t]==3:
            img=gblur(img, 7)
        if transformations[t]==4:
            img=resizing(img, x, y ,1.5,2.5)
    #save the image after applying to different transformations
    cv2.imwrite(str(directory_output+"m"+".jpg"), img)
    print(imagePath)

