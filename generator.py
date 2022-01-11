import cv2
import numpy as np
import random

class generate:
    def batch_1(img):
        #img = np.asarray(img, np.float64)
        images_arrays = []
        """Image arrays format is in list arrays"""
        images_arrays.insert(random.randint(0,len(images_arrays)), generate.rotation(img, -90))
        images_arrays.insert(random.randint(0,len(images_arrays)), generate.rotation(img, 90))
        images_arrays.insert(random.randint(0,len(images_arrays)), generate.vertical_flip(img))
        #images_arrays.insert(random.randint(0,len(images_arrays)), generate.high_brightness(img))
        #images_arrays.insert(random.randint(0,len(images_arrays)), generate.low_brightness(img))
        images_arrays.insert(random.randint(0,len(images_arrays)), generate.sharpen(img))

        return images_arrays

    
    def rotation(image,angle):
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, scale=1)
        rotated_image = cv2.warpAffine(image,rotation_matrix,(width,height))
        return rotated_image

 
    def vertical_flip(image):
        return cv2.flip(image,0)
    
    def high_brightness(image):
        print(image.shape, np.ones(image.shape , dtype=np.uint8).shape)
        return cv2.add(image, np.ones(image.shape , dtype=np.uint8) * .3)

    def low_brightness(image):
        print(image.shape, np.ones(image.shape , dtype=np.uint8).shape)
        return cv2.add(image, np.ones(image.shape , dtype=np.uint8) * -.3)

    def sharpen(image):
        kernel = np.array([ [-1,-1,-1],
                            [-1,10,-1],
                            [-1,-1,-1] ])
        return cv2.filter2D(image,-1,kernel)
        
          