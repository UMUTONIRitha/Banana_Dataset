import cv2
import numpy as np
import random


class generate:
    def batch_1(img):
        """ Requires uint8 image dtype"""
        images_arrays = []
        images_arrays.append(generate.rotation(img, -90))
        images_arrays.append(generate.rotation(img, 90))
        images_arrays.append( generate.vertical_flip(img))
        images_arrays.append(generate.high_brightness(img))
        images_arrays.append(generate.low_brightness(img))
        images_arrays.append( generate.sharpen(img))

        return images_arrays

    
    def rotation(image,angle):
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, scale=1)
        rotated_image = cv2.warpAffine(image,rotation_matrix,(width,height))
        return rotated_image

 
    def vertical_flip(image):
        return cv2.flip(image,0)
    
    def high_brightness(image):
        return cv2.add(image, np.ones(image.shape , dtype=np.uint8) * 40)

    def low_brightness(image):
        return cv2.subtract(image, np.ones(image.shape , dtype=np.uint8) * 40)
        

    def sharpen(image):
        kernel = np.array([ [-1,-1,-1],
                            [-1,10,-1],
                            [-1,-1,-1] ])
        return cv2.filter2D(image,-1,kernel)