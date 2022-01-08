from Maize_Dataset.generator import generate
import cv2
import random
import os
import numpy as np


class compile:
    path = r'Maize_Dataset/src/'
    def __init__(self, classes='binary', resize_shape=(256, 256), normalise=True, load_equal_ratio=True, augmentation_stage=1):
        self.fetch_imgs_paths()
        if classes == 'binary':
            self.data_x, self.data_y = self.load_binary_classes(resize_shape=resize_shape, normalise=normalise)
        elif classes == 'all':
            self.data_x, self.data_y = self.load_by_classes(resize_shape=resize_shape)
        else :
            return ValueError('Invalid class specification')
        
        if augmentation_stage > 0:
            self.data_x, self.data_y = self.generator(self.data_x, self.data_y)
            
       
    @classmethod
    def fetch_imgs_paths(self):
        self.imgs_paths = {}
        for class_folder in os.listdir(f'{self.path}'):
            self.imgs_paths[f'{class_folder}'] = os.listdir(f'{self.path}{class_folder}/')
    
    @classmethod
    def load_binary_classes(self, resize_shape, normalise, fetch_max_equal_ratio=True):
        x_dataset, y_dataset = [], []
        healthy_class_len = len(self.imgs_paths['healthy'])
        class_sample_size = int(healthy_class_len / 3)
        if fetch_max_equal_ratio:
            for class_folder in self.imgs_paths:
                if class_folder == 'healthy':
                    for img_path in self.imgs_paths[class_folder]:
                        position = random.randint(0, len(x_dataset))
                        img_array = cv2.resize(cv2.imread(self.path + class_folder + '/' + img_path), resize_shape, interpolation = cv2.INTER_AREA)
                        img_array = img_array / 256 if normalise == True else img_array
                        x_dataset.insert(position, img_array)
                        y_dataset.insert(position, 0)
                else:
                    sample_pool = random.sample(self.imgs_paths[class_folder], class_sample_size)
                    for img in sample_pool:
                        position = random.randint(0, len(x_dataset))
                        img_array = cv2.resize(cv2.imread(self.path + class_folder + '/' + img), resize_shape, interpolation=cv2.INTER_AREA)
                        img_array = img_array / 256 if normalise == True else img_array
                        x_dataset.insert(position, img_array)
                        y_dataset.insert(position, 1)
                        
                        
        elif fetch_max_equal_ratio == False:
            for class_folder in self.imgs_paths:
                if class_folder == 'healthy':
                    for img_path in self.imgs_paths[class_folder]:
                        position = random.randint(0, len(x_dataset))
                        img_array = cv2.resize(cv2.imread(self.path + class_folder + '/' + img_path), resize_shape, interpolation = cv2.INTER_AREA)
                        img_array = img_array / 256 if normalise == True else img_array
                        x_dataset.insert(position, img_array)
                        y_dataset.insert(position, 0)
                else:
                    for img_path in self.imgs_paths[class_folder]:
                        position = random.randint(0, len(x_dataset))
                        img_array = cv2.resize(cv2.imread(self.path + class_folder + '/' + img_path), resize_shape, interpolation = cv2.INTER_AREA)
                        img_array = img_array / 256 if normalise == True else img_array
                        x_dataset.insert(position, img_array)
                        y_dataset.insert(position, 1)
                                                  
        return x_dataset, y_dataset
    
    @classmethod
    def load_all_classes(self, resize_shape, fetch_max_equal_ratio=True, normalise=256):
        pass
    @classmethod
    def generator(self, data_x, data_y):
        for index, img in enumerate(data_x):
            for generic_img in generate.batch_1(img):
                position = random.randint(0, len(data_x)) 
                data_x.insert(position,generic_img)
                data_y.insert(position, data_y[index]) 
        return data_x, data_y
    @property
    def load(self, train_size=.7, test_size=.3):
        if (train_size + test_size) == 1:
            train_slice = int(len(self.data_x)/.7)
            return np.asarray(self.data_x[:train_slice]), np.asarray(self.data_y[:train_slice]), np.asarray(self.data_x[train_slice:]), np.asarray(self.data_x[train_slice:])
        elif (train_size + test_size) != 1:
            return ValueError('Invalid Split Value') 
        
    
    