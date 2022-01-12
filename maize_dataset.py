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
        
            
       
    @classmethod
    def fetch_imgs_path(path, class_encoding, equal_ratio_to_healthy, shuffle):
    class_folders = os.listdir(path)
    imgs_path_dict = {}
    for class_folder in class_folders:
        files = [path + class_folder + '/' + file for file in os.listdir(path + class_folder + '/')]
        if shuffle == True:
            random.shuffle(files)
        imgs_path_dict[str(class_folder)] = files
    if class_encoding == 'binary':
        temp_dict = {'unhealthy':[], 'healthy':[]}
        if equal_ratio_to_healthy == True:
            ratio = int(len(imgs_path_dict['healthy']) / (len(class_folders) - 1))
            for folder, img_files  in imgs_path_dict.items():
                if folder == 'healthy':
                    temp_dict['healthy'] = random.sample(imgs_path_dict[folder], len(imgs_path_dict['healthy']))
                else:
                    temp_dict['unhealthy'] = temp_dict['unhealthy'] + random.sample(imgs_path_dict[folder], ratio)
        
            imgs_path_dict = temp_dict   
            
        else:
            for folder, img_files  in imgs_path_dict.items():
                if folder == 'healthy':
                    temp_dict['healthy'] = imgs_path_dict[folder]
                else:
                    temp_dict['unhealthy'] = temp_dict['unhealthy'] + imgs_path_dict[folder]
            
            imgs_path_dict = temp_dict
            
        
    return imgs_path_dict
    
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
                        x_dataset.insert(position, img_array / 256.0 if normalise == True else img_array)
                        y_dataset.insert(position, 0)
                        for img in generate.batch_1(img_array):
                            new_pos = random.randint(0, len(x_dataset))
                            x_dataset.insert(new_pos, img / 256.0 if normalise == True else img_array)
                            y_dataset.insert(new_pos, 0)
                            
                        
                else:
                    sample_pool = random.sample(self.imgs_paths[class_folder], class_sample_size)
                    for img in sample_pool:
                        position = random.randint(0, len(x_dataset))
                        img_array = cv2.resize(cv2.imread(self.path + class_folder + '/' + img), resize_shape, interpolation=cv2.INTER_AREA)
                        x_dataset.insert(position, img_array / 256.0 if normalise == True else img_array)
                        y_dataset.insert(position, 1)
                        for img in generate.batch_1(img_array):
                            new_pos = random.randint(0, len(x_dataset))
                            x_dataset.insert(new_pos, img / 256.0 if normalise == True else img_array)
                            y_dataset.insert(new_pos, 1)
                        
                        
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
        for index, img in enumerate(self.data_x):
            for generic_img in generate.batch_1(img):
                position = random.randint(0, len(self.data_x)) 
                data_x.insert(position,generic_img)
                data_y.insert(position, self.data_y[index]) 
        return data_x, data_y
    @property
    def load(self, train_size=.7, test_size=.3):
        if (train_size + test_size) == 1:
            train_slice = int(len(self.x_dataset)/.7)
            return self.x_dataset[:train_slice], self.y_dataset[:train_slice] , self.x_dataset[train_slice:], self.y_dataset[train_slice:]
        elif (train_size + test_size) != 1:
            return ValueError('Invalid Split Value') 
        
  