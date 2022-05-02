from .generator import generate
import cv2
import random
import os
import numpy as np
import shutil


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

def save_generation(img_files, split, folder_path, class_folder, names, augment=1):
    count = 0
    gen_count = 0
    for img_file in img_files:
        img = cv2.imread(img_file)
        cv2.imwrite(folder_path + split + class_folder + '/' + str(names.pop()) + '.jpg', img)
        count += 1
        if augment > 0:
            if augment == 1:
                img_array = generate.batch_1(img)
                for generic_img in img_array:
                    cv2.imwrite(folder_path + split + class_folder + '/' + str(names.pop()) + '.jpg', generic_img)
                    gen_count += 1
        del img_file

    return {'Original':count, 'Generic':gen_count}

def to_split_folder(folder_path, img_path_dict, ratio, augment_train=1, augment_val=0, augment_test=0):
    tree = {'train':{}, 'val':{}, 'test':{}}
    folder_splits = ('train/', 'val/', 'test/')
    try:
        shutil.rmtree(folder_path)
    except FileNotFoundError:
        pass
    os.mkdir(folder_path)
        
    for index, split in enumerate(ratio):
        if sum(ratio) > 1:
            print('Split Ratio is Invalid!')
            break
        os.mkdir(folder_path + folder_splits[index])
        for folder in img_path_dict.keys():
            os.mkdir(folder_path + folder_splits[ratio.index(split)] + folder)
        if ratio.index(split) == 0:
            for class_folder, img_files in img_path_dict.items():
                img_files = img_files[:int(len(img_files)*split)]  
                names = [id for id in range((len(img_files)) * 7)]
                random.shuffle(names)
                tree['train'][class_folder] = save_generation(img_files=img_files, split='train/', folder_path=folder_path, class_folder=class_folder, names=names, augment=augment_train)
        if ratio.index(split) == 1:
            for class_folder, img_files in img_path_dict.items():
                img_files = img_files[int(len(img_files)*ratio[0]):(int(len(img_files)*ratio[0])+int(len(img_files)*split))]                
                names = [id for id in range((len(img_files)) * 7)]
                random.shuffle(names)
                tree['val'][class_folder] = save_generation(img_files=img_files, split='val/', folder_path=folder_path, class_folder=class_folder, names=names, augment=augment_val)
        if ratio.index(split) == 2:
            for class_folder, img_files in img_path_dict.items():
                img_files = img_files[(int(len(img_files)*ratio[0])+(int(len(img_files)*ratio[1]))):]
                names = [id for id in range((len(img_files)) * 7)]
                random.shuffle(names)
                tree['test'][class_folder] = save_generation(img_files=img_files, split='test/', folder_path=folder_path, class_folder=class_folder, names=names, augment=augment_test)
    return tree

class compile:
    path = 'Banana_Dataset/src/'
    temp_file = 'Banana_Dataset/temp/'
    def __init__(self, class_encoding, ratio=[0.7,.2,.1], equal_ratio_to_healthy=False, shuffle=True, augment_train=1, augment_val=0, augment_test=0, random_file_name=True, ):
        self.dataset_dir = fetch_imgs_path(self.path, class_encoding, equal_ratio_to_healthy, shuffle)
        #to split_folders
        self.tree = to_split_folder(folder_path=self.temp_file, img_path_dict=self.dataset_dir, ratio=ratio, augment_train=augment_train, augment_val=augment_val, augment_test=augment_test)

    @property
    def visualise_tree(self):
        for split_folder, split_file in self.tree.items():
            print(split_folder, 'folder')
            for class_folder, class_files in split_file.items():
                print(f'    |--class {class_folder}')
                for ratio_id, ratio in  class_files.items():
                    print(f'          |--contains {ratio} {ratio_id} images')