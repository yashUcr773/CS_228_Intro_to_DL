import os
from os import walk
from PIL import Image
import numpy as np

path_to_read_dataset = './unsplash_images'
path_to_store_labels = './dataset/bw_images/'
path_to_store_original = './dataset/true_images/'
index = 0
resize_shape = (512,512)


if not os.path.exists(path_to_read_dataset):
    raise Exception('path to dataset does not exist')

if not os.path.exists(path_to_store_labels):
    os.makedirs(path_to_store_labels, exist_ok=True)

if not os.path.exists(path_to_store_original):
    os.makedirs(path_to_store_original, exist_ok=True)



for (dirpath, dirnames, filenames) in walk(path_to_read_dataset):

    for filename in filenames:
        img_path = dirpath +'/'+ filename
        
        img = Image.open(img_path)
        img = img.resize(resize_shape)

        aaa = np.array(img)
        if len(aaa.shape) > 3:
            print (aaa.shape)
            print (index)
        
        img.save(f'{path_to_store_original}{index}.jpg')

        img = img.convert("L")
        img.save(f'{path_to_store_labels}{index}.jpg')
        index += 1