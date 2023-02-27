import cv2
import os
from collections import defaultdict
import errno
from tqdm import tqdm

def read_folder_list(path):
    folder_list = os.listdir(path)
    return folder_list

def read_files(directory, ext=('.png', '.jpg', '.jpeg'), skip_empty=True):
    files = defaultdict(list)
    for entry in os.scandir(directory):
        relpath = os.path.relpath(entry.path, directory)
        if entry.is_dir():
            d_files = read_files(entry.path, ext=ext, skip_empty=skip_empty)
            if skip_empty and not len(d_files):
                continue
            files[relpath] = d_files[entry.path]
        elif entry.is_file():
            if ext is None or entry.path.lower().endswith(tuple(ext)):
                files[directory].append(relpath)
    return files

dataset_path = '/home/seok436/data/nextchipDB_SSL/T0001_static_temp_for_zip/'
crop_path = '/home/seok436/data/nextchipDB_SSL/crop/all/'

# dataset_path = 'Y:/SSL_NextchipDB_front/NewNextchipDB_1920x1080/T0001_static/train/'
# crop_path = 'Y:/SSL_NextchipDB_front/NewNextchipDB_1920x1080/T0001_static_640x192/'

img_dict = read_files(dataset_path)

for folder in tqdm(img_dict):
    try:
        if not (os.path.isdir(crop_path + '/' + folder)):
            os.makedirs(os.path.join(crop_path + '/' + folder))
        # if os.path.exists(crop_path + '/' + folder):
        #     continue
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory.")
            raise
    for img_file in img_dict[folder]:
        full_path = dataset_path + folder + '/' + img_file
        image = cv2.imread(full_path)
        ### frontview ###
        dst = image.copy()
        roi = image[0:800, 0:1920]
        image_height = 800
        image_width = 1920
        # resize
        # image_height = 192
        # image_width = 640
        # crop
        image = cv2.resize(roi, (image_width, image_height))
        # resize
        # image = cv2.resize(dst, (image_width, image_height))
        # file_name, ext = os.path.splitext(img_file)
        cv2.imwrite(crop_path + folder + '/' + img_file, image)
print('Work is done!')