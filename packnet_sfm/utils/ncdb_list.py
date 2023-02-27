import os
import numpy as np
import natsort


def read_folder_list(path):
    folder_list = os.listdir(path)
    folder_list = natsort.natsorted(folder_list)
    return folder_list


TP  = ['0001', '0002', '0003']
for i in range(len(TP)):
    db_path = '/home/seok436/data/nextchipDB_SSL/crop/TP_{}_v3'.format(TP[i])
    db_file_list = read_folder_list(db_path)
    for j in range(len(db_file_list)):
        infolder_path = os.path.join(db_path, db_file_list[j])
        infolder_list = read_folder_list(infolder_path)
        for k in infolder_list:
            file_path = os.path.join(infolder_path, k)
            print(file_path)
