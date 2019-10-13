import re
import pathlib
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

# Data : Face Detection Data Set and Benchmark (FDDB)

# in
fddb_paths = 'data/paths.txt'
fddb_annotations = 'data/annotations.txt'

name_num = 1
with open(fddb_annotations, 'r') as annotations:
    while True:
        line = annotations.readline()
        if not line:
            break
        # if line is path
        elif re.compile("[0-9]*/[0-9]*/[0-9]*/").search(line):
            path = line.strip() + '.jpg'
            save_path = 'data/annotatedPics/' + path
            # if save directory is not exist, make it
            directory = '/'.join(save_path.split('/')[:-1])
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            # image read
            image = cv2.imread('data/originalPics/' + path)
            mask = np.zeros(image.shape[:2], dtype='uint8')
        # else if line is information of image
        elif re.compile("[0-9]").search(line):
            face_num = int(line)
            for _ in range(face_num):
                line = annotations.readline()
                major = int(float(line.split()[0]))
                minor = int(float(line.split()[1]))
                angle = int(float(line.split()[2]))
                cx = int(float(line.split()[3]))
                cy = int(float(line.split()[4]))
                #insert ellipse mask
                mask = cv2.ellipse(mask, (cx, cy), (minor, major), angle, 0, 360, 1, -1)
                
            #save masked image    
            image_masked = cv2.bitwise_and(image, image, mask=mask)
            cv2.imwrite(save_path, image_masked)
            #save N * 3channels with labels numpy array
            # 1 is skin, 0 is nonskin
            result = np.dstack((image, mask))
            result = result.reshape(-1, 4)
            np.save(save_path.split('.')[0] + '.npy', result)
                
