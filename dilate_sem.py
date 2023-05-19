import os
import sys
import cv2
import numpy as np

def main(argv):
    if len(argv) < 2:
        print("Usage: %s <path to dataset>" % argv[0])
        return
    sem_file_path = os.path.join(argv[1], "sem")
    filenames = sorted(os.listdir(sem_file_path))
    
    output_file_path = os.path.join(argv[1], "sem_dilated")
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    
    car_label = (150, 143, 9)
    
    for filename in filenames:
        img = cv2.imread(os.path.join(sem_file_path, filename))
        b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
        mask = (b == car_label[0]) & (g == car_label[1]) & (r == car_label[2])
        
        mask = mask.astype(np.uint8)
        
        erosion_kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, erosion_kernel, iterations = 1)
        
        dilation_kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, dilation_kernel, iterations = 2)
        
        mask = mask.astype(np.bool)
        
        img[:,:,:3][mask] = [car_label[0], car_label[1], car_label[2]]
        
        cv2.imwrite(os.path.join(output_file_path, filename), img)

if __name__ == '__main__':
    main(sys.argv)