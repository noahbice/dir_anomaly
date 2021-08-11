import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pydicom


def read(patient_dir):
    sub_dir = os.listdir(patient_dir)
    sub_sub_dir = os.listdir(patient_dir + '/' + sub_dir[0])
    final = patient_dir + '/' + sub_dir[0] + '/' + sub_sub_dir[0] + '/'
    slices = os.listdir(final)
    num_slices = len(slices)
    vol = np.zeros([num_slices, 512, 512])

    for slice_file in slices:
        dcmimage = pydicom.dcmread(final + slice_file)
        n = dcmimage.InstanceNumber
        n = num_slices - n
        img = dcmimage.pixel_array
        vol[n, ::-1, :] = img
    vol += 1000.
    vol /= 2000.
    # plt.imshow(vol[50], cmap='gray')
    # plt.show()
    return vol


if __name__ == '__main__':
    master = 'T:/TCIA_pancreas/manifest-1599750808610/Pancreas-x/'
    patient_list = os.listdir(master)
    patient_counter = 0
    
    print('Converting patient x...')
    for patient in tqdm(patient_list):
        patient_dir = master + patient
        volume = read(patient_dir)
        np.save('./npydata/x/' + str(patient_counter) + '.npy', volume)
        patient_counter += 1
