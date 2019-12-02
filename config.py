import numpy as np

model_name = "trained"
batch_size = 35
opt = "Adam"  # 'SGD' 'AdaGrad'
validation_len = 500  # Number of samples to take from each subset for validation
datafiles_to_load = 3  # Number of files to load and train on at one time
datafile_len = 1500  # Number of samples in the datefile
subset_len = 3 * datafile_len  # Number of samples to load and train on at one time
available_count = 3  # Number of data files available
image_dimensions = np.array([240, 240])
fov = 85

