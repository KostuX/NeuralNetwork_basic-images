
fit = True  # make fit false if you do not want to train the network again
train_dir = 'src/data/train'
test_dir = 'src/data/test'

batch_size = 12
num_classes = 3
img_width = 128
img_height = 128
img_channels = 3

epochs = 5


# augmentation
random_Rotation = 0.06
translation_factors = 0.1
randomZoom = 0.07
