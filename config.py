#[input image]
#frames per sec
FPS_D = 30.05
FPS_H = 30.00
#input imsize
TARGET_IMSIZE = (455,256)
# width_crop, height_crop =


#[input prev action]


#[output label]
dict_act_map={'sit':0, 'stand':1, 'walk':2, 'smell':3, 'run':4}

#[model]
filter_num = 32
kernel_size = (7,7)
strides = (1,1)
activation_conv2d = 'linear'
bnorm_axis = 3
activation_dense = 'sigmoid'
n_dense = 2
optimizer = 'adam'

#[raw data]
# pn_data_info = "C:/home/Amy/Research/DogBehavior/data/data_info_small.txt"
pn_data_info = "C:/home/Amy/Research/DogBehavior/data/data_info_mid.txt"
#pn_data_info = "C:/home/Amy/Research/DogBehavior/data/data_info_mid_small.txt"
