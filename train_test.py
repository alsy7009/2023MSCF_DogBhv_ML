import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras import layers, models
import sys
import fnmatch,os
sys.path.append('C:/home/Amy/Research/DogBehavior/src/py')
from dataIO import load_data_info, load_starttime, load_action, load_aud_stft
from prepdata import prep_data_trainAndtest
from processingdata import downsampling_img, downsampling_aud
import seaborn as sns
import pandas as pd
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import cv2 as cv
import moviepy.editor as mp
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, concatenate, BatchNormalization, Dropout
from keras.utils import np_utils
# TARGET_IMSIZE = (455,256)

# im_resize = 0.8
# TARGET_IMSIZE = (int(im_resize*TARGET_IMSIZE[0]), int(im_resize*TARGET_IMSIZE[1]))
# width_crop, height_crop =

#img
downsample_img_factor = 3
filter_num = 32
kernel_size1 = (7, 7)
kernel_size2 = (3, 3)  # (7,7)
# kernel_size3 = (3,3)
strides1 = (1, 1)
strides2 = (1, 1)
activation_conv2d = 'linear'
bnorm_axis = 3
#aud
downsample_aud_factor = 16
pool_size1 = (3, 3)  # (7,7) #
pool_size2 = (3, 3)  # (7,7) #
# pool_size3 = (2,2)
#motion
level_motion = 0
rows_mot = 6
pool_size_mot = (3, 9)

if 0:
    #test1
    dict_sample_wt={0:878/339, 1:878/522, 2:878/1575, 3:878/1076 }
    epochs = 20
    batch_size = 100
    #dense layer
    n_dense1 = 30
    n_dense2 = 10
    n_dense3 = 4
    n_dense4 = 4
    activation_dense = 'sigmoid'
    output_activation_dense = 'softmax' #'relu'

if 0:
    #test2
    dict_sample_wt={0:878/339, 1:878/522, 2:878/1575, 3:878/1076 }
    epochs = 10
    batch_size = 50
    #dense layer
    n_dense1 = 30
    n_dense2 = 10
    n_dense3 = 4
    n_dense4 = 4
    activation_dense = 'sigmoid'
    output_activation_dense = 'softmax' #'relu'

if 0:
    #test3: 4batches, for each class, its samples equally distributed among batches
    dict_sample_wt={0:878/339, 1:878/522, 2:878/1575, 3:878/1076 }
    epochs = 10
    batch_size = 50
    #dense layer
    n_dense1 = 30
    n_dense2 = 10
    n_dense3 = 4
    n_dense4 = 4
    percentage_dropout = 0.2
    activation_dense = 'sigmoid'
    output_activation_dense = 'softmax' #'relu'

if 0:
    #test4: 1batch
    dict_sample_wt={0:878/339, 1:878/522, 2:878/1575, 3:878/1076 }
    epochs = 1
    batch_size = 50
    #dense layer
    n_dense1 = 30
    n_dense2 = 10
    n_dense3 = 4
    n_dense4 = 4
    percentage_dropout = 0.2
    activation_dense = 'sigmoid'
    output_activation_dense = 'softmax' #'relu'

if 1:
    #test5: same as test3, add a dropout layer before dense layer
    dict_sample_wt={0:878/339, 1:878/522, 2:878/1575, 3:878/1076 }
    epochs = 10
    batch_size = 50
    #dense layer
    n_dense1 = 30
    n_dense2 = 10
    n_dense3 = 4
    n_dense4 = 4
    percentage_dropout_pre = 0.2
    percentage_dropout = 0.2
    activation_dense = 'sigmoid'
    output_activation_dense = 'softmax' #'relu'


optimizer = 'adam'
EPOCHS = 10
checkpoint_filepath = 'C:/home/Amy/Research/DogBehavior/src/py/checkpoint'
checkpoint_filepath_val = 'C:/home/Amy/Research/DogBehavior/src/py/checkpoint_val'


def downsampling_img(X_train, dn_factor=5):
    X_smooth = None
    for k in range(X_train.shape[0]):
        y = cv.blur(X_train[k, :, :, :, ], (dn_factor, dn_factor))[int((dn_factor + 1) / 2)::dn_factor,
            int((dn_factor + 1) / 2)::dn_factor, :]
        if X_smooth is None:
            X_smooth = np.ndarray((1, y.shape[0], y.shape[1], y.shape[2]))
            X_smooth[0, :, :, :] = y
        else:
            X_smooth = np.vstack((X_smooth, np.expand_dims(y, 0)))
        # print(X_smooth.shape)
    return X_smooth



def downsampling_aud(X_train_aud, dn_factor=16):
    fil = np.ones(dn_factor)
    X_smooth = None
    for k in range(X_train_aud.shape[0]):
        y = np.convolve(X_train_aud[k, :], fil, mode='same')[int((dn_factor + 1) / 2)::dn_factor]
        if X_smooth is None:
            X_smooth = np.expand_dims(y, 0)
        else:
            X_smooth = np.vstack((X_smooth, np.expand_dims(y, 0)))
        # print(X_smooth.shape)
    return X_smooth

def load_data_traintest(fn):
    data = pd.read_pickle(fn)

    X_train_img = data['x_img']
    X_train_aud = data['x_aud'].T
    train_motion = data['x_motion']
    Y_train = data['y_output']
    numsamp = X_train_img.shape[0]
    print('numsamp', numsamp)
    #################################
    # one-hot encoding
    Y_train[Y_train == 4] = 2
    Y_train_one_hot_encoding = np_utils.to_categorical(Y_train)
    for label in [0, 1, 2, 3]:
        print(f'label={label}', np.sum(Y_train == label))
    # breakpoint()
    #################################

    # downsample img
    X_train_img = downsampling_img(X_train_img, downsample_img_factor)
    (imh, imw, imc) = (X_train_img.shape[1], X_train_img.shape[2], X_train_img.shape[3])

    # downsmaple aud
    X_train_aud = downsampling_aud(X_train_aud, downsample_aud_factor)
    #     X_test_aud = downsampling_aud(X_test_aud, downsample_aud_factor)
    audsize = X_train_aud.shape[1]

    # motion field: take upper half of motion field
    mx = train_motion[level_motion]['mvx'][:, 0:rows_mot, :]
    my = train_motion[level_motion]['mvy'][:, 0:rows_mot, :]

    # motion field: remove global mean
    for idx_samp in range(train_motion[level_motion]['mvx'].shape[0]):
        mx[idx_samp, :, :] -= np.mean(mx[idx_samp, :, :])
        my[idx_samp, :, :] -= np.mean(my[idx_samp, :, :])

    mv = (mx ** 2 + my ** 2) ** 0.5
    X_train_mot = np.expand_dims(mv, 3).astype(np.float64)
    (mh, mw) = (X_train_mot.shape[1], X_train_mot.shape[2])

    # global max motion: scalar
    X_train_maxmot = np.ndarray((mv.shape[0], 1))
    for idx_samp in range(mv.shape[0]):
        X_train_maxmot[idx_samp, 0] = np.max(mv[idx_samp, :, :])
        # block max motion

    # color
    print('img', X_train_img.shape, imw, imh)
    print('aud', X_train_aud.shape, audsize)
    print('mot', X_train_mot.shape, mw, mh)
    print('maxmot', X_train_maxmot.shape)
    print('y_one_hot', Y_train_one_hot_encoding.shape)
    # breakpoint()

    return (X_train_img, X_train_aud, X_train_mot, X_train_maxmot, Y_train, Y_train_one_hot_encoding)



if __name__ == "__main__":

    if 0:
        fn_train = 'C:/home/Amy/Research/DogBehavior/data/train_batch{batch}.pk'
        fn_val = 'C:/home/Amy/Research/DogBehavior/data/validation_all.pk'
        fn_test = 'C:/home/Amy/Research/DogBehavior/data/test_all.pk'
        TEST = 'test2'
        hepochs = 10
        Nbatch = 10

    if 1:
        fn_train = 'C:/home/Amy/Research/DogBehavior/data/4batches_equaldistribution/train_batch{batch}.pk'
        fn_val = 'C:/home/Amy/Research/DogBehavior/data/4batches_equaldistribution/validation_all.pk'
        fn_test = 'C:/home/Amy/Research/DogBehavior/data/4batches_equaldistribution/test_all.pk'

        # TEST = 'test3'
        # hepochs = 10
        TEST = 'test9'
        hepochs = 40
        Nbatch = 4

    if 0:
        fn_train = 'C:/home/Amy/Research/DogBehavior/data/1batch/train_batch{batch}.pk'
        fn_val = 'C:/home/Amy/Research/DogBehavior/data/1batch/validation_all.pk'
        fn_test = 'C:/home/Amy/Research/DogBehavior/data/1batch/test_all.pk'

        TEST = 'test4'
        hepochs = 40
        Nbatch = 1

    if 0:
        fn_train = 'C:/home/Amy/Research/DogBehavior/data/4batches_equaldistribution/train_batch{batch}.pk'
        fn_val = 'C:/home/Amy/Research/DogBehavior/data/4batches_equaldistribution/validation_all.pk'
        fn_test = 'C:/home/Amy/Research/DogBehavior/data/4batches_equaldistribution/test_all.pk'
        TEST = 'test5'
        hepochs = 10
        Nbatch = 4

    if 0:
        fn_train = 'C:/home/Amy/Research/DogBehavior/data/4batches_equaldistribution/train_batch{batch}.pk'
        fn_val = 'C:/home/Amy/Research/DogBehavior/data/4batches_equaldistribution/validation_all.pk'
        fn_test = 'C:/home/Amy/Research/DogBehavior/data/4batches_equaldistribution/test_all.pk'
        #TEST = 'test6img'
        #TEST = 'test7aud'
        TEST = 'test8mot'
        hepochs = 10
        Nbatch = 4


    if not os.path.exists(f'C:/home/Amy/Research/DogBehavior/src/py/{TEST}'):
        os.mkdir(f'C:/home/Amy/Research/DogBehavior/src/py/{TEST}')
    if not os.path.exists(f'C:/home/Amy/Research/DogBehavior/src/py/{TEST}/models'):
        os.mkdir(f'C:/home/Amy/Research/DogBehavior/src/py/{TEST}/models')
    if not os.path.exists(f'C:/home/Amy/Research/DogBehavior/src/py/{TEST}/result'):
        os.mkdir(f'C:/home/Amy/Research/DogBehavior/src/py/{TEST}/result')

    #load val, test data
    (X_val_img, X_val_aud, X_val_mot, X_val_maxmot, Y_val, Y_val_one_hot_encoding) = load_data_traintest(fn_val)
    (X_test_img, X_test_aud, X_test_mot, X_test_maxmot, Y_test, Y_test_one_hot_encoding) = load_data_traintest(fn_test)

    for hepoch in range(hepochs):

        for batch in range(0,Nbatch):
            print('train, hepoch:', hepoch, 'batch:', batch)
            (X_train_img, X_train_aud, X_train_mot, X_train_maxmot, Y_train, Y_train_one_hot_encoding) = load_data_traintest(fn_train.format(batch=batch))
            sample_wt = np.asarray([dict_sample_wt[y] for y in Y_train.flatten()])

            (imh, imw, imc) = (X_train_img.shape[1], X_train_img.shape[2], X_train_img.shape[3])
            audsize = X_train_aud.shape[1]
            (mh, mw) = (X_train_mot.shape[1], X_train_mot.shape[2])

            #input layer
            input_image = Input(shape=(imh, imw, 3))
            input_aud = Input(shape=(audsize,))
            input_motion = Input(shape=(mh, mw, 1))
            # input_maxmotion=Input(shape=(1,))


            #img: convolution, maxpool
            ##########################
            ###########################
            x_image=Conv2D(filters=filter_num, kernel_size=kernel_size1, strides=strides1)(input_image)
            print('img conv2d', x_image.shape)


            x_image=BatchNormalization(3)(x_image)
            print('img batchnorm', x_image.shape)

            x_image=MaxPool2D(pool_size=pool_size1)(x_image)
            print('img maxpool', x_image.shape, x_image)

            x_image = Flatten()(x_image)
            print(x_image.shape)

            ##########################
            # breakpoint()
            ###########################
            #aud:
            print('aud', input_aud.shape)
            x_aud = BatchNormalization(1)(input_aud)
            print('aud batchnorm', input_aud.shape)

            #motion: maxpool
            x_motion = MaxPool2D(pool_size=pool_size_mot)(input_motion)
            print('motin maxpool', x_motion.shape, x_motion)
            x_motion = Flatten()(x_motion)

            # concatenate flattened image + aud + motion
            # concat = concatenate([x_image, input_aud, x_motion])
            concat = concatenate([x_image, x_aud, x_motion])
            #concat = x_image
            #concat = x_aud

            # dense layer
            if TEST in ['test1','test2','test3','test4']:
                dense1 = Dense(n_dense1, activation_dense)(concat)
                #dense2 = Dense(n_dense2, activation_dense)(dense1)
                # dense3 = Dense(n_dense3, activation_dense)(dense2)
                dropout = Dropout(percentage_dropout)(dense1)
                output = Dense(n_dense4, 'softmax')(dropout) #(dense1)
            if TEST in ['test5','test9']:
                dropout_pre = Dropout(percentage_dropout_pre)(concat)
                dense1 = Dense(n_dense1, activation_dense)(dropout_pre)
                #dense2 = Dense(n_dense2, activation_dense)(dense1)
                # dense3 = Dense(n_dense3, activation_dense)(dense2)
                dropout = Dropout(percentage_dropout)(dense1)
                output = Dense(n_dense4, 'softmax')(dropout) #(dense1)
            if TEST in ['test6img']:
                concat = x_image
                dropout_pre = Dropout(percentage_dropout_pre)(concat)
                dense1 = Dense(n_dense1, activation_dense)(dropout_pre)
                #dense2 = Dense(n_dense2, activation_dense)(dense1)
                # dense3 = Dense(n_dense3, activation_dense)(dense2)
                dropout = Dropout(percentage_dropout)(dense1)
                output = Dense(n_dense4, 'softmax')(dropout) #(dense1)
            if TEST in ['test7aud']:
                concat = x_aud
                dropout_pre = Dropout(percentage_dropout_pre)(concat)
                dense1 = Dense(n_dense1, activation_dense)(dropout_pre)
                #dense2 = Dense(n_dense2, activation_dense)(dense1)
                # dense3 = Dense(n_dense3, activation_dense)(dense2)
                dropout = Dropout(percentage_dropout)(dense1)
                output = Dense(n_dense4, 'softmax')(dropout) #(dense1)
            if TEST in ['test8mot']:
                concat = x_motion
                dropout_pre = Dropout(percentage_dropout_pre)(concat)
                dense1 = Dense(n_dense1, activation_dense)(dropout_pre)
                #dense2 = Dense(n_dense2, activation_dense)(dense1)
                # dense3 = Dense(n_dense3, activation_dense)(dense2)
                dropout = Dropout(percentage_dropout)(dense1)
                output = Dense(n_dense4, 'softmax')(dropout) #(dense1)

            model = Model(inputs=[input_image,input_aud,input_motion], outputs=output)
            #model = Model(inputs=[input_image, input_aud], outputs=output)        # model = Model(inputs=[x_image.input, x_aud.input], outputs=output)
            #model = Model(inputs=input_image, outputs=output)
            #model = Model(inputs=input_aud, outputs=output)
            # keras.utils.plot_model(model, show_shapes=True)

            optimizer = 'adam'
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  ##,
            model.summary()

            #model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_weights_only=True,monitor='val_accuracy',mode='max',save_best_only=True)

            # train_dataset = tf.data.Dataset.from_tensor_slices(([X_train,X_train_aud], Y_train)).batch(476)
            # test_dataset = tf.data.Dataset.from_tensor_slices(([X_test,X_test_aud], Y_test)).batch(476)
            # history = model.fit(train_dataset, epochs=20, validation_data=test_dataset)

            #breakpoint()
            if batch==0:
                if hepoch>0:
                    model.load_weights(f'C:/home/Amy/Research/DogBehavior/src/py/{TEST}/models/model_hepoch{hepoch-1}batch{Nbatch-1}.h5')
            else:
                model.load_weights(f'C:/home/Amy/Research/DogBehavior/src/py/{TEST}/models/model_hepoch{hepoch}batch{batch-1}.h5')
            model.fit([X_train_img,X_train_aud, X_train_mot], Y_train_one_hot_encoding, sample_weight=sample_wt, epochs=epochs, batch_size=batch_size) #, callbacks = [model_checkpoint_callback])
            #model.fit([X_train_img,X_train_aud], Y_train_one_hot_encoding, epochs=20, batch_size=40, callbacks = [model_checkpoint_callback])
            #model.fit(X_train_img, Y_train_one_hot_encoding, epochs=10, batch_size=40, callbacks = [model_checkpoint_callback])
            #model.fit(X_train_aud, Y_train_one_hot_encoding, epochs=10, batch_size=50)

            # breakpoint()

            #save model
            model.save_weights(f'C:/home/Amy/Research/DogBehavior/src/py/{TEST}/models/model_hepoch{hepoch}batch{batch}.h5')
            # model.save(f'C:/home/Amy/Research/DogBehavior/src/py/models/model_hepoch{hepoch}batch{batch}')

            #evaluate
            #model.load_weights(checkpoint_filepath)
            wt=model.get_weights()
            # print(wt)

            #breakpoint()
            dict_result = {'wt':wt}
            for name in ['train', 'val', 'test']:
                if name in ['train']:
                    (X_input_img, X_input_aud, X_input_mot, Y_out) = (X_train_img, X_train_aud, X_train_mot, Y_train)
                elif name in ['val']:
                    (X_input_img, X_input_aud, X_input_mot, Y_out)  = (X_val_img, X_val_aud, X_val_mot, Y_val)
                elif name in ['test']:
                    (X_input_img, X_input_aud, X_input_mot, Y_out)  = (X_test_img, X_test_aud, X_test_mot, Y_test)

                y_pred_one_hot = model.predict([X_input_img, X_input_aud, X_input_mot])
                y_pred = np.argmax(y_pred_one_hot, axis=1)

                for label in [0,1,2,3]:
                    print(name, 'label=',label, np.sum((Y_out.flatten()==label)&(y_pred==label))/np.sum(Y_out.flatten()==label))

                df_cm = pd.DataFrame(index=[0,1,2,3], columns=[0,1,2,3])
                for label_gt in df_cm.index:
                    for label_pred in df_cm.columns:
                        df_cm.loc[label_gt, label_pred] = np.sum( (Y_out.flatten()==label_gt)&(y_pred==label_pred) )

                dict_result[name] = df_cm

                print('hepoch:',hepoch, 'batch:',batch, name, 'accuracy:', np.sum(y_pred==Y_out.flatten())/Y_out.size)
                print(df_cm)

            #save result(hepoch, confusin matrix)
            pd.to_pickle(dict_result, f'C:/home/Amy/Research/DogBehavior/src/py/{TEST}/result/epoch{hepoch}batch{batch}.pk')


