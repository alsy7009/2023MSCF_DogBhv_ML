import pandas as pd
import numpy as np
import os



def split_batch_equal_distribution(pn_dict_data, subdir_motion, n_batches, fn_out):
    """
    use (dict_data, dict_split_info) to generate train batch_i(img,aud,motion) val(img,aud,motion) test(img,aud,motion)
    :param pn_dict_data:
    :param n_batches: number of batches
    :param fn_out:
    :return:
    """
    dict_data, dict_split_info = pd.read_pickle(pn_dict_data)

    numsamp_train = dict_data['train']['x_img'].shape[0]
    batchsize = int(numsamp_train/n_batches)
    y_output = dict_data['train']['y_output']
    y_output[y_output==4] = 2
    #############################3
    breakpoint()
    # y_output = np.expand_dims(np.random.randint(0,3, 3500), 1)
    #######################################
    list_label = sorted(list(set(y_output.flatten())))
    dict_batch_idx_label = {label:{} for label in list_label}
    for label in list_label:
        idx_label = [k for k in range(y_output.size) if y_output[k,0]==label]
        numsamp_label = len(idx_label)
        numsamp_per_batch = int(numsamp_label/n_batches+0.5)
        for idx_batch in range(n_batches):
            # (start,end)=(idx_batch*numsamp_per_batch, min(numsamp_label,(idx_batch+1)*numsamp_per_batch))
            # dict_batch_idx_label[label][idx_batch] = idx_label[start:end]
            dict_batch_idx_label[label][idx_batch] = idx_label[idx_batch::n_batches]
    breakpoint()


    dict_batch_idx = {}
    for idx_batch in range(n_batches):
        list_idx =[]
        for label in list_label:
            list_idx = list_idx + dict_batch_idx_label[label][idx_batch]
        dict_batch_idx[idx_batch] = list_idx

    #################################33
    for idx_batch in range(n_batches):
        list_idx = dict_batch_idx[idx_batch]
        for label in list_label:
            print('batch',idx_batch, 'label',label, np.sum(y_output[list_idx]==label))
    breakpoint()
    #################################33

    # list_idx = []
    # for idx_batch in range(n_batches):
    #     if idx_batch == n_batches-1:
    #         list_idx.append((idx_batch*batchsize, numsamp_train))
    #     else:
    #         list_idx.append((idx_batch*batchsize, (idx_batch+1)*batchsize))

    breakpoint()
    for idx_batch in range(n_batches):
    #     (idx_start,idx_end) = list_idx[idx_batch]
        list_idx = dict_batch_idx[idx_batch]
        ##############################
        print('batch',idx_batch) #, idx_start,'--', idx_end)
        ##############################
        #img,aud,y_output
        # dict_trainbatch={'x_img':dict_data['train']['x_img'][idx_start:idx_end,:,:,:],
        #                         'x_aud':dict_data['train']['x_aud'][:,idx_start:idx_end],
        #                         'y_output':dict_data['train']['y_output'][idx_start:idx_end],
        #                         }
        dict_trainbatch = {'x_img': dict_data['train']['x_img'][list_idx, :, :, :],
                           'x_aud': dict_data['train']['x_aud'][:, list_idx],
                           'y_output': dict_data['train']['y_output'][list_idx],
                           }

        #motion
        dict_motion={ }
        list_idx_valid = []
        #for idx in range(idx_start,idx_end):
        for idx in list_idx:
            (pn_img,frame,channel) = dict_split_info['train'][idx]
            #######################################
            print('batch',idx_batch, idx, 'frame',frame)
            ########################################
            if not os.path.exists(f'{pn_img}/{subdir_motion}/{channel}{str(frame).zfill(5)}.pk'):
                list_idx_valid.append(False)
                continue
            else:
                list_idx_valid.append(True)
            pyramid_motion = pd.read_pickle(f'{pn_img}/{subdir_motion}/{channel}{str(frame).zfill(5)}.pk')
            for (level,motion) in pyramid_motion.items():
                (height,width)=motion['mvx'].shape
                if level not in dict_motion:
                    dict_motion[level]={'mvx':np.ndarray((0,height,width)),'mvy':np.ndarray((0,height,width))}
                for mv in ['mvx','mvy']:
                    arr_motion = np.expand_dims(motion[mv].values,axis=0)
                    dict_motion[level][mv] = np.vstack((dict_motion[level][mv],arr_motion))
        breakpoint()
        dict_trainbatch = {'x_img':dict_trainbatch['x_img'][list_idx_valid,:,:,:],
                            'x_aud':dict_trainbatch['x_aud'][:,list_idx_valid],
                            'y_output':dict_trainbatch['y_output'][list_idx_valid],
                            'x_motion':dict_motion
                           }
        pd.to_pickle(dict_trainbatch, fn_out.format(idx_batch))

    breakpoint()
    #val, test
    for (key, name) in [('val','validation'), ('test','test')]:
        numsamp = dict_data[key]['x_img'].shape[0]
        dict_motion = {}
        list_idx_valid = []
        for idx in range(numsamp):
            (pn_img, frame, channel) = dict_split_info[key][idx]
            #######################################
            print(key, idx, '/', numsamp, 'frame', frame)
            ########################################
            # if (idx_batch==4) and (idx==1636):
            #     breakpoint()
            #########################################
            if not os.path.exists(f'{pn_img}/{subdir_motion}/{channel}{str(frame).zfill(5)}.pk'):
                list_idx_valid.append(False)
                continue
            else:
                list_idx_valid.append(True)
            pyramid_motion = pd.read_pickle(f'{pn_img}/{subdir_motion}/{channel}{str(frame).zfill(5)}.pk')
            for (level, motion) in pyramid_motion.items():
                (height, width) = motion['mvx'].shape
                if level not in dict_motion:
                    dict_motion[level] = {'mvx': np.ndarray((0, height, width)), 'mvy': np.ndarray((0, height, width))}
                for mv in ['mvx', 'mvy']:
                    arr_motion = np.expand_dims(motion[mv].values, axis=0)
                    dict_motion[level][mv] = np.vstack((dict_motion[level][mv], arr_motion))
        breakpoint()
        dict_valtest = {'x_img': dict_data[key]['x_img'][list_idx_valid, :, :, :],
                           'x_aud': dict_data[key]['x_aud'][:,list_idx_valid],
                           'y_output': dict_data[key]['y_output'][list_idx_valid],
                           'x_motion': dict_motion
                           }
        breakpoint()
        pd.to_pickle(dict_valtest, fn_out.replace('train_batch{}', f'{name}_all'))





if __name__ == '__main__':
    if 0:
        pn_dict_data='C:/home/Amy/Research/DogBehavior/data/trainValTestData_stft1_dns5.pk'
        fn_out = 'C:/home/Amy/Research/DogBehavior/data/train_batch{}.pk'
        subdir_motion = 'immotion5'
        split_batch(pn_dict_data=pn_dict_data,subdir_motion=subdir_motion, n_batches=10, fn_out=fn_out)

    if 0:
        pn_dict_data = 'C:/home/Amy/Research/DogBehavior/data/trainValTestData_stft1_dns5.pk'
        fn_out = 'C:/home/Amy/Research/DogBehavior/data/4batches_equaldistribution/train_batch{}.pk'
        subdir_motion = 'immotion5'
        split_batch_equal_distribution(pn_dict_data=pn_dict_data,subdir_motion=subdir_motion, n_batches=4, fn_out=fn_out)

    if 1:
        pn_dict_data = 'C:/home/Amy/Research/DogBehavior/data/trainValTestData_stft1_dns5.pk'
        fn_out = 'C:/home/Amy/Research/DogBehavior/data/1batch/train_batch{}.pk'
        subdir_motion = 'immotion5'
        split_batch_equal_distribution(pn_dict_data=pn_dict_data,subdir_motion=subdir_motion, n_batches=1, fn_out=fn_out)
