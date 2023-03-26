import pandas as pd
import numpy as np
from numpy.random import default_rng
import os
import time
from matplotlib import pyplot as plt
import IPython.display as ipd


import cv2
import moviepy.editor as mp
import librosa
import imageio
import imageio.v3 as iio

from dataIO import load_data_info, load_starttime, load_action, load_aud_stft
from config import *

# C:/home/Amy/Research/DogBehavior/data/img/20221007/3 DH 81 680 H 681 912 DH 913 1130 DH 1230 1536 H 1891 2982 DH 2983 3229 DH 4069 4229 H 4230 4738 DH 4739 4893 H 4894 5023


# import pydub
# import pydub.playback

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io.wavfile import read, write
# from IPython.display import Audio
# from numpy.fft import fft, ifft
#use plt.show()

# def frame_audio_write(fp, store1, store2):
#     """
#
#     :param fp:
#     :param store1:
#     :param store2:
#     :return:
#     """
#     #cv2
#     cap = cv2.VideoCapture(fp)
#     frameNr = 0
#
#     while (True):
#         success, frame = cap.read()
#         if success == True:
#             cv2.imwrite(store1 + frameNr + '_' + cap.get(cv2.CAP_PROP_POS_MSEC)+ store2, frame)
#         else:
#             break
#         frameNr += 1
#     cap.release()
#
#     #moviepy
#     my_clip = mp.VideoFileClip(fp)
#     my_clip.audio.write_audiofile(r"C:/home/Amy/Research/dog_behavior_research/data/video_output/audio1.mp3")

def read_frames_from_video_cv(pn_video, pn_frame_tpl, pn_timestamps, scale_percent, quality):
    """
    convert video to frames, resize
    :param pn_video:
    :return:

    """
    cap = cv2.VideoCapture(pn_video)
    list_timestamps_ms = []
    frame_no = 1

    success = True
    while success:
        success, frame = cap.read()
        if frame is None:
            continue
        print(pn_frame_tpl.format(frame_no), 'start')
        # if frame_no == 182:
        #     breakpoint()
        frame = resize_frame(frame, scale_percent)
        if success:
            p = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            #imwrite("compressed.jpg", img, p);
            cv2.imwrite(pn_frame_tpl.format(frame_no), frame, p)
            #####################
            print(pn_frame_tpl.format(frame_no), 'done')
            ######################
            list_timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            frame_no += 1
    cap.release()
    s_timestamps_ms = pd.Series(data=list_timestamps_ms, index=range(1, len(list_timestamps_ms) + 1))
    pd.to_pickle(s_timestamps_ms, pn_timestamps)

def resize_frame(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def read_frames_from_video_imageio(pn_video):
    # bulk read all frames
    # Warning: large videos will consume a lot of memory (RAM)
    # breakpoint()
    frames = iio.imread(pn_video, plugin="pyav")
    return frames, None



def write_frames_to_files(pn_frame_tpl, list_frames, fn_audio, list_timestamps_ms=None, pn_timestamps=None):
    """
    write img frames to files, write timestamps to pk file
    :param pn_frame_tpl: str, file name to store img frames,
    e.g. 'C:/home/Amy/Research/dog_behavior_research/data/video_output/frame_{}.jpg'
    :param list_frames: list, list of img frames
    :param list_timestamps_ms: list, list of timestamps of frames
    :param pn_timestamps: str, file name to store timestamps
    :return: none

    """
    frame_num = 1
    for frame in list_frames:
        cv2.imwrite(pn_frame_tpl.format(frame_num), frame)
        print(pn_frame_tpl.format(frame_num), 'done')
        frame_num += 1
    if list_timestamps_ms is not None:
        s_timestamps_ms = pd.Series(data=list_timestamps_ms, index=range(1,len(list_timestamps_ms)+1))
        pd.to_pickle(s_timestamps_ms, pn_timestamps)

# for i in range(1,23):
#     fn_list = os.listdir('C:/home/Amy/Research/dog_behavior_research/data/cam data/20221007/' + str(i))
#     for fn in fn_list:
#         frame_audio_write(fp, store1, store2)

# a = pydub.AudioSegment.from_mp3('test.mp3')
# pydub.playback.play(a)

def video_to_frames(list_pn_video, pn_img, loader='cv', scale_percent=30, quality=20):
    """

    :param list_pn_video: ['C:/home/Amy/Research/DogBehavior/data/cam data/20221007', '...']
    :param pn_img:
    :param loader:
    :param scale_percent:
    :param quality:
    :return:
    save img frames to 'C:/home/Amy/Research/DogBehavior/data/img/20220926/0/D1.jpg' ...
    save audio to "C:/home/Amy/Research/DogBehavior/data/img/20220926/0/H_audFile.mp3",
                "C:/home/Amy/Research/DogBehavior/data/img/20220926/0/D_audFile.mp3"

    """

    for pn_video in list_pn_video:
        # pn_video = 'C:/home/Amy/Research/DogBehavior/data/cam data/20221007'
        subdir_img = '{}/{}'.format(pn_img, pn_video.split("/")[-1])
        if not os.path.exists(subdir_img):
            # subdir_img = 'C:/home/Amy/Research/DogBehavior/data/img/20221007
            os.mkdir(subdir_img)
        list_subdir_video = os.listdir(pn_video) # ['1', '2', '3', ...]
        for subdir_video in list_subdir_video:
            final_path_img = subdir_img + '/' + subdir_video # C:/home/Amy/Research/DogBehavior/data/img/20221007/1
            final_path_video = pn_video+'/'+subdir_video # C:/home/Amy/Research/DogBehavior/data/cam data/20221007/1
#            breakpoint()
#             if not os.path.exists(final_path_img):
#                 os.mkdir(final_path_img)

            ################## temp comment #############################3
            # if os.path.exists(final_path_img):
            #     continue
            # else:
            #     os.mkdir(final_path_img)
            #     #####################
            #     print('mkdir', final_path_img)
            #     #####################
            ###################### end temp comment ##########################

            list_fn_video = os.listdir(final_path_video)
            # ['C:/home/Amy/Research/DogBehavior/data/cam data/20221007/1/x.mp4, y.mov]
            for fn_video in list_fn_video:
                if os.path.splitext(fn_video)[1] == '.mp4':
                    fn_video_type = 'D'
                else:
                    fn_video_type = 'H'
#                breakpoint()
                #print(fn_video, 'read frame')
                print(final_path_video + '/' + fn_video, 'read audio')

                #audio
                fn_audio = final_path_img + '/' + fn_video_type + '_audFile.mp3'
                audio_clip = mp.VideoFileClip(final_path_video + '/' + fn_video)
                audio_clip.audio.write_audiofile(fn_audio)
                audio = librosa.load(fn_audio)
                pn_audio = final_path_img + '/' + fn_video_type + '_aud.pk'
                s_audio = pd.Series(data=audio, index=range(1, len(audio) + 1))
                pd.to_pickle(s_audio, pn_audio)


                ###################### temporary comment ########################
                # if loader =='cv':
                #     pn_frame_tpl = final_path_img + '/' + fn_video_type + '{}.jpg'
                #     pn_timestamps = final_path_img + '/' + fn_video_type + '_ts.pk'
                #     read_frames_from_video_cv(final_path_video+'/'+fn_video, pn_frame_tpl, pn_timestamps, scale_percent, quality)
                # elif loader == 'imageio':
                #     list_frames, list_ts = read_frames_from_video_imageio(final_path_video + '/' + fn_video)
                # else:
                #     (list_frames, list_ts) = ([], [])
                #     print('unknown loader')
                ######################## end temporary comment #########################
#                breakpoint()
#                 pn_frame_tpl = final_path_img+'/'+fn_video_type+'{}.jpg'
#                 pn_timestamps = final_path_img+'/'+fn_video_type+'_ts.pk'
#                 write_frames_to_files(pn_frame_tpl, list_frames, list_ts, pn_timestamps)




def write_pair_img(path_img, dict_fps={'D': 30.05, 'H': 30.00}, fn_align='timing.txt', flag_cvshow=True):
    """
    write dog and human view frames side by side. dog left, human right
    :param path_img: pathname of img folder (C:/home/Amy/Research/DogBehavior/data/img/20220926/2)
    :param dict_fps:
    :param fn_align:
    :param flag_cvshow:
    :return: 1. if not exists: create subdir (C:/home/Amy/Research/DogBehavior/data/img/20220926/2/pair),
             2. write pair img (P0001.jpg)
    """
    list_dh = ['D', 'H']
    fns = os.listdir(path_img)

    # breakpoint()

    # num of frames total
    dict_num_frames = {}
    for dh in list_dh:
        fns_dh = [x for x in fns if ((x[0] == dh) & (x[-3:] == 'jpg'))]
        dict_num_frames[dh] = len(fns_dh)

    # breakpoint()

    # total time duration
    dict_dur = {dh: dict_num_frames[dh] / dict_fps[dh] for dh in list_dh}
    print('dur:', dict_dur)

    # aligned start time
    f = open(f'{path_img}/{fn_align}')
    txt = f.readlines()
    dict_starttime_aligned = {'H': float(txt[0].strip()[2:]), 'D': float(txt[1][2:])}
    print('start time:', dict_starttime_aligned)

    # duration start to end
    dict_start_end = {dh: dict_dur[dh] - dict_starttime_aligned[dh] for dh in list_dh}
    print('start to end:', dict_start_end)

    # aligned duration
    aligned_dur = min(dict_start_end.values())
    print('aligned_dur:', aligned_dur)

    # end time
    dict_endtime_aligned = {dh: dict_starttime_aligned[dh] + aligned_dur for dh in list_dh}
    print('end_time_aligned:', dict_endtime_aligned)

    # start and end frame aligned
    dict_startframe_aligned = {dh: round(dict_starttime_aligned[dh] * dict_fps[dh]) for dh in list_dh}
    dict_endframe_aligned = {dh: round(dict_endtime_aligned[dh] * dict_fps[dh]) for dh in list_dh}
    print('startframe aligned:', dict_startframe_aligned)
    print('endframe aligned:', dict_endframe_aligned)

    cnt_pair = 1


    for i in range(dict_startframe_aligned['D'], dict_endframe_aligned['D']):
        j = round((i / dict_fps['D'] - dict_starttime_aligned['D'] + dict_starttime_aligned['H']) * dict_fps['H'] + 0.5)
        #         print(f'(D, H)=({i},{j})')
        fn_d = f'{path_img}/D{i}.jpg'
        fn_h = f'{path_img}/H{j}.jpg'
        img_d = cv2.imread(fn_d)
        img_h = cv2.imread(fn_h)

        min_height = min(img_d.shape[0], img_h.shape[0])
        img_d = cv2.resize(img_d, (int(min_height / img_d.shape[0] * img_d.shape[1]), min_height))
        img_h = cv2.resize(img_h, (int(min_height / img_h.shape[0] * img_h.shape[1]), min_height))

        img_blank = np.ones((img_d.shape[0], 50, 3), np.uint8) * 255
        # img_dh = cv2.hconcat([img_d, img_blank, img_h])
        img_dh = cv2.hconcat([img_d,img_h])

        # cv2.imwrite(f'{path_img}/P{cnt_pair}.jpg', img_dh, [cv2.IMWRITE_JPEG_QUALITY, 70])

        if not os.path.exists(f'{path_img}/pair'):
            os.mkdir(f'{path_img}/pair')
        cv2.imwrite(f'{path_img}/pair/P{str(cnt_pair).zfill(5)}.jpg', img_dh, [cv2.IMWRITE_JPEG_QUALITY, 70])

        cnt_pair += 1

        if 0:
            if flag_cvshow:
                # cv2.imshow(f'({i},{j})', img_dh)
                cv2.imshow('img_dh', img_dh)
                cv2.waitKey()
                print(i, j)
                time.sleep(0.1)
            else:
                img_dh = cv2.cvtColor(img_dh, cv2.COLOR_BGR2RGB)
                plt.imshow(img_dh)
                plt.title(f'({i},{j})')
                plt.show()


def list_valid_dir(path):
    """

    :param path: "C:/home/Amy/Research/DogBehavior/data/img"
    :return:
    """
    # breakpoint()
    valid_dir = []
    list_dir = os.listdir(path)
    for dd in list_dir:
        # dd: '20221007'
        list_dd_dir = os.listdir(f'{path}/{dd}')
        # list_dd_dir: ['0', '1', '2', '3', ...]
        for ddd in list_dd_dir:
            if (os.path.exists(f'{path}/{dd}/{ddd}/D_audFile.mp3') & os.path.exists(f'{path}/{dd}/{ddd}/H_audFile.mp3')):
                valid_dir.append(f'{path}/{dd}/{ddd}')
                print(f'{path}/{dd}/{ddd}')
    print('total valid_dir:', len(valid_dir))
    return valid_dir



def prep_data_trainAndtest(pn_data_info, channel_select='D', target_imsize=TARGET_IMSIZE, stft_winsize=2, stft_hop=1 / 90,
                           fps_D=FPS_D, train_test_split={'test':0.2,'val':0.1}, dns_rate=50,
                           lookback_prevact=10, stepsize_prevact=10):
    """
    pn_data_info: "C:/home/Amy/Research/DogBehavior/data/data_info.txt"
    channel_select: 'D' -- use dog view to generate train and test img, dog view is taken from left half of pair image
                 'H' -- use human view to generate traind and test img, human view is taken from right half of pair image
    target_imsize: (455,256) -- (width, height),  dog/human view img are resized to this size as train/test data
    fps_D: 30.05, fps of dog video (note: even if channel_select='H', still requires fps of dog video)
    train_test_split: 0.7, first 70% used as train, remaining 30% used for test
    dns_rate: 50, take one frame out of every 50 frames (note: adjacent frames highly similar)
    """
    # get start time
    #breakpoint()

    list_data_info = load_data_info(pn_data_info)
    dict_data = {key:{'x_img':np.ndarray((0, target_imsize[1], target_imsize[0], 3)),
                      'x_aud':None,
                      # 'x_prevaction':np.ndarray((0,lookback_prevact)),
                      'y_output':np.ndarray((0, 1)),
                      } for key in ['train','val','test']
                 }
    dict_split_info = {key:[] for key in ['train','val','test']}

    ##########################################3
    # breakpoint()
    # temp = pd.read_pickle('c:/home/Amy/Research/DogBehavior/dict_data_temp.pk')
    # (dict_data, dict_split_info) = temp
    ##############################################

    #breakpoint()
    #loop over valid segments
    pn_img_old =''
    for data_info_seg in list_data_info: #[0:10]: #[0:1]:
        print(f'data_info_seg:',data_info_seg)
        (pn_img, channel, fstart, fend) = data_info_seg

        if channel_select not in channel:
            continue

        #########################################
        #load aud and act for each new video, not new segment
        if pn_img != pn_img_old:
            ##############################################
            #pd.to_pickle((dict_data, dict_split_info), 'c:/home/Amy/Research/DogBehavior/dict_data_temp.pk')
            #############################################
            pn_img_old = pn_img
            # load audio stft
            # breakpoint()
            stft = load_aud_stft(pn_img, stft_winsize=stft_winsize, hop_length_sec=stft_hop, flag_display=False)
            # if X_train_aud is None:
            #     X_train_aud = np.ndarray((stft.shape[0], 0))
            #     X_val_aud = np.ndarray((stft.shape[0], 0))
            #     X_test_aud = np.ndarray((stft.shape[0], 0))
            if dict_data['train']['x_aud'] is None:
                for key in ['train','val','test']:
                    dict_data[key]['x_aud'] = np.ndarray((stft.shape[0], 0))
            #breakpoint()

            # load actions
            df_act = load_action(pn_img, dict_map=dict_act_map )
        ###############################################

        # split data into train, val, test
        #1. random sampling on frame index
        list_index_allframes = list(range(fstart, fend + 1, dns_rate))
        list_index_allframes = [idx for idx in list_index_allframes if df_act.notna().loc[idx,'action']]
        num_test = int(train_test_split['test']*len(list_index_allframes))
        num_val = int(train_test_split['val']*len(list_index_allframes))
        rng = default_rng()
        list_randnum = rng.choice(list_index_allframes, size=num_test+num_val, replace=False)
        list_index_test = sorted(list_randnum[0:num_test])
        list_index_val = sorted(list_randnum[num_test:num_test+num_val])
        list_index_train = sorted(list(set(list_index_allframes)-set(list_randnum)))

        dict_data_seg = {key: {'x_img': np.ndarray((0, target_imsize[1], target_imsize[0], 3)),
                           'x_aud': np.ndarray((stft.shape[0], 0)),
        # 'x_prevaction':np.ndarray((0,lookback_prevact)),
                           'y_output': np.ndarray((0, 1)),
                           } for key in ['train', 'val', 'test']
                     }
        dict_split_info_seg = {key: [] for key in ['train', 'val', 'test']}

        #2. loop over frames in pair img: add to train, val, test
        for frame in list_index_allframes: #range(fstart, fend + 1, dns_rate):
            print(f'frame {frame}')
            # load pair img, split to img_D,img_H assume same size
            fn_img = pn_img + f'/pair/P{str(frame).zfill(5)}.jpg'
            img_pair = cv2.imread(fn_img)
            print(img_pair.shape)
            ncols = int(img_pair.shape[1] / 2)
            if channel_select == 'D':
                img_channel = img_pair[:, 0:ncols, :]
            else:
                img_channel = img_pair[:, ncols::, :]
            #             print(img_channel.shape)
            img_channel = cv2.resize(img_channel, (target_imsize[0], target_imsize[1]))
            img_channel = np.expand_dims(img_channel, axis=0)

            # load starttime of human video
            starttime_H = load_starttime(pn_img)

            # get img timestamp in human video
            # note: frame is the frame index to pair imgs, it comes from frame index to dog imgs
            img_ts = frame / fps_D + starttime_H
            print(f'frame:{frame}  img_ts:{img_ts}')

            # get STFT at timestamp
            stft_frame = librosa.time_to_frames(img_ts)
            cur_stft = stft[:, stft_frame:stft_frame + 1]

            #             plt.plot(cur_stft)

            # breakpoint()

            # prevact
            # arr_prevact = [df_act['action'][frame] for frame in range(frame-(lookback_prevact*stepsize_prevact), frame, stepsize_prevact)]
            # idx = range(frame - (lookback_prevact * stepsize_prevact), frame, stepsize_prevact)
            # arr_prevact = df_act['action'].loc[idx]

            # get action at frame
            act = np.array([[df_act['action'][frame]]])
            # if np.isnan(act):
            #     continue

            if frame in list_index_train:
                dataset = 'train'
            elif frame in list_index_val:
                dataset = 'val'
            else:
                dataset = 'test'

            #record frame index for train, val, test
            # # breakpoint()
            # dict_split_info[dataset].append((pn_img, frame, channel_select))
            #
            # #append img (input)
            # dict_data[dataset]['x_img'] = np.vstack((dict_data[dataset]['x_img'], img_channel))
            # #append aud (input)
            # dict_data[dataset]['x_aud'] = np.hstack((dict_data[dataset]['x_aud'], cur_stft))
            # #append prevact (input)
            # # dict_data[dataset]['x_prevact'] = np.vstack((dict_data[dataset]['x_prevact', arr_prevact]))
            # #append label (output)
            # dict_data[dataset]['y_output'] = np.vstack((dict_data[dataset]['y_output'], act))

            breakpoint()
            dict_split_info_seg[dataset].append((pn_img, frame, channel_select))

            #append img (input)
            dict_data_seg[dataset]['x_img'] = np.vstack((dict_data_seg[dataset]['x_img'], img_channel))
            #append aud (input)
            dict_data_seg[dataset]['x_aud'] = np.hstack((dict_data_seg[dataset]['x_aud'], cur_stft))
            #append prevact (input)
            # dict_data[dataset]['x_prevact'] = np.vstack((dict_data[dataset]['x_prevact', arr_prevact]))
            #append label (output)
            dict_data_seg[dataset]['y_output'] = np.vstack((dict_data_seg[dataset]['y_output'], act))

            print(pn_img, frame, act, dataset, 'x_img:',dict_data_seg[dataset]['x_img'].shape, 'x_aud:', dict_data_seg[dataset]['x_aud'].shape, 'y:',dict_data_seg[dataset]['y_output'].shape)

        breakpoint()
        for dataset in ['train','val','test']:
            dict_data[dataset]['x_img'] = np.vstack((dict_data[dataset]['x_img'], dict_data_seg[dataset]['x_img']))
            #append aud (input)
            dict_data[dataset]['x_aud'] = np.hstack((dict_data[dataset]['x_aud'], dict_data_seg[dataset]['x_aud']))
            #append prevact (input)
            # dict_data[dataset]['x_prevact'] = np.vstack((dict_data[dataset]['x_prevact', arr_prevact]))
            #append label (output)
            dict_data[dataset]['y_output'] = np.vstack((dict_data[dataset]['y_output'], dict_data_seg[dataset]['y_output']))
            print(pn_img, 'all', dataset, 'img', dict_data[dataset]['x_img'].shape, 'aud', dict_data[dataset]['x_aud'].shape, 'y', dict_data[dataset]['y_output'].shape)
    breakpoint()
    return dict_data, dict_split_info

def map_motion_frame(pn_dict_data, dataset, iter):
    dict_split_info = pd.read_pickle(pn_dict_data)[1]
    # motion = dict_split_info[dataset][iter]
    return motion


def color_dif(frame, x_split=3, y_split=2):
    """

    :param frame: frame array from dict_data
    :param x_split:
    :param y_split:
    :return:
    """
    blockh = int(frame.shape[0]/y_split)-1
    blockw = int(frame.shape[1]/x_split)-1
    #center_coords of blocks. block (x,y) indexed by center_coords[y][x]
    center_coords = ([[() for x in range(0,x_split)] for y in range(0,y_split)])
    avg_blocks = {clr:([[() for x in range(0,x_split)] for y in range(0,y_split)]) for clr in ['r','g','b']}
    for y in range(0,y_split):
        for x in range(0,x_split):
            center_x = x*blockw+int(blockw/2)
            center_y = y*blockh+int(blockh/2)
            center_coords[y][x] = (center_y, center_x)
            avg[y][x] = np.mean(frame[center_y-int(blockh/2):center_y+int(blockh/2)+1, center_x-int(blockw/2):center_x+int(blockw/2)+1])




def read_valid_dir(fn_valid_dir="C:/home/Amy/Research/DogBehavior/data/valid_dir.txt"):
    f = open(fn_valid_dir)
    lines = f.readlines()
    list_fn = [x.strip() for x in lines]
    return list_fn

if __name__ == '__main__':
    if 0: #else:
        data = pd.read_pickle('C:/home/Amy/Research/DogBehavior/data/trainTestData_v1.pk')
        (X_train, X_train_aud, Y_train, X_test, X_test_aud, Y_test) = data

    if 0:
        # test write_pair_img()
        path_img = "C:/home/Amy/Research/DogBehavior/data/img/20220926/2"
        write_pair_img(path_img, dict_fps={'D': 30.05, 'H': 30.00}, fn_align='timing.txt', flag_cvshow=True)

    if 0:
        # test list_valid_dir()
        path = "C:/home/Amy/Research/DogBehavior/data/img"
        valid_dir = list_valid_dir(path)

    if 0:
        #rotate H view
        import cv2
        valid_dir =[ #('C:/home/Amy/Research/DogBehavior/data/img/20220926/7',cv2.ROTATE_90_CLOCKWISE),
                     ('C:/home/Amy/Research/DogBehavior/data/img/20221007/2', cv2.ROTATE_90_COUNTERCLOCKWISE),
                     ]
        breakpoint()
        for (pn,rotate) in valid_dir:
            list_fn=os.listdir(pn+'/tmp')
            list_fn=[x for x in list_fn if ((x[0]=='H') and (x[-3::]=='jpg')) ]
            for fn in list_fn:
                img= cv2.imread(f'{pn}/tmp/{fn}')
                img = cv2.rotate(img, rotate)
                cv2.imwrite(f'{pn}/{fn}', img)
            breakpoint()
    if 0:
        # step 4
        # create pair_img for all valid dir
        # path = "C:/home/Amy/Research/DogBehavior/data/img"
        # valid_dir = list_valid_dir(path)
        valid_dir = [
            # 'C:/home/Amy/Research/DogBehavior/data/img/20221007/10',
                     'C:/home/Amy/Research/DogBehavior/data/img/20221007/11',
                     'C:/home/Amy/Research/DogBehavior/data/img/20221007/12',
                     'C:/home/Amy/Research/DogBehavior/data/img/20221007/13',
                     'C:/home/Amy/Research/DogBehavior/data/img/20221007/14'
                     ]
        valid_dir =[ 'C:/home/Amy/Research/DogBehavior/data/img/20220926/7',
                     'C:/home/Amy/Research/DogBehavior/data/img/20221007/2',
                     ]  #correct H view, rotate 90
        breakpoint()
        for path_img in valid_dir:
            if not os.path.exists(f'{path_img}/pair'):
                print(path_img)
                write_pair_img(path_img, dict_fps={'D': 30.05, 'H': 30.00}, fn_align='timing.txt', flag_cvshow=False)


    if 1:
        # use segments defined in pn_data_info to generate all (img,audio,act), split into train,val,test
        # (stft_winsize, dns_rate, fn_out) = (2, 50, 'C:/home/Amy/Research/DogBehavior/data/trainTestData_v1.pk')
        # (stft_winsize, dns_rate, fn_out) = (2, 5, 'C:/home/Amy/Research/DogBehavior/data/trainTestData_v2.pk')
        # (stft_winsize, dns_rate, fn_out) = (0.2, 5, 'C:/home/Amy/Research/DogBehavior/data/trainTestData_v3.pk')
        #(stft_winsize, dns_rate, fn_out) = (0.1, 5, 'C:/home/Amy/Research/DogBehavior/data/trainTestData_v4.pk')

        # (stft_winsize, dns_rate, fn_out) = (0.1, 5, 'C:/home/Amy/Research/DogBehavior/data/trainValTestData_v0.pk')
        (stft_winsize, dns_rate, fn_out) = (1, 5, 'C:/home/Amy/Research/DogBehavior/data/trainValTestData_stft1_dns5.pk')
        dict_data, dict_split_info = prep_data_trainAndtest(pn_data_info, stft_winsize=stft_winsize, dns_rate=dns_rate)
        breakpoint()
        pd.to_pickle((dict_data, dict_split_info), fn_out)

    if 0:
        #correct: two videos Hview rotate by 90degree, replace pairimg,
        correction_dir =[ 'C:/home/Amy/Research/DogBehavior/data/img/20220926/7',
                     'C:/home/Amy/Research/DogBehavior/data/img/20221007/2',
                     ]  #correct H view, rotate 90
        fn_old = 'C:/home/Amy/Research/DogBehavior/data/trainValTestData_stft1_dns5_old.pk'
        (dict_data, dict_split_info) = pd.read_pickle(fn_old)
        breakpoint()
        for key in ['train', 'val', 'test']:
            list_idx = []
            for (pn_img, frame, channel) in dict_split_info[key]:
                if pn_img in correction_dir:
                    list_idx.append(False)
                else:
                    list_idx.append(True)
            breakpoint()
            dict_data[key]['x_img'] = dict_data[key]['x_img'][list_idx,:,:,:]
            dict_data[key]['x_aud'] = dict_data[key]['x_aud'][:,list_idx]
            dict_data[key]['y_output'] = dict_data[key]['y_output'][list_idx]
        breakpoint()
        pd.to_pickle((dict_data, dict_split_info), 'c:/home/Amy/Research/DogBehavior/dict_data_temp.pk')
        breakpoint()
        pn_data_correction_info = "C:/home/Amy/Research/DogBehavior/data/data_info_mid_correct.txt"
        (stft_winsize, dns_rate, fn_out) = (1, 5, 'C:/home/Amy/Research/DogBehavior/data/trainValTestData_stft1_dns5_correction.pk')
        dict_data, dict_split_info = prep_data_trainAndtest(pn_data_correction_info, stft_winsize=stft_winsize, dns_rate=dns_rate)
        pd.to_pickle((dict_data, dict_split_info), fn_out)

        #pd.to_pickle((dict_data, dict_split_info), fn_out)

    if 0:
        fn_dict_data = 'C:/home/Amy/Research/DogBehavior/data/trainValTestData_stft1_dns5.pk'
        dict_data, dict_split_info = pd.read_pickle(fn_dict_data)
        # remove sample with y_output=NaN
        breakpoint()
        for key in ['train', 'val', 'test']:
            list_notnan = np.squeeze(~np.isnan(dict_data[key]['y_output']))
            dict_data[key]['x_img'] = dict_data[key]['x_img'][list_notnan, :, :, :]
            dict_data[key]['x_aud'] = dict_data[key]['x_aud'][:, list_notnan]
            dict_data[key]['y_output'] = dict_data[key]['y_output'][list_notnan]
            dict_split_info[key] = dict_split_info[key][list_notnan]
        pd.to_pickle((dict_data, dict_split_info), fn_dict_data)