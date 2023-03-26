import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras import layers, models
import sys
import fnmatch,os
sys.path.append('C:/home/Amy/Research/DogBehavior/src/py')
import seaborn as sns
import pandas as pd
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import cv2

def load_data_info(pn):
    """
    parse data_info.txt
    :param pn: "C:/home/Amy/Research/DogBehavior/data/data_info.txt"
    :return:
    [['C:/home/Amy/Research/DogBehavior/data/img/20220926/11','DH',  '262','3655'],
     ['C:/home/Amy/Research/DogBehavior/data/img/20220926/13', 'H', '379', '408']
     ....]
    """
    data_info = []
    fp = open(pn)
    lines = fp.readlines()
    for line in lines:
        print(line)
        list_str = line.strip().split(" ")
        num_seg = int((len(list_str)-1)/3)
        pn_data = list_str[0]
        for k in range(num_seg):
            # breakpoint()
            data_info.append([pn_data, list_str[k*3+1], int(list_str[k*3+2]), int(list_str[k*3+3])])
    return data_info

def load_action(pn_img, dict_map={'sit':0, 'stand':1, 'walk':2, 'smell':3, 'run':4}):
    """
    read '/pair/D_bhv.txt' from pn_img='C:/home/Amy/Research/DogBehavior/data/img/20220926/13'
    :param pn_img:
    :return:
    """
    num_frames = len(fnmatch.filter(os.listdir(pn_img + '/pair'), '*.jpg'))
    s_act = pd.Series(np.nan, index=range(1, num_frames + 1))
    s_chgact = pd.Series(np.nan, index=range(1, num_frames + 1))

    pn_actions = pn_img + '/pair/D_bhv.txt'
    f_actions = open(pn_actions)
    lines = f_actions.readlines()

    for line in lines:
        fNo = int(line.strip().split(' ')[0])
        action = line.strip().split(' ')[1]
        s_act[fNo] = action
        s_chgact[fNo] = 1

    for (key, value) in dict_map.items():
        s_act.replace(key, value, inplace=True)
    s_act.ffill(inplace=True)
    df_act = pd.DataFrame()
    df_act['action'] = s_act
    df_act['change_action'] = s_chgact



    return df_act


def load_starttime(pn, channel='H'):
    """

    :param pn: "C:/home/Amy/Research/DogBehavior/data/img/20220926/13"
    :param channel: 'H' or 'D'
    :return: starttime: 2.222
    """
    f = open(f'{pn}/timing.txt')
    lines = f.readlines()

    if channel == 'H':
        linenum = 0
    else:
        linenum = 1

    starttime = float(lines[linenum].strip().split(' ')[-1])
    return starttime


def load_aud_stft(pn_img, stft_winsize, sr=44100, hop_length_sec=1/90, flag_display=False):
    """

    :param pn_img: 'C:/home/Amy/Research/DogBehavior/data/img/20220926/13'
    :param flag_display:
    sr: audio sampling rate, 44100Hz (iphone audio sr)
    stft_winsize: window size for stft, in seconds
    hop_length_sec: time in seconds between adjacent samples
    :return: aud_stft_db, sr
        aud_stft_db: 2d np array, stft of audio wave
        sr: float, sampling rate
    """
    pn_audio = pn_img + '/H_audFile.mp3'
    aud, sr = librosa.load(pn_audio, sr=sr)
    n_fft = int(stft_winsize*sr)
    hop_length = int(hop_length_sec*sr)
    aud_stft = librosa.stft(aud, n_fft=n_fft, hop_length=hop_length)
    aud_stft_db = librosa.amplitude_to_db(abs(aud_stft))
    if flag_display:
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(aud, sr=sr)
        plt.show()
        ipd.Audio(pn_audio)
        plt.show()
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(aud_stft_db, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar()
    return aud_stft_db


if __name__ == "__main__":
    load_aud_stft(pn_img='C:/home/Amy/Research/DogBehavior/data/img/20220926/13', stft_winsize=2)