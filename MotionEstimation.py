import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
#% matplotlib inline
import time


def read_pairimg(pn):
    """
    pn='/C:/home/Amy/Research/DogBehavior/pair'
    :param pn:
    :return:
    """
    list_fns = sorted(os.listdir(pn))
    list_fns = [x for x in list_fns if os.path.splitext(x)[-1] == '.jpg']
    # print(list_fns)
    dict_img = {}
    for fn in list_fns:
        idx = int( fn.split('.')[0][1::])
        img = cv2.imread(f'{pn}/{fn}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dict_img[idx] = img
        ###################
        #print(f'load pairimg:{pn}/{fn}')
        ###################
    return dict_img


def build_pyramid(img, nLevel):
    # pyramid
    dict_pyramid = {0: img}
    for k in range(1, nLevel):
        tmp = cv2.GaussianBlur(dict_pyramid[k - 1], (5, 5), cv2.BORDER_DEFAULT)
        dict_pyramid[k] = tmp[::2, ::2]
    ###############
    # print('build impyramid')
    # for (l, img) in dict_pyramid.items():
    #     print('level:', l, img.shape)
    ###############
    return dict_pyramid


def get_bbox(y, dy, template_size, imh):
    (ymin0, ymax0) = (y - int(template_size / 2), y + int(template_size / 2))
    (ymin1, ymax1) = (ymin0 + dy, ymax0 + dy)
    return (ymin0, ymax0, ymin1, ymax1)


def plot_impyramid(pyramid, title=''):
    list_lev = sorted(list(pyramid.keys()))
    for lev in list_lev:
        if lev == 0:
            tmp = pyramid[lev]
        else:
            img = pyramid[lev]
            img_ex = np.ones((img.shape[0], tmp.shape[1])) * 255
            img_ex[:, 0:img.shape[1]] = img
            tmp = np.vstack((img_ex, tmp))
    plt.imshow(tmp, cmap='gray', vmin=0, vmax=255);
    plt.title(title);
    plt.show()


def plot_motionfield(dict_mv,pn=None):
    """
    plot
    :param dict_mv: dict_mv={'mvx':df_mvx, 'mvy':df_mvy}
    :return:
    """
    (df_mvx, df_mvy) = (dict_mv['mvx'], dict_mv['mvy'])
    arr_mx = df_mvx.values.flatten()
    arr_my = df_mvy.values.flatten()
    arr_origin = [(x,y)  for y in df_mvx.index for x in df_mvx.columns]
    arr_origin = np.asarray(arr_origin).T

    # origin_mv = [([x0, y0], mv) for ((x0, y0), mv) in mv_pt.items()]
    # arr_origin = np.asarray([tup[0] for tup in origin_mv]).T
    # arr_mv = np.asarray([tup[1] for tup in origin_mv])
    # plt.quiver(*arr_origin, arr_mv[:, 0], arr_mv[:, 1]);
    plt.clf()
    plt.quiver(*arr_origin, arr_mx, arr_my)
    #plt.show()
    if pn is not None:
        plt.savefig(pn)

def plot_motionfield_arr(mx, my,pn=None):
    """
    plot
    :param dict_mv: dict_mv={'mvx':df_mvx, 'mvy':df_mvy}
    :return:
    """
    arr_origin = [(x,y)  for y in range(mx.shape[0]) for x in range(mx.shape[1])]
    arr_origin = np.asarray(arr_origin).T

    arr_mx = pd.DataFrame(mx).fillna(np.nan).values.flatten()
    arr_my = pd.DataFrame(my).fillna(np.nan).values.flatten()

    plt.quiver(*arr_origin, arr_mx, arr_my)
    plt.show()


def template_matching(img_prev, img, list_pt, par):
    """

    :param img_prev:
    :param img:
    :return:
    """
    # pyramid levels
    (nLevel, search_range_coarse, template_size)=(par['nLevel'],par['search_range_coarse'],par['template_size'])
    # breakpoint()

    # pyramid
    pyramid_prev = build_pyramid(img_prev, nLevel=nLevel)
    pyramid = build_pyramid(img, nLevel=nLevel)

    ###########################
    if 0:
        print('pyramid_prev:')
        plot_impyramid(pyramid_prev, 'pyramid_prev')
        plot_impyramid(pyramid, 'pyramid')
    ###########################

    dict_pyramid_mv = {}
    list_ptx = sorted(list(set([x for (x,y) in list_pt])))
    list_pty = sorted(list(set([y for (x,y) in list_pt])))
    # nLevel-1: block matching
    mv_pt = {(x, y): (0, 0) for (x, y) in list_pt}
    for level in range(nLevel - 1, -1, -1):
        ##########################
        # print(f'tempmlate matching: level{level}')
        ##########################
        scale = 2 ** level
        (I0, I1) = (pyramid_prev[level], pyramid[level])
        (imw, imh) = (I0.shape[1], I0.shape[0])
        if level == (nLevel-1):
            search_range = search_range_coarse
        else:
            search_range = 1
        for (x0, y0) in list_pt:
            (x, y) = (int(x0 / scale), int(y0 / scale))
            #672 64 2
            # breakpoint()
            # print(x0,y0,level)
            (mx, my) = (mv_pt[(x0, y0)][0]*2, mv_pt[(x0,y0)][1]*2)
            ##############################
            # breakpoint()
            # t0=time.time()
            # df_mse = pd.DataFrame(np.nan, index=range(my-search_range, my + search_range + 1), columns=range(mx - search_range, mx + search_range + 1))
            # for dy in df_mse.index:
            #     (ymin0, ymax0, ymin1, ymax1) = get_bbox(y, dy, template_size, imh)
            #     if ((ymin0 < 0) | (ymax0 >= imh) | (ymin1 < 0) | (ymax1 >= imh)):
            #         continue
            #     # print('bbox y:', ymin0, ymax0, ymin1, ymax1)
            #     for dx in df_mse.columns:
            #         (xmin0, xmax0, xmin1, xmax1) = get_bbox(x, dx, template_size, imw)
            #         if ((xmin0 < 0) | (xmin1 >= imw) | (xmin1 < 0) | (xmin1 >= imw)):
            #             continue
            #         # print('bbox x:', xmin0, xmax0, xmin1, xmax1)
            #         df_mse.loc[dy, dx] = np.mean((I0[ymin0:ymax0 + 1, xmin0:xmax0 + 1] - I1[ymin1:ymax1 + 1, xmin1:xmax1 + 1]) ** 2)
            # s_min = df_mse.min(axis=0)
            # dx = s_min.idxmin()
            # dy = df_mse[dx].idxmin()
            # print(dx,dy)
            ###################################
            # t1=time.time()
            (dx,dy,minerr)=(None,None,None)
            for yshift in range(my-search_range,my+search_range+1):
                (ymin0, ymax0, ymin1, ymax1) = get_bbox(y, yshift, template_size, imh)
                if ((ymin0 < 0) | (ymax0 >= imh) | (ymin1 < 0) | (ymax1 >= imh)):
                    continue
                for xshift in range(mx-search_range, mx+search_range+1):
                    (xmin0, xmax0, xmin1, xmax1) = get_bbox(x, xshift, template_size, imw)
                    if ((xmin0 < 0) | (xmax0 >= imw) | (xmin1 < 0) | (xmax1 >= imw)):
                        continue

                    # print(f'x0:{x0}, y0:{y0}, level:{level}, yshift:{yshift}  xshift:{xshift}')
                    # if xshift==-3 and yshift==-3 and x0==656 and y0==48 and level==3:
                    #     breakpoint()
                    err = np.mean((I0[ymin0:ymax0 + 1, xmin0:xmax0 + 1] - I1[ymin1:ymax1 + 1, xmin1:xmax1 + 1]) ** 2)
                    if minerr is None:
                        (dx,dy,minerr) = (xshift,yshift,err)
                    else:
                        if err<minerr:
                            (dx, dy, minerr) = (xshift, yshift, err)
            # t2=time.time()
            # print(t1-t0, t2-t1)
            # breakpoint()
            #################
            #print(df_mse)
            #print(f'pt0:({x0},{y0}) pt:({x},{y})  mv:({mx},{my})-->({dx},{dy})')
            #breakpoint()
            ###############
            mv_pt[(x0, y0)] = (dx, dy)
            ##################
            # print('mse: level',level, f'({x0},{y0})', f'mv:({dx},{dy})')
            # plt.imshow(df_mse.values, cmap='gray',vmin=0,vmax=df_mse.max().max()); #plt.xticks(df_mse.columns); plt.yticks(df_mse.index);
            # plt.title('mse'); plt.show()
            ##################
        #record in pyramd
        df_mvx = pd.DataFrame(index=list_pty, columns=list_ptx)
        df_mvy = pd.DataFrame()
        for (x0,y0) in list_pt:
            df_mvx.loc[y0,x0] = mv_pt[(x0,y0)][0]
            df_mvy.loc[y0,x0] = mv_pt[(x0, y0)][1]
        dict_pyramid_mv[level] = {'mvx':df_mvx, 'mvy':df_mvy}
        ###########################
        # print('level', level)
        # print(df_mvx)
        # print(df_mvy)
        #breakpoint()
        ##################

    #######################
    if 0:
        breakpoint()
        plot_motionfield(dict_pyramid_mv[0])
    #######################
    return dict_pyramid_mv


def motion_estimation(pn, grid_space=32, step_timescale=5, par={'nLevel':4, 'search_range_coarse':3, 'template_size':5}):
    """
    :param pn: pn='/C:/home/Amy/Research/DogBehavior/pair'
    :return:
        save mvD, mvH to '/C:/home/Amy/Research/DogBehavior/immotion{step_timescale}/D00001.pk', '/C:/home/Amy/Research/DogBehavior/immotion{step_timescale}/H00001.pk'
        mvD, mvH: dict, mvD[level]['mvx'|'mvy']
    """
    img_bound = (int(par['template_size']/2)+par['search_range_coarse'])*(2**(par['nLevel']-1))+2
    print('img_bound', img_bound)
    # breakpoint()

    # load pairimg
    dict_img = read_pairimg(pn)
    (imw, imh) = (dict_img[1].shape[1], dict_img[1].shape[0])

    # breakpoint()
    list_pt = []
    for y in np.arange(img_bound, imh - img_bound, grid_space):
        list_pt = list_pt + [(x, y) for x in np.arange(img_bound, int(imw/2)-img_bound, grid_space)]
    print(sorted(list(set([y for (x, y) in list_pt]))))
    print(sorted(list(set([x for (x, y) in list_pt]))))
    # breakpoint()
    #####################
    # print(list_pt)
    if 0:
        for (k, img) in dict_img.items():
            plt.imshow(img, cmap='gray', vmin=0, vmax=255);
            plt.title(f'frame{k}:{img.shape[0]}x{img.shape[1]}');
            plt.show()
    #####################
    #create dir to save mv00001.pk ...
    pn_immotion = pn.replace('pair',f'immotion{step_timescale}')
    if not os.path.exists( pn_immotion ):
        os.mkdir(pn_immotion)
    print('create', pn_immotion)

    # breakpoint()
    #motion field
    list_frameNo = sorted(list(dict_img.keys()))
    for frameNo in list_frameNo[step_timescale::]:
        # if os.path.exists(f'{pn_immotion}/D{str(frameNo).zfill(5)}.pk') and os.path.exists(f'{pn_immotion}/H{str(frameNo).zfill(5)}.pk'):
        #     continue
        print(f'motion est({frameNo}): frame{frameNo-step_timescale} --> frame{frameNo}')
        #prev img, current img
        (img_prev, img) = (dict_img[frameNo-step_timescale], dict_img[frameNo])
        #split dogview, humanview: D - left half, H-right half
        (imgD_prev, imgD) = (img_prev[:, 0:int(imw/2)], img[:, 0:int(imw / 2)])
        (imgH_prev, imgH) = (img_prev[:, int(imw/2)::], img[:, int(imw / 2)::])
        #motion estimate
        # breakpoint()
        mvD = template_matching(imgD_prev, imgD, list_pt, par)
        mvH = template_matching(imgH_prev, imgH, list_pt, par)

        if 0:
            plot_motionfield(mvD[0], f'{pn_immotion}/mfD{str(frameNo).zfill(5)}.jpg')
            plot_motionfield(mvH[0], f'{pn_immotion}/mfH{str(frameNo).zfill(5)}.jpg')

        pd.to_pickle(mvD, f'{pn_immotion}/D{str(frameNo).zfill(5)}.pk' )
        pd.to_pickle(mvH, f'{pn_immotion}/H{str(frameNo).zfill(5)}.pk' )

        print('save motion field:', f'{pn_immotion}/D|H{str(frameNo).zfill(5)}.pk')


def assemble_frames():
    skipframe = 5
    f = open("C:/home/Amy/Research/DogBehavior/data/valid_dir_small.txt")
    lines = f.readlines()
    lines = [x.strip() for x in lines]
    breakpoint()
    for pn in lines:
        print(pn)
        pn_mot = f'{pn}/immotion{skipframe}'
        list_fn = os.listdir(pn_mot)
        dict_mot = {}
        for channel in ['D', 'H']:
            dict_mot[channel] = {}
            list_chann = [x for x in list_fn if ((x[0]==channel) and (x.split('.')[1]=='pk')) ]
            for fn in list_chann:
                frameNo = int(fn.split('.')[0][1::])
                dict_mot[channel][frameNo] = pd.read_pickle(f'{pn_mot}/{fn}')
        #breakpoint()
        pd.to_pickle(dict_mot, f'{pn}/motionfield{skipframe}.pk')



if __name__ == '__main__':
    # assemble_frames()
    # breakpoint()

    from prepdata import read_valid_dir
    # list_pn = read_valid_dir("C:/home/Amy/Research/DogBehavior/data/valid_dir_small_0.txt")
    # list_pn = read_valid_dir("C:/home/Amy/Research/DogBehavior/data/valid_dir_small_1.txt")
    # list_pn = read_valid_dir("C:/home/Amy/Research/DogBehavior/data/valid_dir_small_2.txt")
    #list_pn = read_valid_dir("C:/home/Amy/Research/DogBehavior/data/valid_dir_small_3.txt")
    list_pn = read_valid_dir("C:/home/Amy/Research/DogBehavior/data/valid_dir_small_correction.txt")
    #breakpoint()
    par={'nLevel':4, 'search_range_coarse':3, 'template_size':5}
    for pn in list_pn:
        motion_estimation(pn+"/pair", grid_space=32, step_timescale=5, par=par)

    #c:\home\Amy\Research\DogBehavior\src\py