__author__ = 'kazuto1011'

import numpy as np

VIDEOS_PER_CATEGORY = 50
CLIPS_PER_VIDEO     = 32
OVERWRAPPED_FRAMES  = 8
DIMENSION           = 4096

INPUT_DIR = '/home/kazuto/Desktop/study/c3d/C3D_video_descriptor/output'
CATEGORY  = ('/read-a-book', '/eat-a-meal', '/gaze-at-a-robot', '/gaze-at-a-tree', '/look-around')

def load_feature(extension='.fc6-1'):

    def _read_binary(file_name):
        with open(file_name, 'rb') as f:
            num,channel,length,height,width = np.fromfile(f, np.int32, count=5)
            dimension = num*channel*length*height*width
            feature   = np.fromfile(f, np.float32, count=dimension)
            return feature.tolist()

    data_list = []
    target_list = []

    for ci in range(0,len(CATEGORY)):
        for vi in range(0,VIDEOS_PER_CATEGORY):
            for fi in range(0,OVERWRAPPED_FRAMES*CLIPS_PER_VIDEO,OVERWRAPPED_FRAMES):
                file_name = INPUT_DIR + CATEGORY[ci] + CATEGORY[ci] + "_%d_%d"%(vi,fi) + extension
                feature = _read_binary(file_name)
                data_list.append(feature)
                target_list.append(ci)

    return {'data':np.array(data_list),'target':np.array(target_list)}

def shuffle_video(dataset):
    split_data   = np.array(zip(*[iter(dataset['data'])]*CLIPS_PER_VIDEO))
    split_target = np.array(zip(*[iter(dataset['target'])]*CLIPS_PER_VIDEO))

    perm = np.random.permutation(VIDEOS_PER_CATEGORY*len(CATEGORY))
    shuffled_data   = split_data[perm]
    shuffled_target = split_target[perm]

    dataset['data']   = shuffled_data.reshape(VIDEOS_PER_CATEGORY*len(CATEGORY)*CLIPS_PER_VIDEO, DIMENSION)
    dataset['target'] = shuffled_target.reshape(VIDEOS_PER_CATEGORY*len(CATEGORY)*CLIPS_PER_VIDEO)

    return dataset
