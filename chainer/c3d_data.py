import numpy as np

N_VIDEO = 50
N_CLIP = 32
FRAME = 8
DIMENSION = 4096

input_dir = '/home/kazuto/Desktop/study/c3d/C3D_video_descriptor/output'
CATEGORY = ('/read-a-book', '/eat-a-meal', '/gaze-at-a-robot', '/gaze-at-a-tree', '/look-around')

def read_binary(file):
    with open(file, 'rb') as f:
        s = np.fromfile(f, np.int32, count=5)
        # num x chanel x length x height x width
        m = s[0]*s[1]*s[2]*s[3]*s[4]
        feature = np.fromfile(f, np.float32, count=m)
        return feature.tolist()

def load_feature():
    dataset = {}
    dataset.setdefault('data', np.array([],np.float32))
    dataset.setdefault('target', np.array([]))

    data_list = []
    target_list = []
    for i in range(0,len(CATEGORY)):
        for j in range(0,N_VIDEO):
            for k in range(0,FRAME*N_CLIP,FRAME):
                file = input_dir + CATEGORY[i] + CATEGORY[i] + "_%d_%d"%(j,k) + ".fc6-1"
                feature = read_binary(file)
                data_list.append(feature)
                target_list.append(i)

    dataset['data'] = np.array(data_list)
    dataset['target'] = np.array(target_list)
    return dataset

def shuffle_video(dataset):
    data = dataset['data']
    target = dataset['target']

    split_data = np.array(zip(*[iter(data)]*N_CLIP))
    split_target = np.array(zip(*[iter(target)]*N_CLIP))

    perm = np.random.permutation(N_VIDEO*len(CATEGORY))
    shuffled_data = split_data[perm]
    shuffled_target = split_target[perm]

    dataset['data'] = shuffled_data.reshape(N_VIDEO*len(CATEGORY)*N_CLIP, DIMENSION)
    dataset['target'] = shuffled_target.reshape(N_VIDEO*len(CATEGORY)*N_CLIP)

    return dataset
