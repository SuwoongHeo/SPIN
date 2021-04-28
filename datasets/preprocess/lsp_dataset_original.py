import os
from os.path import join
import argparse
import numpy as np
import scipy.io as sio
import subprocess
from .read_openpose import read_openpose

def lsp_dataset_original_extract(dataset_path, openpose_path, out_path):
    """
    Annotation order
    0 - Right ankle, 1 - Right knee, 2 - Right hip, 3 - Left hip, 4 - Left knee, 5 - Left ankle,
    6 - Right wrist, 7 - Right elbow, 8 - Right shoulder, 9 - Left shoulder, 10 - Left elbow,
    11 - Left wrist, 12 - Neck, 13 - Head top
    """
    # bbox expansion factor
    scaleFactor = 1.2

    # we use LSP dataset original for training
    imgs = range(1000)

    # structs we use
    imgnames_, scales_, centers_, parts_, openposes_  = [], [], [], [], []

    # annotation files
    annot_file = os.path.join(dataset_path, 'joints.mat')
    joints = sio.loadmat(annot_file)['joints']

    json_path = os.path.join(openpose_path, 'lsp')
    if not os.path.isdir(json_path):
        sub_path = 'images'
        os.makedirs(json_path, exist_ok=True)
        subprocess.run(
            "python /ssd2/swheo/dev/HumanRecon/Preprocessing/run_openpose.py --GPU_ID {0} --input_path {1} --write_json {2} --no_display {3}".format(
                str(1),
                os.path.join(dataset_path, sub_path),
                os.path.join(json_path),
                True).split(' '))
    # go over all the images
    for img_i in imgs:
        # image name
        imgname = 'im%04d.jpg' % (img_i+1)
        imgname_full = join('images', imgname)
        # read keypoints
        part14 = joints[:2,:,img_i].T
        # scale and center
        bbox = [min(part14[:,0]), min(part14[:,1]),
                max(part14[:,0]), max(part14[:,1])]
        center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
        scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
        # update keypoints
        part = np.zeros([24,3])
        part[:14] = np.hstack([part14, np.ones([14,1])])

        # read openpose detections
        json_file = os.path.join(openpose_path, 'lsp',
            imgname.replace('.jpg', '_keypoints.json'))
        openpose = read_openpose(json_file, part, 'lsp')

        # store data
        imgnames_.append(imgname_full)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        openposes_.append(openpose)


    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'lsp_dataset_original_train.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       openpose=openposes_)
