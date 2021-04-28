import os
from os.path import join
import h5py
import numpy as np
from .read_openpose import read_openpose
import subprocess

def mpii_extract(dataset_path, openpose_path, out_path):

    # Original order
    # (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip,
    # 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax,
    # 8 - upper neck, 9 - head top, 10 - r wrist,
    # 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)
    # convert joints to global order to
    joints_idx = [0, 1, 2, 3, 4, 5, 14, 15, 12, 13, 6, 7, 8, 9, 10, 11]

    # structs we use
    imgnames_, scales_, centers_, parts_, openposes_ = [], [], [], [], []

    # annotation files
    annot_file = os.path.join('data', 'train.h5')

    # read annotations
    f = h5py.File(annot_file, 'r')
    centers, imgnames, parts, scales = \
        f['center'], f['imgname'], f['part'], f['scale']

    json_path = os.path.join(openpose_path, 'mpii')
    if not os.path.isdir(json_path):
        sub_path = 'images'
        os.makedirs(json_path, exist_ok=True)
        subprocess.run(
            "python /ssd2/swheo/dev/HumanRecon/Preprocessing/run_openpose.py --GPU_ID {0} --input_path {1} --write_json {2} --no_display {3}".format(
                str(1),
                os.path.join(dataset_path, sub_path),
                os.path.join(json_path),
                True).split(' '))

    # go over all annotated examples
    for center, imgname, part16, scale in zip(centers, imgnames, parts, scales):
        imgname = imgname.decode('utf-8')
        # check if all major body joints are annotated 
        if (part16>0).sum() < 2*len(joints_idx):
            continue
        # keypoints
        part = np.zeros([24,3])
        part[joints_idx] = np.hstack([part16, np.ones([16,1])])
        # read openpose detections
        json_file = os.path.join(openpose_path, 'mpii',
            imgname.replace('.jpg', '_keypoints.json'))
        openpose = read_openpose(json_file, part, 'mpii')
        
        # store data
        imgnames_.append(join('images', imgname))
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        openposes_.append(openpose)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'mpii_train.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       openpose=openposes_)
