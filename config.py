"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

Dataset_Root = '/ssd3/jwhuh/Datasets'
H36M_ROOT = 'h36m'
LSP_ROOT = join(Dataset_Root, 'lsp')
LSP_ORIGINAL_ROOT = join(Dataset_Root, 'lsp_original')
LSPET_ROOT = join(Dataset_Root, 'hr_lspet')
MPII_ROOT = join(Dataset_Root, 'mpii')
COCO_ROOT = join(Dataset_Root, 'coco')
MPI_INF_3DHP_ROOT = join(Dataset_Root, 'mpi-inf-3dhp')
PW3D_ROOT = join(Dataset_Root, '3dpw')
UPI_S1H_ROOT = join(Dataset_Root, 'upi-s1h')
UP_3D_ROOT = join(Dataset_Root, 'up-3d')

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
OPENPOSE_PATH = join(Dataset_Root, 'openpose')

# Path to test/train npz files
DATASET_FILES = [ {'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
                   'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2.npz'),
                   'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
                   '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                   'up-3d': join(DATASET_NPZ_PATH, 'up_3d_lsp_test.npz')
                  },

                  {# 'h36m': join(DATASET_NPZ_PATH, 'h36m_train.npz'), # Unavailable now
                   'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
                   'mpii': join(DATASET_NPZ_PATH, 'mpii_train.npz'),
                   'coco': join(DATASET_NPZ_PATH, 'coco_2014_train.npz'),
                   'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
                   'up-3d': join(DATASET_NPZ_PATH, 'up_3d_trainval.npz')
                  }
                ]

DATASET_FOLDERS = {'h36m': H36M_ROOT,
                   'h36m-p1': H36M_ROOT,
                   'h36m-p2': H36M_ROOT,
                   'lsp-orig': LSP_ORIGINAL_ROOT,
                   'lsp': LSP_ROOT,
                   'lspet': LSPET_ROOT,
                   'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                   'mpii': MPII_ROOT,
                   'coco': COCO_ROOT,
                   '3dpw': PW3D_ROOT,
                    'up-3d': UP_3D_ROOT,
                   'upi-s1h': UPI_S1H_ROOT,
                }

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
STATIC_FITS_DIR = 'data/static_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
