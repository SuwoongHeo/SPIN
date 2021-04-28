#!/usr/bin/python
"""
Preprocess datasets and generate npz files to be used for training testing.
It is recommended to first read datasets/preprocess/README.md
"""
import argparse
import config as cfg
from datasets.preprocess import h36m_extract,\
                                pw3d_extract, \
                                mpi_inf_3dhp_extract, \
                                lsp_dataset_extract,\
                                lsp_dataset_original_extract, \
                                hr_lspet_extract, \
                                mpii_extract, \
                                up_3d_extract, \
                                coco_extract

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', default=True, action='store_true', help='Extract files needed for training')
parser.add_argument('--eval_files', default=False, action='store_true', help='Extract files needed for evaluation')
parser.add_argument('--set_name', default=None, type=str, help='train : mpi_inf, lsp_original, hr_lspet, mpii, coco, up3d, val : h36m, mpi_inf, pw3d, lsp, up3d')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # define path to store extra files
    out_path = cfg.DATASET_NPZ_PATH
    openpose_path = cfg.OPENPOSE_PATH

    do_single_set = args.set_name != None
    if args.train_files:
        if (do_single_set and args.set_name == "up3d") or not do_single_set:
            # up_3d dataset preprocessing (training set)
            up_3d_extract(cfg.UP_3D_ROOT, out_path,'trainval')

        if (do_single_set and args.set_name == "mpi_inf") or not do_single_set:
            # MPI-INF-3DHP dataset preprocessing (training set)
            mpi_inf_3dhp_extract(cfg.MPI_INF_3DHP_ROOT, openpose_path, out_path, 'train', extract_img=False, static_fits=cfg.STATIC_FITS_DIR)

        if (do_single_set and args.set_name == "lsp_original") or not do_single_set:
            # LSP dataset original preprocessing (training set)
            lsp_dataset_original_extract(cfg.LSP_ORIGINAL_ROOT, openpose_path, out_path)

        if (do_single_set and args.set_name == "hr_lspet") or not do_single_set:
            # LSP Extended training set preprocessing - HR version
            hr_lspet_extract(cfg.LSPET_ROOT, openpose_path, out_path)

        if (do_single_set and args.set_name == "mpii") or not do_single_set:
            # MPII dataset preprocessing
            mpii_extract(cfg.MPII_ROOT, openpose_path, out_path)

        if (do_single_set and args.set_name == "coco") or not do_single_set:
            # COCO dataset prepreocessing
            coco_extract(cfg.COCO_ROOT, openpose_path, out_path)

    if args.eval_files:
        if (do_single_set and args.set_name == "h36m") or not do_single_set:
            # Human3.6M preprocessing (two protocols)
            # Todo Insert openpose pose estimation code if needed
            h36m_extract(cfg.H36M_ROOT, out_path, protocol=1, extract_img=True)
            h36m_extract(cfg.H36M_ROOT, out_path, protocol=2, extract_img=False)

        if (do_single_set and args.set_name == "mpi_inf") or not do_single_set:
            # MPI-INF-3DHP dataset preprocessing (test set)
            # Todo Insert openpose pose estimation code if needed
            mpi_inf_3dhp_extract(cfg.MPI_INF_3DHP_ROOT, openpose_path, out_path, 'test')

        if (do_single_set and args.set_name == "pw3d") or not do_single_set:
            # 3DPW dataset preprocessing (test set)
            # Todo Insert openpose pose estimation code if needed
            pw3d_extract(cfg.PW3D_ROOT, out_path)

        if (do_single_set and args.set_name == "lsp") or not do_single_set:
            # LSP dataset preprocessing (test set)
            # Todo Insert openpose pose estimation code if needed
            lsp_dataset_extract(cfg.LSP_ROOT, out_path)

        if (do_single_set and args.set_name == "up3d") or not do_single_set:
            # up_3d dataset preprocessing (training set)
            up_3d_extract(cfg.UP_3D_ROOT, out_path, 'lsp_test')