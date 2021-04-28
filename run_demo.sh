#Script for running demo.py
work_path="/ssd2/swheo/db/lg_project/test" #"/ssd2/swheo/db/lg_project/synthetic_test/"
data_name="00" #"Camera000"
op_name="/keypoints/00.jpg_keypoints.json"
echo ${work_path}/images/${data_name}.jpg
echo ${work_path}${op_name}
#python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.jpg --openpose=examples/im1010_openpose.json
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=${work_path}/images/${data_name}.jpg --openpose=${work_path}${op_name} --outfile ${work_path}/SPIN/${data_name}