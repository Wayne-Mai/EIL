:<<'END'
This is sample bash script for CUB-200-2011 dataset
support model: vgg
END

gpu=3,5,6,7
name=vgg_eil_beta
epoch=80
decay=60
model=vgg16
server=tcp://127.0.0.1:12345
batch=64
wd=1e-4
lr=0.001
data_root="/data/wayne/CUB_200_2011/"
cam_thr=0.15
mode='base'
# notice plot

CUDA_VISIBLE_DEVICES=${gpu} \
python train_level_1.py -a ${model} --dist-url ${server} \
    --multiprocessing-distributed --world-size 1 --pretrained\
    --data ${data_root} --dataset CUB \
    --train-list datalist/CUB/train.txt \
    --test-list datalist/CUB/test.txt \
    --data-list datalist/CUB/ --bbox-mode DANet\
    --task wsol_eval --cam-thr=${cam_thr} --mode=${mode}\
    --batch-size ${batch} --epochs ${epoch} --LR-decay ${decay} \
    --wd ${wd} --lr ${lr} --nest --name ${name} 
