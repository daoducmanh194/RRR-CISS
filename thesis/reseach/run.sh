DATA_ROOT=/kaggle/input/pascalvoc12/pascalvoc12_new/
DATASET=voc
TASK=15-1
EPOCH=10
BATCH=16
LOSS=bce_loss
LR=0.01
THRESH=0.9
MEMORY=0

python main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101 --gpu_id 0 --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH} --freeze  --bn_freeze  \
    --unknown --w_transfer --amp --mem_size ${MEMORY}
    