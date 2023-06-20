DATA_ROOT=/kaggle/input/pascalvoc12/PascalVOC12/
DATASET=voc
TASK=10-1
EPOCH=10
BATCH=16
LOSS=bce_loss
LR=0.01
THRESH=0.7
MEMORY=100

python main.py --data_root ${DATA_ROOT} --model deeplabv3_resnet101_separable --gpu_id 0 --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --overlap --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH} --freeze  --bn_freeze  \
    --unknown --w_transfer --amp --mem_size ${MEMORY}
    