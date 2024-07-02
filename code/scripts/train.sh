# 网络模块个数
cfg=$1
# 批大小
batch_size=32
# 开始轮次
state_epoch=1
# 预训练模型路径
pretrained_model_path='./saved_models/bird'

# 进程数
num_workers=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/train.py \
                    --cfg $cfg \
                    --batch_size $batch_size \
                    --state_epoch $state_epoch \
                    --num_workers $num_workers \
                    --pretrained_model_path $pretrained_model_path \
