# 网络模块个数
cfg=$1
# 批大小
batch_size=32

# 预训练模型路径
pretrained_model='./saved_models/birds/xxx.pth'
# 进程数
num_workers=8

CUDA_VISIBLE_DEVICES=0 python src/test.py \
                    --cfg $cfg \
                    --batch_size $batch_size \
                    --num_workers $num_workers \
                    --pretrained_model_path $pretrained_model \
