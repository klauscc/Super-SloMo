#================================================================
#   God Bless You. 
#   
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2020/03/31
#   description: 
#
#================================================================

python train_ode.py  \
    --dataset_root ~/stor6/dataset/processed_dataset/video-interpolation/adobe240fps \
    --checkpoint_dir ~/stor5/workspace/0331-video-interpolation/ode-slomo-euler-multiplicative \
    --epochs 200 \
    --train_batch_size 6\
    --validation_batch_size 12 \
    --ode_method euler \
    --lr_scheduler exp \
    #--dataset_root /tmp/fengcheng/dataset/processed_dataset/video-interpolation/adobe240fps \
