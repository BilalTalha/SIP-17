1. Train on our vanilla 16
python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py --patch_size 16 --pretrained_weights /Midgard/home/mzhu/code/DINO/Sim-to-Real/Output_withoutpretrain/checkpoint.pth --epochs 50 --batch_size_per_gpu 24 --num_labels 16 --data_path /Midgard/Data/mzhu/DINO/Final_Dataset_Clean --output_dir /Midgard/home/mzhu/code/DINO/Sim-to-Real/Output_pretrained_vanilla16​

2. Train on pretrain ResNet50
python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py --arch resnet50 --pretrained_weights /Midgard/Data/mzhu/DINO/Pretrained_models_FB/dino_resnet50_pretrain.pth --epochs 50 --batch_size_per_gpu 24 --num_labels 16 --data_path /Midgard/Data/mzhu/DINO/Final_Dataset_Clean --output_dir /Midgard/home/mzhu/code/DINO/Sim-to-Real/Output_pretrained_resnet50​

3. Train on pretrain xcit_medium_24_p8
python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py --arch xcit_medium_24_p8 --pretrained_weights /Midgard/Data/mzhu/DINO/Pretrained_models_FB/dino_xcit_medium_24_p8_pretrain.pth --epochs 50 --batch_size_per_gpu 24 --num_labels 16 --data_path /Midgard/Data/mzhu/DINO/Final_Dataset_Clean --output_dir /Midgard/home/mzhu/code/DINO/Sim-to-Real/Output_pretrained_xcit_medium_24p8​

4. Try to train on vit_s8 with adjust global_crops_scle and local crops_scle?

5. Train imagenet on vanilla with adjust global_crops_scle and local crops_scle? 

####vanilla trainings on our dataset
###VIT_SMALL
##patch8
#With Default parameters (we did already i guess)
python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch 'vit_small' --patch_size 8 --data_path /Midgard/Data/mzhu/DINO/Final_Dataset_Clean/train --norm_last_layer True --warmup_teacher_temp 0.04 --teacher_temp 0.04 --use_fp16 True --batch_size_per_gpu 64 --epochs 100 --freeze_last_layer 1 --lr 0.0005 --optimizer 'adamw' --global_crops_scale (0.4, 1.) --local_crops_number 8 --local_crops_scale (0.05, 0.4) --output_dir /Midgard/home/mzhu/code/DINO/Sim-to-Real/Output_vanilla_vit_S8_default

##important##
#With increase in crop area

srun --gres=gpu:4 --mem=40GB --cpus-per-task=4 --constrain=khazadum --mail-use=xiazhu@kth.se --mail-type=BEGIN,END,FAIL --output ./output/%J.out --error ./output/%J.err python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch vit_small --patch_size 8 --batch_size_per_gpu 4 --data_path /Midgard/Data/mzhu/DINO/Final_Dataset_Clean/train --norm_last_layer True --warmup_teacher_temp 0.04 --teacher_temp 0.04 --use_fp16 True --batch_size_per_gpu 64 --epochs 100 --freeze_last_layer 1 --lr 0.0005 --optimizer 'adamw' --global_crops_scale 0.8 1 --local_crops_number 8 --local_crops_scale 0.4 1 --output_dir /Midgard/home/mzhu/code/DINO/Sim-to-Real/Output_vanilla_vit_s8_increaseCropArea

python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch 'vit_small' --patch_size 8 --data_path /Midgard/Data/mzhu/DINO/Final_Dataset_Clean/train --norm_last_layer True --warmup_teacher_temp 0.04 --teacher_temp 0.04 --use_fp16 True --batch_size_per_gpu 64 --epochs 100 --freeze_last_layer 1 --lr 0.0005 --optimizer 'adamw' --global_crops_scale (0.14, 1.) --local_crops_number 8 --local_crops_scale (0.4, 0.1) --output_dir /Midgard/home/mzhu/code/DINO/Sim-to-Real/Output_vanilla_vit_s8_increaseCropArea
#Without multi-crop strategy
python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch 'vit_small' --patch_size 8 --data_path /Midgard/Data/mzhu/DINO/Final_Dataset_Clean/train --norm_last_layer True --warmup_teacher_temp 0.04 --teacher_temp 0.04 --use_fp16 True --batch_size_per_gpu 64 --epochs 100 --freeze_last_layer 1 --lr 0.0005 --optimizer 'adamw' --global_crops_scale (0.14 1) --local_crops_number 0 --local_crops_scale (0.05, 0.4) --output_dir /Midgard/home/mzhu/code/DINO/Sim-to-Real/Output_vanilla_vit_S8_noCrop
##important##

#using author's recommendation for better accuracy
python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch 'vit_small' --patch_size 8 --data_path /Midgard/Data/mzhu/DINO/Final_Dataset_Clean/train --norm_last_layer False --warmup_teacher_temp 0.04 --teacher_temp 0.07 --warmup_teacher_temp_epochs 30 --use_fp16 True --batch_size_per_gpu 64 --epochs 300 --freeze_last_layer 1 --lr 0.0005 --optimizer 'adamw' --global_crops_scale (0.4, 1.) --local_crops_number 8 --local_crops_scale (0.05, 0.4) --output_dir /Midgard/home/mzhu/code/DINO/Sim-to-Real/Output_vanilla_vit_S8_bestParam 

##patch16
#With Default parameters
python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch 'vit_small' --patch_size 16 --data_path /Midgard/Data/mzhu/DINO/Final_Dataset_Clean/train --norm_last_layer True --warmup_teacher_temp 0.04 --teacher_temp 0.04 --use_fp16 True --batch_size_per_gpu 64 --epochs 100 --freeze_last_layer 1 --lr 0.0005 --optimizer 'adamw' --global_crops_scale (0.4, 1.) --local_crops_number 8 --local_crops_scale (0.05, 0.4) --output_dir /Midgard/home/mzhu/code/DINO/Sim-to-Real/Output_vanilla_vit_S16_default  

###VIT_BASE
#With Default parameters
python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch 'vit_base' --patch_size 8 --data_path /Midgard/Data/mzhu/DINO/Final_Dataset_Clean/train --norm_last_layer True --warmup_teacher_temp 0.04 --teacher_temp 0.04 --use_fp16 False --batch_size_per_gpu 64 --epochs 100 --freeze_last_layer 1 --lr 0.0005 --optimizer 'adamw' --global_crops_scale (0.4, 1.) --local_crops_number 8 --local_crops_scale (0.05, 0.4) --output_dir /Midgard/home/mzhu/code/DINO/Sim-to-Real/Output_vanilla_vit_B8_default  

###RESNET
#With Default parameters
python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch 'resnet50' --data_path /Midgard/Data/mzhu/DINO/Final_Dataset_Clean/train --batch_size_per_gpu 64 --lr 0.03 --optimizer 'sgd' --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale (0.14, 1.) --local_crops_number 8 --local_crops_scale (0.05 0.14) /Midgard/home/mzhu/code/DINO/Sim-to-Real/Output_vanilla_ResNet_default  
#Without multi-crop strategy
python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch 'resnet50' --data_path /Midgard/Data/mzhu/DINO/Final_Dataset_Clean/train --batch_size_per_gpu 64 --lr 0.03 --optimizer 'sgd' --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale (0.14, 1.) --local_crops_number 0 --local_crops_scale (0.05 0.14) /Midgard/home/mzhu/code/DINO/Sim-to-Real/Output_vanilla_ResNet_noCrop  


#Imagenet path: "/local_storage/datasets/imagenet/train"
####vanilla trainings on imagenet
