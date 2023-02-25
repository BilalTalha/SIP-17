# Sim-to-Real
 This repository contain files for the thesis "Sim-to-Real: Domain Adaptation for Industrial Parts Inspection using CAD data"

## Dataset
We assume your data is structured with following format:

<ul>
<li>Data
<ul>
<li>train
<ul>
<li>class1
<ul>
<li>image1.jpg</li>
<li>image2.jpg</li>
<li>...</li>
</ul>
</li>
<li>class2
<ul>
<li>image1.jpg</li>
<li>image2.jpg</li>
<li>...</li>
</ul>
</li>
<li>...</li>
</ul>
</li>
<li>val
<ul>
<li>class1
<ul>
<li>image1.jpg</li>
<li>image2.jpg</li>
<li>...</li>
</ul>
</li>
<li>class2
<ul>
<li>image1.jpg</li>
<li>image2.jpg</li>
<li>...</li>
</ul>
</li>
<li>...</li>
</ul>
</li>
<li>test
<ul>
<li>class1
<ul>
<li>image1.jpg</li>
<li>image2.jpg</li>
<li>...</li>
</ul>
</li>
<li>class2
<ul>
<li>image1.jpg</li>
<li>image2.jpg</li>
<li>...</li>
</ul>
</li>
<li>...</li>
</ul>
</li>
</ul>
</li>
</ul>

### wandb login
install and login to wandb
```
pip install wandb
wandb login
```


## Training

### Train Backbone
Please install python version 3.6, PyTorch version 1.7.1, CUDA 11.0, timm and torchvision 0.8.2. Please run following command to train the DINO Model on your custom data: 
```
python main_dino.py --arch vit_small --batch_size_per_gpu 24 --data_path /path/to/Data/train --output_dir /path/to/save/output
```

### Train on Pretrained Backbone
Please run following command to train the DINO Model on your custom data with pretrained backbone: 
```
python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py --patch_size 8 --pretrained_weights /path/to/pretrained/backbone/weights --epochs 50 --batch_size_per_gpu 24 --num_labels 4 --data_path /path/to/Data/Data --output_dir /path/to/save/output  
```
Choose --patch size according to the dowloaded pretrained weights. If weights are for ViT-S/16 or ViT-B/16 then select --patch_size 16. If weights are for ViT-S/8 or ViT-B/8 then select --patch_size 8.  
Choose --arch according to the downloaded pretrained wrights. If weights are for ViT-S then select --arch vit_small. If weights are for ViT-B then select --arch vit_base. 

## Testing/Evaluation

Run following command according to your data and pre-trained weights: 
```
python eval_linear.py --evaluate --arch vit_small --patch_size 16 --batch_size_per_gpu 50 --pretrained_weights /path/to/pretrained/backbone/weights --pretrained_weights_linear /path/to/pretrained/linear_classifier/weights --num_labels 4 --data_path /path/to/Data
```
Note: put test images in folder with name 'val'
