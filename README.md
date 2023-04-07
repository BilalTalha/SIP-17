# Towards Sim-to-Real Industrial Parts Classification with Synthetic Dataset

## Introduction
The **Synthetic Industrial Parts dataset (SIP-17)** is designed for the Sim-to-Real challenge in Industrial Parts Classification.  
  
It comprises 17 objects that represent six typical industry use cases. The first four use cases require the classification of isolated parts and the remaining two require the classification of assembled parts.  
  
For each objects, we provided three types of images: Syn_O, synthetic images without random backgrounds and post-processing; Syn_R, synthetic images with random backgrounds and post-processing; and Real, images captured from cameras in real industrial scenarios.  
  
To facilitate model training and validation, we generated 1,200 synthetic images for each object for training and 300 synthetic images for validation. In total, we have created 33,000 images for both Syn_O and Syn_R. For testing, we captured 566 real images from various industrial scenarios. Our dataset is available at <a href="https://www.kaggle.com/datasets/mandymm/synthetic-industrial-parts-dataset-sip-17">dataset</a>.  
  
To evaluate the performance of the dataset, we benchmarked it using five state-of-the-art models, including ResNet, EfficientNet, ConvNext, VIT, and DINO. Notably, we trained the models only on synthetic data and tested them on real data.
![PDF Image](/Image/image_23.jpg)

## Dataset
We assume your data is structured with the following format:

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


To use the dataset, please download the <a href="https://www.kaggle.com/datasets/mandymm/synthetic-industrial-parts-dataset-sip-17">dataset</a>.

## wandb login
install and login to wandb:

<code>pip install wandb</code>

<code>wandb login</code>

## Training on ResNet/EfficientNet/ConvNext/VIT
To train a superviseed learning model, please run the following command:

```
python train.py --data_path /path/to/Data --output_dir "/path/to/checkpoint.pth"
```

Choose `--output_dir` as the path and name you want to choose for the weights file.

## Testing/Evaluation on ResNet/EfficientNet/ConvNext/VIT
To evaluate a model, please run the following command:

```
python test.py --data_path /path/to/Data --weights "/path/to/checkpoint.pth"
```

Choose <code>--weights</code> as the path to the saved weights file. Note: put test images in a folder named 'val'.

Note: Above commands are applicable for all the models except DINO. Train and test for DINO are given below.

## Training and Testing for DINO

Run following command to train the DINO Model with pretrained backbone: 
```
python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py --patch_size 8 --pretrained_weights /path/to/pretrained/backbone/weights --epochs 50 --batch_size_per_gpu 24 --num_labels 4 --data_path /path/to/Data/Data --output_dir /path/to/save/output  
```
Choose --patch size according to the dowloaded pretrained weights (<a href="https://github.com/facebookresearch/dino">download weights</a>). If weights are for ViT-S/16 or ViT-B/16 then select --patch_size 16. If weights are for ViT-S/8 or ViT-B/8 then select --patch_size 8.  
Choose --arch according to the downloaded pretrained wrights. If weights are for ViT-S then select --arch vit_small. If weights are for ViT-B then select --arch vit_base. 


Run following command according to your data and pre-trained weights: 
```
python eval_linear.py --evaluate --arch vit_small --patch_size 16 --batch_size_per_gpu 50 --pretrained_weights /path/to/pretrained/backbone/weights --pretrained_weights_linear /path/to/pretrained/linear_classifier/weights --num_labels 4 --data_path /path/to/Data
```
Note: put test images in folder with name 'val'

## Pretrained Weights
Pretrained weights for each use case and network are coming soon!

## Credits & How to Cite
If you use this dataset in your research, please cite our paper:

[PAPER CITATION HERE]

The DINO code used in this project is based on the original implementation [1] by Caron et al. [2]. 

[1] https://github.com/facebookresearch/dino

[2] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

## License
This project is licensed under the CC0-1.0 License - see the LICENSE.md file for details.

