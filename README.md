# Towards Sim-to-Real Industrial Parts Classification with Synthetic Dataset  
This repo contains the source code and dataset for our CVPR 2023 workshop paper:  
<a href="https://openaccess.thecvf.com/content/CVPR2023W/VISION/html/Zhu_Towards_Sim-to-Real_Industrial_Parts_Classification_With_Synthetic_Dataset_CVPRW_2023_paper.html"> Towards Sim-to-Real Industrial Parts Classification With Synthetic Dataset </a>  
CVRP 2023, Workshop on Vision-Based Industrial Inspection  

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

## Dependencies  
To train and evaluate on the ResNet/EfficientNet/ConvNext/VIT models, we developed the code with python version 3.8, pytorch 1.12.1, torchvision 0.13.1, and CUDA 11.0+.  
  
To train and evaluate on the DINO model, we follow the dependancies from the original DINO github reposity, used python version 3.6, pytorch 1.7.1, torchvision 0.8.2, and CUDA 11.0+.  

Please also install pandas, sklean, and matplotlib.

## wandb login
install and login to wandb:

<code>pip install wandb</code>

<code>wandb login</code>

## Training on ResNet/EfficientNet/ConvNext/VIT
To train a superviseed learning model, please run the following command:

```
python train.py --data_path /path/to/Data --output_dir "/path/to/checkpoint.pth --epochs 25 --lr 0.001"
```

Choose `--output_dir` as the path and name you want to choose for the weights file.  
Choose `--epochs` (default is 25), `--lr` (default is 0.001), `--momentum` (default is 0.9), `--step_size` (default is 7), `--gamma` (default is 0.1).  

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

You can find our trained weights for the ConvNeXt model on the 15 objects here: 
- [Trained model of ConvNeXt](https://drive.google.com/drive/folders/1M1AlmrsgbAtfVwHUQf0hiudHImOZmzC_?usp=sharing)

For other weights, please watch this repository for updates or contact us for further information.


## Acknowledgement  
The DINO code used in this project is based on the original implementation from the [DINO](https://github.com/facebookresearch/dino) repository. 

## License
This project is licensed under the CC0-1.0 License - see the LICENSE.md file for details.

## Credits & How to Cite
If you use this dataset in your research, please cite our paper.   

```
@InProceedings{Zhu_2023_CVPR,
    author    = {Zhu, Xiaomeng and Bilal, Talha and M\r{a}rtensson, P\"ar and Hanson, Lars and Bj\"orkman, M\r{a}rten and Maki, Atsuto},
    title     = {Towards Sim-to-Real Industrial Parts Classification With Synthetic Dataset},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {4453-4462}
}
```


