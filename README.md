# ISD: Industry Synthetic Dataset for Part Classification

## Introduction
Deep neural networks, with their outperformance, can provide a robust solution for automatic industrial part classification. One drawback of these networks is that they require a large amount of data which is a laborious, time-consuming, and costly process. An alternative is to use synthetic data for training. Synthetic data introduce a domain gap that degrades the performance when tested in real environments. In this paper, we introduce a new dataset, ISD: Industry Synthetic Dataset, for the Sim2Real challenge in industrial part classification. This dataset contains synthetic and real images from multi-domain industrial scenarios. We then evaluate the baseline performance of our dataset with different SOTA supervised and self-supervised neural networks. Our dataset is available at <a href="https://dafd.com/">dataset</a>.
![PDF Image](/Image/data.jpg)

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


To use the dataset, please download the <a href="https://dafd.com/">dataset</a>.

## wandb login
install and login to wandb:

<code>pip install wandb</code>

<code>wandb login</code>

## Training
To train a model, please run the following command:

```
python train.py --data_path /path/to/Data --output_dir "/path/to/checkpoint.pth"
```


Choose `--output_dir` as the path and name you want to choose for the weights file.

## Testing/Evaluation
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

[Paper citation information here]

## License
This project is licensed under the CC0-1.0 License - see the LICENSE.md file for details.

