# EfficientNet
 Sim-to-Real Baseline

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


Please run following command to do transfer learning on your custom data: 
```
python EfficientNet_train.py --data_path /path/to/Data --output_dir "/path/to/checkpoint.pth" 
```
Choose --output_dir as path and name you want to choose for weights's file

## Testing/Evaluation

Run following command according to your data and pre-trained weights: 
```
python EfficientNet_test.py --data_path /path/to/Data --weights "/path/to/checkpoint.pth"
```
Choose --weights as path to saved weights's file
Note: put test images in folder with name 'val'
