# ISD: Industry Synthetic Dataset for Part Classification

## Introduction
Deep neural networks, with their outperformance, can provide a robust solution for automatic industrial part classification. One drawback of these networks is that they require a large amount of data which is a laborious, time-consuming, and costly process. An alternative is to use synthetic data for training. Synthetic data introduce a domain gap that degrades the performance when tested in real environments. In this paper, we introduce a new dataset, ISD: Industry Synthetic Dataset, for the Sim2Real challenge in industrial part classification. This dataset contains synthetic and real images from multi-domain industrial scenarios. We then evaluate the baseline performance of our dataset with different SOTA supervised and self-supervised neural networks. Our dataset is available at <a href="https://dafd.com/">dataset</a>.
![PDF Image](/Images/dataset.jpg)

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


To use the dataset, please download it from https://dafd.com and structure your data in the above format.

## wandb login
nstall and login to wandb:

<code>pip install wandb</code>

<code>wandb login</code>

## Training
To train a model, please run the following command:

<code>python train.py --data_path /path/to/Data --output_dir "/path/to/checkpoint.pth"</code>


Choose `--output_dir` as the path and name you want to choose for the weights file.

## Testing/Evaluation
To evaluate a model, please run the following command:

<code>python test.py --data_path /path/to/Data --weights "/path/to/checkpoint.pth"</code>


Choose <code>--weights</code> as the path to the saved weights file. Note: put test images in a folder named 'val'.

## Credits & How to Cite
If you use this dataset in your research, please cite our paper: 

[Paper citation information here]

## License
This project is licensed under the [license name] License - see the LICENSE.md file for details.

