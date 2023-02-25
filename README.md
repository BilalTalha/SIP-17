ISD: Industry Synthetic Dataset for Part Classification
Introduction
Deep neural networks have shown outstanding performance in automatic industrial part classification. However, training these networks typically requires a large amount of real-world data, which is time-consuming, laborious, and expensive to collect. A potential solution to this problem is to use synthetic data for training. However, synthetic data introduces a domain gap that degrades the performance when tested in real environments. To address this issue, we introduce a new dataset, ISD: Industry Synthetic Dataset, for the Sim2Real challenge in industrial part classification. This dataset contains synthetic and real images from multi-domain industrial scenarios.

Dataset
We assume your data is structured with the following format:

Data
├── train
│   ├── class1
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
├── val
│   ├── class1
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
└── test
    ├── class1
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...

wandb login
Install and login to wandb:

Copy code
pip install wandb
wandb login
Training
Please run the following command to do transfer learning on your custom data:

css
Copy code
python ConvNeXt_train.py --data_path /path/to/Data --output_dir "/path/to/checkpoint.pth"
Choose --output_dir as the path and name you want to choose for the weights file.

Testing/Evaluation
Run the following command according to your data and pre-trained weights:

css
Copy code
python ConvNeXt_test.py --data_path /path/to/Data --weights "/path/to/checkpoint.pth"
Choose --weights as the path to the saved weights file. Note that you should put test images in a folder with the name 'val'.

Credits & How to cite
If you use our dataset or code, please cite our paper:

csharp
Copy code
[insert citation here]
License
This project is licensed under the [insert license here] license.
