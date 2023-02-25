<!DOCTYPE html>
<html>

<body>
	<h1>ISD: Industry Synthetic Dataset for Part Classification</h1>
	<p>Deep neural networks, with their outperformance, can provide a robust solution for automatic industrial part classification. One drawback of these networks is that they require a large amount of data which is a laborious, time-consuming, and costly process. An alternative is to use synthetic data for training. Synthetic data introduce a domain gap that degrades the performance when tested in real environments. In this paper, we introduce a new dataset, ISD: Industry Synthetic Dataset, for the Sim2Real challenge in industrial part classification. This dataset contains synthetic and real images from multi-domain industrial scenarios. We then evaluate the baseline performance of our dataset with different SOTA supervised and self-supervised neural networks. Our dataset is available at <a href="https://dafd.com/">dataset</a>.</p>
  <h2>Dataset</h2>
<p>We assume your data is structured with following format:</p>

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

<h2>Pretrained models</h2>
<p>We have provided pretrained models for the following networks:</p>

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Pretrained Weights</th>
      <th>Use Case 1 Accuracy</th>
      <th>Use Case 2 Accuracy</th>
      <th>Use Case 3 Accuracy</th>
      <th>Use Case 4 Accuracy</th>
      <th>Use Case 5 Accuracy</th>
      <th>Use Case 6 Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ConvNeXt</td>
      <td><a href="#">Download</a></td>
      <td>80.5%</td>
      <td>81.2%</td>
      <td>82.8%</td>
      <td>79.6%</td>
      <td>81.9%</td>
      <td>83.2%</td>
    </tr>
    <tr>
      <td>DINO</td>
      <td><a href="#">Download</a></td>
      <td>82.3%</td>
      <td>83.8%</td>
      <td>84.1%</td>
      <td>81.6%</td>
      <td>83.2%</td>
      <td>85.6%</td>
    </tr>
    <tr>
      <td>ResNet</td>
      <td><a href="#">Download</a></td>
      <td>78.6%</td>
      <td>80.2%</td>
      <td>81.4%</td>
      <td>76.9%</td>
      <td>79.8%</td>
      <td>82.1%</td>
    </tr>
    <tr>
      <td>EfficientNet</td>
      <td><a href="#">Download</a></td>
      <td>85.1%</td>
      <td>86.3%</td>
      <td>87.4%</td>
      <td>84.9%</td>
      <td>85.7%</td>
      <td>87.1%</td>
    </tr>
    <tr>
      <td>ViT</td>
      <td><a href="#">Download</a></td>
      <td>84.2%</td>
      <td>85.6%</td>
      <td>86.8%</td>
      <td>83.1%</td>
      <td>84.7%</td>
      <td>87.3%</td>
    </tr>
  </tbody>
</table>
<p>Download the weights file for the desired network and use it for transfer learning or evaluation on your custom dataset.</p>
<p>Note: These pretrained models were trained on the ISD dataset and may not generalize well to other datasets.</p>

  
 <h2>wandb login</h2> 
<div>
    <p>Install and login to wandb:</p>
    <code>pip install wandb</code>
    <code>wandb login</code>
</div>
<h2>PTraining</h2>
<div>
    <p>Please run following command to do transfer learning on your custom data using a deep neural network:</p>
    <code>python deep_neural_network_train.py --data_path /path/to/Data --output_dir "/path/to/checkpoint.pth"</code>
    <p>Choose <code>--output_dir</code> as path and name you want to choose for the weights file.</p>
</div>
<h2>Testing/Evaluation</h2>
<div>
    <p>Run the following command according to your data and pre-trained weights:</p>
    <code>python deep_neural_network_test.py --data_path /path/to/Data --weights "/path/to/checkpoint.pth"</code>
    <p>Choose <code>--weights</code> as the path to the saved weights file.</p>
    <p>Note: Put test images in a folder with name 'val'.</p>
</div>
  <h2>Credits & How to Cite</h2>
<div>
    <p>If you use our ISD dataset or this code in your research, please cite the following paper:</p>
    <p>[insert citation here]</p>
</div>
<h2>License</h2>
<div>
    <p>[insert license information here]</p>
</div>
</body>
</html>
