<!DOCTYPE html>
<html>
<head>
	<title>ISD: Industry Synthetic Dataset for Part Classification</title>
</head>
<body>
	<h1>ISD: Industry Synthetic Dataset for Part Classification</h1>
	<p>Deep neural networks, with their outperformance, can provide a robust solution for automatic industrial part classification. One drawback of these networks is that they require a large amount of data which is a laborious, time-consuming, and costly process. An alternative is to use synthetic data for training. Synthetic data introduce a domain gap that degrades the performance when tested in real environments. In this paper, we introduce a new dataset, ISD: Industry Synthetic Dataset, for the Sim2Real challenge in industrial part classification. This dataset contains synthetic and real images from multi-domain industrial scenarios. We then evaluate the baseline performance of our dataset with different SOTA supervised and self-supervised neural networks. Our dataset is available at https://dafd.com.</p>
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

<h2>Pretrained Models</h2>
<p>Following pretrained weights are available:</p>

<table>
	<tr>
		<th>Network</th>
		<th>Pretrained Weights</th>
	</tr>
	<tr>
		<td>ConvNext</td>
		<td><a href="https://link-to-convnext-weights">link-to-convnext-weights</a></td>
	</tr>
	<tr>
		<td>DINO</td>
		<td><a href="https://link-to-dino-weights">link-to-dino-weights</a></td>
  </tr>

  
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
  <h2>License/h2>
<div>
    <p>[insert license information here]</p>
</div>

