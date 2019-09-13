# **Traffic Sign Recognition** 

## Writeup

**Traffic Sign Recognition Project**

The goals of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/frequency_per_class.png "Visualization"
[image2]: ./web_data/class_3.jpg "Traffic Sign 1"
[image3]: ./web_data/class_4.jpg "Traffic Sign 2"
[image4]: ./web_data/class_17.jpg "Traffic Sign 3"
[image5]: ./web_data/class_25.jpg "Traffic Sign 4"
[image6]: ./web_data/class_33.jpg "Traffic Sign 5"

## Rubric Points

### Writeup / README

You're reading it!

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images,
* The size of the validation set is 4410 images,
* The size of test set is 12630 images,
* The shape of a traffic sign image is 32x32 pixels
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed per class.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

As a first step, I decided to convert the images to grayscale because it reduce the number of parameters that I have to train and for this kind of project is not absolutely necessary.

As a second step, I normalized the image data using this equation `(pixel - 128)/ 128`, this is a quick way to approximately normalize the data, the normalization helps to optimize the learning process, helping the optimization to be faster.

As a last step I reshape the data, getting it ready for the tranning process, this is to get the arrays in shape for the convolutions in the CNN.

It would be a natural next step to generate more data this due to the imbalanced number of images per class, but in this project I got a great result without doing it.


#### 2. Model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten				| Output 1x400				|
| Fully connected		| Output 1x120        									|
| RELU					|												|
| Dropout				|
| Fully connected		| Output 1x84        									|
| RELU					|												|
| Dropout				|
| Fully connected		| Output 1x43        									|
| Softmax				|         									|



#### 3. Training the algorithm

To train the model, I used an **adam optimizer**, a **batch size** of 128
and 30 **epochs** and use this hyperparameters:

* Learning Rate: 0.001
* Keep Probability: 0.6

#### 4. Finding a 95% accuracy in the validation set.

My final model results were:
* training set accuracy of: 99.8%
* validation set accuracy of: 95.2%
* test set accuracy of: 93.4%

First I use the LeNet architecture in the same way we used it in the MNIST dataset, but I notice a 89% in validation accuracy, but always noticing on average a 10% better behavior in the training data than in the test data, then I sense that there is overfitting in my neural network, so I used dropout in the two fully connected layers during training with a 0.6 keep probability, this help a lot in the final outcome.

I tuned the number of epochs because I was still getting better results with more epochs, and leave it in a good accuracy output.
 

### Testing the Model on New Images

#### 1. Another german traffic signs.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

#### 2. Discuss the model's predictions on these new traffic signs.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 60km/h  	| Speed limit 60km/h   									| 
| Speed limit 70km/h    | Speed limit 70km/h 										|
| No entry				| No entry											|
| Road work	      		| Road work					 				|
| Turn right ahead		| Ahead only      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.9% taking in account that the sample is only of 5 images, but that error is an important error for the people in the self driving car, because it can cause an accident.

#### 3. Certain of the model

For the first image, the model is sure that this is a 60 km/h limit sign (probability of 0.78), and the image does contain a 60 km/h limit sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .78         			| Speed limit 60km/h   									| 
| .17     				| Speed limit 80km/h 										|
| .04					| Speed limit 50km/h											|
| 1x10^-5^    			| Speed limit 100km/h					 				|
| 1x10^-11^			    | Wild animals crossing      							|


But for the last image where I got the false prediction, the model output a Ahead only sign, and the image really contain a Turn right ahead sign, throwing the next softmax probabilities:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .33         			| Ahead only 									| 
| .28     				| Turn left ahead										|
| .12					| Beware of ice/snow											|
| .09  		  			| Turn right ahead					 				|
| .04				    | Children crossing      							|

Here we can conclude a lot of things, first the maximum probability is 33%, comparing with the first image it implies that the model wasn't so sure about a prediction, second in the sum 3 most likely predictions of the first image I got a 99% of certainty, but in the sum of the 5 most likely predictions in the last image I only got 86%, that's another sign that the model wasn't so sure, third it's important to understand that the amount of traning images for the turn right ahead sign is almost the half of the most probably one (ahead only sign), where I can get a reason for the bad prediction, but the second most probably prediction has less training images than the ground truth.