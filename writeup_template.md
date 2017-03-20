# **Traffic Sign Recognition** 


## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/original_distribution.png "data class distribution"
[image2]: ./writeup/nopassing3.5_proc.png "Example1"
[image3]: ./writeup/speed100_proc.png "Example2"
[image4]: ./writeup/balanced_distribution.png "balanced data class distribution"
[image5]: ./writeup/Speed70_aug.png "Augmented Example1"
[image6]: ./writeup/Traffic_signals_aug.png "Augmented Example2"

[image7]: ./writeup/sample_sign1.png "Traffic Sign 1"
[image8]: ./writeup/sample_sign2.png "Traffic Sign 2"
[image9]: ./writeup/sample_sign3.png "Traffic Sign 3"
[image10]: ./writeup/sample_sign4.png "Traffic Sign 4"
[image11]: ./writeup/sample_sign5.png "Traffic Sign 5"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/lcfgr/CarND-Traffic-Sign-Classifier-Project)
---

### Data Set Summary & Exploration

#### 1. The submission includes a basic summary of the data set.

The code for this step is contained in the **2nd** code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. The submission includes an exploratory visualization on the dataset.

The code for this step is contained in the **4th** code cell of the IPython notebook.  

Below we can see how the training data are distributed in the different classes/labels.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. The submission describes the preprocessing techniques used and why these techniques were chosen.

The Pierre Sermanet and Yann LeCun paper accomplished up to 99.17% accurancy in their Traffic Sign Recognition set. Their results and suggestions are:
* MS architecture outperforms SS architecture most of the time
* Use of sophisticated non-linear modules is better than the traditional rectified sigmoid (tanh). They used a pointwised tanh, followed by a subtractive local normalization and a divisive local normalization
* Surprisingly,using no color is often better than using color
* They created a jittered dataset using randomly pertubed in position ([-2,2] pixels), in scale ([0.9,1.1]ratio) and rotation ([15,15] degrees).
* They suggest also affine transformations, brighness, contrast and blur.

By producing a random sampling of 10 images of every class of the training set we can see that the images have multiple features that can cause problem in the image recognition.
Some of these are: low brightness, blur, rotated, zoomed in/out, not centered. Also the training set is not well balanced.

Several pre-processing techniques where tested, including grayscaling, various types of sharpness filters, enhanching the training set with rotated, zoomed in/out and translated images.

Various combinations of the above were tried into a pipeline. To a suprise, most of them did not actually improve the accurancy. The only one that had significant difference was the sharpness filter.

In the end, always the images' pixel values are normalized (values between 0 and 1) so that the neural network can converge and have better accuracy.

The code with the pre-processing filters is contained in the **5th, 6th and 11th** code cell of the IPython notebook.

Below are two examples of the final image processing:  

![alt text][image2]![alt text][image3]


Furthermore a more balanced set was created. The classes/labels do not contain the same ammount of traffic signs. Therefore additional data were generated. In this procedure, a random image was picked each time from each class and was transformed in the ways mentioned  in the Sermanet & LeCun Paper, until all classes were balanced.
	
Below is the balanced distribution of the samples:  

![alt text][image4]

* The size of the balanced training set is 86430


The ** 8th ** code cell of the IPython notebook contains the code for augmenting the data set. Below are two original and  augmented images:

![alt text][image5] ![alt text][image6]

The added images have a random rotation (range 30 degrees), a random translation (range 4 pixels) and a random scale (range 10%) 


####2. The submission provides details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

The code for my final model is located in the ** 13th** cell of the ipython notebook. 

My final model consisted of the following layers:


| Layer         				   	|     Description	        					|
|-------------------------------|---------------------------------------------|
| Input         	|				| 32x32x3 RGB image   							|
| Convolution 1 5x5 | 		 		| 1x1 stride, valid padding, outputs 28x28x80 	|
| RELU 1			|				|	   											|
| Max pooling 1	    |  	   			| 2x2 stride,  outputs 14x14x80 			    |
| Convolution 2 5x5	|       		| 1x1 stride, valid padding, outputs 10x10x80   |
| RELU 2			|				| 	   											|
| max pooling 2		|				| 2x2 stride,  outputs 14x14x80 	   			|
|                   |Max pooling 1b	| 2x2 stride,  outputs 7x17x80 	   				|
|Flatten			|Flatten		| Combine Conv1 and Conv2 Layers, 5920 Outputs  |
| Fully connected 1	|	   			| Outputs 100 									|
| RELU 3			|	   			| 	        									|
| Dropout 1			|	   			| Keep 75%										|
| Fully connected 2	|	   			| Outputs 100  									|
| RELU 4			|	   			| 	        									|
| Dropout 2			|	   			| Keep 80%     									|
| Fully connected 3	|	   			| Outputs 48  									|
|	Softmax			|				|												|
 
The architecture is the following:
We have two convolutional layers, each one with a Relu and max pooling. The output of the first layer is the input of the second layer (after the Relu and max pooling), but is also the input of a second max pooling layer (1b). Then The output of this second max pooling layer (1b) is combined with the output of the second convolutional layer (after the Relu and max pooling). Then the Flattened layer is inputed to 3 full connected layers. The first 2 full connected layers have a Relu and a dropout layer while the last full connected layer is led to the Softmax output.



#### 3. The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.

The code for training the model is located in the **14th** cell of the ipython notebook. 

To train the model, I used an Adam Optimizer. The optimization target was the reduction of the average cross entropy. In the final model, the learning rate of 0.01(default) was used, but I found better the value epsilon =0.1. This value produced slower but more stable converge for the specific model architecture.
The batch size was 128.
The epochs were 200, although from 100 epochs and later the network is almost static.

#### 4. The submission describes the approach to finding a solution. Accuracy on the validation should be 0.93 or greater.

The code for calculating the test accuracy of the model is located in the **15th** cell of the Ipython notebook.

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.961 
* test set accuracy of 0.954

First I began with a classic LeNet architecture.
The final model was based on the Pierre Sermanet and Yann LeCun model architecture.

The architecture was adjusted countless times to be able to minimize the overfitting while maximizing the accuracy. The training and validation accurancy is printed in every step to gain fast understanding of the behaviour of every adjustment.
The dropout layers are very critical to the fine-tuning of the model. It would be probably best if I tried also more complex normalization functions.

The final accuracy is acceptable. The following 2 things should be noted:
1. The results were simular with color and B/W image processing, therefore I chose to keep the color model, since the input contains more information and may lead to better accuracy in data not part of the training/validation sets.
2. The enhanced "balanced" set did not produce better results, even with different architectures (bigger neural networks to "fit" the more data). 

 
### Test a Model on New Images

#### 1. The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to any particular qualities of the images or traffic signs in the images that may be of interest, such as whether they would be difficult for the model to classify.
ose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7]
![alt text][image8] 
![alt text][image9] 
![alt text][image10]
![alt text][image11]

These traffic signs are relatively easy to classify, since there are no objects in front of them and the lighting is good. However all of the images have different camera angles and the original resolutions of the images are not always squared (this can be seen more clearly in the latest, stop sign which seems distorted also in the naked eye). We expect relatively good results.

#### 2. The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.

 Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the **22th** cell of the Ipython notebook.


Here are the results of the prediction:  

Image|Prediction
 --------------------------------------------|-------------------------------------- 
 End of all speed and passing limits      		| End of all speed and passing limits   		
 No entry     									| No entry										
 Priority road									| Priority road									
 Right-of-way at the next intersection      	| Right-of-way at the next intersection			
 Stop											| Stop      									


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.4% . However the particular signs are relatively good images. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for the softmax probabilities is located in the **23rd** cell of the Ipython notebook.

The first image propabilities:


| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| End of all speed and passing limits			|
| 0     				| End of no passing 							|
| 0						| Priority road									|
| 0		      			| End of speed limit (80km/h)	 				|
| 0					    | Road work      								|


The second image propabilities:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| No entry	 									|
| 0     				| Stop 											|
| 0						| No Passing									|
| 0	      				| Yield							 				|
| 0				  		| Priority Road      							|

For the third image ... 

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Priority Road 								|
| 0     				| No entry 										|
| 0						| Roundabout mandatory							|
| 0	  	    			| No passing					 				|
| 0					    | Vehicles over 3.5 metric tons prohibited		|

For the fourth image ... 

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Right-of-way at the next intersection			|
| 0     				| Beware of ice/snow							|
| 0						| Slippery road									|
| 0	     	 			| Double curve					 				|
| 0					    | Pedestrials	      							|

For the fifth image ... 

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Stop sign   									|
| 0    					| Bicycles crossing								|
| 0						| Speed limit (30km/h)							|
| 0	      				| Speed limit (20km/h)			 				|
| 0					    | Speed limit (50km/h) 							|


All the images are precisely predicted. This is probably to the fact that there are no obstacles or very bad conditions, therefore our training set was enough for these relatively normal traffic signs.
