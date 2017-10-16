# **Behavioral Cloning** 

![alt text][image1]

# Project Writeup 

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior.
* Experiment with data transformation / CV (preprocessing) 
* Experiment with data augmentation techniques 
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./examples/img_original.png "Model Visualization"
[image2]: ./examples/img_flipped.png "Grayscaling"
[image3]: ./examples/img_brighness_hsv.png "Recovery Image"
[image4]: ./examples/img_left.png "Recovery Image"
[image5]: ./examples/img_right.png "Recovery Image"
[image6]: ./examples/img_orig_dist.png "Normal Image"
[image7]: ./examples/img_trunc_dist.png "Flipped Image"
[image8]: ./examples/img_balanced_dist.png "Flipped Image"
[image9]: ./examples/img_learning_curve.png "Flipped Image"

### 1. Relevant Files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 video of the car driving on track 1 in autonomous mode 
* writeup.md report summarizing the results

### 2. Solution Design Approach

The overall goal was to build a model capable of driving the car at a higher speeds and staying on the track. 

With data for 2 laps, My first step was to use a convolution neural network model similar to the LeNet architecture. I wanted to see how well this performs before choosing a more complex model. When the LeNet model did not do so well, I implemented the NVIDIA reference model and handled overfitting. The final step was to run the simulator to see how well the car was driving around track one. At the end of the process, the vehicle is able to drive autonomously around the track but went off the track in a couple of situations. It was tremendous improvement. 

To build a more robust model, I realized the data was not well balanced. Also augmenting the data with image processing could provide more data for the model to train on. I used the sample data provided. In addition, I generated training data for 3 laps and 1 lap in reverse track. 

Lastly, to handle the large dataset, I designed a generator that provides batches of data during training and validation. This helps not having to load all images into memory at one. 

### 3. Data Generation and  Preprocessing 

I used the sample data provided. In addition, I generated training data for 3 laps and 1 lap in reverse track using the simulator. 

For every observation, I had  
* The center camera image 

![alt text][image1]

* The left camera image and used a small correction of -0.2 

![alt text][image4]

* The right camera image used a small correction of 0.2 

![alt text][image5]


And for every image, 
* I generated the flipped image 

![alt text][image2]

* brighness change and change to HSV color space 

![alt text][image3]

Therefore, for every center camera images, I generate 11 other images that could be using for training 

Looking at the distribution, the data is highly imbalanced. 

![alt text][image6]

To deal with this, I limited the number of training sample for each steering angle 

![alt text][image7]

And when using the flipped images, the distribution is lot more balanced

![alt text][image8]

I first normalized the data to zero mean and SD of 0.5.I cropped top 75 pixels and bottom 20 pixels to just retain the view of the road thereby reducing the dimensionality 

I finally randomly shuffled the data set and put 25% of the data into a validation set. 

### 4. Model Architecture and Training Strategy

I used MSE to optimize both the below model using Adam Optimizer with validation split of 20%. I experimented with batch sizes of 64 and 128. I cutoff the epochs when the validation error stopped improving or started increasing. 

I first experimented with LeNet architecture. It consisted of
* Convolution Layer with Filter Depth of 32 with Kernel Size 3x3 with ReLU activations 
* Max Pooling with Pool Size of 2x2 
* Convolution Layer with Filter Depth of 64 with Kernel Size 3x3 with ReLU activations
* Max Pooling with Pool Size of 2x2 
* FC Layer 120 units 
* FC Layer 80 units 
* Output 1 unit 

I added dropouts to counter overfitting. The model achieved at validation loos 0.07. However, the when testing in the simulator, the car ran off the road after covering some distance. 

The second model was based on NVIDIA model for autonomous driving. It consisted of the following: 
* Convolution Layer with Filter Depth of 24 with Kernel Size 5x5 with ReLU activations 
* Convolution Layer with Filter Depth of 36 with Kernel Size 5x5 with ReLU activations 
* Convolution Layer with Filter Depth of 48 with Kernel Size 5x5 with ReLU activations 
* Convolution Layer with Filter Depth of 64 with Kernel Size 3x3 with ReLU activations
* Convolution Layer with Filter Depth of 128 with Kernel Size 3x3 with ReLU activations
* FC Layer 128 units 
* FC Layer 64 units 
* FC Layer 8 units 
* Output 1 unit 
 
Looking at the learning curve, I noticed the model was overfitting. The details are covered next in overfitting sections. 


### 5. Model parameter tuning

I used an adam optimizer to train the model, so the learning rate was not tuned manually. I experimented with different batch sizes and settled for 128. I stopped epochs when the validation loss approached training loss. The ideal number of epochs was 4. This is when the validation loss approached training loss and began to increase with more epochs. 

### 6. Overfitting 

I tried 3 different approaches to counter overfitting. 
* Added additional data for 2 laps (one in opposite direction). 
* Added Dropouts 
* Simplified the architecture. I reduced filter depths and number of units in the hidden layer. 

The second model was based on NVIDIA model for autonomous driving. It consisted of the following: 
* Convolution Layer with Filter Depth of 24 with Kernel Size 5x5 with ReLU activations 
* Convolution Layer with Filter Depth of 36 with Kernel Size 5x5 with ReLU activations 
* Convolution Layer with Filter Depth of 48 with Kernel Size 5x5 with ReLU activations 
* Convolution Layer with Filter Depth of 64 with Kernel Size 3x3 with ReLU activations
* Convolution Layer with Filter Depth of 96 with Kernel Size 3x3 with ReLU activations
* Convolution Layer with Filter Depth of 128 with Kernel Size 3x3 with ReLU activations
* FC Layer 64 units 
* FC Layer 32 units 
* FC Layer 8 units 
* Output 1 unit 

I still notice that the NVIDIA model overfits with higher epochs. So early stopping was necessary. 

![alt text][image9]


### 7. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

* Convolution Layer with Filter Depth of 24 with Kernel Size 5x5 with ReLU activations 
* Convolution Layer with Filter Depth of 36 with Kernel Size 5x5 with ReLU activations 
* Convolution Layer with Filter Depth of 48 with Kernel Size 5x5 with ReLU activations 
* Convolution Layer with Filter Depth of 64 with Kernel Size 3x3 with ReLU activations
* Convolution Layer with Filter Depth of 96 with Kernel Size 3x3 with ReLU activations
* Convolution Layer with Filter Depth of 128 with Kernel Size 3x3 with ReLU activations
* FC Layer 64 units 
* FC Layer 32 units 
* FC Layer 8 units 
* Output 1 unit 

When used in the simulator, the car drove really well staying in middle of the road most of the time. The submission consists of 2 videos of these test runs. 

