#**Behavioral Cloning** 

##Project Writeup 

---

**Behavioral Cloning Project**

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior.
* Experiment with data transformation / CV (preprocessing) 
* Experiment with data augmentation techniques 
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

####1. Relevant Files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 video of the car driving on track 1 in autonomous mode 
* writeup.md report summarizing the results


###Data Generation and  Preprocessing 
I generated data using the simulator in training mode. Initially I collected data for 2 laps. In addition, I collected data for 1 lap in the opposite direction. I used images from all 3 cameras using a correction factor 0.25. I also flipped the images and measurements to double the amount of data. 
 
I first normalized the data to zero mean and SD of 0.5.I cropped top 75 pixels and bottom 20 pixels to just retain the view of the road thereby reducing the dimensionality 

###Model Architecture and Training Strategy

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

I added dropouts here as well. This model achieved a validation loss of 0.04. When used in the simulator, the car drove really well staying in middle of the road most of the time. The submission consists of 2 videos of these test runs. 

####2. Overfitting 

I tried 3 different approaches to counter overfitting. 
* Added additional data for 2 laps (one in opposite direction). 
* Added Dropouts 
* Simplified the architecture. I reduced filter depths and number of units in the hidden layer. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. I experimented with different batch sizes and settled for 128. I stopped epochs when the validation loss approached training loss. 

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also used data of driving in the opposite direction of the track so that the model could generalize well. 

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive the car at a higher speeds and staying on the track. 

My first step was to use a convolution neural network model similar to the LeNet architecture. I wanted to see how well this performs before choosing a more complex model. When the LeNet model did not do so well, I implemented the NVIDIA reference model and initially, I noticed the model was overfitting i.e. the validation loss was significantly higher than the training loss. I tried several iteration by generating more data, added dropout and simplifying the layers to counter overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

* Convolution Layer with Filter Depth of 24 with Kernel Size 5x5 with ReLU activations 
* Convolution Layer with Filter Depth of 36 with Kernel Size 5x5 with ReLU activations 
* Convolution Layer with Filter Depth of 48 with Kernel Size 5x5 with ReLU activations 
* Convolution Layer with Filter Depth of 64 with Kernel Size 3x3 with ReLU activations
* Convolution Layer with Filter Depth of 96 with Kernel Size 3x3 with ReLU activations
* FC Layer 128 units 
* FC Layer 32 units 
* FC Layer 8 units 
* Output 1 unit 

####3. Creation of the Training Set & Training Process

I generated data using the simulator in training mode. Initially I collected data for 2 laps. In addition, I collected data for 1 lap in the opposite direction. I used images from all 3 cameras using a correction factor 0.25. I also flipped the images and measurements to double the amount of data. Finally, I added data for one more lap to enable the model to generalize well. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4. This is when the validation loss approached training loss and began to increae with more epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.

