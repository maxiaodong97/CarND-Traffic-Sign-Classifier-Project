## Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report



### Load the data set

Download the data set from 
1. [Training data set](http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip)
2. [Test data set](http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip)
3. [Test result](http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip)

And unzip to data folder, using following function to load the training data and test data.

```python

TRAINING_IMAGE_DIR = 'data/Final_Training/Images'
TEST_IMAGE_DIR = 'data/Final_Test/Images'

def load_train_data():
    data_dir = TRAINING_IMAGE_DIR
    directories = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))])
    labels = []
    images = []
    indexes = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
        indexes.append(len(images))
    return images, labels, indexes


def load_test_data():
    csv_file = os.path.join(TEST_IMAGE_DIR, 'GT-final_test.csv')
    csv = pd.read_csv(csv_file, sep=';')
    labels = csv['ClassId'].values
    files = csv['Filename'].values
    images = []
    for file in files:
        f = os.path.join(TEST_IMAGE_DIR, file)
        images.append(skimage.data.imread(f))
    return images, labels

X_raw, y_raw, indexes = load_train_data()
classes = set(y_raw)
N_CLASSES = len(classes)
X_norm = [skimage.transform.resize(image, (32, 32), mode='constant') for image in X_raw]
y_norm = y_raw
X_train, X_validation, y_train, y_validation = train_test_split(
    X_norm, y_norm, stratify=y_norm, test_size=9209, random_state=0)
X_test, y_test = load_test_data()

n_train = len(X_train)
n_validation = len(X_validation)
n_test = len(X_test)
image_shape = [32, 32, 3]
n_classes = len(y_train)
print("Number of training examples =", n_train)
print("Number of validation examples =", 9209)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```
Here is the output: 
```python
Number of training examples = 30000
Number of validation examples = 9209
Number of testing examples = 12630
Image data shape = [32, 32, 3]
Number of classes = 30000
```

Please note the initial X_raw data is sorted by class label, we use indexes to hold the "seperator". By doing this, we can easily find all images of a class, eg:

```python

def getImageForClass(label):
    if label == 0:
        return X_raw[0: indexes[label]]
    if label == N_CLASSES - 1:
        return X_raw[indexes[label]:]
    return X_raw[indexes[label - 1]: indexes[label]]

```

### Explore, summarize and visualize the data set

Following function is used to plot the samples

```python
def plotImage(images):
    plt.figure(figsize=(15, 15))
    i = 1
    for image in images:
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()


def plotSamplesSummary(images, labels, classes):
    plt.figure(figsize=(15, 15))
    i = 1
    for label in classes:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        plt.imshow(image)
    plt.show()


def plotSamplesSome(images, labels, label, limit):
    plt.figure(figsize=(15, 5))
    i = 1
    start = labels.index(label)
    end = start + labels.count(label)
    images = images[start:end][:limit]
    for image in images:
        plt.subplot(3, 8, i)
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        plt.imshow(image)
    plt.show()


def histogram(X, xDescription, yDescription, title):
    data = [go.Histogram(x=X)]
    layout = go.Layout(
        title=title,
        xaxis=dict(title=xDescription),
        yaxis=dict(title=yDescription),
        bargap=0.1,
        bargroupgap=0.1
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
```

First draw one sample of each class.

```python
plotSamplesSummary(X_raw, y_raw, classes)
```
![alt text](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/images/SampleSummary.png "Sample Summary")

Then we draw the histogram of samples
```python
histogram(y_raw, 'class', 'count', 'Number of Samples per Class')
```
![alt text](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/images/SampleSumaryHistogram.png "Sample Distribution")


### Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


