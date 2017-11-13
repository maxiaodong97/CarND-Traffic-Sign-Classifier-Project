## Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


---
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
---
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

---
### Design and Test a Model Architecture

To design the model, I need to understand the input images further with regarding of size and color

### I first plot the image size distribution: 
```python
histogram([x.shape[0] for x in X_raw], 'width', 'count', 'Image sample width distribution')
```
![alt text](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/images/SampleWidthHistogram.png "Width distribution")

```python
histogram([x.shape[1] for x in X_raw], 'height', 'count', 'Image sample height distribution')
```
![alt text](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/images/SampleHeightHistogram.png "Height distribution")

```python
import numpy as np
print("Average width is = {:.3f}".format(np.mean([x.shape[0] for x in X_raw])))
print("Average Height is = {:.3f}".format(np.mean([x.shape[1] for x in X_raw])))
Average width is = 50.329
Average Height is = 50.836
```

The mean width and height is 50 x 50, but the majority images in 32 x 32. So I decide to try two size 32 x 32 and 48 x 48 to see if larger image gives better result. Use following code to do the resizing:

``` python
X_norm = [skimage.transform.resize(image, (32, 32), mode='constant') for image in X_raw]
```

### then plot the image color
What I found is color does matters, as sign is always show as certain color and some color like red, or blue is indeed a feature of traffic sign. Therefore I decide to not to convert to grey images. 

### Model

LeNet CNN is proved to be a solid method for image classification problem. The foundamental principal is to train a series of different "filters" as capture the image details and then use full connect layer to summarize those details. Here I use two layer convolution layer with max pooling and two full connected layer and one output layer. I use 5 x 5 kernel. As there are so many parameters to tune, I decide to experiment following ones: 

* Baseline model: 2 Conv, 2 Full, 1, output, image size 32 x 32 x 3, adam optimizer.
* Second Try: Baseline + change image size 48 x 48 x 3
* Third Try: Baseline + adding one more full connected layer 
* Fourth Try: Baseline but using adagrad optimizer.
* Fifth Try: Baseline + Augument images (~5000) in training set.  

#### Base line model
```python
mu = 0
sigma = 0.1

weights = {
    'wc1': tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 18), mean=mu, stddev=sigma)),
    'wc2': tf.Variable(tf.truncated_normal(shape=(5, 5, 18, 50), mean=mu, stddev=sigma)),
    'wd1': tf.Variable(tf.truncated_normal(shape=(5 * 5 * 50, 1000), mean=mu, stddev=sigma)),
    'wd2': tf.Variable(tf.truncated_normal(shape=(1000, 300), mean=mu, stddev=sigma)),
    'out': tf.Variable(tf.truncated_normal(shape=(300, N_CLASSES), mean=mu, stddev=sigma))}

biases = {
    'bc1': tf.Variable(tf.zeros([18])),
    'bc2': tf.Variable(tf.zeros([50])),
    'bd1': tf.Variable(tf.zeros([1000])),
    'bd2': tf.Variable(tf.zeros([300])),
    'out': tf.Variable(tf.zeros([N_CLASSES]))}


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='VALID')


def full(x, W, b):
    x = tf.reshape(x, [-1, W.get_shape().as_list()[0]])
    x = tf.add(tf.matmul(x, W), b)
    return tf.nn.relu(x)


def LeNet(x):

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x18.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])

    #  Pooling. Input = 28x28x18. Output = 14x14x18.
    conv1 = maxpool2d(conv1, k=2)

    #  Layer 2: Convolutional. Output = 10x10x50.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

    # Pooling. Input = 10x10x50. Output = 5x5x50.
    conv2 = maxpool2d(conv2, k=2)

    # Flatten. Input = 5x5x50. Output = 1000.
    fc1 = full(conv2, weights['wd1'], biases['bd1'])

    # Layer 3: Fully Connected. Input = 1000. Output = 300.
    fc2 = full(fc1, weights['wd2'], biases['bd2'])

    # Layer 5: Fully Connected. Input = 300. Output = N_CLASSES.
    logits = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return logits
    
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def validationGroup(X_data, y_data):
    X_group = [[] for i in range(0, N_CLASSES)]
    y_group = [[] for i in range(0, N_CLASSES)]
    for i in range(0, len(y_data)):
        X_group[y_data[i]].append(X_data[i])
        y_group[y_data[i]].append(y_data[i])
    return X_group, y_group


def evaluatePerClass(X_data, y_data):
    accuracy = []
    X_group, y_group = validationGroup(X_data, y_data)
    for i in range(0, len(y_group)):
        accuracy.append(evaluate(X_group[i], y_group[i]))
    return accuracy

def plotArray(X):
    trace1 = go.Scatter(x=[i for i in range(0, len(X))], y=X, mode='lines+markers', name='linear')
    trace2 = go.Bar(x=[i for i in range(0, len(X))], y=X, name="bar")
    data = go.Data([trace1, trace2])
    py.iplot(data)

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, N_CLASSES)
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

EPOCHS = 10
BATCH_SIZE = 128

perEpochAccuracy = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        perEpochAccuracy.append(validation_accuracy)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy is {:.3f}".format(validation_accuracy))
        print()

    perClassAccuracy = evaluatePerClass(X_validation, y_validation)
    saver.save(sess, './traffic_signs')
    print("Model saved")


plotArray(perClassAccuracy)
plotArray(perEpochAccuracy)
```

I evalute result based on: 
* validation accuracy changes with each epoch
* validation accuracy of each class

Here is the result:

![alt text](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/images/perEpochAccuracy.png "Validation accuracy per epoch")

![alt text](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/images/perClassAccuracy.png "Validation accuracy per class")

We can see the baseline model seem to work well. 

#### Second Try: Baseline + change image size 48 x 48 x 3
```python
# so we resize to 48 x 48
X_norm = [skimage.transform.resize(image, (48, 48), mode='constant') for image in X_raw]
y_norm = y_raw
X_train, X_validation, y_train, y_validation = train_test_split(
    X_norm, y_norm, stratify=y_norm, test_size=9209, random_state=0)

plotSamplesSummary(X_norm, y_norm, classes)
```
![alt text](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/images/SampleSummary48.png "Image summary 48x48")

Rerun the model, here is the result:
```python
Training...

EPOCH 1 ...
Validation Accuracy = 0.945

EPOCH 2 ...
Validation Accuracy = 0.983

EPOCH 3 ...
Validation Accuracy = 0.989

EPOCH 4 ...
Validation Accuracy = 0.989

EPOCH 5 ...
Validation Accuracy = 0.985

EPOCH 6 ...
Validation Accuracy = 0.989

EPOCH 7 ...
Validation Accuracy = 0.990

EPOCH 8 ...
Validation Accuracy = 0.975

EPOCH 9 ...
Validation Accuracy = 0.990

EPOCH 10 ...
Validation Accuracy = 0.991

Model saved
```

![alt text](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/images/perEpochAccuracy48x48.png "validation accuracy 48x48")

We can see that accuracy increased faster than 32x32, but each epoch is much slower. The overall accuracy is also increased. 

#### Third Try: Baseline + adding one more full connected layer 
```python
weights = {
    'wc1': tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 18), mean=mu, stddev=sigma)),
    'wc2': tf.Variable(tf.truncated_normal(shape=(5, 5, 18, 100), mean=mu, stddev=sigma)),
    'wd1': tf.Variable(tf.truncated_normal(shape=(5 * 5 * 100, 2000), mean=mu, stddev=sigma)),
    'wd2': tf.Variable(tf.truncated_normal(shape=(2000, 1000), mean=mu, stddev=sigma)),
    'wd3': tf.Variable(tf.truncated_normal(shape=(1000, 300), mean=mu, stddev=sigma)),
    'out': tf.Variable(tf.truncated_normal(shape=(300, N_CLASSES), mean=mu, stddev=sigma))}

biases = {
    'bc1': tf.Variable(tf.zeros([18])),
    'bc2': tf.Variable(tf.zeros([100])),
    'bd1': tf.Variable(tf.zeros([2000])),
    'bd2': tf.Variable(tf.zeros([1000])),
    'bd3': tf.Variable(tf.zeros([300])),
    'out': tf.Variable(tf.zeros([N_CLASSES]))}
```

With one more full connected layer, here is the result. 
```python
Training...

EPOCH 1 ...
Validation Accuracy = 0.963

EPOCH 2 ...
Validation Accuracy = 0.972

EPOCH 3 ...
Validation Accuracy = 0.983

EPOCH 4 ...
Validation Accuracy = 0.986

EPOCH 5 ...
Validation Accuracy = 0.991

EPOCH 6 ...
Validation Accuracy = 0.988

EPOCH 7 ...
Validation Accuracy = 0.992

EPOCH 8 ...
Validation Accuracy = 0.988

EPOCH 9 ...
Validation Accuracy = 0.973

EPOCH 10 ...
Validation Accuracy = 0.983
```
I found the accuracy is reduced, I suspect it may require smaller learning rate for this model. So I retry with smaller learning rate and more epochs. 
```python
rate = 0.0001
EPOCHS = 30
...
```

Here is the new result

```python
Training...

EPOCH 1 ...
Validation Accuracy = 0.773

EPOCH 2 ...
Validation Accuracy = 0.906

EPOCH 3 ...
Validation Accuracy = 0.934

EPOCH 4 ...
Validation Accuracy = 0.945

EPOCH 5 ...
Validation Accuracy = 0.956

EPOCH 6 ...
Validation Accuracy = 0.966

EPOCH 7 ...
Validation Accuracy = 0.971

EPOCH 8 ...
Validation Accuracy = 0.976

EPOCH 9 ...
Validation Accuracy = 0.969

EPOCH 10 ...
Validation Accuracy = 0.978

EPOCH 11 ...
Validation Accuracy = 0.979

EPOCH 12 ...
Validation Accuracy = 0.982

EPOCH 13 ...
Validation Accuracy = 0.983

EPOCH 14 ...
Validation Accuracy = 0.989

EPOCH 15 ...
Validation Accuracy = 0.989

EPOCH 16 ...
Validation Accuracy = 0.988

EPOCH 17 ...
Validation Accuracy = 0.979

EPOCH 18 ...
Validation Accuracy = 0.981

EPOCH 19 ...
Validation Accuracy = 0.985

EPOCH 20 ...
Validation Accuracy = 0.990

EPOCH 21 ...
Validation Accuracy = 0.991

EPOCH 22 ...
Validation Accuracy = 0.991

EPOCH 23 ...
Validation Accuracy = 0.991

EPOCH 24 ...
Validation Accuracy = 0.990

EPOCH 25 ...
Validation Accuracy = 0.991

EPOCH 26 ...
Validation Accuracy = 0.991

EPOCH 27 ...
Validation Accuracy = 0.991

EPOCH 28 ...
Validation Accuracy = 0.991

EPOCH 29 ...
Validation Accuracy = 0.991

EPOCH 30 ...
Validation Accuracy = 0.991

```
The model stops improving at 0.991.
![alt text](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/images/perEpochAccuracyOneMoreFull.png "validtion accuracy with one additional full layer")

#### Fourth Try: Baseline but using adagrad optimizer.
```python
optimizer = tf.train.AdagradOptimizer(learning_rate=rate)
```
Here is the output
```python
Training...

EPOCH 1 ...
Validation Accuracy = 0.284

EPOCH 2 ...
Validation Accuracy = 0.429

EPOCH 3 ...
Validation Accuracy = 0.498

EPOCH 4 ...
Validation Accuracy = 0.550

EPOCH 5 ...
Validation Accuracy = 0.589

EPOCH 6 ...
Validation Accuracy = 0.617

EPOCH 7 ...
Validation Accuracy = 0.666

EPOCH 8 ...
Validation Accuracy = 0.695

EPOCH 9 ...
Validation Accuracy = 0.712

EPOCH 10 ...
Validation Accuracy = 0.740

Model saved
```
Adagrad is slower than Adam in our case.

![alt_text](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/images/adagrad.png "adagrad")


#### Fifth Try: Baseline + Augument images (~5000) in training set.  
In the sample distribution graph previously we saw, some samples are too few, which causes some validation accuracy for a perticular class is low.  here I tried to add more training samples for those labels. 
```python
# some classes are very few images, we will augument those classes that has count less than 500
weakLabels = [label for label in classes if y_raw.count(label) < 500]
```

I use another python package called "from imgaug import augmenters as iaa" to do the job.
![alt_text](https://github.com/aleju/imgaug, "imgaug"), I didn't use flip as it is not normal to have a fliped sign.

```python
def sometimes(aug):
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    return iaa.Sometimes(0.5, aug)


def createAugumentor():
    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    seq = iaa.Sequential(
        [
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                # scale images to 80-120% of their size, individually per axis
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # translate by -20 to +20 percent (per axis)
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                mode=ia.ALL
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                # convert images into their superpixel representation
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                    # blur image using local means with kernel sizes between 2 and 7
                    iaa.AverageBlur(k=(2, 7)),
                    # blur image using local medians with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255),
                                          per_channel=0.5),  # add gaussian noise to images
                iaa.OneOf([
                    # randomly remove up to 10% of the pixels
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True),  # invert color channels
                # change brightness of images (by -10 to 10 of original value)
                iaa.Add((-10, 10), per_channel=0.5),
                iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization((0.5, 2.0))
                    )
                ]),
                # improve or worsen the contrast
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                # move pixels locally around (with random strengths)
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                # sometimes move parts of the image around
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq


def augumentImages(labels):
    augimages = []
    auglabels = []
    augmentor = createAugumentor()
    for label in labels:
        X_classes = getImageForClass(label)
        augimages_raw = augmentor.augment_images(X_classes)
        for image in augimages_raw: 
            augimages.append(skimage.transform.resize(image, (32, 32), mode='constant'))
            auglabels.append(label)
    return augimages, auglabels


augImages, augLabels = augumentImages(weakLabels)
print("Augmented images: {}".format(len(augImages)))

```
Here is the some of the result generated: 

![alt_text](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/images/AugSpeedLimit20.png "Auged")

Augmented images: 5730

Now rerun the model

```python
X_train, X_validation, y_train, y_validation = train_test_split(
    X_norm, y_norm, stratify=y_norm, test_size=9209, random_state=0)
X_train.extend(augImages)
y_train.extend(augLabels)

```

Here is the result:
![alt_text](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/images/perEpochAccuracyAugumented.png)

![alt_text](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/images/perClassAccuracyAugumented.png)

As we can see, validation accuracy is high for every class, not just the mean.

#### My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x50   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x50    				|
| Fully connected		| input 5x5x50, output 1000     				|
| RELU					|												|
| Fully connected		| input 1000, output 300        				|
| RELU					|												|
| Fully connected		| input 300, output 43          				|
|						|												|
 

To train the model, I used an adam optimizer, 128 batch size and 10 epochs, 0.001 learning rate.  

After train, here is the test result:

```python
# first normalize
X_test = [skimage.transform.resize(image, (32, 32), mode='constant') for image in X_test]
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

* training set accuracy of 0.991
* validation set accuracy of 0.979
* test set accuracy of 0.918

### Test a Model on New Images

Here is the normlized 5 german traffic sign I found from the web: 

![alt_text](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/images/fiveNewGermanNorm.png)

Run predictions: 
```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    prediction=tf.argmax(logits,1)
    best = sess.run([prediction],feed_dict={x:X_new_norm})
    print(best)

[array([14,  5,  1, 31, 33])]
```
The output matches the label as expected.

```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./traffic_signs.meta')
    saver.restore(sess, "./traffic_signs")
    test_accuracy = evaluate(X_new_norm, y_new)
    print("New data Test Accuracy = {:.3f}".format(test_accuracy))

```
New data Test Accuracy = 1.000

Now check the top 5 matches for each pictures: 

```python
softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=5)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./traffic_signs.meta')
    saver.restore(sess, "./traffic_signs")
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: X_new_norm})
    my_top_k = sess.run(top_k, feed_dict={x: X_new_norm})
    print(my_top_k)

```
Here is the output

```python
TopKV2(values=array(
    [[  1.00000000e+00,   5.76974232e-13,   2.17618566e-13, 8.68988312e-14,   2.45220697e-14],
     [  9.92458224e-01,   5.94804296e-03,   9.76459007e-04, 4.93374187e-04,   4.36481096e-05],
     [  1.00000000e+00,   3.97764275e-08,   8.02838563e-14, 1.51854949e-14,   1.17160853e-17],
     [  9.99554217e-01,   4.26491664e-04,   1.93246815e-05, 1.41658990e-10,   4.47433757e-11],
     [  1.00000000e+00,   3.52916156e-12,   2.39754207e-13, 1.30219891e-13,   1.24684985e-13]], dtype=float32), indices=array([
       [14,  1, 25, 13, 29],
       [ 5,  3, 13,  0,  1],
       [ 1,  0,  2,  5, 29],
       [31,  1, 21, 11,  5],
       [33, 11, 30,  1, 13]], dtype=int32))
```

I found it can not accurately classify type 22 and 27, which is probably caused by the original image contains website name and water marks, which is not part of training set before.

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

```python
def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

```
Taking first layer as example: 
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./traffic_signs.meta')
    saver.restore(sess, "./traffic_signs")
    outputFeatureMap([X_new_norm[0]], conv2d(x, weights['wc1'], biases['bc1']))
```

![alt_text](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/images/featureMapConv1.png)

We use 6 5x5 filter for each RGB color. Therefore it is 18 output.  The feature map shows resulting images after apply 5x5 convolution matrix.  Feature map captures all the details that is unique to the "stop sign" in this case.




