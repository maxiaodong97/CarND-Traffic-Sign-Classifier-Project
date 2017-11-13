import os
import pandas as pd
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa

TRAINING_IMAGE_DIR = 'data/Final_Training/Images'
TEST_IMAGE_DIR = 'data/Final_Test/Images'
NEW_IMAGE_DIR = 'data/New_Test'


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


def getImageForClass(label):
    if label == 0:
        return X_raw[0: indexes[label]]
    if label == N_CLASSES - 1:
        return X_raw[indexes[label]:]
    return X_raw[indexes[label - 1]: indexes[label]]


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


plotSamplesSummary(X_raw, y_raw, classes)

for label in classes:
    X_classes = getImageForClass(label)
    plotImage(X_classes[0:8])

X_norm = [skimage.transform.resize(image, (32, 32), mode='constant') for image in X_raw]
y_norm = y_raw
X_train, X_validation, y_train, y_validation = train_test_split(
    X_norm, y_norm, stratify=y_norm, test_size=9209, random_state=0)
X_test, y_test = load_test_data()

histogram(y_raw, 'class', 'count', 'Number of Samples per Class')
histogram([x.shape[0] for x in X_raw], 'width', 'count', 'Image sample width distribution')
histogram([x.shape[1] for x in X_raw], 'height', 'count', 'Image sample height distribution')


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


def plotArray(X):
    trace1 = go.Scatter(x=[i for i in range(0, len(X))], y=X, mode='lines+markers', name='linear')
    trace2 = go.Bar(x=[i for i in range(0, len(X))], y=X, name="bar")
    data = go.Data([trace1, trace2])
    py.iplot(data)


plotArray(perClassAccuracy)

plotArray(perEpochAccuracy)


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
            # apply the following augmenters to most images
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


augImages, augLabels = augumentImages([0, 20, 26])

plotImage(augImages[0:24])


X_train.extend(augImages)
y_train.extend(augLabels)


def load_new_data():
    csv_file = os.path.join(NEW_IMAGE_DIR, 'manifest.csv')
    csv = pd.read_csv(csv_file, sep=';')
    labels = csv['ClassId'].values
    files = csv['Filename'].values
    images = []
    for file in files:
        f = os.path.join(NEW_IMAGE_DIR, file)
        images.append(skimage.data.imread(f))
    return images, labels


X_new, y_new = load_new_data()

plotImage(X_new)

X_new_norm = [skimage.transform.resize(image, (32, 32), mode='constant') for image in X_new]

plotImage(X_new_norm)


X_test = [skimage.transform.resize(image, (32, 32), mode='constant') for image in X_test]
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_new_norm, y_new)
    print("New data Test Accuracy = {:.3f}".format(test_accuracy))


plotImage(getImageForClass(14)[0:8])
plotImage(getImageForClass(5)[0:8])
plotImage(getImageForClass(1)[0:8])
plotImage(getImageForClass(31)[0:8])
plotImage(getImageForClass(33)[0:8])

softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=5)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: X_new_norm})
    my_top_k = sess.run(top_k, feed_dict={x: X_new_norm})
    print(my_top_k)

print(y_new)
