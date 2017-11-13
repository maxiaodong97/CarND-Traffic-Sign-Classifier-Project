## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This project will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I will then try out the model on images of German traffic signs on the web.


Here is the list of file for submission:

1. [Jupiter notebook](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
2. [HTML format of Jupiter notebook] (https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)
3. [Python code](https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/traffic_sign.py)
4. [Writeup] (https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup.md)
5. [5 Images download from web] (https://github.com/maxiaodong97/CarND-Traffic-Sign-Classifier-Project/tree/master/New_Test)

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set from 
[Training data set](http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip)
[Test data set] (http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip)
[Test result] (http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip)

Unzip to data folder 
```sh
data/ 
    New_test/
    Final_Test/
    FInal_Training/
```

To run jupiter notebook:
```sh
source activate carnd-term1
(carnd-term1) MacOS:CarND-Traffic-Sign-Classifier-Project xma$ jupyter notebook Traffic_Sign_Classifier.ipynb 
[I 19:13:23.999 NotebookApp] Serving notebooks from local directory: /Users/xma/sd/CarND-Traffic-Sign-Classifier-Project
[I 19:13:23.999 NotebookApp] 0 active kernels
[I 19:13:23.999 NotebookApp] The Jupyter Notebook is running at:
[I 19:13:23.999 NotebookApp] http://localhost:8888/?token=f6b99f24fbb4dea7af3b6bd01889870e437941ca103237b3
[I 19:13:23.999 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 19:13:24.000 NotebookApp] 
```

