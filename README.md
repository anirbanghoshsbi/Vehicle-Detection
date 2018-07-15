# Vehicle-Detection
Vehicle Detection using Histogram of Oriented Gradients Descriptor + SVM  (or Random Forest or Deep Learning)

Using Computer Vision and Machine Learning algorithm , we train a image classifier that can categorise and label the content of the image.
The basics of Object Detection is global image classification and local object detection.

# Scope of the Project 
1. To Locate object of Choice in the image and draw a bounding box around it
2. To train a binary classifier to detect whether the template is present in the source image or not.

# Template Matching 
Template matching is the most simplest way to locate the template in a given image. However , it faces a big drawback and it is that it in template  matching we try to find the similiarities matrices over the raw pixel intensities instead of the _abstract features vectors_ like in Histogram of Oriented Gradient Descriptor (HOG).

# Histogram Of Oriented Gradient Descriptor (HOG)
Steps :
1. Experiment  Preparation : To start the experiment , we would consider the Caltech 101 dataset as a beginning point.We would make a classifier that can classify any object in the Caltech 101 dataset. However , any other data can also be classified using the same pipeline.
In the current experiment we would try to locate cars in images. So this classifier would be a binary classifier which would predict the presence or absence of cars.

However to work on this problem we would make a `json configuration` file that would have all the configuration details so that we dont have to put them in command line each time.

```
{
	/****
	* DATASET PATHS
	****/
	"image_dataset": "datasets/caltech101/101_ObjectCategories/car_side",
	"image_annotations": "datasets/caltech101/Annotations/car_side",
	"image_distractions": "datasets/sceneclass13",

	/****
	* FEATURE EXTRACTION
	****/
	"features_path": "output/cars/car_features.hdf5",
	"percent_gt_images": 0.5,
	"offset": 5,
	"use_flip": true,
	"num_distraction_images": 500,
	"num_distractions_per_image": 10,

	/****
	* HISTOGRAM OF ORIENTED GRADIENTS DESCRIPTOR
	****/
	"orientations": 9,
	"pixels_per_cell": [4, 4],
	"cells_per_block": [2, 2],
	"normalize": true,

	/****
	* OBJECT DETECTOR
	****/
	"window_step": 4,
	"overlap_thresh": 0.3,
	"pyramid_scale": 1.5,
	"window_dim": [96, 32],
	"min_probability": 0.7,

	/****
	* LINEAR SVM
	****/
	"classifier_path": "output/cars/model.cpickle",
	"C": 0.01
}
```

# Let us go through the configuration file

`image_dataset` path to where our “positive example” images reside on disk.In our case it is the `cars` .
`image_annotations` path to where the annotations of the bounding boxes of the `image_dataset` reside.

We require these bounding boxes so that we can extract the histogram of oriented gradients `HOG` features from the region of interest (roi) and then use a classifier to classify the images.
`image_distractors` the negative examples for training the `////???////`. We will choose the scene-13 as these will not contain any car images in it . _It is one of the requirement that we donot have any positive image in this database as it would pollute the classifier_.

`FEATURE EXTRACTION` path is containing all the possible steps for feature extraction like where to store the extracted feature or number of negative images or whether the image is to flipped or not so on...

`HISTOGRAM OF ORIENTED GRADIENTS DESCRIPTOR`
HOG features were first introduced by Dalal and Triggs in their CVPR 2005 paper, Histogram of Oriented Gradients for Human Detection. 
The most important parameters for the HOG descriptor are the `orientations` , `pixels_per_cell` , and the `cells_per_block` . These three parameters (along with the size of the input image) _effectively control the dimensionality of the resulting feature vector._

We’ll be using `pixels_per_cell=(4, 4)` , `cells_per_block=(2, 2)` , `9 orientations per histogram`, and square-root normalization applied to the image prior to description.

 `OBJECT DETECTOR`  has the following functions :

Looping over all layers of the image pyramid.
Applying our sliding window at each layer of the pyramid.
Extracting HOG features from each window.
Passing the extracted HOG feature vectors to our model for classification.
Maintaining a list of bounding boxes that are reported to contain an object of interest with sufficient probability.
these five points  are the knobs that we can dial to control our object detector.
"window_step": 4 (step size of the window),	"overlap_thresh": 0.3,	"pyramid_scale": 1.5 (controls the scale of our image pyramid),	"window_dim": [96, 32] (gives the dimension of our window) ,	"min_probability": 0.7(70% probability of the desired image).


The `Linear SVM` section can be found at the very bottom of the file. We have added two parameters for our classifier. The first is the output classifier_path  — this is the path to where our classifier will be stored after training.


