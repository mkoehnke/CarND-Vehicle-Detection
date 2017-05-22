## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/dataset_output.jpg
[image2a]: ./output_images/hog1_output.jpg
[image2b]: ./output_images/hog2_output.jpg
[image2c]: ./output_images/hog3_output.jpg
[image2d]: ./output_images/spatial_bin_output.jpg
[image2e]: ./output_images/histogram_output.jpg
[image3]: ./output_images/window_search_output.jpg
[image4a]: ./output_images/test1_output.jpg
[image4b]: ./output_images/test2_output.jpg
[image4c]: ./output_images/test3_output.jpg
[image4d]: ./output_images/test4_output.jpg
[image4e]: ./output_images/test5_output.jpg
[image4f]: ./output_images/test6_output.jpg
[image5]: ./output_images/heatmap_output.jpg
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first to fourth code cells of the IPython notebook.

I started by reading in all the `car` and `non-car` images.  Here is an example of one of each of the `car` and `non-car` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I also experimented with different color spaces and channels. I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for all color channels:


![alt text][image2a]
![alt text][image2b]
![alt text][image2c]

In addition, I used Spatially Binned Features. For this I resized all images to size of 16x16 px and used the `ravel()` function to create feature vector. The following plot shows a visualization of those features for the car image above:

![alt text][image2d]

Another method I explored was looking at color histograms of pixel intensity as features. I ended up using 16 histogram bins, which gave the the following result for the RGB color space:

![alt text][image2e]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, ran the whole stack of identifying vehicles with the provided test images and always went back to tweak parameters again.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the code cells 7-9 of the IPython notebook.

I trained a linear SVM using the provided dataset of car and non-car images. For those images, I created labels with the value **1** and **0**. In order to determine the accuracy, I split the dataset into 80% training set and 20% validation set.
This resulted in a Feature vector length of 6108 and a Test Accuracy of SVC = 0.98958333.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the code cells 10-12 of the IPython notebook.

I decided to search only the street (lower) section of the image (y: 400-565) at scales up to 1.5. The following image shows a visualization of the search region and windows:

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4a]
![alt text][image4b]
![alt text][image4c]
![alt text][image4d]
![alt text][image4e]
![alt text][image4f]

I tried to optimize the performance of the classifier by focusing the area for the sliding window search to the street only.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in the code cells 19-21 of the IPython notebook.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap and the identified bounding boxes then overlaid on a frame of the video:

![alt text][image5]

In order to minimize the occurence of false positives and make the rectangles more stable, I combined the heatmap of previous frames in the `draw_boxes_with_heatmap` function.

--- 

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I often saw the problem of wobbly, unstable bounding boxes. Also there are still a few false positives in the video that I couldn't elimiate due to limited time. Another problem is that cars on the left side of the highway got also detected and which might confuse the car.

The pipeline will probably fail to identify a car correctly, if this car is very close in front of the camera. This is due to the limited sliding window search area. Also bad light and weather conditions might result in wrong classification.

Possible ways to improve the pipeline:

- Increase the dataset by e.g. mirroring the existing images
- Put a mask on top of the image to reduce unwanted vehicle detections on the left side of the highway.
- Increase the window search area to also detect vehicles that are very close to the camera.