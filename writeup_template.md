##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Content

- `test.ipynb` - Jupyter notebook with code to run the project
- `utils.py` - All functions defined for the project
- `project_video_proc.mp4` - the output video

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video]: ./project_video_proc.mp4
[test6]: ./examples/test6.png
[test5]: ./examples/test5.png
[test4]: ./examples/test4.png
[test3]: ./examples/test3.png
[test2]: ./examples/test2.png
[test1]: ./examples/test1.png
[windows]: ./examples/windows.png
[hog_car]: ./examples/hog_car.png
[hog]: ./examples/hog.png
[heatmap]: ./examples/heatmap.png
[roi]: ./examples/roi.png
[bound]: ./examples/bound.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for the hog extraction function is contained in lines 36 through 50 of the file called `utils.py`.

In the 3rd cell of the notebook `test.ipynb`, I use the function extract_features to extract all the features including spatial binning, color histograms and hog features.

I started by reading in all the images `vehicle/non-vehicle`

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).
Here is an example of a training image and it's HOG

![alt text][hog_car]
![alt text][hog]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and noticed that what worked best for the classifier was that I use all the channels to create the HOG features and increase the orientations bin size as much as possible. But this had the added downside of making training extremely slow and the feature vector could blow out of proportions.

So after experimentation, I settled for only using the L channel in the LUV colorspace for extracting the HOG features and using 8 as the orientation bin size. THough increasing the bin size increased accuracy but the classifier was displaying enough accuracy for our purposes.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used gridsearch to try to fine tune the model and discovered that rbf worked better but as it took longer time, I instead focused on using the simpler linear kernel and increased the accuracy by tuning the feature extraction process.

I trained a linear SVM using the `hinge` loss as it works great with SVMs. The code is in the third block of `test.ipynb`. I used all three features types, the spatial binning with size (16, 16), color histogram and HOG for L channel only with following config:

```
orient = 8  # HOG orientations
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # L channel
pix_per_cell = 8
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Basic sliding window algorithm implemented is the same which is presented in Udacity's lectures. It allows to search a car in a desired region of the frame with a desired window size (each subsampled window is rescaled to 64x64 px for classification).

The window size and overlap had been manually tuned as it should account for expected cars in the frame and should mimic real perspective.

for testing images, I used

```
y_start_stop=[400, 640]
xy_window=(128, 128)
```

For the actual frame processing, I use multiscale windows with a comination of region of interest as detected by previous location of a car in the frame. I use different regions to detect new cars arriving into the frame and cars far off along the road.

The code is in `process_frame` function, line 389 in `utils.py`.

There are some sample results for a fixed window size (128x128 px) and overlap for the provided test images:

![alt text][test1]
![alt text][test2]
![alt text][test3]
![alt text][test4]
![alt text][test5]
![alt text][test6]

As we can see on examples above, the classifier successfully finds cars on the test images. However, there is a false positive example, so, we will need to apply a kind of filter (such as heat map) and the classifier failed to find a car on th 3rd image because it is too small for it. That is why, we will need to use multi scale windows.

![alt text][windows]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched in three ways: (lines 398 - 427, `utils.py`)
* detecting new vehicles, larger scale
* detecting distant vehicles, smaller scale
* tracking and redetecting previous detections using ROI at multuple scales

# Region of Interest:

![alt text][roi]

This optimizes the detection pipeline as the whole image doesn't need to be searched and also dramatically increase the accuracy due to multi scaled search for various expected vehicle sizes.

I cut down on the feature space as much as possible to optimize it further by only using a single channle HOG and keeping orientation bin at a minimum of 8.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_proc.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this starts at line 324 in `utils.py`.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid:

![alt text][test5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][heatmap]

### Here the resulting bounding boxes are drawn onto the last frame:
![alt text][bound]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

- I got a boost in classifier accuracy after I augmented the data with flipped images.
- To reduce false positives, I implemented a threshold approach using heat maps as suggested in the lectures
- I try to use smallest number of windows to search in the image to optimize performance by only searching for where a new car might arrive and tracking previous detections with a region of interest
- I skip every alternate frame and just reuse results to increase performance as cars or not so fast that we could lose track of them
- I created the frame heat map by applying a basic low pass filter. (line 338 in `utils.py`) this increases robustness of the detections over multiple frames. (using a weighted average of the previous heat map and the current detection)

Some limitations:

- the pipeline is not real time ~ (6 fps) and it can be further optimized by reducing feature space and search windows.
- in case of car overlap, the algorithm detects it as one major giant blob and fails. This may be resolved by fine tracking cars positions.
- The classifier can be further improved by additional data augmentation, hard negative mining and classifier parameter tuning.
- the algorithm may fail in case of difficult lighting, which could be partly resolved by the improving the classifier.
