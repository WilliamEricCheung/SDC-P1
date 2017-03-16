**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images/processed_solidWhiteCurve.jpg "SolidWhiteCurveOutput"

[image2]: ./test_images/processed_solidWhiteRight.jpg "SolidWhiteRightOutput"

[image3]: ./test_images/processed_solidYellowCurve.jpg "SolidYellowCurveOutput"

[image4]: ./test_images/processed_solidYellowCurve2.jpg "SolidYellowCurve2Output"

[image5]: ./test_images/processed_solidYellowLeft.jpg "SolidYellowLeftOutput"

[image6]: ./test_images/processed_whiteCarLaneSwitch.jpg "WhiteCarLaneSwitchOutput"


---

### Reflection

###1. A Description of my pipeline 


###The pipeline I built to do lane lines detection consisted of 7 steps.

###Step 1 Color selection

Since the lines are either white or yellow, the pipeline is only interested in these two colors. 
The input images are 8-bit RGB images. To extract white color, a simple RGB filter will work. So the current pipeline will filter out all the pixels whose RGB values are below [200,200,200] to generate a white color mask. Yellow is a little trickier. Initially, I used a simple RGB filter, [200,200,30], to generate a color mask that has both yellow and white color in it. However, it performs poorly in the challenging video part due to change in the lighting condition. To tackle this issue, I converted RGB images into HSV images, and generate a yellow color mask in HSV space. At the end, the white mask and yellow mask are combined to do a color selection on the original images. 

###Step 2 Grayscaling

The pipeline will convert the output from the color selection into a grayscale image for edge detection later

###Step 3 Gaussian smoothing

The pipeline will apply Gaussian smoothing on the grayscale image to prepare it for Canny edge detection

###Step 4 Edge detection

The pipeline will use Canny edge detection on the output from previous step

###Step 5 Region of interest selection

The pipeline will draw a polygon on the image to select region of interest

###Step 6 Hough Transform line detection

The output of the Hough Transform consist of many segmented lines

###Step 7 Draw line

For all the lines from the Hough Transform line detection, it can be represented by two endpoints. The draw_line function will first take in all the endpoints and then separate them into two groups, one group contains endpoints from the left line region and one group contains endpoints from the right line region. Then the draw_line function will apply linear regression on both groups to find out left and right line. At the end, the two lines will be extended to the top and bottom edges of region of interest and be shown on the output image 


Here are some outputs of my pipeline: 

![alt text][image1]

![alt text][image2]

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6]


###2. Potential shortcomings with my current pipeline


One potential shortcoming would be the draw_line function would perform poorly when the lane lines are not centered in the image. Right now the draw_line function use center of the image as the reference line to separate endpoints of the outputs from Hough transform. Also my region of interest selection will also be problematic when the position of the lines shifts dramatically. 

Another shortcoming could be the color selection method. If the images are taken from a rainy day or during the night, then my color selection method might fail because my color selection filters use hardcoded values, which won't adapt to change in lighting condition and other environment changes automatically. 


###3. Possible improvements to my pipeline

A possible improvement would be to use SVM to separate points from the left and right lines.

Another potential improvement could be to use machine learning technique to find region of interest dynamically so the pipeline can become more adaptive. 
