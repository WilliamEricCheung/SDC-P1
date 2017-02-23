#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

import os
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def original_draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_lines(img, lines, color=[255, 0, 0], thickness=13):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if lines is None:
    	return

    left_slope_data_x = np.array([])
    left_slope_data_y = np.array([])
    right_slope_data_x = np.array([])
    right_slope_data_y = np.array([])

    for line in lines:
        for x1,y1,x2,y2 in line:
        	minimum = x1
        	if x2 < x1:
        		minimum = x2
        	if  minimum > 0.5*img.shape[1] :
        		left_slope_data_x = np.append(left_slope_data_x,x1)
        		left_slope_data_x = np.append(left_slope_data_x,x2)
        		left_slope_data_y = np.append(left_slope_data_y,y1)
        		left_slope_data_y = np.append(left_slope_data_y,y2)        		
        	else:
        		right_slope_data_x = np.append(right_slope_data_x,x1)
        		right_slope_data_x = np.append(right_slope_data_x,x2)
        		right_slope_data_y = np.append(right_slope_data_y,y1)
        		right_slope_data_y = np.append(right_slope_data_y,y2) 	

    (left_a, left_b) = np.polyfit(left_slope_data_x, left_slope_data_y, 1)
    (right_a, right_b) = np.polyfit(right_slope_data_x, right_slope_data_y, 1)

    y1 = int(0.63*img.shape[0])
    y2 = img.shape[1]
    
    lx1 = int((y1 - left_b) / left_a) 
    lx2 = int((y2 - left_b) / left_a)

    rx1 = int((y1 - right_b) / right_a)
    rx2 = int((y2 - right_b) / right_a) 

    cv2.line(img, (lx1, y1), (lx2, y2), color, thickness)
    cv2.line(img, (rx1, y1), (rx2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    original_draw_lines(line_img, lines)

    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

#plt.imshow(gray, cmap='gray')
# if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

# Define color selection criteria

# for white lines
red_threshold_white_profile = 200
green_threshold_white_profile = 200
blue_threshold_white_profile = 200

rgb_threshold_white_profile = [red_threshold_white_profile, green_threshold_white_profile, blue_threshold_white_profile]

# for yellow lines
red_threshold_yellow_profile = 200
green_threshold_yellow_profile = 200
blue_threshold_yellow_profile = 200

rgb_threshold_yellow_profile = [red_threshold_yellow_profile, green_threshold_yellow_profile, blue_threshold_yellow_profile]

#Gaussian Smoothing
kernel_size = 3

#Canny parameters
low_threshold = 50
high_threshold = 150

#Hough Transform Parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/360 # angular resolution in radians of the Hough grid
threshold = 20     # minimum number of votes (intersections in Hough grid cell)
min_line_len = 100 #minimum number of pixels making up a line
max_line_gap = 200    # maximum gap in pixels between connectable line segments
#line_image = np.copy(image)*0 # creating a blank to draw lines on


dirs = os.listdir("test_images/")

for img in dirs:
	#print(img)
	#reading in an image
	path = "test_images/" + img
	#print(path)

	image = mpimg.imread(path)
	image = image[:,:,:3]

	plt.imshow(image)
	plt.show()

	#printing out some stats and plotting
	#print('This image is:', type(image), 'with dimensions:', image.shape)


	# Color Selection

	imgHSV = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

	#plt.imshow(imgHSV)
	#plt.show()

	lower_yellow = np.array([20,80,80])
	
	upper_yellow = np.array([25,255,255])

	mask = cv2.inRange(imgHSV,lower_yellow,upper_yellow)

	imgHSV_Yellow_mask = cv2.bitwise_and(image,image, mask = mask)

	#plt.imshow(imgHSV_Yellow_mask)
	#plt.show()

	color_select_yellow = np.copy(imgHSV_Yellow_mask)
	color_select_white = np.copy(image)

	# Mask pixels below the threshold
	color_thresholds = (image[:,:,0] < rgb_threshold_white_profile[0]) | (image[:,:,1] < rgb_threshold_white_profile[1]) | (image[:,:,2] < rgb_threshold_white_profile[2])

	color_select_white[color_thresholds] = [0,0,0]
	
	color_select = cv2.bitwise_and(cv2.bitwise_or(color_select_yellow, color_select_white), image)

	#plt.imshow(color_select)
	#plt.show()

	#grayscale the image
	gray = grayscale(color_select)
	#plt.imshow(gray)
	#plt.show()

	# Define a kernel size and apply Gaussian smoothing
	blur_gray = gaussian_blur(gray,kernel_size)

	#plt.imshow(blur_gray)
	#plt.show()

	#Canny Edge detection
	edges = canny(blur_gray, low_threshold, high_threshold)

	plt.imshow(edges)

	plt.show()

	#Defining a four sided polygon to mask
	imshape = image.shape
	vertices = np.array([[(0,imshape[0]),(int (0.45*imshape[1]), int (0.6*imshape[0])), (int(0.55*imshape[1]), int (0.6*imshape[0])), (imshape[1],imshape[0])]], dtype=np.int32)
	
	masked_image = region_of_interest(edges, vertices)
	#masked_image = region_of_interest(image, vertices)

	#plt.imshow(masked_image)

	#plt.show()

	line_img = hough_lines(masked_image, rho, theta, threshold, min_line_len, max_line_gap)

	plt.imshow(line_img)

	plt.show()


	# Create a "color" binary image to combine with line image
	#color_edges = np.dstack((edges, edges, edges)) 
	#output = weighted_img(line_img, color_edges, 0.8, 1, 0)

	# Draw the lines on the edge image
	output = weighted_img(line_img, image, 1, 1, 0)
	

	plt.imshow(output)

	plt.show()

	new_filename = "processed_" + img

	#print(new_filename)
	plt.imsave("test_images/" + new_filename,output) 
