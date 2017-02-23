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



# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

# Define color selection criteria

# for white lines
red_threshold_white_profile = 200
green_threshold_white_profile = 200
blue_threshold_white_profile = 200

rgb_threshold_white_profile = [red_threshold_white_profile, green_threshold_white_profile, blue_threshold_white_profile]

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
min_line_len = 10 #minimum number of pixels making up a line
max_line_gap = 200    # maximum gap in pixels between connectable line segments


dirs = os.listdir("test_images/")

for img in dirs:
    #reading in an image
    path = "test_images/" + img

    image = mpimg.imread(path)
    image = image[:,:,:3]

    # Color Selection

    imgHSV = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20,80,80])
    upper_yellow = np.array([25,255,255])
    mask = cv2.inRange(imgHSV,lower_yellow,upper_yellow)
    imgHSV_Yellow_mask = cv2.bitwise_and(image,image, mask = mask)
    
    color_select_yellow = np.copy(imgHSV_Yellow_mask)
    color_select_white = np.copy(image)

    # Mask pixels below the threshold
    color_thresholds = (image[:,:,0] < rgb_threshold_white_profile[0]) | (image[:,:,1] < rgb_threshold_white_profile[1]) | (image[:,:,2] < rgb_threshold_white_profile[2])
    color_select_white[color_thresholds] = [0,0,0]
    color_select = cv2.bitwise_and(cv2.bitwise_or(color_select_yellow, color_select_white), image)

    #grayscale the image
    gray = grayscale(color_select)

    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussian_blur(gray,kernel_size)

    #Canny Edge detection
    edges = canny(blur_gray, low_threshold, high_threshold)

    #Defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(int (0.45*imshape[1]), int (0.6*imshape[0])), (int(0.55*imshape[1]), int (0.6*imshape[0])), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_image = region_of_interest(edges, vertices)

    line_img = hough_lines(masked_image, rho, theta, threshold, min_line_len, max_line_gap)

    # Draw the lines on the edge image
    output = weighted_img(line_img, image, 1, 1, 0)

    new_filename = "processed_" + img

    plt.imsave("test_images/" + new_filename,output) 

