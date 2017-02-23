# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
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
    
    #color selection
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
    #masked_image = region_of_interest(image, vertices)
    
    line_img = hough_lines(masked_image, rho, theta, threshold, min_line_len, max_line_gap)
    
    # Draw the lines on the edge image
    output = weighted_img(line_img, image, 1, 1, 0)
    
    return output