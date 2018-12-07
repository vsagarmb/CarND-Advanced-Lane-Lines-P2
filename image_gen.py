import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

dist_pickle = pickle.load(open('camera_cal/cal_pickle.p','rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

def cal_undistort(img, mtx, dist):        
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
    
    
def abs_sobel_thresh(im, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    if orient == 'y':
        sobelx = sobely
    
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return sxbinary

def mag_thresh(im, sobel_kernel=3, mag_thresh=(0, 255)):
    
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_sobelxy = np.sqrt((sobelx**2)+(sobely**2))
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    sxybinary = np.zeros_like(scaled_sobel)
    sxybinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return sxybinary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def rgb_select(img, r_thresh=(0,255),g_thresh=(0,255),b_thresh=(0,255)):
    r_channel = img[:, :, 2]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 0]

    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 1

    cv2.imshow("r_binary", r_binary)

    g_binary = np.zeros_like(g_channel)
    g_binary[(g_channel >= g_thresh[0]) & (g_channel <= g_thresh[1])] = 1

    cv2.imshow("g_binary", g_binary)

    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1

    cv2.imshow("b_binary", b_binary)
    return combineChannels(combineChannels(r_binary, g_binary), b_binary)    

def s_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def combineGradients(image):

    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, 'x', ksize, 10, 160)

    grady = abs_sobel_thresh(image, 'y', ksize, 10, 160)

    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(10, 160))

    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.79, 1.2))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined

    #return combineChannels(combined, s_select(image, (170, 255)))   

def combineChannels(chan1, chan2):
    combined = np.zeros_like(chan1)
    combined[(chan1 == 1) & (chan2 == 1)] = 1
    return combined

def WarpImgConstants():

    img = cv2.imread('./test_images/straight_lines1.jpg')

    # Undistort the image
    undist = cal_undistort(img, mtx, dist)

    xSize, ySize, _ = undist.shape
    copy = undist.copy()

    bottomY = 720
    topY = 455


    left1 = (190, bottomY)
    left1_x, left1_y = left1
    left2 = (585, topY)
    left2_x, left2_y = left2

    right1 = (705, topY)
    right1_x, right1_y = right1

    right2 = (1130, bottomY)
    right2_x, right2_y = right2

    color = [255, 0, 0]
    w = 2

    cv2.line(copy, left1, left2, color, w)
    cv2.line(copy, left2, right1, color, w)
    cv2.line(copy, right1, right2, color, w)
    cv2.line(copy, right2, left1, color, w)    

    src = np.float32([ 
        [left2_x, left2_y],
        [right1_x, right1_y],
        [right2_x, right2_y],
        [left1_x, left1_y]
    ])
    nX = undist.shape[1]
    nY = undist.shape[0]
    img_size = (nX, nY)
    offset = 200
    dst = np.float32([
        [offset, 0],
        [img_size[0]-offset, 0],
        [img_size[0]-offset, img_size[1]], 
        [offset, img_size[1]]
    ])

    img_size = (nX, nY)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    dist_pickle['M'] = M
    dist_pickle['Minv'] = Minv

    pickle.dump(dist_pickle,open('camera_cal/cal_pickle.p', 'wb'))

'''
    print(M)
    print(Minv)        

    warped = cv2.warpPerspective(undist, M, img_size)
    cv2.imwrite('./output_images/warped1.jpg', warped)

    img = cv2.imread('test_images\\test6.jpg')  
    undist = cal_undistort(img, mtx, dist)
    
    warped = cv2.warpPerspective(undist, M, img_size)
    cv2.imwrite('./output_images/warped1.jpg', warped)
'''


def changePerspective(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)
    cv2.imwrite('./output_images/warped.jpg', warped)
    
    return warped

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def show_histogram(out_img, histogram, ploty, left_fitx, right_fitx):
    plt.figure(figsize=(12.8, 7.2))
    plt.imshow(out_img.astype(np.uint8), cmap=None)
    plt.plot(720 - histogram, color='blue', linewidth=5.0)
    plt.plot(left_fitx, ploty, color='yellow', linewidth=5.0)
    plt.plot(right_fitx, ploty, color='yellow', linewidth=5.0)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('history.jpg')    


def findLines(image, nwindows=9, margin=110, minpix=50):
    """
    Find the polynomial representation of the lines in the `image` using:
    - `nwindows` as the number of windows.
    - `margin` as the windows margin.
    - `minpix` as minimum number of pixes found to recenter the window.
    - `ym_per_pix` meters per pixel on Y.
    - `xm_per_pix` meters per pixels on X.
    
    Returns (left_fit, right_fit, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)
    """    
    
    # Make a binary and transform image
    binary_warped = combineAndTransform(image)
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Fit a second order polynomial to each
    left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)    

    show_histogram(out_img, histogram, ploty, left_fitx, right_fitx)  

    return (left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)


def calculateCurvature(yRange, left_fit_cr):
    """
    Returns the curvature of the polynomial `fit` on the y range `yRange`.
    """    
    return ((1 + (2*left_fit_cr[0]*yRange*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])

def drawLine(img, left_fit, right_fit):
    """
    Draw the lane lines on the image `img` using the poly `left_fit` and `right_fit`.
    """
    Minv = dist_pickle['Minv']
    yMax = img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros_like(img).astype(np.uint8)
    
    # Calculate points.
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)


def combineAndTransform(img):
    undist = cal_undistort(img, mtx, dist)
    #cv2.imwrite('./output_images/undistorted.jpg', undist)

    gradient_img = combineGradients(undist)    
    stacked = np.dstack((gradient_img, gradient_img, gradient_img))*255
    #cv2.imwrite('./output_images/threshold.jpg', stacked)
    cv2.imwrite('stacked_gradient_img.jpg', stacked)    

    return changePerspective(gradient_img, dist_pickle['M'])

def composeImageBase(image, left_curvature, right_curvature, dif_from_vehicle): 

    combImg = cv2.imread('stacked_gradient_img.jpg')

    historyImg = cv2.imread('history.jpg')

    ratio = 0.29

    combImg_r = cv2.resize(combImg, None, fx = ratio, fy = ratio, interpolation=cv2.INTER_AREA)

    persp_f = cv2.resize(historyImg, None, fx = ratio, fy = ratio, interpolation=cv2.INTER_AREA)

    image[0:combImg_r.shape[0],0:combImg_r.shape[1]] = combImg_r

    offset_x = 0
    width_x = persp_f.shape[0]

    offset_y = image.shape[1]-persp_f.shape[1]
    width_y = persp_f.shape[1] 

    image[offset_x:offset_x+width_x,offset_y:offset_y+width_y] = persp_f

    base_x = 380
    basy_y = 65

    if dif_from_vehicle > 0:
        message = '{:.2f}m right'.format(dif_from_vehicle)
    else:
        message = '{:.2f}m left'.format(-dif_from_vehicle)

    cv2.putText(image, "Offset from Center: {}".format(message), (base_x, basy_y), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
    cv2.putText(image, "Left Curvature: {:6.2f}m".format(left_curvature), (base_x, basy_y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
    cv2.putText(image, "Right Curvature: {:6.2f}m".format(right_curvature), (base_x, basy_y+100), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)

    return image