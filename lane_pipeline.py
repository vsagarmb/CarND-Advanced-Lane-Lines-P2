from image_gen import *
from camera_cal import *
from moviepy.editor import VideoFileClip

class Lane():
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.left_fit_m = None
        self.right_fit_m = None
        self.left_curvature = None
        self.right_curvature = None


def calculateLanes(img):
    """
    Calculates the lane on image `img`.
    """
    yRange = 719

    left_fit, right_fit, left_fit_m, right_fit_m, _, _, out_img, _, _ = findLines(img)    

    # Calculate curvature
    left_curvature = calculateCurvature(yRange, left_fit_m) 
    right_curvature = calculateCurvature(yRange, right_fit_m)
    
    # Calculate vehicle center
    xMax = img.shape[1]*xm_per_pix
    yMax = img.shape[0]*ym_per_pix
    vehicleCenter = xMax / 2
    lineLeft = left_fit_m[0]*yMax**2 + left_fit_m[1]*yMax + left_fit_m[2]
    lineRight = right_fit_m[0]*yMax**2 + right_fit_m[1]*yMax + right_fit_m[2]
    lineMiddle = lineLeft + (lineRight - lineLeft)/2
    dif_from_vehicle = lineMiddle - vehicleCenter
    
    return (left_fit, right_fit, left_fit_m, right_fit_m, left_curvature, right_curvature, dif_from_vehicle)

def displayLanes(img, left_fit, right_fit, left_fit_m, right_fit_m, left_curvature, right_curvature, dif_from_vehicle):
    """
    Display the lanes information on the image.
    """
    output = drawLine(img, left_fit, right_fit)
    
    return output
    
    
def videoPipeline(inputVideo, outputVideo):
    """
    Process the `inputVideo` frame by frame to find the lane lines, draw curvarute and vehicle position information and
    generate `outputVideo`
    """
    myclip = VideoFileClip(inputVideo)
    #myclip = myclip.subclip(23, 24)
    
    leftLane = Lane()
    rightLane = Lane()

    #computeCameraCal()
    WarpImgConstants()    

    def process_image(img):           

        left_fit, right_fit, left_fit_m, right_fit_m, left_curvature, right_curvature, dif_from_vehicle = calculateLanes(img)

        if left_curvature > 10000:
            left_fit = leftLane.left_fit
            left_fit_m = leftLane.left_fit_m
            left_curvature = leftLane.left_curvature
        else:
            leftLane.left_fit = left_fit
            leftLane.left_fit_m = left_fit_m
            leftLane.left_curvature = left_curvature
        
        if right_curvature > 10000:
            right_fit = rightLane.right_fit
            right_fit_m = rightLane.right_fit_m
            right_curvature = rightLane.right_curvature
        else:
            rightLane.right_fit = right_fit
            rightLane.right_fit_m = right_fit_m
            rightLane.right_curvature = right_curvature
            
        img = displayLanes(img, left_fit, right_fit, left_fit_m, right_fit_m, left_curvature, right_curvature, dif_from_vehicle)

        compImage = composeImageBase(img, left_curvature, right_curvature, dif_from_vehicle)        

        return compImage

    clip = myclip.fl_image(process_image)
    clip.write_videofile(outputVideo, audio=False)

# Project video
videoPipeline('project_video.mp4', 'output_project_video.mp4')