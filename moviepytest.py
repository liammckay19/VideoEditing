
from glob import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import pygame
import gizeh as gz
from math import sin
from moviepy.editor import VideoFileClip, concatenate, CompositeVideoClip
from scipy.fft import fft, fftfreq

video = VideoFileClip(glob("Finger*")[0])
audio = video.audio

(w, h), d = video.size, video.duration
center=  np.array([w/2, h/2])
CHUNK = 10*2


def detect_face(image):
    # Get user supplied values
    cascPath = "haarcascade_frontalface_default.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image

def invert_green_blue(image):
    return image[:,:,[0,2,1]]

def drawContourFX(image):

    edged = cv2.Canny(image,200,200)
    contours, hierarchy = cv2.findContours(edged, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (255, 255, 255), 1)
    return image


def convexHullFX(get_frame, t):
    import cv2 as cv
    import numpy as np
    import argparse
    import random as rng
    image = get_frame(t)
    rng.seed(12345)
    def thresh_callback(val):
        threshold = val
        # Detect edges using Canny
        canny_output = cv.Canny(src_gray, threshold, threshold * 2)
        # Find contours
        contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Find the convex hull object for each contour
        hull_list = []
        for i in range(len(contours)):
            hull = cv.convexHull(contours[i])
            hull_list.append(hull)
        # Draw contours + hull results
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv.drawContours(drawing, contours, i, color)
            cv.drawContours(drawing, hull_list, i, color)
        return drawing
    # Convert image to gray and blur it
    src_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3,3))
    # Create Window
    max_thresh = 255
    thresh = 100 # initial threshold
    return thresh_callback(thresh)

def polygon(get_frame, t):
    """ Transforms a frame (given by get_frame(t)) into a different
    frame, using vector graphics."""

    surface = gz.Surface(w,h)
    fill = (gz.ImagePattern(get_frame, pixel_zero=center)
            .scale(1.5, center=center))
    nfaces,angle,f = [5, sin(t)/10, 3.0/6]
    xy = (f*w, h*(.5+ .05*np.sin(2*np.pi*(t/d+f))))
    shape = gz.regular_polygon(w/2,nfaces, xy = xy,
            fill=fill.rotate(angle, center))
    shape.draw(surface)
    return surface.get_npimage()

def zoom_face(get_frame, t):
    face_choice=0

    image = get_frame(t)
    image = detect_face(image)

    return image




# pygame.sndarray.array(video.audi)
def effect(get_frame, t):
    image = get_frame(t)
    image = polygon(image, t)
    blue_channel = image[:,:,0]
    rows,cols,_ = image.shape
    volume = np.int(np.abs(audio.get_frame(t)[0])*50)
    M = np.float32([[1,0,5+volume],[0,1,0]])
    
    blue_channel = cv2.warpAffine(blue_channel,M,(cols,rows))
    # create empty image with same shape as that of image image
    blue_img = np.zeros(image.shape)

    #assign the red channel of image to empty image
    blue_img[:,:,0] = blue_channel
    
    red_channel = image[:,:,2]

    # create empty image with same shape as that of image image
    red_img = np.zeros(image.shape)

    #assign the red channel of image to empty image
    red_img[:,:,2] = red_channel
    blue_img = np.asarray(blue_img, np.float64)
    red_img = np.asarray(red_img, np.float64)
    image = np.asarray(image, np.float64)
    final = cv2.addWeighted(blue_img,0.5,red_img,0.5,0)
    final = np.asarray(final, np.float64)
    final = cv2.addWeighted(final,0.7,image,0.3,0.4)

    return final

def preview_fix(video):
    aud = video.audio.set_fps(44100)
    subclip = video.without_audio().set_audio(aud)
    subclip.preview()

def clip_based_on_peaks(video, duration, threshold, count=10):
    peaks = np.argwhere(audio.to_soundarray(fps=44000) > threshold)
    # print(peaks[0:10])
    clip_list=[]
    for i in range(0,len(peaks),len(peaks)//count):
        start_clip,_ = peaks[i]
        # end_clip,_ = peaks[i+1]
        start_clip = start_clip//44000
        # end_clip = end_clip//44000
        print(start_clip)
        start_cut = start_clip
        end_cut = start_clip + duration
        if start_cut < video.duration > end_cut:
            clip_list.append(video.subclip(start_cut, end_cut))
        
    mega_clipped = CompositeVideoClip(
        [clip.set_start(i*duration) for i,clip in enumerate(clip_list)]
    ) # start at t=9s
    return mega_clipped


def clip_based_on_dips(video, duration, threshold, count=10):
    peaks = np.argwhere(audio.to_soundarray(fps=44000) < threshold)
    # print(peaks[0:10])
    clip_list=[]
    for i in range(0,len(peaks),len(peaks)//count):
        start_clip,_ = peaks[i]
        # end_clip,_ = peaks[i+1]
        start_clip = start_clip//44000
        # end_clip = end_clip//44000
        print(start_clip)
        start_cut = start_clip
        end_cut = start_clip + duration
        if start_cut < video.duration > end_cut:
            clip_list.append(video.subclip(start_cut, end_cut))
        
    mega_clipped = CompositeVideoClip(
        [clip.set_start(i*duration) for i,clip in enumerate(clip_list)]
    ) # start at t=9s
    return mega_clipped

# def clip_out_silence(video, duration, threshold_lower, threshold_upper, all_, count=10):
#     peaks = np.argwhere(threshold_lower < audio.to_soundarray(fps=44000)[0:] < threshold_upper)
#     print(peaks)
#     clip_list=[]
#     for i in range(0,len(peaks),len(peaks)//count):
#         start_clip,_ = peaks[i]
#         # end_clip,_ = peaks[i+1]
#         start_clip = start_clip//44000
#         # end_clip = end_clip//44000
#         print(start_clip)
#         start_cut = start_clip
#         end_cut = start_clip + duration
#         if start_cut < video.duration > end_cut:
#             clip_list.append(video.subclip(start_cut, end_cut))
        
#     mega_clipped = CompositeVideoClip(
#         [clip.set_start(i*duration) for i,clip in enumerate(clip_list)]
#     ) # start at t=9s
#     return mega_clipped
modifiedClip = video.fl( convexHullFX )
# preview_fix(modifiedClip)
modifiedClip.write_videofile("geometry_class.webm", audio=True)

# preview_fix(clip_based_on_peaks(video, 2, 0.8))
# preview_fix(clip_based_on_peaks(video, 0.1, 0.92, count=50))
# clip_based_on_dips(video, 0.1, 0.01, count=200).write_videofile("theglitch_concert1.webm", audio=True)

# modifiedClip.write_videofile("concert1.mp4", audio=True) # Many options...