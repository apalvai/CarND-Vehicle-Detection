import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from scipy.ndimage.measurements import label

from classifier import detect_vehicles_using_hog_sub_sampling, load_training_data, load_classifier

# load training data and classifier
features, y, X_scaler = load_training_data()
svc = load_classifier()

# heat_windows
boxes = None
boxes_limit = 10

# create an array to maintain an array for heats
heats = None
heats_limit = 10

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    
    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def update_heats(heat):
    global heats
    if heats is None:
        heats = []
    
    heats.append(heat)
    if len(heats) > heats_limit:
        heats = heats[-heats_limit:]
        # print('heats: ', len(heats))

def update_boxes(new_boxes):
    global boxes
    if boxes is None:
        boxes = []
    
    boxes.extend(new_boxes)
    if len(boxes) > boxes_limit:
        boxes = boxes[-boxes_limit:]
    # print('boxes: ', len(boxes))

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        if nonzerox.size == 0 or nonzeroy.size == 0:
            continue
        try:
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        except:
            pass
    # Return the image
    return img

def detect_vehicles_using_heat_maps(image):
    # get the vehicles enclosing boxes in the image
    features, y, X_scaler, svc = get_training_data_and_classifier()
    box_list = detect_vehicles_using_hog_sub_sampling(image, features, y, X_scaler, svc)
    
    # update boxes with box_list
    update_boxes(box_list)
    
    # create heat map
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat, boxes)
    
    # Append heat to the heats list
    update_heats(heat)
    
    # Compute the mean heat from previous N heat maps
    heat = np.mean(np.array(heats), axis = 0)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)
    
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    
    # Perform the operation
    heatmap = heatmap.astype(np.int8)
    output = cv2.connectedComponentsWithStats(heatmap)
    labels = (np.array(output[1]), output[0])
    # print('labels: ', labels)
    
    # Find final boxes from heatmap using label function
    # labels = label(heatmap)
    # print('labels: ', labels)
    
    return heatmap, labels

def get_training_data_and_classifier():
    return features, y, X_scaler, svc

def image_with_vehicles(image):
    # get the labels using heat maps
    heatmap, labels = detect_vehicles_using_heat_maps(image)
    
    # draw the labels on image
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    return draw_img
    # return draw_img, heatmap

def test():
    image_names = glob.glob('../test_images/*.jpg')
    
    for image_name in image_names:
        # Read in image similar to one shown above
        image = mpimg.imread(image_name)
        
        # draw the labels on image
        draw_img, heatmap = image_with_vehicles(image)
        
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show()

#test()

from moviepy.editor import VideoFileClip

white_output = '../white.mp4'
clip1 = VideoFileClip('../project_video.mp4')
white_clip = clip1.fl_image(image_with_vehicles)
white_clip.write_videofile(white_output, audio=False)

