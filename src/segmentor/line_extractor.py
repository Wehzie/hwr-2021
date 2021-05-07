# %% 
import cv2
import numpy as np
from scipy.signal import argrelmin, find_peaks


class BoundingBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def get_bounding_boxes(img, max_height=250, min_area=10000):
    ''' Takes a thresholded image and returns an array of bounding boxes. 
        Bounding boxes with height smaller than max_height are split, boxes 
        with area smaller than min_area are discarded.
    '''
    proj = np.sum(img,1)
    peaks = find_peaks(proj, prominence=np.max(proj)/8)[0]
    
    # Add cross-through lines for text line center of gravity
    for y in peaks:
        cv2.line(img, (0, y), (img.shape[1], y), 255, 30)  

    # Estimate connected components
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
     
    boxes = []
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        keep_height = h < max_height
        keep_area = area > min_area
        if keep_height and keep_area:
            boxes.append(BoundingBox(x, y, w, h))
        elif keep_area and not keep_height:
            boxes += split_bounding_box(BoundingBox(x, y, w, h))   
    return boxes 

def split_bounding_box(box):
    ''' Just splits too large bounding box in the middle, might need more sophisticated method '''
    box1 = BoundingBox(box.x, box.y, box.w, box.h//2)
    box2 = BoundingBox(box.x, box.y+box.h//2, box.w, box.h//2)
    return [box1, box2] 

def clean_boxes(img, box, offset):
    ''' Takes an image, bounding box and offset width as input and returns an image of the bounding box without
    shapes introducing from other text lines'''
    
    img_pad = np.pad(img, ((offset, offset), (0, 0)))
    sub_img = img_pad[box.y:(box.y+2*offset+box.h), (box.x):(box.x+box.w)]
    
    cv2.rectangle(sub_img,(0,0),(sub_img.shape[1], sub_img.shape[0]), 255, 1)
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(sub_img, 8, cv2.CV_32S)
    comp_id = np.argmax(stats[1:-1,cv2.CC_STAT_AREA])+1
    
    componentMask = (labels == comp_id).astype("uint8") * 255
    res = np.where(componentMask, 0, sub_img)
    return res

def extract_lines(filename):
    '''Loads a given file from image-data and saves extracted lines
    to the lines folder
    '''
    img = cv2.imread(f"image-data/{filename}", cv2.IMREAD_GRAYSCALE)

    # Threshold and invert
    img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]
    img = cv2.bitwise_not(img)
    lines_img = img.copy()
    
    boxes = get_bounding_boxes(lines_img)
    for i, box in enumerate(boxes):
        clean = clean_boxes(img, box, 10)
        cv2.imwrite(f"lines/{filename}_L{i}.png", cv2.bitwise_not(clean))


if __name__ == "__main__":
    # Change this to loop which iterates over all binary images in image_data
    filename = 'P344-Fg001-R-C01-R01-binarized.jpg'
    extract_lines(filename)
# %%
