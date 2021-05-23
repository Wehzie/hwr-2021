import glob
import os
import cv2 as cv
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks


class BoundingBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def get_bounding_boxes(img, max_height=250, min_area=10000):
    """
    Takes a thresholded image and returns an array of bounding boxes.
    Bounding boxes with height smaller than max_height are split,
    boxes with area smaller than min_area are discarded.
    """
    proj = np.sum(img, 1)
    peaks = find_peaks(proj, prominence=np.max(proj) / 8)[0]

    # Add cross-through lines for text line center of gravity
    for y in peaks:
        cv.line(img, (0, y), (img.shape[1], y), 255, 30)

    # Estimate connected components
    (num_labels, labels, stats, centroids) = cv.connectedComponentsWithStats(
        img, 8, cv.CV_32S
    )

    boxes = []
    for i in range(1, num_labels):
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        area = stats[i, cv.CC_STAT_AREA]
        keep_height = h < max_height
        keep_area = area > min_area
        if keep_height and keep_area:
            boxes.append(BoundingBox(x, y, w, h))
        elif keep_area and not keep_height:
            boxes += split_bounding_box(BoundingBox(x, y, w, h))
    return boxes


def split_bounding_box(box):
    """
    Just splits too large bounding box in the middle,
    might need more sophisticated method
    """
    box1 = BoundingBox(box.x, box.y, box.w, box.h // 2)
    box2 = BoundingBox(box.x, box.y + box.h // 2, box.w, box.h // 2)
    return [box1, box2]


def clean_boxes(img, box, offset):
    """
    Takes an image, bounding box and offset width as input and returns an image
    of the bounding box without shapes introducing from other text lines
    """

    img_pad = np.pad(img, ((offset, offset), (0, 0)))
    sub_img = img_pad[box.y : (box.y + 2 * offset + box.h), (box.x) : (box.x + box.w)]

    cv.rectangle(sub_img, (0, 0), (sub_img.shape[1], sub_img.shape[0]), 255, 1)
    (num_labels, labels, stats, centroids) = cv.connectedComponentsWithStats(
        sub_img, 8, cv.CV_32S
    )
    comp_id = np.argmax(stats[1:-1, cv.CC_STAT_AREA]) + 1

    component_mask = (labels == comp_id).astype("uint8") * 255
    res = np.where(component_mask, 0, sub_img)
    return res


def correct_rotation(image_path, threshold):
    """
    Rotate the image so the lines are (mostly) horizontal,
    instead of (slightly) tilted.
    TODO: Dynamically set threshold for better hough line detection in some
     images while reducing unneccesarily high number of lines in others.
    """
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        print("Image not found")
        return

    edges = cv.Canny(img, 50, 200)
    lines = cv.HoughLines(
        edges, 1, np.pi / 180, threshold, min_theta=0.45 * np.pi, max_theta=0.55 * np.pi
    )

    while (lines is None or not (10 < len(lines) < 100)) and threshold > 10:
        if lines is not None and len(lines) > 100:
            threshold += 10
        else:
            threshold -= 10

        print(f"Changing threshold to {threshold}")
        lines = cv.HoughLines(
            edges, 1, np.pi / 180, threshold, min_theta=0.45 * np.pi, max_theta=0.55 * np.pi
        )


    hough = img.copy()
    for r, theta in lines[:, 0, :]:
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * r
        y0 = b * r

        x1 = int(x0 + 8000 * (-b))
        y1 = int(y0 + 8000 * (a))
        x2 = int(x0 - 8000 * (-b))
        y2 = int(y0 - 8000 * (a))
        cv.line(hough, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imwrite(f"img/{os.path.basename(image_path)}_hough.png", hough)

    thetas = lines[:, 0, 1] - 0.5 * np.pi
    rotation = np.degrees(np.mean(thetas))
    # Other rotation finding techniques that are currently not in use, using
    # binning instead of simple averaging. Seems like it had a worse effect.
    # bins = np.array([np.radians(deg) for deg in range(-20, 21)])
    # binned = np.digitize(thetas, bins)
    # rotation = np.degrees(bins[stats.mode(binned).mode[0]])
    # rotation = np.degrees(np.mean(bins[np.array(Counter(binned).most_common(3))[:, 0]]))
    print(f"Avg. rotation based on {lines[:, 0, 1].size} lines: {rotation} deg.")
    return ndimage.rotate(img, rotation, cval=255)


def extract_lines(image_path):
    """
    Loads a given file from image-data and saves extracted lines
    to the lines folder
    """

    file_name = os.path.basename(image_path)
    rotated_img = correct_rotation(image_path, 100)
    cv.imwrite(f"img/{file_name}_rotated.png", rotated_img)

    #Threshold and invert
    img = cv.threshold(rotated_img, 127, 255, cv.THRESH_BINARY)[1]
    img = cv.bitwise_not(img)
    lines_img = img.copy()

    boxes = get_bounding_boxes(lines_img)

    for i, box in enumerate(boxes):
        # clean = clean_boxes(img, box, 10)
        clean = img[box.y : box.y + box.h, box.x : box.x + box.w]
        cv.imwrite(f"lines/{file_name}_L{i}.png", cv.bitwise_not(clean))


if __name__ == "__main__":
    # Change this to loop which iterates over all binary images in image_data
    for img in glob.glob("../../data/image-data/*binarized.jpg"):
        print(f"Processing {os.path.basename(img)}")
        extract_lines(img)
