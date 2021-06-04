from pathlib import Path

import cv2 as cv
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from tap import Tap


class ArgParser(Tap):
    input_dir: Path  # Directory that contains dead sea scroll pictures
    output_dir: Path  # Directory where extracted pictures are saved

    def configure(self):
        self.add_argument("input_dir")
        self.add_argument("output_dir")


class BoundingBox:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x: int = x
        self.y: int = y
        self.w: int = w
        self.h: int = h


def get_bounding_boxes(img: np.ndarray, max_height=250, min_area=1000) -> list:
    """
    Takes a thresholded image and returns an array of bounding boxes.
    Bounding boxes with height smaller than max_height are split,
    boxes with area smaller than min_area are discarded.
    """
    img = img.copy()
    proj = np.sum(img, 0)   # 0 = x-axis
    #print(len(proj))
    #out_str = ""
    #for i in range(0, len(proj), 30):
    #    out_str += str(proj[i])
    #print(out_str)
    #import matplotlib.pyplot as plt
    #plt.plot(proj)
    #plt.show()

    # a word is a connection of non-zeroes but 50 zeroes are allowed
    boxes = []
    height, width = img.shape 
    box = BoundingBox(None, 0, None, height)
    for val, i in enumerate(proj):
        # word beginning
        if val != 0:
            box.x = i
        # word end
        #if box.x != None and 

    #peaks = find_peaks(proj, prominence=np.max(proj), width=1)
    #print(peaks)
    exit()

    # Add cross-through lines for text line center of gravity
    #for y in peaks:
    #    cv.line(img, (0, y), (img.shape[1], y), 255, 30)

    # Estimate connected components
    def old():
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
            #boxes.append(BoundingBox(x, y, w, h))
            keep_height = h < max_height
            keep_area = area > min_area
            if keep_height and keep_area:
                boxes.append(BoundingBox(x, y, w, h))
            elif keep_area and not keep_height:
                boxes += split_bounding_box(BoundingBox(x, y, w, h))
    return boxes


def split_bounding_box(box: BoundingBox):
    """
    TODO: check if box is large AND bimodal, then split on valley.
    Splitting only on height leads to some large single lines being split.

    Just splits too large bounding box in the middle,
    might need more sophisticated method
    """
    box1 = BoundingBox(box.x, box.y, box.w, box.h // 2)
    box2 = BoundingBox(box.x, box.y + box.h // 2, box.w, box.h // 2)
    return box1, box2


def clean_boxes(img: np.ndarray, box: BoundingBox) -> np.ndarray:
    """
    Takes an image, bounding box and offset width as input and returns an image
    of the bounding box without shapes intruding from other text lines
    """
    offset: int = 3
    img_pad: np.ndarray = np.pad(img, ((offset, offset), (0, 0)))
    sub_img: np.ndarray = img_pad[box.y : (box.y + 2 * offset + box.h), (box.x) : (box.x + box.w)]
    cv.rectangle(sub_img, (0, 0), (sub_img.shape[1], sub_img.shape[0]), 255, 2)

    (num_labels, labels, stats, centroids) = cv.connectedComponentsWithStats(
        sub_img, 8, cv.CV_32S
    )
    if len(stats) > 2:
        comp_id = np.argmax(stats[1:-1, cv.CC_STAT_AREA]) + 1
    else:
        # This happens only once, seems due to bad box splitting.
        # Might not be necessary once better splitting implemented
        print("No components found")
        return sub_img

    component_mask = (labels == comp_id).astype("uint8") * 255
    res = np.where(component_mask, 0, sub_img)
    return res


def overlay_hough_lines(img: np.ndarray, lines):
    """
    Add the extracted hough lines to the original image.
    Only used for testing purposes.
    """
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

    return hough


def extract_words(img_path: Path, out_dir: Path) -> None:
    """
    Loads a line image and saves extracted words.
    """
    img = cv.imread(str(img_path.resolve()), cv.IMREAD_GRAYSCALE)
    assert np.all(img) != None, f"Image {img_path.name} not found!"

    # Threshold and invert
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
    img = cv.bitwise_not(img)

    boxes = get_bounding_boxes(img)

    for i, box in enumerate(boxes):
        #clean = clean_boxes(img, box)
        clean = img[box.y : box.y + box.h, box.x : box.x + box.w]
        cv.imwrite(
            str((out_dir / f"{img_path.stem}_W{i}.png").resolve()),
            cv.bitwise_not(clean),
        )


if __name__ == "__main__":
    args = ArgParser(
        description="Extract words from lines of binarized DSS images."
    ).parse_args()

    # TODO: let loop iterates over all lines of all fragments
    for img in args.input_dir.glob("*binarized_L*"):
        print(f"Processing {img.name}")
        extract_words(img, args.output_dir)
