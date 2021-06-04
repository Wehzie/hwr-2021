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
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def get_bounding_boxes(img: np.ndarray, max_height=250, min_area=10000):
    """
    Takes a thresholded image and returns an array of bounding boxes.
    Bounding boxes with height smaller than max_height are split,
    boxes with area smaller than min_area are discarded.
    """
    img = img.copy()
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


def clean_boxes(img: np.ndarray, box: BoundingBox):
    """
    Takes an image, bounding box and offset width as input and returns an image
    of the bounding box without shapes intruding from other text lines
    """
    offset = 3
    img_pad = np.pad(img, ((offset, offset), (0, 0)))
    sub_img = img_pad[box.y : (box.y + 2 * offset + box.h), (box.x) : (box.x + box.w)]
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


def correct_rotation(img: np.ndarray, hough_threshold: int):
    """
    Rotate the image so the lines are (mostly) horizontal,
    instead of (slightly) tilted.
    TODO: Dynamically set threshold for better hough line detection in some
     images while reducing unneccesarily high number of lines in others.
    """

    edges = cv.Canny(img, 50, 200)
    lines = cv.HoughLines(
        edges,
        1,
        np.pi / 180,
        hough_threshold,
        min_theta=0.45 * np.pi,
        max_theta=0.55 * np.pi,
    )

    while (lines is None or not 10 < len(lines) < 100) and hough_threshold > 10:
        hough_threshold += 10 if lines is not None and len(lines) > 100 else -10

        # print(f"Changing threshold to {threshold}")
        lines = cv.HoughLines(
            edges,
            1,
            np.pi / 180,
            hough_threshold,
            min_theta=0.45 * np.pi,
            max_theta=0.55 * np.pi,
        )

    # hough_img = overlay_hough_lines(img, lines)
    # cv.imwrite(f"img/{img_path.stem}_hough.png", hough_img)

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


def extract_lines(img_path: Path, out_dir: Path):
    """
    Loads a given file from image-data and saves extracted lines
    to the lines folder
    """

    img = cv.imread(str(img_path.resolve()), cv.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Image {img_path.name} not found")
        return

    rotated_img = correct_rotation(img, 100)

    # Threshold and invert
    img = cv.threshold(rotated_img, 127, 255, cv.THRESH_BINARY)[1]
    img = cv.bitwise_not(img)

    boxes = get_bounding_boxes(img)

    for i, box in enumerate(boxes):
        clean = clean_boxes(img, box)
        # clean = img[box.y : box.y + box.h, box.x : box.x + box.w]
        cv.imwrite(
            str((out_dir / f"{img_path.stem}_L{i}.png").resolve()),
            cv.bitwise_not(clean),
        )


if __name__ == "__main__":
    args = ArgParser(
        description="Extract lines from binarized dead sea scroll pictures"
    ).parse_args()

    # TODO: let loop iterate over all binarized images
    for img in args.input_dir.glob("*binarized.jpg"):
        print(f"Processing {img.name}")
        extract_lines(img, args.output_dir)
