import math
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from multiprocessing import Pool
import operator

import cv2
import numpy as np
import pytesseract


def show_image(image):
    try:
        cv2.imshow('Image', image)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()


def distance_between(p1, p2):
    return np.sqrt(sum((x2 - x1) ** 2 for x1, x2 in zip(p1, p2)))


def preprocess_image(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    show_image(rgb)
    blur_size = (5, 5)
    blur_std = 5
    blur = cv2.GaussianBlur(rgb, blur_size, blur_std)
    show_image(blur)
    # canny = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # Canny
    canny = cv2.Canny(blur, 1, 1)
    show_image(canny)
    return canny


def draw_corners(image, corners):
    cornered_image = image.copy()
    for corner in corners:
        x, y = corner
        cornered_image = cv2.circle(cornered_image, (x, y), radius=4, color=(0, 255, 0), thickness=-1)
    show_image(cornered_image)


def find_corners_of_board(preprocessed_image):
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    bounding = contours[0]
    bottom_right, _ = max(
        enumerate([pt[0][0] + pt[0][1] for pt in bounding]),
        key=operator.itemgetter(1)
    )
    top_left, _ = min(
        enumerate([pt[0][0] + pt[0][1] for pt in bounding]),
        key=operator.itemgetter(1)
    )
    bottom_left, _ = min(
        enumerate([pt[0][0] - pt[0][1] for pt in bounding]),
        key=operator.itemgetter(1)
    )
    top_right, _ = max(
        enumerate([pt[0][0] - pt[0][1] for pt in bounding]),
        key=operator.itemgetter(1)
    )
    corners_idx = [top_left, top_right, bottom_right, bottom_left]
    corners_raw = [bounding[corner][0] for corner in corners_idx]
    return corners_raw


def crop_and_warp(image, corners):
    src = np.array(corners, dtype='float32')
    side = max(distance_between(corners[i], corners[(i + 1) % 4]) for i in range(4))
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, m, (int(side), int(side)))
    show_image(warped)
    return warped


def get_cell(image, r, c):
    divisions = 15
    side = image.shape[:1]
    cell_length = side[0] / divisions
    top_offset = int(cell_length * r)
    left_offset = int(cell_length * c)
    bottom_offset = int(min(cell_length * (r + 1), side[0]))
    right_offset = int(min(cell_length * (c + 1), side[0]))
    return image[top_offset:bottom_offset, left_offset:right_offset]


def lin_contrast_stretching(image):
    # Default values
    a = 0
    b = 255

    # Max and min in each level of matrix
    x = np.max(image, axis=0)
    x = np.max(x, axis=0)
    y = np.min(image, axis=0)
    y = np.min(y, axis=0)
    image = image.astype(float)
    row, column, height = image.shape

    for i in range(row):
        for j in range(column):
            image[i][j][0] = (((b - a) / (x[0] - y[0])) * (image[i][j][0] - y[0])) + a
            image[i][j][1] = (((b - a) / (x[1] - y[1])) * (image[i][j][1] - y[1])) + a
            image[i][j][2] = (((b - a) / (x[2] - y[2])) * (image[i][j][2] - y[2])) + a

    image = image.astype(np.uint8)

    return image


def preprocess_cell_image(image):
    # 205 -> 204 with blur
    roi = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
    # Contrast + gray -> 210 correct
    roi = lin_contrast_stretching(roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    roi = cv2.resize(roi, (0, 0), fx=1.5, fy=1.5)
    thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return 255 - thresh


def extract_char_worker(image, r, c):
    # Unpack due to process pool executor constraints
    roi = get_cell(image, r, c)
    roi = preprocess_cell_image(roi)
    custom_config = r'--oem 1 --psm 11 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ" '
    text = pytesseract.image_to_string(roi, lang='eng', config=custom_config).strip()
    return r, c, text


def unpack_and_extract(args):
    return extract_char_worker(*args)


def get_ocr_board(image):
    # Multiprocess to run pytesseract
    ocr_text = [[''] * 15 for _ in range(15)]
    board_range = range(15)
    with ProcessPoolExecutor(max_workers=8) as executor:
        for r, c, text in executor.map(
                unpack_and_extract,
                product([image], board_range, board_range),
                chunksize=5
        ):
            ocr_text[r][c] = text
    return ocr_text


def read_board_checker(filename='default'):
    if filename == 'default':
        return [
            [''] * 15,
            [''] * 9 + ['H'] + [''] * 5,
            [''] * 9 + ['U'] + [''] * 5,
            [''] * 9 + ['E', 'G', 'G'] + [''] * 3,
            [''] * 9 + ['V'] + [''] * 5,
            [''] * 3 + ['P', 'E', 'Q', 'U', 'E', 'N', 'O', 'S'] + [''] * 4,
            [''] * 3 + ['I'] + [''] * 6 + ['M'] + [''] * 4,
            [''] * 2 + ['L', 'E', 'G'] + [''] * 5 + ['A'] + [''] * 3 + ['B'],
            [''] * 3 + ['R'] + [''] * 6 + list('LIBRO'),
            [''] + list('SONRISA') + [''] * 2 + ['L'] + [''] * 3 + ['O'],
            [''] * 3 + ['O'] + [''] * 2 + ['M'] + [''] * 7 + ['K'],
            [''] * 6 + ['I', '', 'R'] + [''] * 6,
            [''] * 6 + list('LEER') + [''] * 5,
            [''] * 6 + ['E', '', 'A'] + [''] * 6,
            [''] * 8 + ['D'] + [''] * 6
        ]
    with open(filename) as f:
        content = [[x if x != '_' else '' for x in row.split()] for row in f.read().split('\n')]
        return content


def create_board(image, expected_file='default'):
    # image = preprocess_image(image.copy())
    expected = read_board_checker(expected_file)
    ocr_text = get_ocr_board(image)
    # Run stats on detection
    correctly_detected = 0
    undetected = 0
    hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    total_occupied = 0
    unoccupied_max_std = 0
    occupied_min_std = math.inf
    for r in range(15):
        for c in range(15):
            # Experiment with hue
            hue_roi = get_cell(h, r, c)
            std_dev = np.std(hue_roi)
            if expected[r][c]:
                total_occupied += 1
                occupied_min_std = min(occupied_min_std, std_dev)
            else:
                unoccupied_max_std = max(unoccupied_max_std, std_dev)
            # End of experiment
            text = ocr_text[r][c]
            if not text and expected[r][c]:
                undetected += 1
                print(f'Found nothing at {r},{c} but expected {expected[r][c]}.')
                cell = get_cell(image, r, c)
                show_image(cell)
                show_image(preprocess_cell_image(cell))
            elif text and expected[r][c]:
                if text != expected[r][c]:
                    print(f'Found {text} at {r},{c} instead of {expected[r][c]}')
                    cell = get_cell(image, r, c)
                    show_image(cell)
                    show_image(preprocess_cell_image(cell))
                else:
                    correctly_detected += 1
            elif not text and not expected[r][c]:
                correctly_detected += 1
            else:
                print(f'Expected nothing but found {text} at {r}, {c}.')
    print(f'Summary matches -> {correctly_detected} correctly detected; {undetected} undetected.')
    print(f'Experiment summary: Hue std dev for unoccupied, occupied was {unoccupied_max_std}, {occupied_min_std}.')


if __name__ == '__main__':
    in_to_out_mapper = {
        "images/sample_board_new.png": "default",
        "images/board1_srdewan.jpeg": "checkers/board1.txt"
    }
    # filename = "images/sample_board_new.png"
    filename = "images/board1_srdewan.jpeg"
    bgr_image = cv2.imread(filename)
    preprocess = preprocess_image(bgr_image)
    corners = find_corners_of_board(preprocess)
    draw_corners(bgr_image, corners)
    warped = crop_and_warp(bgr_image, corners)
    create_board(warped, expected_file=in_to_out_mapper[filename])
