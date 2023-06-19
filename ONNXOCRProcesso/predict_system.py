import cv2
import numpy as np
import logging
from typing import List, Optional, Iterable
import copy
from .detection import TextDetector
from .classification import TextClassifier
from .recognition import TextRecognizer

log = logging


def adjust_image(img, corners):
    image_width = int(
        max(
            np.linalg.norm(corners[0] - corners[1]),
            np.linalg.norm(corners[2] - corners[3])))
    image_height = int(
        max(
            np.linalg.norm(corners[0] - corners[3]),
            np.linalg.norm(corners[1] - corners[2])))
    points = np.float32([[0, 0], [image_width, 0],
                         [image_width, image_height],
                         [0, image_height]])
    M = cv2.getPerspectiveTransform(corners, points)
    result_img = cv2.warpPerspective(
        img,
        M, (image_width, image_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    if result_img.shape[0] * 1.0 / result_img.shape[1] >= 1.5:
        result_img = np.rot90(result_img)

    return result_img


class TextProcessor:
    def __init__(self, use_angle_classifier=True, det_model=None, rec_model=None, ort_providers=None):
        self.detector = TextDetector(det_model_path=det_model, ort_providers=ort_providers)
        self.recognizer = TextRecognizer(rec_model_path=rec_model, ort_providers=ort_providers)
        self.use_angle_classifier = use_angle_classifier
        if self.use_angle_classifier:
            self.classifier = TextClassifier(ort_providers=ort_providers)

    def whitelist_chars(self, chars: Optional[Iterable[str]]):
        self.recognizer.set_char_whitelist(chars)

    def single_line_ocr(self, img):
        result = self.multi_line_ocr([img])
        if result:
            return result[0]

    def multi_line_ocr(self, image_list: List[np.ndarray]):
        tmp_image_list = []
        for img in image_list:
            if img.shape[0] * 1.0 / img.shape[1] >= 1.5:
                img = np.rot90(img)
            tmp_image_list.append(img)
        recognition_result, elapse = self.recognizer(tmp_image_list)
        return recognition_result

    def detect_and_recognize(self, img: np.ndarray, drop_score=0.5, ratio=None, threshold=None):
        original_img = img.copy()
        detected_boxes, elapse = self.detector(img, ratio, threshold)
        log.debug("Number of detected boxes: {}, Elapsed time: {}".format(len(detected_boxes), elapse))
        if detected_boxes is None:
            return []
        cropped_image_list = []
        detected_boxes = box_sort(detected_boxes)
        for box in detected_boxes:
            temp_box = copy.deepcopy(box)
            cropped_img = adjust_image(original_img, temp_box)
            cropped_image_list.append(cropped_img)
        if self.use_angle_classifier:
            cropped_image_list, angle_list, elapse = self.classifier(cropped_image_list)
            log.debug("Number of classified images : {}, Elapsed time: {}".format(len(cropped_image_list), elapse))

        recognition_result, elapse = self.recognizer(cropped_image_list)
        log.debug("Number of recognition results: {}, Elapsed time: {}".format(len(recognition_result), elapse))
        result = []
        for box, rec_result, cropped_img in zip(detected_boxes, recognition_result, cropped_image_list):
            text, score = rec_result
            if score >= drop_score:
                result.append(Result(box, cropped_img, text, score))
        return result


class Result:
    box: List[int]
    image: np.ndarray
    text: str
    score: float

    def __init__(self, box, image, text, score):
        self.box = box
        self.image = image
        self.text = text
        self.score = score

    def get_box(self):
        return self.box

    def __str__(self):
        return 'Result[%s, %s]' % (self.text, self.score)

    def __repr__(self):
        return self.__str__()


def box_sort(boxes):
    num_boxes = boxes.shape[0]
    sorted_boxes = sorted(boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp

    return _boxes
