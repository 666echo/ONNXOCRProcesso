from ONNXOCRProcesso.predict_system import TextProcessor
import cv2
import numpy as np

text_sys = TextProcessor()

img_path = './test.png'
img = cv2.imread(img_path)
res = text_sys.detect_and_recognize(img)
for idx, boxed_result in enumerate(res, start=1):
    print("Result {}:".format(idx))
    print("Category: {}".format(boxed_result.text))
    print("Confidence: {:.3f}".format(boxed_result.score))
    print("Coordinates: {}\n".format(boxed_result.box))
    pts = np.array(boxed_result.get_box(), np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

