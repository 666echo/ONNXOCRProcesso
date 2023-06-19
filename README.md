# ONNXOcrInference

ONNXOcrInference: 是一个基于ONNX（开放神经网络交换）模型的光学字符识别（OCR）工具。它充分利用了ONNX的跨平台优势，可以对图像进行OCR操作，识别并提取图像中的文本信息。

这个工具使用预训练的ONNX模型进行文本检测和识别，支持多种语言，并能处理各种类型和质量的图像。无论您是需要从文档中提取文本，还是从自然场景的图像中识别文字，ONNXOcrInference都是一个强大、快速和精确的解决方案。

ONNXOcrInference同时提供了对个别图片和批量图片的处理能力，并且可以将识别结果进行可视化，使其更加易于理解和分析。整个工具的设计目标是快速、高效且用户友好，无论你是初学者还是有经验的开发者，都能轻松上手。

在未来的开发中，我们计划不断优化和扩展ONNXOcrInference的功能，包括更高效的图像处理算法，更广泛的语言支持，以及更多的自定义选项等。

欢迎试用ONNXOcrInference，并向我们提供反馈和建议，我们非常珍视您的意见，并会用它们来改进我们的工具。让我们一起通过ONNXOcrInference，开启OCR的全新篇章。

# 开始

请首先安装Python和pip，然后通过pip安装本项目所需的依赖库。你可以通过以下命令进行安装：

> pip install -r requirements.txt


## 以下是一个简单的使用示例，演示如何使用ONNXOcrInference从图像中识别和提取文本信息。

 
```
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
```
