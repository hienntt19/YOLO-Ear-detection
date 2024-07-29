import os
import sys
import cv2
import torch

# from utils.convert_annotations import denormalize
import math
import os


IMAGE_WIDTH = 2976
IMAGE_HEIGHT = 1984

def denormalize(x_norm, y_norm, w_norm, h_norm):
    w = round(w_norm * IMAGE_WIDTH)
    h = round(h_norm * IMAGE_HEIGHT)
    x = round(x_norm * IMAGE_WIDTH - w / 2)
    y = round(y_norm * IMAGE_HEIGHT - h / 2)
    return [x, y, w, h]


class Detector:
    weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), '', 'best.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

    # By calling with image_name, the model resizes the picture as appropriate
    def detect(self, image_name):
        detections = []
        results = self.model(image_name, size=416)
        img = cv2.imread(image_name)
        img_width, img_height = img.shape[:2]

        for tensor1, tensor2 in zip(results.xywhn, results.xywh):
            for result1, result2 in zip(tensor1, tensor2):
                x_norm, y_norm, w_norm, h_norm, _, _ = result1.numpy()
                x_, y_, w_, h_, _, _ = result2.numpy()
                detections.extend([denormalize(x_norm, y_norm, w_norm, h_norm)])

                # denormalize(results.xywh) != results.xywh !!
                # print(denormalize(x_norm, y_norm, w_norm, h_norm))
                # print(x_, y_, w_, h_)

        return detections


if __name__ == '__main__':
    # file_name = sys.argv[1]
    # img = cv2.imread(file_name)
    # detector = Detector()
    # detected_loc = detector.detect(file_name)
    # for x, y, w, h in detected_loc:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (128, 255, 0), 4)
    # cv2.imwrite(file_name + '.detected.jpg', img)

    root_data = 'D:\Poliface data'

    result_folder = 'D:\Test data\\final_result_10_person_ears'

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for folder in os.listdir(root_data):
        if not folder.startswith('V'):
            continue
        for session in os.listdir(os.path.join(root_data, folder)):
            for light in os.listdir(os.path.join(root_data, folder, session)):
                for emotion in os.listdir(os.path.join(root_data, folder, session, light)):
                    for i in os.listdir(os.path.join(root_data, folder, session, light, emotion)):
                        if not i.startswith('C'):
                            continue

                        img_path = os.path.join(root_data, folder, session, light, emotion, i)
                        img = cv2.imread(img_path)

                        detector = Detector()
                        detected_loc = detector.detect(img_path)
                        for x, y, w, h in detected_loc:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (128, 255, 0), 4)


                        cv2.imwrite(os.path.join(result_folder, img_path.replace('\\','_')), img)

                        # output_path = os.path.join(result_folder, f'{folder}_{session}_L{light}_E0{emotion}_C{i}.JPG')
                        # cv2.imwrite(output_path, cv2.cvtColor(label_img, cv2.COLOR_RGB2BGR))



