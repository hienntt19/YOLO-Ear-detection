import os
import cv2
import torch
from PIL import Image, UnidentifiedImageError

def denormalize(x_norm, y_norm, w_norm, h_norm, img_width, img_height):
    w = round(w_norm * img_width)
    h = round(h_norm * img_height)
    x = round(x_norm * img_width - w / 2)
    y = round(y_norm * img_height - h / 2)
    return [x, y, w, h]

class Detector:
    weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), '', 'best.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

    def detect(self, image_name):
        detections = []
        try:
            img = Image.open(image_name)
        except UnidentifiedImageError:
            print(f"Cannot identify image file '{image_name}'")
            return []

        results = self.model(img, size=416)
        img_cv = cv2.imread(image_name)
        img_height, img_width = img_cv.shape[:2]

        for tensor1, tensor2 in zip(results.xywhn, results.xywh):
            for result1, result2 in zip(tensor1, tensor2):
                x_norm, y_norm, w_norm, h_norm, _, _ = result1.numpy()
                x_, y_, w_, h_, _, _ = result2.numpy()
                detections.extend([denormalize(x_norm, y_norm, w_norm, h_norm, img_width, img_height)])

        return detections

def get_boxes(img_path, text_size=1, text_th=2):
    detector = Detector()
    boxes = detector.detect(img_path)
    if not boxes:
        return None
    cls = classify_boxes(boxes, img_path)

    if len(boxes) != len(cls):
        print(f"Warning: Mismatch between boxes and classes for {img_path}")
        print(f"Boxes: {boxes}")
        print(f"Classes: {cls}")
        return None

    img = cv2.imread(img_path)
    if img is None:
        return None
    img_height, img_width = img.shape[:2]

    export_to_txt_(img_path, boxes, cls, img_width, img_height)
    return img

def classify_boxes(boxes, image_name):
    if len(boxes) == 1:
        if any(c in image_name for c in ['C1.JPG', 'C2.JPG', 'C3.JPG', 'C4.JPG', 'C5.JPG', 'C6.JPG', 'C14.JPG', 'C15.JPG', 'C16.JPG', 'C21.JPG', 'C22.JPG', 'C23.JPG']):
            return ['right ear']
        elif any(c in image_name for c in ['C8.JPG', 'C9.JPG', 'C10.JPG', 'C11.JPG', 'C12.JPG', 'C13.JPG', 'C18.JPG', 'C19.JPG', 'C20.JPG', 'C25.JPG', 'C26.JPG', 'C27.JPG']):
            return ['left ear']
    elif len(boxes) == 2:
        xmin_1 = boxes[0][0]
        xmin_2 = boxes[1][0]
        if xmin_1 < xmin_2:
            return ['left ear', 'right ear']
        else:
            return ['right ear', 'left ear']
    return ['right ear'] * len(boxes)

def export_to_txt_(img_path, boxes, cls, img_width, img_height):
    text_label = 'D:\\Test data\\ears_label'

    if not os.path.exists(text_label):
        os.makedirs(text_label)

    output_file = 'labels_10_person_ears.txt'  # One file for all labels
    output_path = os.path.join(text_label, output_file)

    with open(output_path, 'a') as f:  # Open in append mode
        img_path_new = img_path.replace('D:\\Poliface data\\', '')
        for i in range(len(boxes)):
            class_id = 0 if cls[i] == 'left ear' else 1
            x_min, y_min, box_width, box_height = boxes[i]
            f.write(f'{img_path_new} {class_id} {x_min} {y_min} {box_width} {box_height}\n')


if __name__ == '__main__':
    root_data = 'D:\\Poliface data'
    result_folder = 'D:\\Test data\\final_result_10_person_ears_test'

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

                        img = get_boxes(img_path)
                        if img is None:
                            continue

                        # cv2.imwrite(os.path.join(result_folder, img_path.replace('\\', '_')), img)
