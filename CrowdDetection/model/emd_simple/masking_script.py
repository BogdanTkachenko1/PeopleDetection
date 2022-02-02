import os
import argparse
import shutil
import time

import cv2
import numpy as np
import megengine as mge
from megengine import jit

from config import config
import dataset
import network
from set_nms_utils import set_cpu_nms

if_set_nms = True
top_k = 2


class PeopleFillingModel:
    def __init__(self, weights_path, color, threshold, fill):
        if not os.path.exists(weights_path):
            raise FileNotFoundError('Weights file was not found')

        if not (0 < threshold < 1):
            raise ValueError('Detection threshold must be in range (0; 1)')

        self.threshold = threshold

        if len(color) != 3 or min(color) < 0 or max(color) > 255:
            raise ValueError('Color must be tuple of 3 integers in range [0; 255]')

        self.color = color

        self.fill = fill

        self.network = network.Network()
        self.network.eval()
        self.network.load_state_dict(mge.load(weights_path)['state_dict'])

        @jit.trace(symbolic=False)
        def forward_pass():
            return self.network(self.network.inputs)

        self.forward_pass = forward_pass

    @staticmethod
    def load_image(path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if config.eval_resize == False:
            resized_img, scale = image, 1
        else:
            resized_img, scale = dataset.resize_img_by_short_and_max_size(
                image, config.eval_image_short_size, config.eval_image_max_size)

        original_height, original_width = image.shape[0:2]
        height, width = resized_img.shape[0:2]
        transposed_img = np.ascontiguousarray(
            resized_img.transpose(2, 0, 1)[None, :, :, :],
            dtype=np.float32)
        im_info = np.array([height, width, scale, original_height, original_width],
                           dtype=np.float32)[None, :]
        return image, transposed_img, im_info

    @staticmethod
    def get_coordinates(coordinates):
        return np.array([max(0, round(coordinate)) for coordinate in coordinates])

    def fill_people_on_image(self, image_path):
        original_image, transposed_image, image_info = self.load_image(image_path)

        self.network.inputs['image'].set_value(transposed_image.astype(np.float32))
        self.network.inputs['im_info'].set_value(image_info)

        predicted_boxes = self.forward_pass().numpy()
        predicted_boxes = predicted_boxes[predicted_boxes[:, -1] >= self.threshold]

        n = np.ceil(predicted_boxes.shape[0] / top_k)
        idents = np.tile(np.arange(n)[:, None], (1, top_k)).reshape(-1, 1)
        predicted_boxes = np.hstack((predicted_boxes, idents[:len(predicted_boxes)]))
        keep = predicted_boxes[:, -2] > self.threshold
        predicted_boxes = predicted_boxes[keep]
        keep = set_cpu_nms(predicted_boxes, 0.5)

        if predicted_boxes.shape[0]:
            predicted_boxes = np.apply_along_axis(self.get_coordinates, 1, predicted_boxes[keep][:, :-1])

        result_image = np.asarray(original_image, dtype=np.int32)

        for box in predicted_boxes:
            box = list(map(round, box))
            if self.fill:
                result_image = cv2.rectangle(result_image, (box[0], box[3]), (box[2], box[1]), self.color, 1)
            else:
                result_image[box[1]:box[3], box[0]:box[2]] = self.color

        return result_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_weights', '-w', default=None, type=str)
    parser.add_argument('--source_folder_path', '-s', default=None, type=str)
    parser.add_argument('--destination_folder_path', '-d', default=None, type=str)
    parser.add_argument('--color', '-c', default=0, nargs='+', type=int)
    parser.add_argument('--fill', '-f', default=True, type=bool)
    parser.add_argument('--thresh', '-t', default=0.3, type=float)
    args = parser.parse_args()

    model = PeopleFillingModel(args.resume_weights, args.color, args.thresh, args.fill)

    source_folder_path = args.source_folder_path
    source_folder_name = source_folder_path.split('/')[-1]

    destination_folder_path = os.path.join(args.destination_folder_path, source_folder_name + '_outputs')

    if os.path.exists(destination_folder_path):
        shutil.rmtree(destination_folder_path)

    os.mkdir(destination_folder_path)

    for image_name in os.listdir(source_folder_path):
        if image_name.split('.')[-1] in ('jpg', 'jpeg', 'png'):
            cv2.imwrite(os.path.join(destination_folder_path, image_name),
                        model.fill_people_on_image(os.path.join(source_folder_path, image_name)))


