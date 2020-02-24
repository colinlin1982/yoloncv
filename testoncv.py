# -*- coding:utf-8 -*-

import os
import numpy as np
import cv2
import glob
import tkinter as tk
from PIL import Image, ImageTk


classes_file = "/Volumes/HD/Applications/darknet/Pascal/voc.names"
cfg_file = "net.cfg"
weights_file = "/Volumes/HD/Applications/darknet/Pascal/backup/yolov3-lite-416_55.35.weights"
img_dir = "/Volumes/HD/Applications/darknet/Pascal/VOCdevkit/VOC2007"
input_shape = (416, 416)
conf_threshold = 0.5
nms_threshold = 0.6


class YoloDetector:

    # Get the names of the output layers
    @staticmethod
    def getOutputsNames(net):
        # Get the names of all the layers in the network
        layers_names = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def __init__(self):
        self.classes = None
        self.net = None
        self.input_shape = (416, 416)

    def load_classes(self, names_file):
        with open(names_file, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        print(self.classes)

    def init_net(self, cfg, weights, input_shape=(416,416)):
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.input_shape = input_shape

    def detect_image(self, image):
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, self.input_shape, (0, 0, 0), 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        detections = self.net.forward(self.getOutputsNames(self.net))
        # Put efficiency information. The function getPerfProfile returns the
        # overall time for inference(t) and the timings for each of the layers(in layersTimes)
        cost, _ = self.net.getPerfProfile()

        return detections, cost

    # Draw the predicted bounding box
    def draw_box(self, image, class_id, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0))

        if conf > 0:
            label = '%.2f' % conf
        else:
            label = ""

        # Get the label for the class name and its confidence
        if self.classes:
            assert (class_id < len(self.classes))
            label = '%s:%s' % (self.classes[class_id], label)

        # Display the label at the top of the bounding box
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    def draw_annotations(self, image, image_path:str):
        annotation_path = image_path[:-3] + "txt"
        if not os.path.exists(annotation_path):
            annotation_path = annotation_path.replace("JPEGImages", "labels").replace("images", "labels")
        if not os.path.exists(annotation_path):
            return

        img_h, img_w = image.shape[:2]
        # draw annotations
        with open(annotation_path, "r") as fr:
            while True:
                line = fr.readline()
                if not line:
                    break

                annotation = line.strip().split(" ")
                center_x = float(annotation[1]) * img_w
                center_y = float(annotation[2]) * img_h
                width = float(annotation[3]) * img_w
                height = float(annotation[4]) * img_h

                left = max(int(center_x - width / 2), 0)
                top = max(int(center_y - height / 2), 0)
                right = min(img_w - 1, int(left + width))
                bottom = min(img_h - 1, int(top + height))
                self.draw_box(image, int(annotation[0]), -1.0, left, top, right, bottom)

    def draw_detections(self, image, detections, conf_thresh=0.5, nms_thresh=0.6):
        """Remove the bounding boxes with low confidence using non-maxima suppression"""
        img_h = image.shape[0]
        img_w = image.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        class_ids = []
        confidences = []
        boxes = []
        for out in detections:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > conf_thresh:
                    center_x = int(detection[0] * img_w)
                    center_y = int(detection[1] * img_h)
                    width = int(detection[2] * img_w)
                    height = int(detection[3] * img_h)
                    left = max(int(center_x - width / 2), 0)
                    top = max(int(center_y - height / 2), 0)
                    width = min(img_w - left, width)
                    height = min(img_h - top, height)
                    class_ids.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            right = min(img_w -1, left + box[2])
            bottom = min(img_h -1, top + box[3])
            self.draw_box(image, class_ids[i], confidences[i], left, top, right, bottom)


# init detector
detector = YoloDetector()
detector.load_classes(classes_file)
detector.init_net(cfg_file, weights_file, input_shape)
file_list = glob.glob(img_dir + "/*/*.jpg")


class Viewer:

    def __init__(self, init_conf, init_nms):
        init_size = (1080, 700)

        root = tk.Tk()
        root.title("Test YOLO on OpenCV")
        root.geometry('%dx%d+20+20' % init_size)
        root.config(cursor="arrow")

        self.conf_var = tk.DoubleVar()
        self.conf_var.set(init_conf)
        self.nms_var = tk.DoubleVar()
        self.nms_var.set(init_nms)

        container_top0 = tk.Label(root)
        container_top0.pack(side=tk.TOP, expand=tk.NO, fill=tk.X, padx=10, pady=10)
        label_conf = tk.Label(container_top0, text="Conf", anchor="s")
        label_conf.pack(side=tk.LEFT, expand=tk.NO, fill=tk.Y)
        scale_conf = tk.Scale(container_top0, from_=0.1, to=1.0, resolution=0.1,
                              length=300, width=7, troughcolor="lightblue", bd=1, relief="flat",
                              sliderrelief='flat', variable=self.conf_var, command=self.on_thresh_change,
                              orient=tk.HORIZONTAL)
        scale_conf.pack(side=tk.LEFT, expand=tk.NO, fill=tk.NONE)
        label_nms = tk.Label(container_top0, text="NMS", anchor="s")
        label_nms.pack(side=tk.LEFT, expand=tk.NO, fill=tk.Y)
        scale_nms = tk.Scale(container_top0, from_=0.1, to=1.0, resolution=0.1,
                              length=300, width=7, troughcolor="lightblue", bd=1, relief="flat",
                              sliderrelief='flat', variable=self.nms_var, command=self.on_thresh_change,
                              orient=tk.HORIZONTAL)
        scale_nms.pack(side=tk.LEFT, expand=tk.NO, fill=tk.NONE)

        container_top1 = tk.Label(root)
        container_top1.pack(side=tk.TOP, expand=tk.NO, fill=tk.X, padx=10, pady=10)
        btn_prev = tk.Button(container_top1, text="Prev", command=self.show_prev)
        btn_prev.pack(side=tk.LEFT, expand=tk.NO, fill=tk.NONE, ipadx=15, ipady=5)
        btn_next = tk.Button(container_top1, text="Next", command=self.show_next)
        btn_next.pack(side=tk.LEFT, expand=tk.NO, fill=tk.NONE, ipadx=15, ipady=5)

        container_center = tk.Label(root)  # initialize image panel
        container_center.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.BOTH, padx=10, pady=5)
        canvas_left = tk.Label(container_center, bg="gray")
        canvas_left.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)
        canvas_right = tk.Label(container_center, bg="black")
        canvas_right.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.BOTH)

        self.root, self.canvas_left, self.canvas_right = root, canvas_left, canvas_right
        self.cur_index = -1

    def _update_image(self, canvas, cv_img):
        cv2image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        image_file = ImageTk.PhotoImage(img)
        canvas.image = image_file
        canvas.config(image=image_file)

    def show_images(self):
        if file_list is None or len(file_list) == 0:
            return
        if self.cur_index < 0 or self.cur_index >= len(file_list):
            self.cur_index = 0

        # each image
        img = cv2.imread(file_list[self.cur_index])
        detections, cost = detector.detect_image(img)

        # resize image to fit
        tw, th = self.canvas_right.winfo_width() - 20, self.canvas_right.winfo_height()
        ih, iw = img.shape[:2]
        scale = min(tw*1.0/iw, th*1.0/ih)
        fit_img = cv2.resize(img, (int(iw*scale), int(ih*scale)), interpolation=cv2.INTER_LINEAR)
        ann_img = fit_img.copy()

        # Remove the bounding boxes with low confidence
        detector.draw_detections(fit_img, detections, float(self.conf_var.get()), float(self.nms_var.get()))
        label = 'Inference time: %.2f ms' % (cost * 1000.0 / cv2.getTickFrequency())
        cv2.putText(fit_img, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        self._update_image(self.canvas_left, fit_img)

        detector.draw_annotations(ann_img, file_list[self.cur_index])
        self._update_image(self.canvas_right, ann_img)

    def show_next(self):
        if self.cur_index == len(file_list) - 1:
            return

        self.cur_index += 1
        self.show_images()

    def show_prev(self):
        if self.cur_index < 1:
            return

        self.cur_index -= 1
        self.show_images()

    def on_thresh_change(self, event):
        self.show_images()


viewer = Viewer(conf_threshold, nms_threshold)
viewer.root.mainloop()
