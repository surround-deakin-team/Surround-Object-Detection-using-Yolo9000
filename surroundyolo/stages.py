from surround import Stage, SurroundData
import cv2
import numpy as np

classes = None
with open('yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

class YoloData(SurroundData):
    output_data = None

    def __init__(self, input_data):
        print('YoloData init')
        self.input_data = input_data
        self.errors = []

class ValidateData(Stage):
    def operate(self, surround_data, config):
        print('ValidateData operate')

        Width = surround_data.input_data.shape[1]
        Height = surround_data.input_data.shape[0]
        scale = 0.00392
        print("Read Image Height and Width")

        net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
        print("Read Yolo Weights and Config")

        blob = cv2.dnn.blobFromImage(surround_data.input_data, scale, (416,416), (0,0,0), True, crop=False)
        print("Create Input Blob")

        net.setInput(blob)
        print("Set Input Blob")

        outs = net.forward(get_output_layers(net))
        print("Get predictions from output layer")

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4


        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_bounding_box(surround_data.input_data, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        cv2.imshow("object detection", surround_data.input_data)
        cv2.waitKey()
    
        cv2.imwrite("object-detection.jpg", surround_data.input_data)
        cv2.destroyAllWindows()




