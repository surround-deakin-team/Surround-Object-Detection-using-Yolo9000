from surround import Stage, SurroundData
import cv2
import numpy as np

# create data class for object detection
class YoloData(SurroundData):
    #define class variables
    output_data = None
    image=None
    classes=None
    COLORS=None
    net=None
    outs=None
    class_ids=[]
    confidences=[]
    boxes=[]
    indices=None

    #assign class variables on initialization
    def __init__(self, image,classes,COLORS,net):
        self.image=image
        self.classes=classes
        self.COLORS=COLORS
        self.net=net
        self.errors = []
        self.outs=None
        self.class_ids=[]
        self.confidences=[]
        self.boxes=[]
        self.indices=None

#first stage to  create and set input blob
class InputBlob(Stage):
    def operate(self, surround_data, config):
        scale = 0.00392
        # create input blob 
        blob = cv2.dnn.blobFromImage(surround_data.image, scale, (416,416), (0,0,0), True, crop=False)

        # set input blob for the network
        surround_data.net.setInput(blob)

# second stage to get the output layer names in the architecture
# run inference through the networknand gather predictions from output layers
class OutputLayer(Stage):
    def operate(self, surround_data, config):
        layer_names = surround_data.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in surround_data.net.getUnconnectedOutLayers()]
        surround_data.outs = surround_data.net.forward(output_layers)

# third stage to get the confidence, class id, bounding box params
# for each detetion from each output layer 
# and ignore weak detections (confidence < 0.5) 
class Confidence(Stage):
    def operate(self, surround_data, config):
        # initialization
        Width = surround_data.image.shape[1]
        Height = surround_data.image.shape[0]
        for out in surround_data.outs:
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
                    surround_data.class_ids.append(class_id)
                    surround_data.confidences.append(float(confidence))
                    surround_data.boxes.append([x, y, w, h])

# fourth stage to apply non-max suppression                   
class NMS(Stage):
    def operate(self, surround_data, config):
        conf_threshold = 0.5
        nms_threshold = 0.4
        surround_data.indices = cv2.dnn.NMSBoxes(surround_data.boxes, surround_data.confidences, conf_threshold, nms_threshold)

# fifth stage to draw bounding box on the detected object with class name
class DrawBoxes(Stage):
    def operate(self, surround_data, config):
        for i in surround_data.indices:
            i = i[0]
            box = surround_data.boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            label = str(surround_data.classes[surround_data.class_ids[i]])
            color = surround_data.COLORS[surround_data.class_ids[i]]
            cv2.rectangle(surround_data.image, (round(x),round(y)), (round(x+w),round(y+h)), color, 2)
            cv2.putText(surround_data.image, label, (round(x)-10,round(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

class ObjectCount(Stage):
    def operate(self, surround_data, config):
        unique_elements, counts_elements = np.unique(surround_data.class_ids, return_counts=True)
        place=0
        for i in range(len(unique_elements)):
            cv2.putText(surround_data.image, str(surround_data.classes[unique_elements[i]]+"="+str(counts_elements[i])), (20,20+place), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            place+=20