import os
import json
from surround import Surround, Wrapper, AllowedTypes
from stages import InputBlob, YoloData, OutputLayer, Confidence, NMS, DrawBoxes, ObjectCount

class PipelineWrapper(Wrapper):
    def __init__(self):
        #initialize surround and pass stages as parameters
        surround = Surround([InputBlob(),OutputLayer(),Confidence(),NMS(),DrawBoxes(),ObjectCount()], __name__)
        super().__init__(surround)

    #wrapper method to run stages
    def run(self, image, classes, COLORS, net):
        #initialize data from YoloData
        data = YoloData(image, classes, COLORS, net)
        #process data
        self.surround.process(data)
        #return image with object detection
        return data.image
