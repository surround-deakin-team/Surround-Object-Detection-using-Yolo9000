import logging
from .wrapper import PipelineWrapper
import os
import json
logging.basicConfig(level=logging.INFO)

#import required packages for object detection
import cv2
import numpy as np

def main():

    #read the image
    image = cv2.imread('data/img.jpg')

    # read class names from text file
    classes = None
    with open('models/yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # generate different colors for different classes 
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # read pre-trained model and config file
    net = cv2.dnn.readNet('models/yolov3.weights', 'models/yolov3.cfg')
    
    #create surround wrapper
    wrapper = PipelineWrapper()

    #get wrapper configuration
    config = wrapper.get_config()

    #get output after running the wrapper
    output = wrapper.run(image,classes,COLORS,net)

    # display output image 
    cv2.imshow("object detection", output)

    # wait until any key is pressed
    cv2.waitKey()

    # save output image to disk
    cv2.imwrite("output/object-detection.jpg", image)

    # release resources
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
