import logging
from .wrapper import PipelineWrapper
import os
import json
logging.basicConfig(level=logging.INFO)

#import required packages for object detection
import cv2
import numpy as np

def main():

    #read video
    cap = cv2.VideoCapture('data/traffic.mp4')

    # Define the codec and create VideoWriter object
    v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter('output/output.mp4', fourcc, 15.0, (v_width,v_height))
    
    # read class names from text file
    classes = None
    with open('models/yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # generate different colors for different classes 
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # read pre-trained model and config file
    net = cv2.dnn.readNet('models/yolov3.weights', 'models/yolov3.cfg')
    
    while(cap.isOpened()):
        ret, image = cap.read()
        if ret==True:
            #create surround wrapper
            wrapper = PipelineWrapper()

            #get wrapper configuration
            config = wrapper.get_config()

            #get output after running the wrapper
            output = wrapper.run(image,classes,COLORS,net)

            #write frame
            vout.write(output)

            # display output image    
            cv2.imshow("object detection", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    # release resources
    cap.release()
    vout.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
