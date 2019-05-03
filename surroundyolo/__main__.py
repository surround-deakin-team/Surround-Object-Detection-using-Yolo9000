import logging
from .wrapper import PipelineWrapper
import os
import json
import cv2
logging.basicConfig(level=logging.INFO)

def main():
    image = cv2.imread('img.jpg')
    print('Enter Main')
    wrapper = PipelineWrapper()
    config = wrapper.get_config()
    print('After wrapper get_config')
    output = wrapper.run(image)
    #with open(os.path.join(config["output_path"], "output.txt"), 'w') as f:
     #   f.write(output["output"])
    logging.info(output)

if __name__ == "__main__":
    main()
