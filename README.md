# Surround-Object-Detection-using-Yolo9000-Video-Feature

Yolo9000 is an object detection model that can detect upto 9000 objects in an image. This branch shows implementation of object detection in a video file using surround framework.

# Project Folders and Files
* data folder contain the video file (in this example name of video file is traffic.mp4)
* models folder contain yolo confihuration file **yolov3.cfg**, yolo classes file **yolov3.txt** and predefined yolo weights file **yolov3.weights** (this is a large file and is downlaoded separatley which is shown below)
* output folder contain the output file generated after running the model.
* surroundyolo folder contain the the **init**, **main**, **config** (suuround configurations), **stages** and **wrapper** files.

# Project Flow
* In the main file, all the model files and video file is read, and a video writer is crteated. Then each frame is from the video file is sent to the the pipeline created for object detection. Output for each frame is recieved and passed to video writer to write the output video file. This process will go on till all the frames from the input image are passed to the wrapper.
* In the wrapper file PipelineWrapper class is defined using surround Wrapper class. The purpose of this class is to create surround object with all the stages, Initialialize Data for obejct detection and process surround stages.
* In stages file, first YoloData class is created which derived from SurroundData class. The purpose of this class is to define all the variables that will be used in all the stages. Then all the stages are defined for the object detection program. 6 different stages are defined for object detection example. Explanation of each stage is provided in stages file.

#Instructions to Run
* prerequisites
    1. python3
    2. surround (refer to surround installation guide)
    3. opencv (refer to yolo9000 section in surround documentation for installing opencv)
    4. numpy (refer to yolo9000 section in surround documentation for installing numpy)

* clone the videoFeature branch by entering this command on cli, *'git clone -b videoFeature --single-branch https://github.com/surround-deakin-team/Surround-Object-Detection-using-Yolo9000.git'*

* change working directory to models folder in the newly created project

* when in the models folder enter this command on cli to download the yolo weights file, *'curl https://pjreddie.com/media/files/yolov3.weights -o yolov3.weights'*

* change working directory to project main directory

* when in project main directory enter this commad in cli to run the project, *'python3 -m surroundyolo'* (to run on windows type python instead of python3)

* a new popup window will open with object detection video in it

* once the program is finished running, output video can be found in project output folder

Thanks

