
# About SurroundAI


Surround AI is the python framework which is designed for flexible usage in Artificial Intelligence(AI). It is designed to support data scientist in their progress. Each and every scientist use different algorithm to solve different problems. There are no standard way for them to analyse altogether in a single module. To provide a standard solution surround frame work is built. Evolution of machine learning pipeline is not possible without re-constructing the coding whereas surround package will provide a pipeline without any alterations.

problems that where addressed at Applied Artificial Intelligence Institute:

There were same changes required to refactor code again and again, which was written by data scientist to make it ready for implement. That means there was no standard script, no proper way to handle configuration and no standard pipeline architecture.

The models which are existing are serving the model rather than end-to-end solution. The model needs to me clubbed with multiple models and glue code to tie these models together.

Existing models don’t allow for the evolution of a machine learning pipeline without re-engineering the solution. Ex: using a cloud API for the first release before training a custom model much later.

Code was commonly being commented out to run other branches.

Why surround?
It is designed to support data scientist in their progress.
Each and every scientist use different algorithm to solve different problems. There are no standard way for them to analyse altogether in a single module.
To provide a standard solution surround framework is built.
Every machine learning pipeline can be accessed for getting the appropriate solution instead of reffering many machine learning packages.
Usage
Helping the data scientists in analytics instead
using glue codes for their research.
Easy interaction between several machine learning pipelines.
Provides end-to-end solution instead of providing solution for the models.

# Yolo9000

YOLO is an object detection system targeted for real-time processing.YOLO9000 extends YOLO to detect objects over 9000 classes using hierarchical classification with a 9418 node WordTree. It combines samples from COCO and the top 9000 classes from the ImageNet. YOLO samples four ImageNet data for every COCO data. It learns to find objects using the detection data in COCO and to classify these objects with ImageNet samples.

During the evaluation, YOLO test images on categories that it knows how to classify but not trained directly to locate them, i.e. categories that do not exist in COCO. YOLO9000 evaluates its result from the ImageNet object detection dataset which has 200 categories. It shares about 44 categories with COCO. Therefore, the dataset contains 156 categories that have never been trained directly on how to locate them. YOLO extracts similar features for related object types. Hence, we can detect those 156 categories by simply from the feature values.

YOLO9000 gets 19.7 mAP overall with 16.0 mAP on those 156 categories. YOLO9000 performs well with new species of animals not found in COCO because their shapes can be generalized easily from their parent classes. However, COCO does not have bounding box labels for any type of clothing so the test struggles with categories like “sunglasses”.


   * Grid Cell
   YOLO divides the input image into an S×S grid. Each grid cell predicts only one object.Each grid cell predicts a fixed        number of boundary boxes.However, the one-object rule limits how close detected objects can be. For that, YOLO does have      some limitations on how close objects can be.
   
   * Benefits of YOLO
   Fast. Good for real-time processing.
   Predictions (object locations and classes) are made from one single network. Can be trained end-to-end to improve accuracy.
   YOLO is more generalized. It outperforms other methods when generalizing from natural images to other domains like artwork.
   Region proposal methods limit the classifier to the specific region. YOLO accesses to the whole image in predicting            boundaries. With the additional context, YOLO demonstrates fewer false positives in background areas.
   YOLO detects one object per grid cell. It enforces spatial diversity in making predictions.

# surroundyolo

In this project we are trying to build an object detection system with the help of yolo9000  using surround framework, Yolo9000 is a python library which can detect the objects and we are trying to incorporate Yolo9000 with the surround framework.The business problem we identified for this project was to identify the traffic at a particular point of timeframe and analyse the data after the process. With the pretrained model and configuration of yolo9000 we are trying to identify the objects and process the information passed.

There are some dependancy for the projects which are 

Numpy
OpenCV
Predefined weight (https://pjreddie.com/media/files/yolov3.weights)

  # Installing Numpy
  
       # Windows

         pip3 install numpy
       
       # MAC

         pip3 install numpy

  # Installing openCV
  
       # Windows

         pip3 install openCV
       
       # MAC

         pip3 install openCV
 
  # Installing Predefined weights
  
  Change the directory to the project folder Windows

     cd <set path to project folder> curl https://pjreddie.com/media/files/yolov3.weights -o yolov3.weights

  MAC

     cd <set path to project folder> curl https://pjreddie.com/media/files/yolov3.weights -o yolov3.weights
     
 Note: _ Make sure the name of the yolov3.weights is similar to the weight file in yolo project._

 # How to run the project
 
 Place the image which needs to be used for the object detection in "data" folder . Using terminal open the workspace of project then 
 
  # Windows

         python -m surroundyolo
       
  # MAC

         python3 surroundyolo.py


The output image will be displayed with the details of each object.
