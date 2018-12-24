# Deep Learning License Plate Recognition
Making use of Google’s TensorFlow framework to create a system that can detect and read license plates in pre-recorded videos and live
camera feed. The core of the system consists of two models in the form of so called frozen graphs. Making them work together, we were able to achieve 
a very good accuracy on detecting license plates as well as descent results on predicting the text written on them. Finally, the system
can be set to check predicted text against a database and display the retrieved information.<br/>
![alt Demo](https://raw.githubusercontent.com/georgi-valkov/Deep-Learning-License-Plate-Recognition/master/assets/screenshot.png)
## Installation
* Install Miniconda3 according to instruction for your OS which can be found 
[here](https://conda.io/docs/user-guide/install/index.html).
* Create a Virtual Environment with python 3.6
```` 
 conda create -n name_of_environemnt python=3.6
 
 ````
 * Activate environment
 ````
 source activate name_of_environment
 ````
 * Install Dependencies
    - For Training: <br/>
    [Tensorflow Object Detection Repository](https://github.com/tensorflow/models/tree/master/research/object_detection) <br/>
    And follow instruction for installing Tensorflow for GPU<br/>
    Clone Tensorflow Research Repository<br/>
    For training the reading model we used [Attention-based OCR](https://github.com/tensorflow/models/tree/master/research/attention_ocr) included in TensorFlow repository
    - For Detection and Reading <br/>
    We are going to use environment with installed Tesnsorflow for CPU.<br/>
 ````
 # Upgrade pip first
 pip install --upgrade pip
 # Install requirements from requirements.txt
 pip install -r requirements.txt
 
 # Note: I found that installing dependencies with
 # conda instead of pip sometimes gives better performance
 # when running the application
````

## Training

### Object Detection Model

* Data Collection and Labeling<br/>
    Data was extracted from videos in the form of images. Then, we manually labeled 2884 images by creating bounding
    boxes around every license plate occurrence on each image using [LabelImg](https://github.com/tzutalin/labelImg#labelimg)<br/>
    You can download this data set (actual images and label files) from [here](https://mega.nz/#!0QU1DQ4B!6Ui3CDaiJbME3GUraW5pT82LeHc2prkomzQ8j7TGyoU)
* Utilizing Transfer Learning<br/>
    We used one of [these pre-trained models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) 
    on our data set.
    
### Text Prediction Model

* Data Collection and Labeling<br/>
    Data was extracted from videos that we took ourselves as well as videos found on youtube using the object detection model.
    The idea was to find every occurrence of a license plate in each frame cut it out from the picture and save it. Thus, we
    ended up with huge data set of images, and even though there were a lot of duplicates in terms of the license plate number,
    because of the fact they were extracted from a video we got a great variety of shadow and lighting conditions for each image.<br/>
    Then we sent around 500 000 images to the [Google Vision API](https://cloud.google.com/vision/) for labeling.
    Data set that includes all of the images and label files in json format that were returned from by the API you can download [here.](https://mega.nz/#!MZlmVY7b!zqVRfu_8ul0NMyyq8Dxyx3QoYKTwGNY9A9TyeD7Cq0c)<br/>
    The raw data set can be also used for a single character extraction since label that was returned from Googl's Vision API
    contains coordinates of the bounding box of every present character as well as well as certainty measure.<br/>
    The refined data set only contains images that were used to train the reading model and a single txt file that contains labels in format
    "[img_file_name] [label]". You can download those ~250k images and their labels by clicking [here.](https://mega.nz/#!1V1m3a5D!INPNmAxJY16_LEOxO5TXAOmo7B_gcpOYxP1iw30xHnI)<br/>
    Note: Since the data for this part of the project was machine labeled keep in mind that it contains errors that will be transferred
    to the model you're training.   
    
* Training Method and Parameters
 
    - Model Parameters
    ```
    # Optimization
    NUM_EPOCH = 1000000
    BATCH_SIZE = 65
    INITIAL_LEARNING_RATE = 1.0

    # Network parameters
    CLIP_GRADIENTS = True
    MAX_GRADIENT_NORM = 5.0  # Clip Gradients to This Normalization
    TARGET_EMBEDDING_SIZE = 10  # Embedding Dimension For Each Target
    # Decreasing this improves performance but it may worsen accuracy
    # For shorter strings sweet spot might be lower than 128
    ATTN_NUM_HIDDEN = 128  # Number of Hidden Units in Attention Decoder Cell
    ATTN_NUM_LAYERS = 2  # Number of Layers in Attention Decoder Cell

    CHANNELS = 3  # Number of Color Channels From Source Image - RGB
    ```
    Using GRU (Gated Recurrent Unit) instead of LSTM
    
## References
[TensorFlow Documentation](https://www.tensorflow.org/api_docs/)<br/>
[Attention-based Extraction of Structured
Information from Street View Imagery](https://arxiv.org/pdf/1704.03549.pdf)<br/>
[License Plate Recognition via Deep Neural Networks](https://arxiv.org/pdf/1806.10447.pdf)