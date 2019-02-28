# Deploying a Keras Tensorflow Model to Android
A Tutorial that shows you how to deploy a trained deep learning model to Android mobile app 

## **Step 1. Convert Keras model(.h5) to a Tensorflow Lite FlatBuffer(.tflite)**

- For TensorFlow **1.12.0**, follow the steps [here](https://www.tensorflow.org/lite/convert/python_api#exporting_a_tfkeras_file_) 

- For previous version, things are not as straightforward, and you need to follow the steps below:
  ###### 1. Convert Keras model file(.h5) to TensorFlow protobuf (.pb) file 
    Follow [this tutorial](https://github.com/amir-abdi/keras_to_tensorflow) to convert your .h5 model file to .pb file 
  ###### 2. Convert .pb file to Tensorflow Lite FlatBuffer(.tflite)
    Use the Python code below to convert .pb file to .tflite:
    
    ```
    import tensorflow as tf
    graph_def_file = "/path/to/Downloads/mobilenet_v1_1.0_224/frozen_graph.pb"
    input_arrays = ["input"]                                # input node name
    output_arrays = ["MobilenetV1/Predictions/Softmax"]     # output node name

    converter = tf.contrib.lite.TocoConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)
    ```
    If you don't know the input/output nodes' names, do the following:
    ```
      >>> import tensorflow as tf 
      >>> g = tf.GraphDef()
      >>> g.ParseFromString(open(“path/to/mymodel.pb”, “rb”).read())
      >>> [n for n in g.node if n.name.find(“input”) != -1] # same for output or any other node you want to make sure is ok
    ```  

*Note: The windows version of TensorFlow might not have the package you need. If so, you will have to perform this conversion in Linux environment. 
  
## **Step 2. Put the .tflite model file to Android project folder**
The .tflite file need to be put in the "assets" folder of the project as shown below:

![alt text](https://github.com/Yu-Hang/Deploying-a-Keras-Tensorflow-Model-to-Android/blob/master/asset_folder.png "Description goes here")

Path in the project: /app/src/main/assets

## **Step 3. Add TensorFLow Lite library to your Android project**
To build an Android App that uses TensorFlow Lite, the first thing you’ll need to do is add the tensorflow-lite library to your app. This can be done by adding the following line to your build.gradle file’s dependencies section:
```
	implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
```  

## **Step 4. Load and run model in Android**
In this final step, we write code in the Android project to load and run the model for inference.
###### 1. load model file    

[classifier = TFLiteImageClassifier.create(getAssets(), MODEL_FILE, LABEL_FILE, INPUT_SIZE);](https://github.com/tensorflow/tensorflow/blob/fdbaab6f506a1829cbadaf79482ffc95a7342b37/tensorflow/lite/examples/android/app/src/main/java/org/tensorflow/demo/ClassifierActivity.java#L102)
  
  See the TFLiteImageClassifier.create() function [here](https://github.com/tensorflow/tensorflow/blob/f38eea2aec56f7cdbee11d354e5753a097943c94/tensorflow/lite/examples/android/app/src/main/java/org/tensorflow/demo/TFLiteImageClassifier.java#L85)
  
###### 2. run model for inference

*The input image need to be in Bitmap ARGB_8888 format for this.

We fisrt need to put the Bitmap data into a ByteBuffer, which is the input data format to the inference function.
```
private int[] intValues;
private ByteBuffer imgData = null;
...
...

private void convertBitmapToByteBuffer(Bitmap bitmap) {
    if (imgData == null) {
      return;
    }
    imgData.rewind();
    
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    
    // Convert the image to floating point.
    int pixel = 0;
    long startTime = SystemClock.uptimeMillis();
    for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
      for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
        final int val = intValues[pixel++];
        imgData.put((byte) ((val >> 16) & 0xFF));	// R	# In Android Bitmap ARGB_8888, each pixel is stored on 4 bytes(32 bits).
        imgData.put((byte) ((val >> 8) & 0xFF));  	// G	# Hence, each int val stores the ARGB values for one pixel.
        imgData.put((byte) (val & 0xFF));         	// B	# '>>' and '& 0xFF' together gives the RGB values      
      }
    }
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
  }
  ```

Finally, we run inference. In order to reduce the runtime, multiple input images should be inferenced at once

```
int[] dims = new int[4];
dims[0] = batchSize;		// number of images to be inferenced 
dims[1] = DIM_IMG_SIZE;		// image height
dims[2] = DIM_IMG_SIZE;		// image width
dims[3] = DIM_PIXEL_SIZE;	// number of channels

tfLite.resizeInput(0, dims);    // resize input tensor

float[][] labelProb = new float[current_batchSize][2]; // output probabilities

tfLite.run(imgData, labelProb); // run inference
```
