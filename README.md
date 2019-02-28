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
folder path: \app\src\main\assets
![alt text](/asset folder.png "Description goes here")

