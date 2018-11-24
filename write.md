## Project: Follow Me

My solution for the Robotics NanoDegree [Project #4](https://github.com/udacity/RoboND-DeepLearning-Project).

[//]: # (Image References)

[image1]: ./misc_images/train01.png
[image2]: ./misc_images/following_sample.png
[image3]: ./misc_images/patrol_non_target_sample.png
[image4]: ./misc_images/patrol_with_target_sample.png
[image5]: ./misc_images/following_sample.png

---
### Description

The objective is to train a deep neural network to identify and track a target in simulation (`follow me` applications).

### Data

Download the following files:

 * [Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip) 
 * [Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)
 * [Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

Once the initial sample data is downloaded the `data` directory must follow the structure:

```
├───data
│   ├───figures
│   ├───raw_sim_data
│   │   ├───train
│   │   │   └───run1
│   │   └───validation
│   │       └───run1
│   ├───runs
│   │   ├───following_images_run_1
│   │   ├───patrol_non_targ_run_1
│   │   └───patrol_with_targ_run_1
│   ├───sample_evaluation_data
│   │   ├───following_images
│   │   │   ├───images
│   │   │   └───masks
│   │   ├───patrol_non_targ
│   │   │   ├───images
│   │   │   └───masks
│   │   └───patrol_with_targ
│   │       ├───images
│   │       └───masks
│   ├───train
│   │   ├───images
│   │   ├───masks
│   │   └───run1
│   ├───validation
│   │   ├───images
│   │   ├───masks
│   │   └───run1
│   └───weights
```

where the important folders to consider are:

```
data/runs - contains the results of prediction runs
data/train/images - contains images for the training set
data/train/masks - contains masked (labeled) images for the training set
data/validation/images - contains images for the validation set
data/validation/masks - contains masked (labeled) images for the validation set
data/weights - contains trained TensorFlow models
```

### Network Architecture

The project builds a FCN (Fully Convolutional Network) comprised of an encoder, 1x1 convolution and decoder so as to preserve the spatial information throughout the entire network.

#### Encoder

The Encoder extracts features from the image that later will be used by a decoder. It reduces the input to a deeper 1x1 convolution layer, in contrast to a flat fully connected layer that would be used for basic classification of images.

I used the `separable_conv2d_batchnorm()` function with different filters options, initially tested with two 16 and 32 filter layers obtaining bad results.

Through experimentation and using the recommended downsample factor of 2, I:

 * added a new layer 16,32,64 with better results but still not good enough.
 * double the filters based on comments from the #udacity_deep_learning slack channel; the result is a 3 layers of 32,64,128 filters.

#### 1x1 Convolutions

The 1x1 convolution takes the output channels from the previous step and then combines them into a 256 output layer using the `conv2d_batchnorm()` function.

Internally it uses a standard convolution and includes a batch normalization with the ReLU activation function applied to the layers as it is important to normalize the input.

Batch normalization is based on the idea that, instead of just normalizing the inputs to the network, we normalize the inputs to layers within the network.

#### Decoder

The decoder upscales the output of the encoders such that is the same size as the original image. It results in segmentation or prediction of each individual pixel in the original image. Essentially, it's a reverse convolution (Transposed Convolution or deconvolution) in which the forward and backward passes are swapped.

Similarly to the previous section, the final deconvolution layers are backward organized using 128,64,32 filters to get back to the original layers (image size).

I used the suggestions on the jupyter notebook; upsampling (`upsample_bilinear()` function) by a factor of 2, a concatenation layer step and finally two separable convolution layers to extract some more spatial information from previous layers.

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    # Upsample the small input layer using the bilinear_upsample() function.
    small_out_ip_layer = bilinear_upsample(small_ip_layer)
    # Concatenate the upsampled and large input layers using layers.concatenate
    conc_layers = layers.concatenate([small_out_ip_layer, large_ip_layer])
    # Add some number of separable convolution layers
    output_layer_0 = separable_conv2d_batchnorm(conc_layers, filters)
    output_layer_1 = separable_conv2d_batchnorm(output_layer_0, filters)
    
    return output_layer_1
```

#### Model Summary

Finally the model is a 3 layered encoder/decoder starting with a 32 filter upsampled by a factor of 2 to get a 128 filter (using strides 2):

```
def fcn_model(inputs, num_classes):

    # Add Encoder Blocks.
    encode_layer_0 = encoder_block(inputs, 32, 2)
    encode_layer_1 = encoder_block(encode_layer_0, 64, 2)
    encode_layer_2 = encoder_block(encode_layer_1, 128, 2)
    
    # Add 1x1 Convolution layer using conv2d_batchnorm().
    one_by_one_layer = conv2d_batchnorm(encode_layer_2, 256, 1, 1)
    
    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    decode_layer_2 = decoder_block(one_by_one_layer, encode_layer_1, 128)
    decode_layer_1 = decoder_block(decode_layer_2, encode_layer_0, 64)
    decode_layer_0 = decoder_block(decode_layer_1, inputs, 32)
    
    x = decode_layer_0
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)
```

Using Keras all the layers can be explained as follows (`model.summary()` function):

```
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 160, 160, 3)       0         
separable_conv2d_keras_1 (Se (None, 80, 80, 32)        155       
batch_normalization_1 (Batch (None, 80, 80, 32)        128       
separable_conv2d_keras_2 (Se (None, 40, 40, 64)        2400      
batch_normalization_2 (Batch (None, 40, 40, 64)        256       
separable_conv2d_keras_3 (Se (None, 20, 20, 128)       8896      
batch_normalization_3 (Batch (None, 20, 20, 128)       512       
conv2d_1 (Conv2D)            (None, 20, 20, 256)       33024     
batch_normalization_4 (Batch (None, 20, 20, 256)       1024      
bilinear_up_sampling2d_1 (Bi (None, 40, 40, 256)       0         
concatenate_1 (Concatenate)  (None, 40, 40, 320)       0         
separable_conv2d_keras_4 (Se (None, 40, 40, 128)       43968     
batch_normalization_5 (Batch (None, 40, 40, 128)       512       
separable_conv2d_keras_5 (Se (None, 40, 40, 128)       17664     
batch_normalization_6 (Batch (None, 40, 40, 128)       512       
bilinear_up_sampling2d_2 (Bi (None, 80, 80, 128)       0         
concatenate_2 (Concatenate)  (None, 80, 80, 160)       0         
separable_conv2d_keras_6 (Se (None, 80, 80, 64)        11744     
batch_normalization_7 (Batch (None, 80, 80, 64)        256       
separable_conv2d_keras_7 (Se (None, 80, 80, 64)        4736      
batch_normalization_8 (Batch (None, 80, 80, 64)        256       
bilinear_up_sampling2d_3 (Bi (None, 160, 160, 64)      0         
concatenate_3 (Concatenate)  (None, 160, 160, 67)      0         
separable_conv2d_keras_8 (Se (None, 160, 160, 32)      2779      
batch_normalization_9 (Batch (None, 160, 160, 32)      128       
separable_conv2d_keras_9 (Se (None, 160, 160, 32)      1344      
batch_normalization_10 (Batc (None, 160, 160, 32)      128       
conv2d_2 (Conv2D)            (None, 160, 160, 3)       867       
=================================================================
Total params: 131,289
Trainable params: 129,433
Non-trainable params: 1,856
```

#### Hyperparameters

The learning rate (`learning_rate` parameter) was tuned manually; the selected value is 0.001, other values tested were 1e-2, 1e-3, 1e-4.

The `batch_size` tested was 16 and 32.

Initially, the `num_epochs` parameters used was 2 or 3 just to make sure everything worked and to fine tune other parameters. Then, I tested with 10, 30, 40 and 50. I noticed that in some cases the last model is not the best model so I modified the Keras model configuration to use a callback function and early stop the training process if after 5 epochs there is no improvement in the loss function.

```python
#stop training if the validation error stops improving.
early_stop = keras.callbacks.EarlyStopping(monitor=monitor_value, 
                                           min_delta=min_delta, 
                                           patience=patience, 
                                           verbose=verbose, 
                                           mode='auto') # the direction is automatically inferred from the name of the monitored quantity.
```

The `steps_per_epoch` parameter used is 200 while the selected `validation_steps` parameter is 50.

As the training was done on a machine with an Intel i7 with 8 cores, the `workers` parameter selection is 5.

```
learning_rate = 1e-3
batch_size = 32
num_epochs = 10
steps_per_epoch = 200
```

### Training

All training was done on the [model_training.ipynb](code/model_training.ipynb) notebook. From the original code some changes were done. Using the [callbacks](https://keras.io/callbacks/) functions:

 * checkpoint to save the model after each epoch and keep the best.
 * stop training if the validation error stops improving.

The training data result is as follows:

![Training sample][image1]

### Prediction

There are three different predictions available from the helper code provided:

 * patrol_with_targ: Test how well the network can detect the hero from a distance. 
![patrol with target][image4] 
 * patrol_non_targ: Test how often the network makes a mistake and identifies the wrong person as the target.
![patrol no target][image3] 
 * following_images: Test how well the network can identify the target while following them.
![following sample][image2]
 
### Evaluation

To score the network, two types of error are measured:

 * intersection over the union for the pixelwise classifications is computed for the target channel. 
 * determine whether the network detected the target person or not.

Using the above, the number of detection true_positives, false positives, false negatives are counted.

#### Target following

Scores for while the quad is following behind the target:

```
number of validation samples intersection over the union evaulated on 542
average intersection over union for background is 0.9948830588215676
average intersection over union for other people is 0.33597065519705943
average intersection over union for the hero is 0.9060225071131832
number true positives: 539, number false positives: 0, number false negatives: 0
```

#### Target not visible

Scores for images while the quad is on patrol and the target is not visable:

```
number of validation samples intersection over the union evaulated on 270
average intersection over union for background is 0.9806611376347832
average intersection over union for other people is 0.5967161249589469
average intersection over union for the hero is 0.0
number true positives: 0, number false positives: 72, number false negatives: 0
```

#### Target far away

```
number of validation samples intersection over the union evaulated on 322
average intersection over union for background is 0.9958197155237585
average intersection over union for other people is 0.4255548065562906
average intersection over union for the hero is 0.22432693807704132
number true positives: 136, number false positives: 2, number false negatives: 165
```

#### Score weight

Sum all the true positives, etc from the three datasets to get a score weight:

```
0.7385120350109409
```

#### Intersection over Union

The IoU for the dataset that never includes the hero is excluded from grading:

```
0.5651747225951123
```

#### Final Score

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))`:

```
0.4173883345204603
```

### Discussion

The basic pipeline was implemented, the best result obtained was 41% accuracy from 50 epochs taking 131:50 minutes to train.

Other ideas that could have been tested:

 * Change to HSV color scheme as it is more robust to lightning changes.
 * Use pre-trained networks such as VGG as an Encoder.
 * add data Augmentation based on the current data collection.

### Troubleshooting

### Training Time

The model was run on a local machine using a NVidia GTX 960M card. I started using the CPU for training but decided to swith to the TensorFlow GPU based implementation because it was impossible to iterate fast.
Some number (just for reference) of the first tries once the initial network was built:

epochs | Type | Minutes
--- | --- | ---
3 | CPU | 83:52
3 | GPU | 7:39
30 | GPU | 82:48
30 | GPU | 150:14

#### Model Graph

I could not print the model graph because the following error:

```
ImportError: Failed to import pydot. You must install pydot and graphviz for `pydotprint` to work.
```

All the components were installed, checked (pydot, pydot_ng, graphviz) and Validated:

```sh
pip install pydot
pip install pydot_ng
conda install -c anaconda graphviz
```

```
Package              Version
-------------------- ---------
graphviz             0.10.1
pydot                1.3.0
pydot-ng             2.0.0
```

But still the problem continued.

#### Environment changes

Initially the project was tested with TensorFlow 1.5 and 1.8 but the following error occurs:

```
    778           initializer = initializer(dtype=dtype)
    779         init_val = lambda: initializer(  # pylint: disable=g-long-lambda
--> 780             shape.as_list(), dtype=dtype, partition_info=partition_info)
    781         variable_dtype = dtype.base_dtype	
TypeError: __call__() got an unexpected keyword argument 'partition_info'
```

After several attempts to fix it, I switched to TensorFlow 1.2.1 with GPU support on a new environments.

### Resources

* [Project Baseline](https://github.com/ladrians/RoboND-DeepLearning-Project-P4)
* [Original Repository](https://github.com/udacity/RoboND-DeepLearning-Project)
* [Rubric](https://review.udacity.com/#!/rubrics/1155/view)
* [QuadSim Simulator](https://github.com/udacity/RoboND-DeepLearning/releases/latest)
