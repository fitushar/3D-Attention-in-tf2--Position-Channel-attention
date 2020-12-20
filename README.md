# 3D-Attention-in-tf2--Position-Channel-attention
In recent years we have seen a numbers of implementation with some adiitional attention to tha task(classification/segmentation).
This repo contains the 3D implementation of the commonly used attention mechanism for imaging.





##  Additive Attention Gate (AG)-3D
```
Oktay, Ozan, et al. "Attention u-net: Learning where to look for the pancreas." arXiv preprint arXiv:1804.03999 (2018).

```
![dsc](https://github.com/fitushar/3D-Attention-in-tf2--Position-Channel-attention/blob/main/Figures/AG.PNG)

### code
```ruby
def Attention_mechanism(X,G,out_filters,kernel_size=1,strides=(1, 1, 1),use_bias=False,
                 kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='uniform'),
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                 bias_regularizer=None,
                 **kwargs):

    conv_params={'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    ###input from the resolution.
    Original_x=G
    ###
    X1=tf.keras.layers.Conv3D(filters=out_filters,kernel_size=1,strides=1,**conv_params)(X)
    X1=tf.keras.layers.BatchNormalization()(X1)
    G1=tf.keras.layers.Conv3D(filters=out_filters,kernel_size=1,strides=1,**conv_params)(G)
    G1=tf.keras.layers.BatchNormalization()(G1)
    ##Adding
    X1_G1=X1+G1
    #Applying Relu
    A1=tf.nn.relu6(X1_G1)
    #Applying convolution again
    A1=tf.keras.layers.Conv3D(filters=out_filters,kernel_size=1,strides=1,**conv_params)(A1)
    #Sigmoid
    A1=tf.keras.activations.sigmoid(A1)
    final_attention=tf.math.multiply(Original_x,A1)

    return final_attention


```

## Postion Attention-3D
```
Fu, Jun, et al. "Dual attention network for scene segmentation." 
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
```
![dsc](https://github.com/fitushar/3D-Attention-in-tf2--Position-Channel-attention/blob/main/Figures/PA.PNG)
### code
```ruby
def Position_attention(postion_attention_input):

    #--Getting the Shape of the inputs
    in_shp = postion_attention_input.get_shape().as_list()


    C1=tf.keras.layers.Conv3D(filters=int(in_shp[4]/8),kernel_size=1,strides=(1,1,1))(postion_attention_input)
    C1_shp = C1.get_shape().as_list()

    ##--first-Batch
    F1_HWDxC=tf.reshape(C1, [-1, C1_shp[1]*C1_shp[2]*C1_shp[3],C1_shp[4]])
    print(F1_HWDxC.get_shape())

    ##--Seconr-Batch
    F2_CxHWD=tf.transpose(F1_HWDxC,perm=[0, 2, 1])
    F2_CxHWD=tf.matmul(F1_HWDxC,F2_CxHWD)
    F2_CxHWD=tf.keras.activations.softmax(F2_CxHWD)
    print(F2_CxHWD.get_shape())


    ##--thir-Batch
    C2=tf.keras.layers.Conv3D(filters=in_shp[4],kernel_size=1,strides=(1,1,1))(postion_attention_input)
    F3_HWDxC=tf.reshape(C2, [-1, in_shp[1]*in_shp[2]*in_shp[3],in_shp[4]])
    F3xF2=tf.matmul(F2_CxHWD,F3_HWDxC)
    F3=tf.reshape(F3xF2,[-1, in_shp[1],in_shp[2],in_shp[3],in_shp[4]])
    print(F3.get_shape())
    print(postion_attention_input.get_shape())
    
    postion_attention_output=tf.keras.layers.Multiply()([postion_attention_input,F3])
    postion_attention_output=tf.keras.layers.Conv3D(filters=in_shp[4],kernel_size=1,strides=(1,1,1))(postion_attention_output)


    return postion_attention_output
```

## Channel Attention-3D
```
Fu, Jun, et al. "Dual attention network for scene segmentation." 
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
```
![dsc](https://github.com/fitushar/3D-Attention-in-tf2--Position-Channel-attention/blob/main/Figures/CA.PNG)
### code
```ruby
def Channel_attention(Channel_attention_input):

    in_shp = Channel_attention_input.get_shape().as_list()

    ##--first-Batch
    channel_C1=tf.reshape(Channel_attention_input, [-1, in_shp[1]*in_shp[2]*in_shp[3],in_shp[4]])

    ##--Seconr-Batch
    channel_C2=tf.transpose(channel_C1,perm=[0, 2, 1])
    channel_C2=tf.matmul(channel_C1,channel_C2)
    channel_C2=tf.keras.activations.softmax(channel_C2)

    channel_C3=tf.matmul(channel_C2,channel_C1)
    channel_C3=tf.reshape(channel_C3,[-1, in_shp[1],in_shp[2],in_shp[3],in_shp[4]])

    Channel_attention_output=tf.keras.layers.Multiply()([channel_C3,Channel_attention_input])
    Channel_attention_output=tf.keras.layers.Conv3D(filters=in_shp[4],kernel_size=1,strides=(1,1,1))( Channel_attention_output)

    return Channel_attention_output

```
