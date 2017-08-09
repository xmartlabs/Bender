# Supported Layers

This is a list of the layers/ nodes that are currently supported. We plan to keep adding more and hope that the community will help with this as well.

> In Metal, the basic unit which holds data between layers is the MTLTexture which is contained in an MPSImage. A texture can have an arbitrary amount of feature channels (it is not limited to three or four)
 

The implemented layers are:
* __Add__: Sums two textures element-wise. Both textures have to be of the same size
* __Concat__: Concatenates the inputs along certain axis
* __Convolution__: Wrapper for MPSCNNConvolution
* __ConvTranspose__: Represents a transposed convolution
* __FullyConnected__: Represents a Fully connected (or Dense) layer
* __InstanceNorm__: [Instance Normalization](https://arxiv.org/abs/1607.08022)
* __LRN__: Local Response Normalization
* __Pooling__: Implements _max_ and _average_ pooling
* __Neuron__: Activation neurons such as ReLU, TanH, Sigmoid, Linear
* __Softmax__: Implements a Softmax
* __SpatialNorm__: Spatial Normalization


Also there are some layers that can be used for pre and post processing or which are helpers for special cases:
* __BGRAtoRGBA__: Transforms an image from BGRA to RGBA order
* __Crop__: Crops an image
* __Dummy__: Helper node used in special cases. Will be removed from the execution list in `network.initialize()`.
* __Identity__: Returns the same image. Used in some special cases
* __ImageLinearTransform__: Helper used when the output of the network is an image (not an array of probabilities)
* __Scale__: Scales a texture to a defined size
* __Start__: Starting node of a network


Last but not least, there is a _CompositeLayer_ included:
* __Residual__: A Residual layer contains a group of sublayers which are executed and the output is the result of this sublayers added to the input of the residual layer.

## TensorFlow Mapping

The following are the TensorFlow ops that are mapped to Bender layers:
* __Add__ -> Add (works for two textures/tensors of the same size)
* __ConcatV2__ -> Concat
* __Conv2D__ -> Convolution
* __Conv2Dtranspose__ -> ConvTranspose
* __Matmul + BiasAdd__ -> FullyConnected
* __MaxPooling and AvgPooling__ -> Pooling
* __Relu, Tanh, Sigmoid__ -> Neuron
* __Softmax__ -> Softmax


