
# Bender API

This document explains the basic API in __Bender__.

 * [Creating a network]
 * [Running a network]
 * [ParameterLoader]
 * [Adding new layers]
 * [MetalShaderManager]
 * [Composing Layers]


## Creating a network

To create a network model you can create it from scratch or [import](Documentation/Importing.md) it from a TensorFlow graph. We will explain how to create a network from scratch:

```swift
let network = Network(device: device, 
                      inputSize: inputSize,
                      parameterLoader: loader)

network.start
    ->> Convolution(convSize: ConvSize(outputChannels: 16, kernelSize: 3, stride: 2))
    ->> InstanceNorm()
    ->> Convolution(convSize: ConvSize(outputChannels: 32, kernelSize: 3, stride: 2), neuronType: .relu)
    ->> InstanceNorm()
    ->> FullyConnected(neurons: 128)
    ->> Neuron(type: .tanh)
    ->> FullyConnected(neurons: 10)
    ->> Softmax()

network.initialize()
```

First, we have to create the `network` which receives the MTLDevice (GPU), an inputSize and a parameter loader. The network comes with a `start` node which is the starting point of the network. The `inputSize` is the size expected by the first layer in the network. If the images you pass the network to be processed are not of the expected size then the `start` node will resize them accordingly.

The `parameterLoader` is responsible for loading the weights for each layer. It will be explained in detail further below.

After creating the `network` you can add layers to it with the `->>` operator.
You can also add parallel paths to your network by adding an array of subgraphs like this:

```swift
previousLayer
    ->> [Convolution(convSize: someSize), LocalResponseNorm(),
         Convolution(convSize: otherSize), InstanceNorm()]
    ->> Add()
```

In this case the output of `previousLayer` is passed to two different Convolution layers and then tthrough a normalization layer and after that they are added with the `Add` layer.

After you finish adding layers to your network, you must call `network.initialize()` to finish setting up your network.


## Running a network

To run a network call `run(...)`:

```swift
let commandQueue: MTLCommandQueue = ...

// get image from somewhere
let image = MPSImage(...)

network.run(inputImage: image, queue: commandQueue) { outputImage in
    ...
}
```


## ParameterLoader

A network created from scratch needs a `ParameterLoader` which is responsible for loading the parameters for its layers. There are two types of ParameterLoader implemented in Bender: `PerLayerBinaryLoader`, which expects a different file for each parameter, and `SingleBinaryLoader` that takes a single file and gets all parameters from it.

This loaders should cover most cases but if you want to implement a different loader then you have to create a class conforming to `ParameterLoader`:

```swift
public protocol ParameterLoader {

    /// Weight checkpoint. This variable is the prefix of the weight files.
    var checkpoint: String { get set }

    /// Loads weights for a single buffer
    ///
    /// - Parameters:
    ///   - id: NetworkLayer id
    ///   - modifier: The type of weights to load (e.g. bias, scale, shift). Use to distinguish the different parameters needed for a layer.
    ///   - size: Amount of floats to load
    /// - Returns: A pointer to the loaded floats
    func loadWeights(for id: String, modifier: String, size: Int) -> UnsafePointer<Float>

}
```

> Note: the `modifier` argument of `loadWeights` is passed when a layer requests more than one parameter (i.e. weights and bias). It is used to differentiate between these.


## Adding new layers

To add a new layer you must create a subclas of `NetworkLayer` and override some of the following functions:

```swift
func initialize(network: Network, device: MTLDevice)
func execute(commandBuffer: MTLCommandBuffer)
func updatedCheckpoint(device: MTLDevice) // optional
```

#### Initializing the layer

`initialize` is called when the network is initialized and it should set up everything for later execution. It should load the weights (as we do not want to do this in every execution loop) and also create the `outputImage` and `outputSize`. Bender uses `MPSTemporaryImage` for images used internally in a layer and `MPSImage` for inter-layer communication. Apple suggest using only MPSTemporaryImage's for images used and consumed in one MTLCommandBuffer but we experienced some problems with them and creating the MPSImage's at initialization time should not have performance hits at execution time.

`outputImage` and `outputSize` are two variables defined in `NetworkLayer` which must be instantiated at initialization time or before. `outputImage` is the image passed to the next nodes and `outputSize` is the size of this image.

#### Executing the layer

The `execute` function is called in each run loop. It should execute the layer and store the result in `outputImage`. You must override this function.

#### Updating checkpoints

Sometimes you might have more than one set of learned parameters. In that case, this function will be called if the network is asked to change its parameters (also known as checkpoint in TF).

You should override this method if your layer depends on learned parameters and your app allows to change checkpoints.

### Implementing custom Metal kernels

If your custom layer needs a custom Metal kernel function then you should create a `.metal` file and implement it there. Then in your `initialize` function you can get the `MTLCopmutePipelineState` for that function by calling:

```swift
pipeline = MetalShaderManager.shared.getFunction(name: "my_function")
```

There is a lot to learn about Metal and its special considerations and differences with CUDA but we won't document that here. You can go to the WWDC videos to get started if you are not familiar with Metal.


## MetalShaderManager

The MetalShaderManager keeps all the custom kernel functions that a app has loaded so that different layers that use the same Metal function will effectively use the same `MTLComputePipelineState`.

It also manages the function constants passed to these Metal functions. If you have a function that relies on funciton constants then you can pass them to the MetalShaderManager when you get your function. The function to get a kernel function is this:

```swift 
/// Get a MTLComputePipelineState with a Metal function of the given name
///
/// - Parameters:
///   - name: name of the function
///   - bundle: Bundle where the shader function was compiled. Used to get the correct library
///   - constants: functions constants passed to this function
/// - Returns: a MTLComputePipelineState for the requested function
func getFunction(name: String, in bundle: Bundle = Bundle.main, constants: [FunctionConstantBase]? = nil) -> MTLComputePipelineState
```

The function constants are created like this:

```swift
let c1 = FunctionConstant(index: 0, type: MTLDataType.ushort, value: 2)
let c2 = FunctionConstant(index: 1, type: MTLDataType.float, value: 0.5)
```

## Composing layers

One thing we realized is that it is useful to have single nodes that perform only a convolution or only a normalization but, on the other hand, in a single network we might want to run the same normalization after each convolution and possibly add an activation neuron behind. We also want to easily support residual layers.

Therefore, we support composite layers which basically are just a set of layers which we want to reuse in a network.

For example if we want to create a residual network like the one used for [style transfer](http://cs.stanford.edu/people/jcjohns/eccv16/) we could define it like:

```swift

```

[Creating a network]: #creating-a-network
[Running a network]: #running-a-network
[ParameterLoader]: #parameterloader
[Adding new layers]: #adding-new-layers
[MetalShaderManager]: #metalshadermanager
[Composing Layers]: #composing-layers