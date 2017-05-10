//
//  ComputeFunctions.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/4/17.
//
//

import Metal

open class MetalShaderManager {

    public static var shared = MetalShaderManager()
    var device: MTLDevice
    var palladiumLibrary: MTLLibrary
    var mainLibrary: MTLLibrary
    var pipelines = [ComputePipelineDef: MTLComputePipelineState]()

    private init() {
        self.device = MTLCreateSystemDefaultDevice()!
        self.palladiumLibrary = device.makeMyLibrary(bundle: Bundle(for: MetalShaderManager.self))
        self.mainLibrary = device.makeMyLibrary(bundle: Bundle.main)
    }

    open func getFunction(name: String, in bundle: Bundle = Bundle.main, constants: [FunctionConstantBase]? = nil) -> MTLComputePipelineState {
        let library = bundle == Bundle.main ? mainLibrary : palladiumLibrary
        let pipelineDef = ComputePipelineDef(function: name, constants: constants ?? [])

        if let pipelineState = pipelines[pipelineDef] {
            debugPrint("Reusing pipeline state \(name)")
            return pipelineState
        }

        do {
            debugPrint("Creating pipeline state \(name)")
            var function: MTLFunction
            if let constants = constants {
                let values = MTLFunctionConstantValues()
                for constant in constants {
                    var val = constant.getValue()
                    print("VAL: \(val)")
                    values.setConstantValue(&val, type: constant.type, at: constant.index)
                }
                function = try library.makeFunction(name: name, constantValues: values)
            } else {
                guard let loadedFunction = library.makeFunction(name: name) else {
                    fatalError("Function \(name) does not exist")
                }
                function = loadedFunction
            }
            let pipeline = try device.makeComputePipelineState(function: function)
            pipelines[pipelineDef] = pipeline
            return pipeline
        } catch {
            fatalError("Unable to create pipeline state, check metal shaders")
        }
    }

}
