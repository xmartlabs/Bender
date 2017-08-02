//
//  ComputeFunctions.swift
//  Bender
//
//  Created by Mathias Claassen on 5/4/17.
//
//

import Metal
import MetalPerformanceShadersProxy

/// Manages the references to the different Metal compute kernels
public class MetalShaderManager {

    /// MetalShaderManager singleton instance
    public static var shared = MetalShaderManager()
    var device: MTLDevice
    var benderLibrary: MTLLibrary
    var mainLibrary: MTLLibrary?
    var pipelines = [ComputePipelineDef: MTLComputePipelineState]()

    private init() {
        self.device = MTLCreateSystemDefaultDevice()!
        self.benderLibrary = device.makeMyLibrary(bundle: Bundle(for: MetalShaderManager.self))!
        self.mainLibrary = device.makeMyLibrary(bundle: Bundle.main)
    }

    /// Get a MTLComputePipelineState with a Metal function of the given name
    ///
    /// - Parameters:
    ///   - name: name of the function
    ///   - bundle: Bundle where the shader function was compiled. Used to get the correct library
    ///   - constants: functions constants passed to this function
    /// - Returns: a MTLComputePipelineState for the requested function
    public func getFunction(name: String, in bundle: Bundle = Bundle.main, constants: [FunctionConstantBase]? = nil) -> MTLComputePipelineState {
        let library = bundle == Bundle.main ? mainLibrary! : benderLibrary
        let pipelineDef = ComputePipelineDef(function: name, constants: constants ?? [])

        if let pipelineState = pipelines[pipelineDef] {
            return pipelineState
        }

        do {
            var function: MTLFunction
            if let constants = constants {
                let values = MTLFunctionConstantValues()
                for constant in constants {
                    var val = constant.getValue()
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
