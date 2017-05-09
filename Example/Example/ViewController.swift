//
//  ViewController.swift
//  Example
//
//  Copyright © 2016 Xmartlabs SRL. All rights reserved.
//

import AVFoundation
import MetalKit
import MetalPerformanceShaders
import UIKit
import Palladium

class ViewController: UIViewController {
    var styleNet: Network!
    var device: MTLDevice!

    override func viewDidLoad() {
        super.viewDidLoad()
        self.device = MTLCreateSystemDefaultDevice()!
        setupStyleNet()

    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

    func setupStyleNet() {
        styleNet = Network(device: device, checkpoint: "udnie", inputSize: LayerSize(f: 3, w: 256))
        measure("define graph") {
            let g = styleNet.start ->> Convolution(convSize: ConvSize(outputChannels: 3, kernelSize: 3, stride: 1), weightsFile: "", id: "a")
                ->> BGRAtoRGBA(device: device, id: "b")
                ->> [BGRAtoRGBA(device: device, id: "c") ->> BGRAtoRGBA(device: device, id: "d"),
                     BGRAtoRGBA(device: device, id: "e")]
                ->> Add(device: device, id: "f")
                ->> ResidualLayer(device: device, convSize: ConvSize(outputChannels: 3, kernelSize: 3, stride: 1),
                                  id: "res1", weightFiles: "", "", "", "", "", "")
                ->> [BGRAtoRGBA(device: device, id: "g")
                    ->> [BGRAtoRGBA(device: device, id: "h"),
                         BGRAtoRGBA(device: device, id: "i") ->> BGRAtoRGBA(device: device, id: "j") ->> BGRAtoRGBA(device: device, id: "k")]
                    ->> Add(device: device, id: "l"),
                     BGRAtoRGBA(device: device, id: "m") ->> BGRAtoRGBA(device: device, id: "n")]
                ->> Add(device: device, id: "o")
            print("Network goes from: \(g.input.id ?? "nil") to \(g.output.id ?? "nil")")
        }


        styleNet.initialize()
        print("finished")

    }

}
