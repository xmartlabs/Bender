//
//  UnitTestViewController.swift
//  Example
//
//  Created by Mathias Claassen on 5/26/17.
//
//

import UIKit
import Bender

class TestViewController: UIViewController {

//    let testRunner = BenderTestRunner()

    var device: MTLDevice!

    override func viewDidLoad() {
        super.viewDidLoad()
//        testRunner.run()
        device = MTLCreateSystemDefaultDevice()

        let url = Bundle.main.url(forResource: "test_concat", withExtension: "pb")!
        let inputSize = LayerSize(f: 3, w: 3, h: 5)
        let nnet = Network(device: device, inputSize: inputSize, parameterLoader: SingleBinaryLoader(checkpoint: "lala"))
        let converter = TFConverter.default()
        nnet.convert(converter: converter, url: url, type: .text)

        nnet.initialize()
    }

}
