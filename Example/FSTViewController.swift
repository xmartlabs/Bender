//
//  FSTViewController.swift
//  Example
//
//  Created by Santiago Fernandez on 9/4/17.
//
//

import MetalBender
import UIKit

class FSTViewController: UIViewController {
    
    var network: Network!
    let inputSize: LayerSize = LayerSize(h: 256, w: 256, f: 3)
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let url = Bundle.main.url(forResource: "fst", withExtension: "pb")!
        
        let converter = TFConverter.default(verbose: true)
        network = Network.load(url: url, converter: converter, inputSize: inputSize, parameterLoader: nil, performInitialize: true)
    }
    
}
