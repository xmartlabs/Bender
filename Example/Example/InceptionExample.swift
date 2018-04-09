//
//  InceptionExample.swift
//  Example
//
//  Created by Mathias Claassen on 2/7/18.
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy
import MetalKit
import MetalBender
import UIKit

@available(iOS 11.0, *)
class InceptionViewController: UIViewController, UINavigationControllerDelegate {

    var inception: Network?
    var mobilenetwork: Network!
    var model: mobilenet!

    var selectedIndex = 0
    var commandQueue: MTLCommandQueue!

    var textureLoader: MTKTextureLoader!
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var label: UILabel!
    var classLabels: [String]?
    var inceptionLabels: [String]?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.commandQueue = Device.shared.makeCommandQueue()
        textureLoader = MTKTextureLoader(device: Device.shared)
        setupNetwork()
        classLabels = readClassLabels("mobilenet_labels")
        inceptionLabels = readClassLabels("imagenet_label_strings")
        model = mobilenet()
    }

    func setupNetwork() {
        guard let mobilenetUrl = Bundle.main.url(forResource: "mobilenet_frozen", withExtension: "pb") else {
            return
        }

        mobilenetwork = Network.load(url: mobilenetUrl, inputSize: LayerSize(h: 224, w: 224, f: 3), performInitialize: false)

        // after adding all our layers we are able to initialize the network
        mobilenetwork.addPreProcessing(layers: [Neuron(type: ActivationNeuronType.custom(neuron: MPSCNNNeuronLinear(device: Device.shared, a: 2.0, b: -1)), id: "scale_neuron")])
        mobilenetwork.permanentOutputNodes = mobilenetwork.nodes
        mobilenetwork.initialize()

        guard let inceptionUrl = Bundle.main.url(forResource: "inception_v3", withExtension: "pb") else {
            print("Download Inception from: http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz" +
                  " and freeze it using Benderthon or download a frozen .pb file. Leave it as 'inception_v3.pb' in the Example/data folder")
            return
        }
        inception = Network.load(url: inceptionUrl, inputSize: LayerSize(h: 299, w: 299, f: 3), performInitialize: false)

        // after adding all our layers we are able to initialize the network
        inception?.addPreProcessing(layers: [Neuron(type: ActivationNeuronType.custom(neuron: MPSCNNNeuronLinear(device: Device.shared, a: 2.0, b: -1)))])
        inception?.initialize()
    }

    @IBAction func runNetwork(_ sender: Any) {
        if selectedIndex == 1 {
            if let buffer = imageView.image?.toCVPixelBuffer() {
                print(buffer)
                let pred = try! model.prediction(input__0: buffer).MobilenetV1__Predictions__Reshape_1__0
                let floatPtr = pred.dataPointer.bindMemory(to: Double.self, capacity: pred.count)
                let arr = Array(UnsafeBufferPointer(start: floatPtr, count: pred.count))
                let top5 = arr.argsort(by: > ).prefix(5)
                label.text = top5.reduce("") {
                    return $0 + (classLabels?[$1] ?? "__") + "\n"
                }
            }
        } else {
            guard let image = imageView.image?.toMPS(loader: textureLoader) else {
                return
            }
            var model: Network!
            let labels: [String]!
            if selectedIndex == 0 {
                model = mobilenetwork
                labels = classLabels
            } else if let inception = inception {
                model = inception
                labels = inceptionLabels
            } else {
                return
            }
            model.run(input: image, queue: commandQueue) { [weak self] result in
                let arr = result!.toArray()
                let top5 = arr.argsort(by: > ).prefix(5)
                self?.label.text = top5.reduce("") {
                    return $0 + labels[$1] + "\n"
                }
            }
        }
    }

    @IBAction func changeModel(_ sender: Any) {
        if let control = sender as? UISegmentedControl {
            selectedIndex = control.selectedSegmentIndex
        } else {
            selectedIndex = 0
        }
    }

    @IBAction func camera(_ sender: Any) {

        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            return
        }

        let cameraPicker = UIImagePickerController()
        cameraPicker.delegate = self
        cameraPicker.sourceType = .camera
        cameraPicker.allowsEditing = false

        present(cameraPicker, animated: true)
    }

    @IBAction func openLibrary(_ sender: Any) {
        let picker = UIImagePickerController()
        picker.allowsEditing = false
        picker.delegate = self
        picker.sourceType = .photoLibrary
        present(picker, animated: true)
    }

    func readClassLabels(_ filename: String) -> [String]? {
        guard let path = Bundle.main.path(forResource: filename, ofType: "txt") else {
            return nil
        }

        do {
            let data = try String(contentsOfFile: path, encoding: .utf8)
            let myStrings = data.components(separatedBy: .newlines)
            return myStrings
        } catch {
            print(error)
            return nil
        }
    }

}

@available(iOS 11.0, *)
extension InceptionViewController: UIImagePickerControllerDelegate {
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        picker.dismiss(animated: true)
        guard let image = info["UIImagePickerControllerOriginalImage"] as? UIImage else {
            return
        }
        imageView.image = image
    }

}
