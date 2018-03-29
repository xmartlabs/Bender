//
//  UnitTestViewController.swift
//  Example
//
//  Created by Mathias Claassen on 5/26/17.
//
//

import UIKit

class TestViewController: UIViewController {

    let testRunner = BenderTestRunner()
    @IBOutlet weak var imageRecognitionButton: UIButton!

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        if #available(iOS 11, *) {
            imageRecognitionButton.isEnabled = true
        } else {
            imageRecognitionButton.isEnabled = false
            imageRecognitionButton.setTitle("Image recognition (iOS 11+ only)", for: .normal)
        }
    }

    @IBAction func run(_ sender: Any) {
        testRunner.run()
    }

}
