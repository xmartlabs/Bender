//
//  UnitTestViewController.swift
//  Example
//
//  Created by Mathias Claassen on 5/26/17.
//
//

import UIKit

class ViewController: UIViewController {

    let testRunner = PalladiumTestRunner()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        testRunner.run()
    }

}
