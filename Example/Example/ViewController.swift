//
//  ViewController.swift
//  Example
//
//  Copyright © 2016 Xmartlabs SRL. All rights reserved.
//

import Palladium

class ViewController: UIViewController {

    let testRunner = PalladiumTestRunner()

    override func viewDidLoad() {
        super.viewDidLoad()
        testRunner.run()
    }

}
