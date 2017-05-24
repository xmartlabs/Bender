//
//  TFInstanceNormOptimizer.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/24/17.
//
//

import Foundation

/// Creates an Instance norm from a series of nodes. Must be executed after Variable processor
public class TFInstanceNormOptimizer: TFDeleteSubgraphOptimizer {

    /*  Takes:
     Input --> Moments --> Set_of_nodes --> Mul   -->   Add  --> Output
                                             ^           ^
                                         Variable     Variable

     Returns:
             Input   -->   InstanceNormAdd  -->  Output
                                ^       ^
         Variable -> InstanceNormMul | Variable

     */

    struct INormNodes {
        var input: TFNode?
        var output: TFNode?
        var toStrip: [TFNode] = []
    }

    public var regex: Regex = try! Regex("moments(_\\d+)?/")

    public init() {}

    public func optimize(graph: TFGraph) {
        var mappings = [String: INormNodes]()
        for node in graph.nodes {
            if isInSubgraph(node) {
                var currentValue = mappings[id(for: node)] ?? INormNodes()
                if let inputs = (node.incomingNodes() as? [TFNode])?.filter({ !isInSubgraph($0) }), inputs.count == 1 {
                    if currentValue.input == nil {
                        currentValue.input = inputs.first
                    } else {
                        assert(currentValue.input!.isEqual(to: inputs.first!))
                    }
                } else if let outputs = (node.outgoingNodes() as? [TFNode])?.filter({ !isInSubgraph($0) }), outputs.count == 1 {
                    // check if output nodes form instance norm
                    if let add = outputs.first, add.nodeDef.op.isTFAddOp,
                        let addOut = (add.outgoingNodes() as? [TFNode]), addOut.count == 1,
                        let pow = addOut.first, pow.nodeDef.op.isTFPowOp,
                        let powOut = (pow.outgoingNodes() as? [TFNode]), powOut.count == 1,
                        let trueDiv = powOut.first, trueDiv.nodeDef.op.isTFRealDivOp,
                        let trueDivOut = (trueDiv.outgoingNodes() as? [TFNode]), trueDivOut.count == 1,
                        let sub = trueDiv.incomingNodes().filter({ ($0 as? TFNode)?.nodeDef.op.isTFSubOp ?? false }).first,
                        let mul = trueDivOut.first, mul.nodeDef.op.isTFMulOp,
                        let mulOut = (mul.outgoingNodes() as? [TFNode]), mulOut.count == 1,
                        let addFinal = mulOut.first, addFinal.nodeDef.op.isTFAddOp,
                        // check for incoming variables
                        let _ = mul.incomingNodes().filter({ ($0 as? TFNode)?.nodeDef.op.isTFVariableV2Op ?? false }).first,
                        let _ = addFinal.incomingNodes().filter({ ($0 as? TFNode)?.nodeDef.op.isTFVariableV2Op ?? false }).first
                    {
                        currentValue.toStrip.append(contentsOf: [sub, add, pow, trueDiv] as! [TFNode])
                        // change final add to Instancenorm node
                        addFinal.nodeDef.op = Constants.Ops.InstanceNormAdd
                        mul.nodeDef.op = Constants.Ops.InstanceNormMul
                        currentValue.output = addFinal
                    }
                }
                currentValue.toStrip.append(node)
                mappings[id(for: node)] = currentValue
            }
        }

        // wire together
        for id in mappings.keys {
            if let data = mappings[id], let input = data.input, let output = data.output {
                output.addIncomingEdge(from: input)
                for node in data.toStrip {
                    node.strip()
                }
            }
        }
    }
    
}
