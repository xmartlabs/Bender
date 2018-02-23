//
//  TFConvOptimizer.swift
//  Bender
//
//  Created by Mathias Claassen on 5/23/17.
//
//

/// Combines Conv2d with BiasAdd. Should be executed after TFConvDilationOptimizer and Variable Processor
class TFConvOptimizer: TFOptimizer {

    /*  Takes:
        Conv2D --> BiasAdd (or Add) [--> Neuron]
          ^           ^
        Variable    Variable

     Returns:
        Variable -> Conv2D(+add-ons) <- Variable

     */
    func optimize(graph: TFGraph) {
        for node in graph.nodes {
            if node.nodeDef.isTFConvOp || node.nodeDef.isTFDepthwiseConvOp,
                let out = node.outgoingNodes() as? [TFNode], out.count == 1,
                let biasAdd = out.first {
                if biasAdd.nodeDef.isTFBiasAddOp || biasAdd.nodeDef.isTFAddOp,
                    let biasVar = (biasAdd.incomingNodes() as? [TFNode])?.first(where: { $0.nodeDef.isTFVariableOrConstOp }) {
                    node.addIncomingEdge(from: biasVar)
                    for output in biasAdd.outgoingNodes() {
                        output.replace(incomingEdge: biasAdd, with: node)
                    }
                    biasAdd.strip()
                }

                // add neuron data
                addNeuronIfThere(node: node)
            }
        }
    }

}

/// Adds dilations to Convolution. Should be executed after Variable Processor
class TFConvDilationOptimizer: TFOptimizer {

    /*  Takes:
     SpaceToBatch --> Conv2D --> BatchToSpace
        ^               ^              ^
     Consts          Variable        Consts

     Returns:
     Variable -> Conv2D(+add-ons)

     */
    func optimize(graph: TFGraph) {
        for node in graph.nodes {
            if node.nodeDef.isTFConvOp || node.nodeDef.isTFDepthwiseConvOp,
                let ins = node.incomingNodes() as? [TFNode],
                let stb = ins.first(where: { $0.nodeDef.isTFSpaceToBatch }),
                let out = node.outgoingNodes() as? [TFNode], out.count == 1,
                let bts = out.first(where: { $0.nodeDef.isTFBatchToSpace }) {
                if let btsConsts = (bts.incomingNodes() as? [TFNode])?.filter({ $0.nodeDef.isTFConstOp }) {
                    _ = btsConsts.map { $0.strip() }
                    bts.removeFromGraph()
                }

                if let stbConsts = (stb.incomingNodes() as? [TFNode])?.filter({ $0.nodeDef.isTFConstOp }) {
                    if let blockShape = stbConsts.first(where: { $0.nodeDef.name.hasSuffix("/block_shape")}),
                        let shape: [Int32] = blockShape.nodeDef.valueData()?.toArray(), shape.count == 2 {
                        node.nodeDef.attr["dilations"]?.list.i[1] = Int64(shape[0])
                        node.nodeDef.attr["dilations"]?.list.i[2] = Int64(shape[1])
                    }
                    _ = stbConsts.map { $0.strip() }
                    stb.removeFromGraph()
                }

            }
        }
    }

}
