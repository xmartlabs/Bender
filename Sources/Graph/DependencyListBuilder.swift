//
//  DependencyListBuilder.swift
//  Bender
//
//  Created by Diego Ernst on 6/12/17.
//
//

class DependencyListBuilder<T: Node> {

    private var nodes = [T]()

    /// Builds the dependency list for a graph of T given an input node
    func list(from input: T) -> [T] {
        return list(from: [input])
    }

    /// Builds the dependency list for a graph of T given an inputs node list
    func list(from inputs: [T]) -> [T] {
        for input in inputs {
            buildExecutionList(node: input)
        }
        return nodes
    }

    private func buildExecutionList(node: T) {
        guard !node.incomingNodes().contains (where: { incoming in
            return !nodes.contains(where: { $0.isEqual(to: incoming) })
        }) else { return }
        nodes.append(node)
        for node in node.outgoingNodes().flatMap({ $0 as? T }) {
            buildExecutionList(node: node)
        }
    }

}
