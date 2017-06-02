//
//  Regex.swift
//  Bender
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

/// Holds and evaluates a NSRegularExpression
public class Regex {

    let expression: NSRegularExpression
    let pattern: String

    public init(_ pattern: String) throws {
        self.pattern = pattern
        self.expression = try NSRegularExpression(pattern: pattern, options: .caseInsensitive)
    }


    /// Returns if the expression matches the String passed as parameter
    public func test(_ input: String) -> Bool {
        let matches = expression.matches(in: input, options: [], range: NSRange(location: 0, length: input.characters.count))
        return matches.count > 0
    }

    /// Returns the range of the first match of the expression in the String passed as parameter
    public func match(_ input: String) -> NSRange {
        let range = expression.rangeOfFirstMatch(in: input, options: [], range: NSRange(location: 0, length: input.characters.count))
        return range
    }

}
