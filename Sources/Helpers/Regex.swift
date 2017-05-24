//
//  Regex.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

public class Regex {

    let expression: NSRegularExpression
    let pattern: String

    public init(_ pattern: String) throws {
        self.pattern = pattern
        self.expression = try NSRegularExpression(pattern: pattern, options: .caseInsensitive)
    }

    public func test(_ input: String) -> Bool {
        let matches = expression.matches(in: input, options: .withoutAnchoringBounds, range: NSRange(location: 0, length: input.characters.count))
        return matches.count > 0
    }

    public func match(_ input: String) -> NSRange {
        let range = expression.rangeOfFirstMatch(in: input, options: .withoutAnchoringBounds, range: NSRange(location: 0, length: input.characters.count))
        return range
    }

}
