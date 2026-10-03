import XCTest
import Metal
@testable import PMK

final class FailClosedTests: XCTestCase {
    func testProbeRejectsBrokenLayoutAndProductionHash() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: root.appendingPathComponent("metal/k3sg.metal"), encoding: .utf8)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1; options.fastMathEnabled = false
        options.preprocessorMacros = ["VARIANT": 3, "BM": 64, "BN": 64, "BK": 16, "WM": 2, "WN": 2, "PF": 2].mapValues { NSNumber(value: $0) }
        for (from, to) in [("out[lane * 2] = te[0]", "out[lane * 2] = 999.f"),
                           ("out[0] = v0 ^ v8", "out[0] = v0 ^ v8 ^ 1u")] {
            XCTAssertTrue(source.contains(from))
            let broken = source.replacingOccurrences(of: from, with: to)
            let lib = try device.makeLibrary(source: broken, options: options)
            let pipeline = try device.makeComputePipelineState(function: XCTUnwrap(lib.makeFunction(name: "k3sg")))
            XCTAssertThrowsError(try runProbe(device: device, queue: queue, library: lib, pipeline: pipeline))
        }
    }
}
