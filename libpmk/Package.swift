// swift-tools-version: 5.9
import PackageDescription
let package = Package(name: "libpmk", platforms: [.macOS(.v14)], products: [
    .library(name: "pmk", type: .dynamic, targets: ["PMK"])
], targets: [
    .target(name: "CPMK", path: "include", publicHeadersPath: "."),
    .target(name: "PMK", dependencies: ["CPMK"], path: ".", exclude: ["include", "tests", "README.md"],
            sources: ["Sources/PMK"], resources: [.copy("metal"), .copy("resources/probe")]),
    .testTarget(name: "PMKTests", dependencies: ["PMK", "CPMK"], path: "tests/PMKTests",
        // The optimized @testable Job-construction harness crashes under Swift 6.3.3.
        // Keep tests unoptimized; the PMK library still builds with release optimization.
        swiftSettings: [.unsafeFlags(["-Onone"])])
])
