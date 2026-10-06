// swift-tools-version: 5.9
// SPDX-License-Identifier: Apache-2.0
import PackageDescription
let package = Package(name: "libpmk", platforms: [.macOS(.v14)], products: [
    .library(name: "pmk", type: .dynamic, targets: ["PMK"]),
    .executable(name: "pmk_v4_diag", targets: ["PMKV4Diag"])
], targets: [
    .target(name: "CPMK", path: "include", publicHeadersPath: "."),
    .target(name: "PMK", dependencies: ["CPMK"], path: ".", exclude: ["include", "tests", "tools", "README.md"],
            sources: ["Sources/PMK"], resources: [.copy("metal"), .copy("resources/probe"), .copy("resources/g3_na"), .copy("resources/v4_probe")]),
    .executableTarget(name: "PMKV4Diag", dependencies: ["PMK", "CPMK"], path: "tools",
                      sources: ["pmk_v4_diag.swift"]),
    .testTarget(name: "PMKTests", dependencies: ["PMK", "CPMK"], path: "tests/PMKTests",
        // The optimized @testable Job-construction harness crashes under Swift 6.3.3.
        // Keep tests unoptimized; the PMK library still builds with release optimization.
        swiftSettings: [.unsafeFlags(["-Onone"])])
])
