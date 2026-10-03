#!/usr/bin/env python3
"""Compile the real G3 vector suite as a standalone, fail-closed executable.

Only the build copy is adapted: XCTest assertions become fatal checks, vector
lookup becomes relocatable, and successful completion emits device admission
metadata. No XCTest/SwiftPM/compiler is needed on the target Mac.
"""
import pathlib
import subprocess
import sys

root, out = map(pathlib.Path, sys.argv[1:])
work = out / '.build-work'
lib = work / 'libpmk'
staging = work / 'g3-helper'
staging.mkdir(exist_ok=True)
source = (lib / 'tests/PMKTests/KernelTests.swift').read_text()
source = source.replace('import XCTest', 'import Foundation').replace('@testable import PMK\n', '')
source = source.replace('final class KernelTests: XCTestCase', 'final class KernelTests')
# These are real assertions: every mismatch terminates the process nonzero.
source = source.replace('XCTAssertEqual', 'requireEqual').replace('XCTAssertTrue', 'requireTrue')
source = source.replace('XCTAssertThrowsError', 'requireThrowsError')
source = source.replace('XCTUnwrap', 'unwrapOrThrow').replace('XCTFail', 'failTest')
old = '''let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()'''
assert old in source
source = source.replace(old, 'let root = URL(fileURLWithPath: ProcessInfo.processInfo.environment["PMK_B4_ROOT"]!)')
source = source.replace('bench/k3sg/studio/vectors/', 'vectors/g3/')
marker = '''                print("G3: PASS \\(name) \\(vector.cases.count) cases")
            }'''
assert marker in source
source = source.replace(marker, marker + '''
            let record: [String: Any] = [
                "gpu_name": context.device.name,
                "device_class": context.deviceClass,
                "cache_key": context.cacheKey,
                "os_build": context.osBuild,
                "g3_passed": true,
                "last_probe_unix": Date().timeIntervalSince1970,
                "valid_hours": 6,
                "source": "B4 full G3: v1, v2, v3, v4 slots/counters/overflow/canaries"
            ]
            let data = try JSONSerialization.data(withJSONObject: ["devices": [record]], options: [.sortedKeys])
            print(String(data: data, encoding: .utf8)!)''')
support = r'''
func failTest(_ message: String) -> Never { fatalError(message) }
func requireEqual<T: Equatable>(_ lhs: T, _ rhs: T, _ message: String = "G3 equality assertion failed") {
    if lhs != rhs { failTest(message) }
}
func requireTrue(_ value: Bool, _ message: String = "G3 boolean assertion failed") {
    if !value { failTest(message) }
}
func requireThrowsError<T>(_ expression: @autoclosure () throws -> T,
                          _ handler: (Error) -> Void) {
    do {
        _ = try expression()
        failTest("G3 expected an error")
    } catch {
        handler(error)
    }
}
func unwrapOrThrow<T>(_ value: T?) throws -> T {
    guard let value else { throw PMKError("G3: missing required value") }
    return value
}
'''
(staging / 'main.swift').write_text(source + support + '\nKernelTests().testK3SGBenchVectorsMatchOracleSlotsAndCounters()\n')
# Build-only C module map and resource accessor, with no absolute runtime fallback.
(staging / 'module.modulemap').write_text(f'module CPMK {{ header "{lib / "include/libpmk.h"}" export * }}\n')
(staging / 'Resources.swift').write_text('''import Foundation
extension Bundle {
    static var module: Bundle {
        guard let path = ProcessInfo.processInfo.environment["PMK_RESOURCE_BUNDLE"],
              let bundle = Bundle(path: path) else { fatalError("missing PMK_RESOURCE_BUNDLE") }
        return bundle
    }
}
''')
# Match the production C validator while keeping the Swift harness unoptimized.
subprocess.run(['xcrun', 'clang', '-O3', '-target', 'arm64-apple-macos14.0',
                '-c', str(lib / 'include/anchor.c'), '-o', str(staging / 'anchor.o')], check=True)
# -Onone matches the upstream G3 harness requirement for Swift 6.3 Job construction.
subprocess.run(['xcrun', 'swiftc', '-Onone', '-target', 'arm64-apple-macos14.0',
                '-I', str(staging), *map(str, sorted((lib / 'Sources/PMK').glob('*.swift'))),
                str(staging / 'Resources.swift'), str(staging / 'main.swift'), str(staging / 'anchor.o'),
                '-o', str(out / 'bin/g3-admit')], check=True)
