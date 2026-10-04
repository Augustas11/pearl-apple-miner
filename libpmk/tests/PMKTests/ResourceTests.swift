// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import PMK

final class ResourceTests: XCTestCase {
    private func makeBundle(_ root: URL, file: String) throws -> URL {
        let bundle = root.appendingPathComponent("libpmk_PMK.bundle")
        let fileURL = bundle.appendingPathComponent(file)
        try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try "test".write(to: fileURL, atomically: true, encoding: .utf8)
        return bundle
    }

    private func temporaryRoot(_ name: String = #function) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("pmk-resource-tests")
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent(name)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }

    func testEnvironmentBundleWins() throws {
        let root = try temporaryRoot()
        let envBundle = try makeBundle(root.appendingPathComponent("env"), file: "metal/k3sg.metal")
        let dylibDir = root.appendingPathComponent("lib")
        _ = try makeBundle(dylibDir, file: "metal/k3sg.metal")

        let url = try pmkResourceURL(
            "metal/k3sg.metal",
            environment: ["PMK_RESOURCE_BUNDLE": envBundle.path],
            loadedLibraryURL: dylibDir.appendingPathComponent("libpmk.dylib")
        )

        XCTAssertEqual(url.path, envBundle.appendingPathComponent("metal/k3sg.metal").path)
    }

    func testBundleNextToLoadedDylibWorks() throws {
        let root = try temporaryRoot()
        let dylibDir = root.appendingPathComponent("relocated")
        let bundle = try makeBundle(dylibDir, file: "v4_probe/manifest.json")

        let url = try pmkResourceURL(
            "v4_probe/manifest.json",
            environment: [:],
            loadedLibraryURL: dylibDir.appendingPathComponent("libpmk.dylib")
        )

        XCTAssertEqual(url.path, bundle.appendingPathComponent("v4_probe/manifest.json").path)
    }

    func testMissingEnvironmentBundleThrowsCleanResourceError() throws {
        let root = try temporaryRoot()
        let missing = root.appendingPathComponent("missing.bundle")

        XCTAssertThrowsError(try pmkResourceURL(
            "metal",
            environment: ["PMK_RESOURCE_BUNDLE": missing.path],
            loadedLibraryURL: nil
        )) { error in
            guard let resourceError = error as? PMKResourceError else {
                return XCTFail("expected PMKResourceError, got \(error)")
            }
            XCTAssertTrue(resourceError.description.contains("PMK_RESOURCE_BUNDLE path does not exist"))
            XCTAssertTrue(resourceError.description.contains(missing.path))
        }
    }
}
