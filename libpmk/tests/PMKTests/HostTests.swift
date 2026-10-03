import XCTest
@testable import PMK
import CPMK

final class HostTests: XCTestCase {
    func testABI() {
        XCTAssertEqual(MemoryLayout<pmk_slot>.size, 104)
        XCTAssertEqual(MemoryLayout<pmk_job_desc>.size, 200)
        XCTAssertEqual(MemoryLayout<pmk_result>.size, 72)
        XCTAssertEqual(pmkInit(nil, nil, 0), Int32(PMK_INVALID))
        XCTAssertEqual(pmkInitDiagnostic(nil, nil, 0), Int32(PMK_INVALID))
        XCTAssertEqual(pmkPoll(nil, nil), Int32(PMK_INVALID))
        XCTAssertEqual(pmkJobWaitCallback(nil), Int32(PMK_INVALID))
        XCTAssertEqual(pmkJobRelease(nil), Int32(PMK_INVALID))
        XCTAssertEqual(pmkProbe(nil, nil, 0), Int32(PMK_INVALID))
        XCTAssertEqual(pmkProbeRefresh(nil, nil, 0), Int32(PMK_INVALID))
    }

    func testCallbackBarrierDrainsReturnAndRejectsCallbackThreadWait() {
        let barrier = CallbackBarrier()
        barrier.arm()
        let callbackEntered = DispatchSemaphore(value: 0)
        let allowCallbackReturn = DispatchSemaphore(value: 0)
        let callbackReturned = DispatchSemaphore(value: 0)
        let externalWaitReturned = DispatchSemaphore(value: 0)

        DispatchQueue.global().async {
            barrier.beginCallback()
            XCTAssertFalse(barrier.wait())
            callbackEntered.signal()
            allowCallbackReturn.wait()
            barrier.endCallback()
            callbackReturned.signal()
        }
        XCTAssertEqual(callbackEntered.wait(timeout: .now() + 1), .success)
        DispatchQueue.global().async {
            if barrier.wait() { externalWaitReturned.signal() }
        }
        XCTAssertEqual(externalWaitReturned.wait(timeout: .now() + 0.05), .timedOut)
        allowCallbackReturn.signal()
        XCTAssertEqual(callbackReturned.wait(timeout: .now() + 1), .success)
        XCTAssertEqual(externalWaitReturned.wait(timeout: .now() + 1), .success)
    }

    func testUnarmedCallbackBarrierCanBeDropped() {
        weak var dropped: CallbackBarrier?
        autoreleasepool {
            let barrier = CallbackBarrier()
            dropped = barrier
            XCTAssertTrue(barrier.wait())
        }
        XCTAssertNil(dropped)
    }

    func testG3AdmissionRequiresFreshDeviceClassRecord() throws {
        XCTAssertFalse(requiresG3Admission(deviceClass: "Apple10"))
        XCTAssertTrue(requiresG3Admission(deviceClass: "Apple9"))
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("g3-admission.json")
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        let json = """
        {"devices":[{"gpu_name":"Apple M3 Ultra","device_class":"Apple9","cache_key":"abc","os_build":"26A1","g3_passed":true,"last_probe_unix":1799999900}]}
        """
        try json.write(to: file, atomically: true, encoding: .utf8)
        XCTAssertNoThrow(try requireG3Admission(deviceName: "Apple M3 Ultra", deviceClass: "Apple9",
                                                cacheKey: "abc", osBuild: "26A1", now: now,
                                                path: file.path))
        XCTAssertNoThrow(try requireG3Admission(deviceName: "Apple M3 Ultra", deviceClass: "Apple9",
                                                cacheKey: "abc", osBuild: "26A1",
                                                now: now.addingTimeInterval(25 * 3600),
                                                path: file.path))
        XCTAssertThrowsError(try requireG3Admission(deviceName: "Apple M3 Ultra", deviceClass: "Apple8",
                                                    cacheKey: "abc", osBuild: "26A1", now: now,
                                                    path: file.path))
        XCTAssertThrowsError(try requireG3Admission(deviceName: "Apple M3 Ultra", deviceClass: "Apple9",
                                                    cacheKey: "abc", osBuild: "26A2", now: now,
                                                    path: file.path))
        try """
        {"devices":[{"gpu_name":"Apple M3 Ultra","device_class":"Apple9","cache_key":"abc","os_build":"26A1","g3_passed":true,"last_probe_unix":1799999900,"valid_hours":24}]}
        """.write(to: file, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try requireG3Admission(deviceName: "Apple M3 Ultra", deviceClass: "Apple9",
                                                    cacheKey: "abc", osBuild: "26A1",
                                                    now: now.addingTimeInterval(25 * 3600),
                                                    path: file.path))
        XCTAssertThrowsError(try requireG3Admission(deviceName: "Different GPU", deviceClass: "Apple9",
                                                    cacheKey: "abc", osBuild: "26A1", now: now,
                                                    path: file.path))
        try """
        {"devices":[{"gpu_name":"Apple M3 Ultra","device_class":"Apple9","cache_key":"abc","os_build":"26A1","g3_passed":true,"last_probe_unix":1800000100,"valid_hours":24}]}
        """.write(to: file, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try requireG3Admission(deviceName: "Apple M3 Ultra", deviceClass: "Apple9",
                                                    cacheKey: "abc", osBuild: "26A1", now: now,
                                                    path: file.path))
        try """
        {"devices":[{"gpu_name":"Apple M3 Ultra","device_class":"Apple9","cache_key":"abc","os_build":"26A1","g3_passed":true,"last_probe_unix":1799999900,"valid_hours":0}]}
        """.write(to: file, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try requireG3Admission(deviceName: "Apple M3 Ultra", deviceClass: "Apple9",
                                                    cacheKey: "abc", osBuild: "26A1", now: now,
                                                    path: file.path))
    }

    func testProbeRefreshIntervalPolicyUsesMonotonicTime() {
        XCTAssertEqual(probeRefreshIntervalSeconds([:]), 24 * 3600)
        XCTAssertEqual(probeRefreshIntervalSeconds(["PMK_PROBE_REFRESH_HOURS": "0.5"]), 1800)
        XCTAssertEqual(probeRefreshIntervalSeconds(["PMK_PROBE_REFRESH_HOURS": "-1"]), 24 * 3600)
        XCTAssertFalse(shouldRefreshProbe(last: 100, now: 110, interval: 20))
        XCTAssertTrue(shouldRefreshProbe(last: 100, now: 120, interval: 20))
        XCTAssertFalse(shouldRefreshProbe(last: 120, now: 100, interval: 0))
    }

    func testLargeProductionOverflowFailsClosedInsteadOfRecovering() {
        XCTAssertTrue(canRecoverOverflow(tileCount: overflowRecoveryTileLimit, overflow: 1))
        XCTAssertFalse(canRecoverOverflow(tileCount: overflowRecoveryTileLimit + 1, overflow: 1))
        XCTAssertTrue(canRecoverOverflow(tileCount: overflowRecoveryTileLimit + 1, overflow: 0))
    }

    func testStringBoundaries() {
        var s = [CChar](repeating: 7, count: 5)
        s.withUnsafeMutableBufferPointer { putString("long value", $0.baseAddress, 4) }
        XCTAssertEqual(s, [108, 111, 110, 0, 7])
    }

    func testDiagnosticStringIsBoundedPrintableAndRedacted() {
        let input = "rpc_password=supersecret token=abc\0\n" + String(repeating: "x", count: 700)
        let output = diagnosticString(input)
        XCTAssertLessThanOrEqual(output.count, 512)
        XCTAssertFalse(output.contains("supersecret"))
        XCTAssertFalse(output.contains("abc"))
        XCTAssertFalse(output.contains("\0"))
        XCTAssertTrue(output.contains("rpc_password=<redacted>"))
        XCTAssertTrue(output.contains("token=<redacted>"))
    }

    func testAllocationFailureRecordsContextDiagnostic() throws {
        let context = try Context(requireAdmission: false)
        let opaque = Unmanaged.passUnretained(context).toOpaque()
        var pointer: UnsafeMutableRawPointer?
        XCTAssertEqual(pmkBufferAlloc(opaque, UInt64(maxBufferBytes), &pointer), Int32(PMK_RESOURCE))
        var error = [CChar](repeating: 0, count: 256)
        XCTAssertEqual(pmkContextError(opaque, &error, UInt64(error.count)), 0)
        XCTAssertTrue(String(cString: error).contains("buffer allocation rejected"))
    }

    func testValidSignalBytesAcceptsExactRangeForEveryByteValue() {
        for raw in UInt8.min...UInt8.max {
            let value = Int8(bitPattern: raw)
            let bytes = [value]
            let expected = value >= -64 && value <= 64
            bytes.withUnsafeBufferPointer {
                XCTAssertEqual(validSignalBytes($0.baseAddress!, bytes.count), expected, "byte \(raw)")
            }
        }
    }

    func testValidSignalBytesHandlesUnalignedSIMDLoadsAndTails() {
        var storage = [Int8](repeating: 0, count: 64 + 130)
        for offset in 0..<64 {
            for length in 0...130 {
                for i in 0..<length {
                    storage[offset + i] = Int8((i % 129) - 64)
                }
                storage.withUnsafeBufferPointer { raw in
                    let pointer = raw.baseAddress!.advanced(by: offset)
                    XCTAssertTrue(validSignalBytes(pointer, length), "offset \(offset) length \(length)")
                }
                if length > 0 {
                    storage[offset + length - 1] = 65
                    storage.withUnsafeBufferPointer { raw in
                        let pointer = raw.baseAddress!.advanced(by: offset)
                        XCTAssertFalse(validSignalBytes(pointer, length), "high offset \(offset) length \(length)")
                    }
                    storage[offset + length - 1] = -65
                    storage.withUnsafeBufferPointer { raw in
                        let pointer = raw.baseAddress!.advanced(by: offset)
                        XCTAssertFalse(validSignalBytes(pointer, length), "low offset \(offset) length \(length)")
                    }
                }
            }
        }
    }
}
