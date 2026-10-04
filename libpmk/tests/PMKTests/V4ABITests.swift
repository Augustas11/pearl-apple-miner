// SPDX-License-Identifier: Apache-2.0
// Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
import XCTest
import CPMK
@testable import PMK

final class V4ABITests: XCTestCase {
    func testV4ABIInvalidHandlesAndAdmissionPolicyHelpers() {
        XCTAssertEqual(MemoryLayout<pmk_v4_stats>.size, 80)
        XCTAssertEqual(MemoryLayout<pmk_v4_result>.size, 168)
        XCTAssertEqual(pmkV4Init(nil, nil, 0), Int32(PMK_INVALID))
        XCTAssertEqual(pmkV4InitDiagnostic(nil, nil, 0), Int32(PMK_INVALID))
        XCTAssertEqual(pmkV4Probe(nil, nil, 0), Int32(PMK_INVALID))
        XCTAssertEqual(pmkV4AdmissionMetadata(nil, nil, 0), Int32(PMK_INVALID))
        XCTAssertEqual(pmkV4WriteAdmissionRecord(nil, nil, 0, nil, 0), Int32(PMK_INVALID))
        XCTAssertEqual(pmkV4RunJob(nil, nil, nil, nil, nil), Int32(PMK_INVALID))
        XCTAssertEqual(pmkV4RunCodesDiagnostic(nil, nil, nil, nil, nil), Int32(PMK_INVALID))
        XCTAssertEqual(pmkV4Poll(nil, nil), Int32(PMK_INVALID))
        XCTAssertEqual(pmkV4JobWaitCallback(nil), Int32(PMK_INVALID))
        XCTAssertEqual(pmkV4JobRelease(nil), Int32(PMK_INVALID))
    }
}
