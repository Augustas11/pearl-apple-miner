import XCTest
@testable import PMK

final class PMKTests: XCTestCase {
    func testCPUOracleKeyedCompressDeterministic() {
        let key: [UInt32] = [
            0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
            0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,
        ]
        let transcript = Array(UInt32(0)..<UInt32(16))
        XCTAssertEqual(
            CPUOracle.hash(words: transcript, key: key),
            [3037689664, 1341097246, 1897670623, 1201585010,
             3828720245, 3772246512, 1380648733, 650772819]
        )
    }
}
