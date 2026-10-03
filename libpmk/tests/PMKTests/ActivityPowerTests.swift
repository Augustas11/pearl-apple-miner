import Foundation
import XCTest
import CPMK
@testable import PMK

private final class RecordingActivityManager: ActivityManaging {
    private(set) var options: ProcessInfo.ActivityOptions = []
    private(set) var reason = ""
    private(set) var ended = false
    private let token = NSObject()

    func beginActivity(options: ProcessInfo.ActivityOptions, reason: String) -> NSObjectProtocol {
        self.options = options
        self.reason = reason
        return token
    }

    func endActivity(_ activity: NSObjectProtocol) {
        XCTAssertTrue(activity === token)
        ended = true
    }
}

final class ActivityPowerTests: XCTestCase {
    func testActivityUsesRequestedSleepPolicyAndEndsToken() {
        let manager = RecordingActivityManager()
        var handle: ActivityHandle? = ActivityHandle(reason: "active mining", manager: manager)

        XCTAssertTrue(manager.options.contains(.userInitiated))
        XCTAssertTrue(manager.options.contains(.idleSystemSleepDisabled))
        XCTAssertFalse(manager.options.contains(.idleDisplaySleepDisabled))
        XCTAssertFalse(manager.options.contains(.latencyCritical))
        XCTAssertEqual(manager.reason, "active mining")
        XCTAssertFalse(manager.ended)

        handle = nil
        XCTAssertNil(handle)
        XCTAssertTrue(manager.ended)
    }

    func testActivityCABIHandlesNullReasonAndNullEnd() {
        let handle = pmkActivityBegin(nil)
        XCTAssertNotNil(handle)
        pmkActivityEnd(handle)
        pmkActivityEnd(nil)
    }

    func testPowerSourceClassificationCoversEveryABIValue() {
        XCTAssertEqual(classifyPowerSource(providingType: kIOPMACPowerKey, hasInternalBattery: true), .ac)
        XCTAssertEqual(classifyPowerSource(providingType: kIOPMBatteryPowerKey, hasInternalBattery: true), .battery)
        XCTAssertEqual(classifyPowerSource(providingType: kIOPMACPowerKey, hasInternalBattery: false), .desktop)
        XCTAssertEqual(classifyPowerSource(providingType: kIOPMUPSPowerKey, hasInternalBattery: false), .desktop)
        XCTAssertEqual(classifyPowerSource(providingType: nil, hasInternalBattery: false), .unknown)

        XCTAssertEqual(PowerSource.unknown.rawValue, Int32(PMK_POWER_SOURCE_UNKNOWN.rawValue))
        XCTAssertEqual(PowerSource.ac.rawValue, Int32(PMK_POWER_SOURCE_AC.rawValue))
        XCTAssertEqual(PowerSource.battery.rawValue, Int32(PMK_POWER_SOURCE_BATTERY.rawValue))
        XCTAssertEqual(PowerSource.desktop.rawValue, Int32(PMK_POWER_SOURCE_DESKTOP.rawValue))
    }

    func testLivePowerSourceReturnsDefinedABIValue() {
        XCTAssertTrue([
            Int32(PMK_POWER_SOURCE_UNKNOWN.rawValue),
            Int32(PMK_POWER_SOURCE_AC.rawValue),
            Int32(PMK_POWER_SOURCE_BATTERY.rawValue),
            Int32(PMK_POWER_SOURCE_DESKTOP.rawValue),
        ].contains(pmkPowerSource()))
    }
}
