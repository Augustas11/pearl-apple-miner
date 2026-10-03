import Foundation
import IOKit.ps

private let defaultActivityReason = "Pearl Metal mining"

protocol ActivityManaging: AnyObject {
    func beginActivity(options: ProcessInfo.ActivityOptions, reason: String) -> NSObjectProtocol
    func endActivity(_ activity: NSObjectProtocol)
}

extension ProcessInfo: ActivityManaging {}

final class ActivityHandle {
    private let manager: ActivityManaging
    private let activity: NSObjectProtocol

    init(reason: String, manager: ActivityManaging = ProcessInfo.processInfo) {
        self.manager = manager
        activity = manager.beginActivity(
            options: [.userInitiated, .idleSystemSleepDisabled],
            reason: reason
        )
    }

    deinit {
        manager.endActivity(activity)
    }
}

func activityReason(_ reason: UnsafePointer<CChar>?) -> String {
    guard let reason, let value = String(validatingUTF8: reason), !value.isEmpty else {
        return defaultActivityReason
    }
    return value
}

@_cdecl("pmk_activity_begin")
public func pmkActivityBegin(_ reason: UnsafePointer<CChar>?) -> UnsafeMutableRawPointer? {
    Unmanaged.passRetained(ActivityHandle(reason: activityReason(reason))).toOpaque()
}

@_cdecl("pmk_activity_end")
public func pmkActivityEnd(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    Unmanaged<ActivityHandle>.fromOpaque(handle).release()
}

enum PowerSource: Int32 {
    case unknown = 0
    case ac = 1
    case battery = 2
    case desktop = 3
}

func classifyPowerSource(providingType: String?, hasInternalBattery: Bool) -> PowerSource {
    switch providingType {
    case kIOPMBatteryPowerKey:
        return .battery
    case kIOPMACPowerKey, kIOPMUPSPowerKey:
        return hasInternalBattery ? .ac : .desktop
    default:
        return .unknown
    }
}

func currentPowerSource() -> PowerSource {
    guard let snapshot = IOPSCopyPowerSourcesInfo()?.takeRetainedValue() else {
        return .unknown
    }
    let providingType = IOPSGetProvidingPowerSourceType(snapshot)?.takeUnretainedValue() as String?
    guard let sources = IOPSCopyPowerSourcesList(snapshot)?.takeRetainedValue() as? [Any] else {
        return .unknown
    }
    let hasInternalBattery = sources.contains { source in
        guard let description = IOPSGetPowerSourceDescription(snapshot, source as CFTypeRef)?.takeUnretainedValue()
                as? [String: Any] else {
            return false
        }
        return description[kIOPSTypeKey] as? String == kIOPSInternalBatteryType
    }
    return classifyPowerSource(providingType: providingType, hasInternalBattery: hasInternalBattery)
}

@_cdecl("pmk_power_source")
public func pmkPowerSource() -> Int32 {
    currentPowerSource().rawValue
}
