// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal
import CryptoKit
import CPMK

struct PMKError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}
let maxBufferBytes = 2_000_000_000
let slotWords = 26
let guardWords = 8 * slotWords
let canary: UInt32 = 0xa5a5a5a5
let overflowRecoveryTileLimit: UInt64 = 8_192
let defaultProbeRefreshHours: TimeInterval = 24
let profileEnvironmentKey = "PMK_PROFILE_JSON"
let legacyProfileEnvironmentKey = "PMK_PROFILE"
let splitProfileEnvironmentKey = "PMK_PROFILE_SPLIT_CB"

func words<T>(_ value: T) -> [UInt32] {
    withUnsafeBytes(of: value) { Array($0.bindMemory(to: UInt32.self)) }
}
func putString(_ s: String, _ dst: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) {
    guard let dst, capacity > 0, capacity <= UInt64(Int.max) else { return }
    let bytes = Array(s.utf8.prefix(Int(capacity) - 1))
    for (i, b) in bytes.enumerated() { dst[i] = CChar(bitPattern: b) }
    dst[bytes.count] = 0
}
func diagnosticString(_ value: String, limit: Int = 512) -> String {
    let redacted = value
        .replacingOccurrences(of: #"(?i)(password|passwd|rpc_password|token|secret|key)=\S+"#,
                              with: "$1=<redacted>",
                              options: .regularExpression)
    let printable = redacted.unicodeScalars.map { scalar -> Character in
        if CharacterSet.controlCharacters.contains(scalar) && scalar != "\n" && scalar != "\t" {
            return " "
        }
        return Character(scalar)
    }
    return String(String(printable).prefix(limit))
}
func currentThreadID() -> UInt64 {
    var id: UInt64 = 0
    pthread_threadid_np(nil, &id)
    return id
}

/// Tracks the complete return of the foreign completion callback. Keeping this
/// separate from job release preserves callback-owned release while giving an
/// external release owner an explicit lifetime barrier.
final class CallbackBarrier {
    private let group = DispatchGroup()
    private let lock = NSLock()
    private var callbackThreadID: UInt64?
    private var armed = false

    func arm() {
        lock.lock()
        precondition(!armed, "callback barrier armed twice")
        armed = true
        group.enter()
        lock.unlock()
    }

    func beginCallback() {
        lock.lock()
        precondition(armed, "unarmed callback barrier")
        callbackThreadID = currentThreadID()
        lock.unlock()
    }

    func endCallback() {
        lock.lock()
        callbackThreadID = nil
        lock.unlock()
        group.leave()
    }

    func isCallbackThread() -> Bool {
        lock.lock()
        let value = callbackThreadID == currentThreadID()
        lock.unlock()
        return value
    }

    /// Returns false instead of deadlocking when called by the callback itself.
    func wait() -> Bool {
        guard !isCallbackThread() else { return false }
        group.wait()
        return true
    }
}
func syncComplete(_ cb: MTLCommandBuffer) throws {
    let sem = DispatchSemaphore(value: 0)
    cb.addCompletedHandler { _ in sem.signal() }
    cb.commit(); sem.wait()
    guard cb.status == .completed else { throw PMKError("GPU: \(String(describing: cb.error))") }
}

func currentOSBuild() throws -> String {
    var osSize = 0
    guard sysctlbyname("kern.osversion", nil, &osSize, nil, 0) == 0 else { throw PMKError("OS build unavailable") }
    var os = [CChar](repeating: 0, count: osSize)
    guard sysctlbyname("kern.osversion", &os, &osSize, nil, 0) == 0 else { throw PMKError("OS build unavailable") }
    return String(cString: os)
}

func probeRefreshIntervalSeconds(_ env: [String: String] = ProcessInfo.processInfo.environment) -> TimeInterval {
    guard let raw = env["PMK_PROBE_REFRESH_HOURS"], let hours = Double(raw), hours >= 0 else {
        return defaultProbeRefreshHours * 3600
    }
    return hours * 3600
}

func shouldRefreshProbe(last: TimeInterval, now: TimeInterval, interval: TimeInterval) -> Bool {
    now >= last && now - last >= interval
}

func metalDeviceClass(_ device: MTLDevice) -> String {
    if device.supportsFamily(.apple10) { return "Apple10" }
    if device.supportsFamily(.apple9) { return "Apple9" }
    if device.supportsFamily(.apple8) { return "Apple8" }
    if device.supportsFamily(.apple7) { return "Apple7" }
    return "unsupported"
}

func requiresG3Admission(deviceClass: String) -> Bool {
    ["Apple7", "Apple8", "Apple9"].contains(deviceClass)
}

func canRecoverOverflow(tileCount: UInt64, overflow: UInt32) -> Bool {
    overflow == 0 || tileCount <= overflowRecoveryTileLimit
}

func envFlag(_ name: String, _ env: [String: String] = ProcessInfo.processInfo.environment) -> Bool {
    guard let raw = env[name]?.lowercased() else { return false }
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}

func monotonicNow() -> UInt64 {
    DispatchTime.now().uptimeNanoseconds
}

func milliseconds(_ start: UInt64, _ end: UInt64) -> Double {
    Double(end - start) / 1_000_000.0
}

func validSignalBytes(_ pointer: UnsafePointer<Int8>, _ count: Int) -> Bool {
    pmk_valid_signal_bytes(pointer, UInt64(count)) != 0
}

struct StageTiming {
    let name: String
    let commandBuffer: MTLCommandBuffer
}

final class JobProfile {
    var enabled: Bool
    var splitCommandBuffers: Bool
    var submittedAtNs: UInt64 = 0
    var cpuValidateMs: Double = 0
    var cpuReserveMs: Double = 0
    var cpuAllocMs: Double = 0
    var cpuEncodeMs: Double = 0
    var cpuSubmitMs: Double = 0
    var cpuCompleteMs: Double = 0
    var stages: [StageTiming] = []

    init(enabled: Bool, splitCommandBuffers: Bool) {
        self.enabled = enabled
        self.splitCommandBuffers = splitCommandBuffers
    }
}

func jsonString(_ value: String) -> String {
    String(data: try! JSONEncoder().encode(value), encoding: .utf8)!
}

func requireG3Admission(deviceName: String, deviceClass: String, cacheKey: String,
                        osBuild: String, now: Date,
                        path: String? = ProcessInfo.processInfo.environment["PMK_G3_ADMISSION_FILE"]) throws {
    guard requiresG3Admission(deviceClass: deviceClass) else { return }
    guard let path, !path.isEmpty else {
        throw PMKError("DO NOT MINE: missing G3 admission file for \(deviceName) \(deviceClass)")
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let devices = root["devices"] as? [[String: Any]] else {
        throw PMKError("DO NOT MINE: malformed G3 admission file")
    }
    guard let record = devices.first(where: { ($0["gpu_name"] as? String) == deviceName }) else {
        throw PMKError("DO NOT MINE: no G3 pass recorded for \(deviceName)")
    }
    let lastProbe = (record["last_probe_unix"] as? NSNumber)?.doubleValue ?? 0
    let nowUnix = now.timeIntervalSince1970
    guard (record["g3_passed"] as? Bool) == true,
          (record["device_class"] as? String) == deviceClass,
          (record["cache_key"] as? String) == cacheKey,
          (record["os_build"] as? String) == osBuild,
          lastProbe.isFinite, lastProbe > 0,
          nowUnix.isFinite, lastProbe <= nowUnix else {
        throw PMKError("DO NOT MINE: stale or mismatched G3 admission for \(deviceName)")
    }
    if let validity = record["valid_hours"] {
        guard let validHours = validity as? NSNumber,
              validHours.doubleValue.isFinite,
              validHours.doubleValue > 0,
              nowUnix - lastProbe <= validHours.doubleValue * 3600 else {
            throw PMKError("DO NOT MINE: stale or mismatched G3 admission for \(deviceName)")
        }
    }
}

final class Context {
    let device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary
    let k3: MTLComputePipelineState
    let cacheKey: String
    let deviceClass: String
    let osBuild: String
    let requireAdmission: Bool
    let probeRefreshInterval: TimeInterval
    let profileEnabled: Bool
    let profileSplitCommandBuffers: Bool
    let completionQueue = DispatchQueue(label: "pmk.completion", qos: .utility, attributes: .concurrent)
    let lock = NSLock()
    let pipelineLock = NSLock()
    var buffers: [UInt: MTLBuffer] = [:]
    var usedBytes = 0, jobs = 0
    var healthy = true
    var lastProbeUptime: TimeInterval
    var lastDiagnostic = ""
    var lastDiagnosticByThread: [UInt64: String] = [:]
    let budget: Int
    var noisePipelines: [String: MTLComputePipelineState] = [:]

    init(requireAdmission: Bool = true) throws {
        guard let dev = MTLCreateSystemDefaultDevice(), dev.hasUnifiedMemory,
              dev.supportsFamily(.apple7), let q = dev.makeCommandQueue() else {
            throw PMKError("DO NOT MINE: Apple7+ unified-memory Metal device required")
        }
        self.requireAdmission = requireAdmission
        profileEnabled = envFlag(profileEnvironmentKey) || envFlag(legacyProfileEnvironmentKey)
        profileSplitCommandBuffers = profileEnabled && envFlag(splitProfileEnvironmentKey)
        deviceClass = metalDeviceClass(dev)
        device = dev; queue = q
        budget = Int(ProcessInfo.processInfo.physicalMemory / 4)
        let sourceURL = try pmkResourceURL("metal")
        let source = try String(contentsOf: sourceURL.appendingPathComponent("k3sg.metal"), encoding: .utf8)
            + "\n" + String(contentsOf: sourceURL.appendingPathComponent("noise.metal"), encoding: .utf8)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.fastMathEnabled = false
        options.preprocessorMacros = [
            "VARIANT": NSNumber(value: 3),
            "BM": NSNumber(value: 64),
            "BN": NSNumber(value: 64),
            "BK": NSNumber(value: 16),
            "WM": NSNumber(value: 2),
            "WN": NSNumber(value: 2),
            "PF": NSNumber(value: 2),
        ]
        let settings = "MSL3.1;fastMath=false;VARIANT=3;BM=64;BN=64;BK=16;WM=2;WN=2;PF=2;probe-v1"
        osBuild = try currentOSBuild()
        probeRefreshInterval = probeRefreshIntervalSeconds()
        lastProbeUptime = ProcessInfo.processInfo.systemUptime
        cacheKey = SHA256.hash(data: Data((source + settings + dev.name + osBuild).utf8)).map { String(format: "%02x", $0) }.joined()
        library = try dev.makeLibrary(source: source, options: options)
        guard let function = library.makeFunction(name: "k3sg") else { throw PMKError("Missing K3") }
        k3 = try dev.makeComputePipelineState(function: function)
        guard k3.threadExecutionWidth == 32, k3.maxTotalThreadsPerThreadgroup >= 128 else { throw PMKError("DO NOT MINE: unexpected execution width") }
        _ = try runProbe(device: dev, queue: q, library: library, pipeline: k3)
        if requireAdmission {
            try requireG3Admission(deviceName: dev.name, deviceClass: deviceClass,
                                   cacheKey: cacheKey, osBuild: osBuild,
                                   now: Date())
        }
    }

    func refreshProbe() throws {
        guard jobs == 0 else { throw PMKError("probe refresh busy") }
        guard try currentOSBuild() == osBuild else { throw PMKError("OS build changed since G3 admission") }
        _ = try runProbe(device: device, queue: queue, library: library, pipeline: k3)
        if requireAdmission {
            try requireG3Admission(deviceName: device.name, deviceClass: deviceClass,
                                   cacheKey: cacheKey, osBuild: osBuild,
                                   now: Date())
        }
        lastProbeUptime = ProcessInfo.processInfo.systemUptime
        healthy = true
    }

    func refreshProbeIfIdleDue(nowUptime: TimeInterval = ProcessInfo.processInfo.systemUptime) throws {
        guard jobs == 0 else { return }
        guard try currentOSBuild() == osBuild else { throw PMKError("OS build changed since G3 admission") }
        if shouldRefreshProbe(last: lastProbeUptime, now: nowUptime, interval: probeRefreshInterval) {
            try refreshProbe()
        }
    }

    func buffer(_ length: Int) -> MTLBuffer? {
        guard length > 0, length < maxBufferBytes, length <= device.maxBufferLength else { return nil }
        return device.makeBuffer(length: length, options: .storageModeShared)
    }
    func recordDiagnostic(_ value: String) {
        let diagnostic = diagnosticString(value)
        lastDiagnostic = diagnostic
        lastDiagnosticByThread[currentThreadID()] = diagnostic
    }
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        pipelineLock.lock()
        defer { pipelineLock.unlock() }
        if let p = noisePipelines[name] { return p }
        guard let f = library.makeFunction(name: name) else { throw PMKError("Missing \(name)") }
        let p = try device.makeComputePipelineState(function: f)
        noisePipelines[name] = p
        return p
    }
}

final class Job {
    let context: Context, desc: pmk_job_desc
    let rawA: MTLBuffer, rawBt: MTLBuffer, a: MTLBuffer, b: MTLBuffer
    let ctr: MTLBuffer, blocks: MTLBuffer, shares: MTLBuffer, sink: MTLBuffer
    var scratch: [MTLBuffer] = []
    let reservation: Int
    let lock = NSLock()
    var result = pmk_result()
    var diagnostic = ""
    var done = false
    var recoveredBlocks: UnsafeMutablePointer<pmk_slot>?
    var recoveredShares: UnsafeMutablePointer<pmk_slot>?
    let profile: JobProfile
    let callbackBarrier = CallbackBarrier()

    init?(context c: Context, desc d: pmk_job_desc, rawA: MTLBuffer, rawBt: MTLBuffer, reservation: Int, profile p: JobProfile? = nil) {
        context = c; desc = d; self.rawA = rawA; self.rawBt = rawBt; self.reservation = reservation
        profile = p ?? JobProfile(enabled: false, splitCommandBuffers: false)
        let lengths = [Int(d.m) * Int(d.k), Int(d.n) * Int(d.k), 8,
                       (Int(d.block_capacity) * slotWords + guardWords) * 4,
                       (Int(d.share_capacity) * slotWords + guardWords) * 4, 16]
        let allocStart = monotonicNow()
        var allocations: [MTLBuffer] = []
        for length in lengths {
            guard let buffer = c.buffer(length) else { return nil }
            allocations.append(buffer)
        }
        a = allocations[0]; b = allocations[1]; ctr = allocations[2]
        blocks = allocations[3]; shares = allocations[4]; sink = allocations[5]
        memset(ctr.contents(), 0, 8)
        for (buf, capacity) in [(blocks, d.block_capacity), (shares, d.share_capacity)] {
            let guardStart = Int(capacity) * slotWords
            buf.contents().bindMemory(to: UInt32.self, capacity: buf.length / 4)
                .advanced(by: guardStart)
                .initialize(repeating: canary, count: guardWords)
        }
        profile.cpuAllocMs = milliseconds(allocStart, monotonicNow())
        result.abi_version = 1; result.job_id = d.job_id; result.status = Int32(PMK_PENDING)
    }
    deinit {
        recoveredBlocks?.deallocate(); recoveredShares?.deallocate()
    }
    func encodeK3(_ cb: MTLCommandBuffer) throws {
        guard let e = cb.makeComputeCommandEncoder() else { throw PMKError("No compute encoder") }
        var p: [UInt32] = [desc.m, desc.n, desc.k, desc.block_capacity, desc.share_capacity, 0, 0, 0]
        p += words(desc.a_seed); p += words(desc.block_bound); p += words(desc.share_bound)
        e.setComputePipelineState(context.k3)
        e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1)
        e.setBytes(&p, length: 128, index: 2)
        for (i, buf) in [ctr, blocks, shares, sink].enumerated() { e.setBuffer(buf, offset: 0, index: i + 3) }
        e.dispatchThreadgroups(MTLSize(width: Int(desc.n) / 64, height: Int(desc.m) / 64, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        e.endEncoding()
    }
    func complete(_ cb: MTLCommandBuffer) {
        let completeStart = monotonicNow()
        var result = self.result
        if let first = profile.stages.first?.commandBuffer, let last = profile.stages.last?.commandBuffer {
            result.gpu_start_time = first.gpuStartTime; result.gpu_end_time = last.gpuEndTime
        } else {
            result.gpu_start_time = cb.gpuStartTime; result.gpu_end_time = cb.gpuEndTime
        }
        if profile.stages.isEmpty {
            result.status = cb.status == .completed ? 0 : Int32(PMK_GPU_FAILED)
        } else {
            result.status = profile.stages.allSatisfy { $0.commandBuffer.status == .completed } ? 0 : Int32(PMK_GPU_FAILED)
        }
        if result.status != 0 {
            let failed = profile.stages.first { $0.commandBuffer.status != .completed }?.commandBuffer ?? cb
            let status = failed.status.rawValue
            let error = failed.error.map { String(describing: $0) } ?? "no Metal error"
            diagnostic = diagnosticString("command buffer status=\(status): \(error)")
        }
        if result.status == 0 {
            let counts = ctr.contents().bindMemory(to: UInt32.self, capacity: 2)
            result.block_count = counts[0]; result.share_count = counts[1]
            let tiles = UInt64(desc.m) * UInt64(desc.n) / 32
            for (buf, capacity) in [(blocks, desc.block_capacity), (shares, desc.share_capacity)] {
                let ptr = buf.contents().bindMemory(to: UInt32.self, capacity: buf.length / 4)
                if !(0..<guardWords).allSatisfy({ ptr[Int(capacity) * slotWords + $0] == canary }) {
                    result.status = Int32(PMK_GPU_FAILED)
                    diagnostic = diagnosticString("result guard canary mismatch")
                }
            }
            if UInt64(counts[0]) > tiles || UInt64(counts[1]) > tiles {
                result.status = Int32(PMK_GPU_FAILED)
                diagnostic = diagnosticString("result counter exceeds tile count")
            }
            if result.status == 0 {
                result.overflow = (counts[0] > desc.block_capacity ? 1 : 0) | (counts[1] > desc.share_capacity ? 2 : 0)
                result.block_stored = min(counts[0], desc.block_capacity)
                result.share_stored = min(counts[1], desc.share_capacity)
                result.blocks = UnsafePointer(blocks.contents().assumingMemoryBound(to: pmk_slot.self))
                result.shares = UnsafePointer(shares.contents().assumingMemoryBound(to: pmk_slot.self))
                if result.overflow != 0 {
                    guard canRecoverOverflow(tileCount: tiles, overflow: result.overflow) else {
                        result.status = Int32(PMK_GPU_FAILED)
                        diagnostic = diagnosticString("overflow recovery limit exceeded")
                        emitProfile(result: result)
                        profile.stages.removeAll(keepingCapacity: false)
                        lock.lock(); self.result = result; done = true; lock.unlock()
                        context.lock.lock(); context.healthy = false; context.lock.unlock()
                        return
                    }
                    let cpu = CPUOracle.noised(m: Int(desc.m), n: Int(desc.n), k: Int(desc.k),
                        rawA: rawA.contents().assumingMemoryBound(to: Int8.self), rawBt: rawBt.contents().assumingMemoryBound(to: Int8.self),
                        aKey: words(desc.a_seed), bKey: words(desc.b_seed))
                    let noiseMatches = cpu.a.withUnsafeBytes { memcmp($0.baseAddress!, a.contents(), cpu.a.count) == 0 }
                        && cpu.b.withUnsafeBytes { memcmp($0.baseAddress!, b.contents(), cpu.b.count) == 0 }
                    let recovered = cpu.a.withUnsafeBufferPointer { ap in cpu.b.withUnsafeBufferPointer { bp in
                        CPUOracle.tiles(m: Int(desc.m), n: Int(desc.n), k: Int(desc.k),
                            a: ap.baseAddress!, b: bp.baseAddress!, key: words(desc.a_seed),
                            block: words(desc.block_bound), share: words(desc.share_bound))
                    } }
                    if !noiseMatches || recovered.block.count != Int(counts[0]) || recovered.share.count != Int(counts[1]) {
                        result.status = Int32(PMK_GPU_FAILED)
                        diagnostic = diagnosticString("overflow recovery oracle mismatch")
                    } else {
                        func storage(_ rows: [[UInt32]]) -> UnsafeMutablePointer<pmk_slot> {
                            let ptr = UnsafeMutablePointer<pmk_slot>.allocate(capacity: max(1, rows.count))
                            for (i, row) in rows.enumerated() {
                                row.withUnsafeBytes { raw in
                                    _ = memcpy(ptr.advanced(by: i), raw.baseAddress!, 104)
                                }
                            }
                            return ptr
                        }
                        recoveredBlocks = storage(recovered.block); recoveredShares = storage(recovered.share)
                        result.blocks = UnsafePointer(recoveredBlocks); result.shares = UnsafePointer(recoveredShares)
                        result.block_stored = counts[0]; result.share_stored = counts[1]; result.recovered = 1
                    }
                }
            }
        }
        if result.status != 0 { context.lock.lock(); context.healthy = false; context.lock.unlock() }
        profile.cpuCompleteMs = milliseconds(completeStart, monotonicNow())
        emitProfile(result: result)
        profile.stages.removeAll(keepingCapacity: false)
        lock.lock(); self.result = result; done = true; lock.unlock()
    }

    private func emitProfile(result: pmk_result) {
        guard profile.enabled else { return }
        var fields: [String] = []
        func add(_ key: String, _ value: String) { fields.append("\"\(key)\":\(value)") }
        add("event", jsonString("pmk_job_profile"))
        add("job_id", "\(desc.job_id)")
        add("status", "\(result.status)")
        add("m", "\(desc.m)")
        add("n", "\(desc.n)")
        add("k", "\(desc.k)")
        add("split_command_buffers", profile.splitCommandBuffers ? "true" : "false")
        add("cpu_validate_ms", String(format: "%.6f", profile.cpuValidateMs))
        add("cpu_reserve_ms", String(format: "%.6f", profile.cpuReserveMs))
        add("cpu_alloc_ms", String(format: "%.6f", profile.cpuAllocMs))
        add("cpu_encode_ms", String(format: "%.6f", profile.cpuEncodeMs))
        add("cpu_submit_ms", String(format: "%.6f", profile.cpuSubmitMs))
        add("cpu_complete_ms", String(format: "%.6f", profile.cpuCompleteMs))
        if profile.submittedAtNs != 0 {
            add("submit_to_complete_wall_ms", String(format: "%.6f", milliseconds(profile.submittedAtNs, monotonicNow())))
        }
        var stageJSON: [String] = []
        for stage in profile.stages {
            let cb = stage.commandBuffer
            let gpuMs = cb.gpuEndTime >= cb.gpuStartTime ? (cb.gpuEndTime - cb.gpuStartTime) * 1000.0 : 0.0
            stageJSON.append("{\"name\":\(jsonString(stage.name)),\"status\":\(cb.status.rawValue),\"gpu_start\":\(String(format: "%.9f", cb.gpuStartTime)),\"gpu_end\":\(String(format: "%.9f", cb.gpuEndTime)),\"gpu_ms\":\(String(format: "%.6f", gpuMs))}")
        }
        add("gpu_stages", "[\(stageJSON.joined(separator: ","))]")
        FileHandle.standardError.write(("{\(fields.joined(separator: ","))}\n").data(using: .utf8)!)
    }
}
