import Foundation
import Metal
import CPMK

private func context(_ p: UnsafeMutableRawPointer) -> Context { Unmanaged<Context>.fromOpaque(p).takeUnretainedValue() }
private func job(_ p: UnsafeMutableRawPointer) -> Job { Unmanaged<Job>.fromOpaque(p).takeUnretainedValue() }

@_cdecl("pmk_init")
public func pmkInit(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>?, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    initContext(out, errorBuffer, capacity, requireAdmission: true)
}

@_cdecl("pmk_init_diagnostic")
public func pmkInitDiagnostic(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>?, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    initContext(out, errorBuffer, capacity, requireAdmission: false)
}

private func initContext(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>?, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ capacity: UInt64,
                         requireAdmission: Bool) -> Int32 {
    guard let out else { return Int32(PMK_INVALID) }; out.pointee = nil
    do {
        let c = try Context(requireAdmission: requireAdmission); out.pointee = Unmanaged.passRetained(c).toOpaque()
        putString("", errorBuffer, capacity); return 0
    } catch let resourceError as PMKResourceError {
        putString("DO NOT MINE: \(resourceError)", errorBuffer, capacity); return Int32(PMK_RESOURCE)
    } catch let probeError as PMKProbeError {
        putString("DO NOT MINE: \(probeError)", errorBuffer, capacity)
        if case .resource = probeError {
            return Int32(PMK_RESOURCE)
        }
        return Int32(PMK_PROBE_FAILED)
    } catch let initError {
        putString("DO NOT MINE: \(initError)", errorBuffer, capacity); return Int32(PMK_PROBE_FAILED)
    }
}
@_cdecl("pmk_probe")
public func pmkProbe(_ handle: UnsafeMutableRawPointer?, _ key: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    guard let handle else { return Int32(PMK_INVALID) }
    let c = context(handle); c.lock.lock(); defer { c.lock.unlock() }
    // Success is cached only in this initialized context. New processes always probe;
    // no mutable on-disk cache can authorize mining without a known-answer check.
    guard c.healthy else { return Int32(PMK_PROBE_FAILED) }
    putString(c.cacheKey, key, capacity); return 0
}
@_cdecl("pmk_probe_refresh")
public func pmkProbeRefresh(_ handle: UnsafeMutableRawPointer?, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    guard let handle else { return Int32(PMK_INVALID) }
    let c = context(handle); c.lock.lock(); defer { c.lock.unlock() }
    do {
        try c.refreshProbe()
        putString("", errorBuffer, capacity)
        return 0
    } catch {
        putString("DO NOT MINE: \(error)", errorBuffer, capacity)
        if c.jobs == 0 {
            c.healthy = false
            return Int32(PMK_PROBE_FAILED)
        }
        return Int32(PMK_BUSY)
    }
}
@_cdecl("pmk_destroy")
public func pmkDestroy(_ handle: UnsafeMutableRawPointer?) {
    if let handle { Unmanaged<Context>.fromOpaque(handle).release() }
}
@_cdecl("pmk_buffer_alloc")
public func pmkBufferAlloc(_ handle: UnsafeMutableRawPointer?, _ bytes: UInt64, _ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>?) -> Int32 {
    guard let handle, let out else { return Int32(PMK_INVALID) }; out.pointee = nil
    let c = context(handle); c.lock.lock(); defer { c.lock.unlock() }
    c.recordDiagnostic("")
    guard bytes > 0, bytes < UInt64(maxBufferBytes), bytes <= UInt64(c.budget - c.usedBytes) else {
        c.recordDiagnostic("buffer allocation rejected: bytes=\(bytes), used=\(c.usedBytes), budget=\(c.budget)")
        return Int32(PMK_RESOURCE)
    }
    guard let b = c.buffer(Int(bytes)) else {
        c.recordDiagnostic("Metal buffer allocation failed: bytes=\(bytes)")
        return Int32(PMK_RESOURCE)
    }
    c.usedBytes += b.length
    c.buffers[UInt(bitPattern: b.contents())] = b; out.pointee = b.contents(); return 0
}
@_cdecl("pmk_buffer_release")
public func pmkBufferRelease(_ handle: UnsafeMutableRawPointer?, _ pointer: UnsafeMutableRawPointer?) -> Int32 {
    guard let handle, let pointer else { return Int32(PMK_INVALID) }
    let c = context(handle); c.lock.lock(); defer { c.lock.unlock() }
    // A registered input cannot be freed while any job may retain its pointer for proving.
    guard c.jobs == 0 else { return Int32(PMK_BUSY) }
    guard let b = c.buffers.removeValue(forKey: UInt(bitPattern: pointer)) else { return Int32(PMK_INVALID) }
    c.usedBytes -= b.length; return 0
}

@_cdecl("pmk_run_job")
public func pmkRunJob(_ handle: UnsafeMutableRawPointer?, _ descriptor: UnsafePointer<pmk_job_desc>?,
                      _ callback: (@convention(c) (UnsafeMutableRawPointer?, UnsafeMutableRawPointer?) -> Void)?,
                      _ user: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>?) -> Int32 {
    runJob(handle, descriptor, callback, user, out, diagnostic: false)
}

@_cdecl("pmk_run_job_na")
public func pmkRunJobNA(_ handle: UnsafeMutableRawPointer?, _ descriptor: UnsafePointer<pmk_job_desc>?,
                        _ callback: (@convention(c) (UnsafeMutableRawPointer?, UnsafeMutableRawPointer?) -> Void)?,
                        _ user: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>?) -> Int32 {
    guard let handle else { return Int32(PMK_INVALID) }
    guard context(handle).kernel == .na else { return Int32(PMK_INVALID) }
    return runJob(handle, descriptor, callback, user, out, diagnostic: false)
}

@_cdecl("pmk_run_job_diagnostic")
public func pmkRunJobDiagnostic(_ handle: UnsafeMutableRawPointer?, _ descriptor: UnsafePointer<pmk_job_desc>?,
                                _ callback: (@convention(c) (UnsafeMutableRawPointer?, UnsafeMutableRawPointer?) -> Void)?,
                                _ user: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>?) -> Int32 {
    runJob(handle, descriptor, callback, user, out, diagnostic: true)
}

@_cdecl("pmk_v3_kernel_metadata")
public func pmkV3KernelMetadata(_ handle: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    guard let handle else { return Int32(PMK_INVALID) }
    let c = context(handle)
    c.lock.lock()
    let json = """
    {"kernel":"\(c.kernel.rawValue)","function":"\(c.kernel.functionName)","pattern_id":\(c.kernel.patternId),"device_class":"\(c.deviceClass)","device_name":\(jsonString(c.device.name)),"cache_key":\(jsonString(c.cacheKey)),"os_build":\(jsonString(c.osBuild)),"tile_m":\(c.kernel.tileM),"tile_n":\(c.kernel.tileN)}
    """
    c.lock.unlock()
    putString(json, out, capacity)
    return 0
}

@_cdecl("pmk_kernel_metadata")
public func pmkKernelMetadata(_ handle: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    pmkV3KernelMetadata(handle, out, capacity)
}

private func runJob(_ handle: UnsafeMutableRawPointer?, _ descriptor: UnsafePointer<pmk_job_desc>?,
                    _ callback: (@convention(c) (UnsafeMutableRawPointer?, UnsafeMutableRawPointer?) -> Void)?,
                    _ user: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>?,
                    diagnostic: Bool) -> Int32 {
    guard let handle, let descriptor, let out else { return Int32(PMK_INVALID) }; out.pointee = nil
    let c = context(handle)
    let d = descriptor.pointee
    let maxK: UInt32 = diagnostic ? 65_536 : 8_192
    guard d.abi_version == 1, d.cert_version == 3, d.rank == 128,
           d.m >= 64, d.n >= 64, d.m < (1 << 24), d.n < (1 << 24),
           d.m % UInt32(c.kernel.tileM) == 0, d.n % UInt32(c.kernel.tileN) == 0,
           d.k >= 2048, d.k <= maxK, d.k % 128 == 0,
           d.block_capacity >= 4, d.share_capacity >= 64, let ap = d.a, let bp = d.bt else { return Int32(PMK_INVALID) }
    let na = UInt64(d.m) * UInt64(d.k), nb = UInt64(d.n) * UInt64(d.k)
    let tiles = UInt64(d.m) * UInt64(d.n) / c.kernel.tileElements
    let slotBytes = (UInt64(d.block_capacity) + UInt64(d.share_capacity)) * 104 + UInt64(guardWords * 8)
    guard na < UInt64(maxBufferBytes), nb < UInt64(maxBufferBytes), d.a_bytes >= na, d.bt_bytes >= nb,
           tiles <= UInt64(UInt32.max), UInt64(d.block_capacity) * 104 + UInt64(guardWords * 4) < UInt64(maxBufferBytes),
           UInt64(d.share_capacity) * 104 + UInt64(guardWords * 4) < UInt64(maxBufferBytes) else { return Int32(PMK_RESOURCE) }
    // Include worst-case CPU overflow recovery (nested Swift rows + final flat slots),
    // both noised operands, K1 intermediates and slot guards before allocating anything.
    let needsRecovery = UInt64(d.block_capacity) < tiles || UInt64(d.share_capacity) < tiles
    let recoveryTiles = needsRecovery ? min(tiles, UInt64(overflowRecoveryTileLimit)) : 0
    let recovery = needsRecovery ? recoveryTiles * 768 + na + nb + UInt64(d.m + d.n) * 128 + UInt64(d.k) * 32 : 0
    let reservation = na + nb + UInt64(d.m + d.n) * 128 + UInt64(d.k) * 16 + slotBytes + recovery + 24
    let profile = JobProfile(enabled: false, splitCommandBuffers: false)
    c.lock.lock()
    c.recordDiagnostic("")
    let reserveStart = monotonicNow()
    do {
        try c.refreshProbeIfIdleDue()
    } catch {
        c.healthy = false
        c.lock.unlock()
        return Int32(PMK_PROBE_FAILED)
    }
    guard c.healthy else { c.lock.unlock(); return Int32(PMK_PROBE_FAILED) }
    guard c.jobs < 3 else { c.lock.unlock(); return Int32(PMK_BUSY) }
    guard reservation <= UInt64(c.budget - c.usedBytes),
          let rawA = c.buffers[UInt(bitPattern: ap)], let rawBt = c.buffers[UInt(bitPattern: bp)],
          d.a_bytes <= UInt64(rawA.length), d.bt_bytes <= UInt64(rawBt.length) else {
        c.recordDiagnostic("job resource reservation rejected: reservation=\(reservation), used=\(c.usedBytes), budget=\(c.budget)")
        c.lock.unlock(); return Int32(PMK_RESOURCE)
    }
    c.usedBytes += Int(reservation); c.jobs += 1
    profile.enabled = c.profileEnabled
    profile.splitCommandBuffers = c.profileSplitCommandBuffers
    profile.cpuReserveMs = milliseconds(reserveStart, monotonicNow())
    c.lock.unlock()
    // Caller grants immutable access until release. Validate before GPU arithmetic.
    let validateStart = monotonicNow()
    guard validSignalBytes(ap, Int(na)), validSignalBytes(bp, Int(nb)) else {
        c.lock.lock()
        c.usedBytes -= Int(reservation); c.jobs -= 1
        c.recordDiagnostic("job input signal bytes outside [-64,64]")
        c.lock.unlock()
        return Int32(PMK_INVALID)
    }
    profile.cpuValidateMs = milliseconds(validateStart, monotonicNow())
    return autoreleasepool { () -> Int32 in
        do {
            guard let j = Job(context: c, desc: d, rawA: rawA, rawBt: rawBt, reservation: Int(reservation), profile: profile) else { throw PMKError("job allocation failed") }
            let encodeStart = monotonicNow()
            var completionBuffer: MTLCommandBuffer
            if profile.splitCommandBuffers {
                try j.prepareNoise()
                try j.encodeNoiseStages { name in
                    guard let cb = c.queue.makeCommandBuffer() else { throw PMKError("no command buffer for \(name)") }
                    cb.label = name
                    profile.stages.append(StageTiming(name: name, commandBuffer: cb))
                    return cb
                }
                guard let k3cb = c.queue.makeCommandBuffer() else { throw PMKError("no command buffer for k3") }
                k3cb.label = "k3"
                try j.encodeK3(k3cb)
                profile.stages.append(StageTiming(name: "k3", commandBuffer: k3cb))
                completionBuffer = k3cb
            } else {
                guard let cb = c.queue.makeCommandBuffer() else { throw PMKError("no command buffer for k1_k2_k3") }
                try j.encodeNoise(cb)
                try j.encodeK3(cb)
                profile.stages.append(StageTiming(name: "k1_k2_k3", commandBuffer: cb))
                completionBuffer = cb
            }
            profile.cpuEncodeMs = milliseconds(encodeStart, monotonicNow())
            // Arm only after every failable encode step, immediately before the
            // retained handle becomes externally visible and completion is installed.
            j.callbackBarrier.arm()
            let opaque = Unmanaged.passRetained(j).toOpaque(); out.pointee = opaque
            completionBuffer.addCompletedHandler { completed in
                // CPU recovery must not block Metal's completion delivery thread.
                j.context.completionQueue.async {
                    autoreleasepool {
                        j.complete(completed)
                        j.callbackBarrier.beginCallback()
                        callback?(opaque, user)
                        j.callbackBarrier.endCallback()
                    }
                }
            }
            let submitStart = monotonicNow()
            let submittedStages = profile.stages
            for stage in submittedStages.dropLast() {
                stage.commandBuffer.commit()
            }
            profile.submittedAtNs = monotonicNow()
            profile.cpuSubmitMs = milliseconds(submitStart, profile.submittedAtNs)
            submittedStages.last?.commandBuffer.commit()
            return 0
        } catch {
            c.lock.lock()
            c.usedBytes -= Int(reservation); c.jobs -= 1
            c.recordDiagnostic("job encode failed: \(error)")
            c.lock.unlock()
            return Int32(PMK_RESOURCE)
        }
    }
}
@_cdecl("pmk_poll")
public func pmkPoll(_ handle: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<pmk_result>?) -> Int32 {
    guard let handle, let out else { return Int32(PMK_INVALID) }
    let j = job(handle); j.lock.lock(); defer { j.lock.unlock() }
    guard j.done else { return Int32(PMK_PENDING) }
    out.pointee = j.result; return j.result.status
}
@_cdecl("pmk_context_error")
public func pmkContextError(_ handle: UnsafeMutableRawPointer?, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    guard let handle else { return Int32(PMK_INVALID) }
    let c = context(handle); c.lock.lock(); defer { c.lock.unlock() }
    putString(c.lastDiagnosticByThread[currentThreadID()] ?? c.lastDiagnostic, errorBuffer, capacity)
    return 0
}
@_cdecl("pmk_job_error")
public func pmkJobError(_ handle: UnsafeMutableRawPointer?, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    guard let handle else { return Int32(PMK_INVALID) }
    let j = job(handle); j.lock.lock(); defer { j.lock.unlock() }
    putString(j.diagnostic, errorBuffer, capacity)
    return 0
}
@_cdecl("pmk_job_wait_callback")
public func pmkJobWaitCallback(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard let handle else { return Int32(PMK_INVALID) }
    return job(handle).callbackBarrier.wait() ? 0 : Int32(PMK_BUSY)
}
@_cdecl("pmk_job_release")
public func pmkJobRelease(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard let handle else { return Int32(PMK_INVALID) }
    let j = job(handle); j.lock.lock()
    guard j.done else { j.lock.unlock(); return Int32(PMK_BUSY) }
    j.lock.unlock()
    j.context.lock.lock(); j.context.usedBytes -= j.reservation; j.context.jobs -= 1; j.context.lock.unlock()
    Unmanaged<Job>.fromOpaque(handle).release(); return 0
}
