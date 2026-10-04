// SPDX-License-Identifier: Apache-2.0
// Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.

import Foundation
import Darwin
import Metal
import CryptoKit
import CPMK

private let v4SlotWords = 26
private let v4GuardWords = 8 * v4SlotWords
private let v4Canary: UInt32 = 0xa5a5a5a5
private let v4ExactCellsGate: UInt64 = 100_000_000
private let v4AdmissionValidSeconds: TimeInterval = 7 * 24 * 3600
private let v4VendorPin = "f696760b259500ecb608469ea3953aeabbe78948"

private func v4Context(_ p: UnsafeMutableRawPointer) -> V4Context {
    Unmanaged<V4Context>.fromOpaque(p).takeUnretainedValue()
}

private func v4Job(_ p: UnsafeMutableRawPointer) -> V4Job {
    Unmanaged<V4Job>.fromOpaque(p).takeUnretainedValue()
}

private func words8<T>(_ tuple: T) -> [UInt32] {
    withUnsafeBytes(of: tuple) { Array($0.bindMemory(to: UInt32.self).prefix(8)) }
}
private func words16<T>(_ tuple: T) -> [UInt32] {
    withUnsafeBytes(of: tuple) { Array($0.bindMemory(to: UInt32.self).prefix(16)) }
}


private struct V4StartupProbeTile: Decodable {
    let row: UInt32
    let col: UInt32
    let message: String
    let hash: String
    let kind: String
}

private struct V4StartupProbeManifest: Decodable {
    let schema: String
    let upstream_pin: String
    let m: UInt32
    let n: UInt32
    let k: UInt32
    let rank: UInt32
    let jackpot_key: String
    let block_bound: String
    let share_bound: String
    let blocks_expected: UInt32
    let shares_expected: UInt32
    let tiles: [V4StartupProbeTile]
    let files: [String: String]
}

private func v4HexData(_ hex: String) throws -> Data {
    guard hex.count % 2 == 0 else { throw PMKError("odd-length v4 probe hex") }
    var out = Data(capacity: hex.count / 2)
    var i = hex.startIndex
    while i < hex.endIndex {
        let j = hex.index(i, offsetBy: 2)
        guard let byte = UInt8(hex[i..<j], radix: 16) else { throw PMKError("invalid v4 probe hex") }
        out.append(byte)
        i = j
    }
    return out
}

private func v4Words8FromHex32(_ hex: String) throws -> (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) {
    let bytes = try v4HexData(hex)
    guard bytes.count == 32 else { throw PMKError("v4 probe key/bound is not 32 bytes") }
    let words = bytes.withUnsafeBytes { raw -> [UInt32] in
        let base = raw.bindMemory(to: UInt8.self).baseAddress!
        return (0..<8).map { i in
            UInt32(base[i * 4]) | (UInt32(base[i * 4 + 1]) << 8) | (UInt32(base[i * 4 + 2]) << 16) | (UInt32(base[i * 4 + 3]) << 24)
        }
    }
    return (words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7])
}

private func v4HexFromWords(_ words: [UInt32]) -> String {
    var bytes = [UInt8](); bytes.reserveCapacity(words.count * 4)
    for word in words {
        let le = word.littleEndian
        bytes.append(UInt8(truncatingIfNeeded: le))
        bytes.append(UInt8(truncatingIfNeeded: le >> 8))
        bytes.append(UInt8(truncatingIfNeeded: le >> 16))
        bytes.append(UInt8(truncatingIfNeeded: le >> 24))
    }
    return bytes.map { String(format: "%02x", $0) }.joined()
}

private func v4SHA256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func v4StartupProbeDirectory() throws -> URL {
    try pmkResourceURL("v4_probe")
}

private func v4LoadStartupProbe() throws -> (URL, V4StartupProbeManifest, Data) {
    let dir = try v4StartupProbeDirectory()
    let manifestData = try Data(contentsOf: dir.appendingPathComponent("manifest.json"))
    let manifest = try JSONDecoder().decode(V4StartupProbeManifest.self, from: manifestData)
    guard manifest.schema == "pmk-v4-startup-probe-v1", manifest.upstream_pin == v4VendorPin else {
        throw PMKError("v4 startup probe manifest provenance mismatch")
    }
    for (name, expectedHash) in manifest.files {
        let data = try Data(contentsOf: dir.appendingPathComponent(name))
        guard v4SHA256Hex(data) == expectedHash else {
            throw PMKError("v4 startup probe file hash mismatch: \(name)")
        }
    }
    return (dir, manifest, manifestData)
}

private func v4StartupProbeDigest() throws -> String {
    let (dir, manifest, manifestData) = try v4LoadStartupProbe()
    var material = Data("pmk-v4-startup-probe\n".utf8)
    material.append(manifestData)
    for name in manifest.files.keys.sorted() {
        material.append(Data(name.utf8)); material.append(0)
        material.append(try Data(contentsOf: dir.appendingPathComponent(name)))
    }
    return v4SHA256Hex(material)
}

private func v4LoadedLibraryInfo() throws -> (path: String, sha256: String) {
    guard let libraryURL = pmkLoadedLibraryURL() else {
        throw PMKError("DO NOT MINE: unable to identify loaded native v4 image")
    }
    let path = libraryURL.path
    let data = try Data(contentsOf: libraryURL)
    return (path, v4SHA256Hex(data))
}

private func copyBuffer<T>(_ device: MTLDevice, _ pointer: UnsafePointer<T>?, _ count: Int) -> MTLBuffer? {
    guard count > 0, let pointer else { return nil }
    guard count <= Int.max / MemoryLayout<T>.stride else { return nil }
    let bytes = count * MemoryLayout<T>.stride
    guard bytes <= device.maxBufferLength else { return nil }
    return device.makeBuffer(bytes: pointer, length: bytes, options: .storageModeShared)
}

private func emptyBuffer(_ device: MTLDevice, _ bytes: Int) -> MTLBuffer? {
    guard bytes > 0, bytes < maxBufferBytes, bytes <= device.maxBufferLength else { return nil }
    return device.makeBuffer(length: bytes, options: .storageModeShared)
}

private func unwrap<T>(_ value: T?, _ message: String) throws -> T {
    guard let value else { throw PMKError(message) }
    return value
}

private func productFitsBuffer(_ values: [UInt64], stride: UInt64, device: MTLDevice) -> Bool {
    var total: UInt64 = stride
    for value in values {
        if value == 0 { return false }
        if total > UInt64.max / value { return false }
        total *= value
    }
    return total <= UInt64(Int.max) && total <= UInt64(device.maxBufferLength)
}

private func product(_ values: UInt64...) -> UInt64? {
    var total: UInt64 = 1
    for value in values {
        if total > UInt64.max / value { return nil }
        total *= value
    }
    return total
}

private func addBytes(_ total: inout UInt64, _ value: UInt64?) -> Bool {
    guard let value, total <= UInt64.max - value else { return false }
    total += value
    return true
}

private func v4ResourceBudgetOK(m: UInt32, n: UInt32, k: UInt32, r: UInt32, blockCap: UInt32, shareCap: UInt32, fused: Bool) -> Bool {
    let m = UInt64(m), n = UInt64(n), k = UInt64(k), r = UInt64(r)
    var bytes: UInt64 = 0
    for value in [
        product(m, k), product(n, k),                 // codes
        product(m, k, 4), product(n, k, 4),           // decoded floats
        product(n, k, 4),                             // transpose/scratch
        product(m, n, 4),                             // full C diagnostic/readback
        product(m / 32, n / 32, 8, 4),                // stats
        product(UInt64(blockCap) + UInt64(v4GuardWords / v4SlotWords), UInt64(v4SlotWords), 4),
        product(UInt64(shareCap) + UInt64(v4GuardWords / v4SlotWords), UInt64(v4SlotWords), 4),
    ] {
        if !addBytes(&bytes, value) { return false }
    }
    if fused {
        for value in [
            product(m, k), product(n, k), product(m, r), product(n, r), product(k, r), product(k, r),
            product(m, 4), product(n, 4),
        ] {
            if !addBytes(&bytes, value) { return false }
        }
    }
    let physical = ProcessInfo.processInfo.physicalMemory
    return bytes <= UInt64(maxBufferBytes) && bytes.saturatingMultiplied(by: 3) <= physical / 4
}

private extension UInt64 {
    func saturatingMultiplied(by other: UInt64) -> UInt64 {
        self > UInt64.max / other ? UInt64.max : self * other
    }
}

struct V4Params {
    var words = [UInt32](repeating: 0, count: 32)

    init(m: UInt32, n: UInt32, k: UInt32, r: UInt32, blockCap: UInt32, shareCap: UInt32,
         key: [UInt32], block: [UInt32], share: [UInt32]) {
        words[0] = m; words[1] = n; words[2] = k; words[3] = r
        words[4] = blockCap; words[5] = shareCap
        for i in 0..<8 {
            words[8 + i] = key[i]
            words[16 + i] = block[i]
            words[24 + i] = share[i]
        }
    }
}

final class V4Context {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    let pipelines: [String: MTLComputePipelineState]
    let cacheKey: String
    let osBuild: String
    let deviceClass: String
    let requireAdmission: Bool
    let libraryPath: String
    let librarySHA256: String
    let probeRefreshInterval: TimeInterval
    var lastProbeUptime: TimeInterval
    let completionQueue = DispatchQueue(label: "pmk.v4.completion", qos: .utility, attributes: .concurrent)
    let lock = NSLock()
    var healthy = true
    var jobs = 0
    var lastDiagnostic = ""

    init(requireAdmission: Bool) throws {
        guard let dev = MTLCreateSystemDefaultDevice(), dev.hasUnifiedMemory,
              dev.supportsFamily(.apple7), let q = dev.makeCommandQueue() else {
            throw PMKError("DO NOT MINE: Apple7+ unified-memory Metal device required")
        }
        device = dev
        queue = q
        self.requireAdmission = requireAdmission
        deviceClass = metalDeviceClass(dev)
        osBuild = try currentOSBuild()
        let loadedLibrary = try v4LoadedLibraryInfo()
        libraryPath = loadedLibrary.path
        librarySHA256 = loadedLibrary.sha256
        probeRefreshInterval = probeRefreshIntervalSeconds()
        lastProbeUptime = ProcessInfo.processInfo.systemUptime
        let sourceURL = try pmkResourceURL("metal")
        let source = try String(contentsOf: sourceURL.appendingPathComponent("v4.metal"), encoding: .utf8)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.fastMathEnabled = false
        options.preprocessorMacros = ["V4_EB": 2, "V4_EW": 1].mapValues { NSNumber(value: $0) }
        let settings = "MSL3.1;fastMath=false;V4_EB=2;V4_EW=1;libpmk-v4-kernel-e"
        let probeDigest = try v4StartupProbeDigest()
        cacheKey = SHA256.hash(data: Data((source + settings + dev.name + osBuild + probeDigest + librarySHA256).utf8)).map {
            String(format: "%02x", $0)
        }.joined()
        library = try dev.makeLibrary(source: source, options: options)
        var built: [String: MTLComputePipelineState] = [:]
        for name in ["v4_quantize_operand", "v4_transpose_b", "v4_decode_codes", "v4_tilemax", "v4_kernel_e", "v4_lottery", "v4_fp8_roundtrip"] {
            guard let f = library.makeFunction(name: name) else { throw PMKError("Missing \(name)") }
            built[name] = try dev.makeComputePipelineState(function: f)
        }
        pipelines = built
        if pipelines["v4_kernel_e"]?.threadExecutionWidth != 32 {
            throw PMKError("DO NOT MINE: unexpected v4 execution width")
        }
        try runV4FP8Selftest(device: dev, queue: q, pipeline: try unwrap(pipelines["v4_fp8_roundtrip"], "Missing v4_fp8_roundtrip"))
        try runStartupProbe()
        if requireAdmission {
            try requireV4Admission(deviceName: dev.name, deviceClass: deviceClass, cacheKey: cacheKey,
                                   osBuild: osBuild, librarySHA256: librarySHA256, now: Date())
        }
    }

    private func runStartupProbe() throws {
        let (dir, manifest, _) = try v4LoadStartupProbe()
        guard manifest.m == 64, manifest.n == 64, manifest.k == 4096, manifest.rank == 32,
              manifest.tiles.count == 24,
              manifest.blocks_expected == 8, manifest.shares_expected == 16 else {
            throw PMKError("v4 startup probe manifest shape mismatch")
        }
        func load(_ name: String) throws -> Data { try Data(contentsOf: dir.appendingPathComponent(name)) }
        let aClean = try load("a_clean.bin")
        let bClean = try load("b_clean.bin")
        let aNoiseE = try load("a_noise_e.bin")
        let bNoiseE = try load("b_noise_e.bin")
        let aNoiseF = try load("a_noise_f.bin")
        let bNoiseF = try load("b_noise_f.bin")
        let aAlpha = try load("a_alpha.bf16")
        let bAlpha = try load("b_alpha.bf16")
        let aBeta = try load("a_beta.bf16")
        let bBeta = try load("b_beta.bf16")
        let expectedA = try load("a.bin")
        let expectedB = try load("b.bin")
        let expectedC = try load("c_b200.bin")
        guard aClean.count == Int(manifest.m) * Int(manifest.k),
              bClean.count == Int(manifest.n) * Int(manifest.k),
              aNoiseE.count == Int(manifest.m) * Int(manifest.rank),
              bNoiseE.count == Int(manifest.n) * Int(manifest.rank),
              aNoiseF.count == Int(manifest.k) * Int(manifest.rank),
              bNoiseF.count == Int(manifest.k) * Int(manifest.rank),
              aAlpha.count == Int(manifest.m) * 2, bAlpha.count == Int(manifest.n) * 2,
              aBeta.count == Int(manifest.m) * 2, bBeta.count == Int(manifest.n) * 2,
              expectedA.count == Int(manifest.m) * Int(manifest.k),
              expectedB.count == Int(manifest.n) * Int(manifest.k),
              expectedC.count == Int(manifest.m) * Int(manifest.n) * 4 else {
            throw PMKError("v4 startup probe file size mismatch")
        }
        let ns = [aClean, aNoiseE, aNoiseF, aAlpha, aBeta, bClean, bNoiseE, bNoiseF, bAlpha, bBeta].map { $0 as NSData }
        var desc = pmk_v4_job_desc()
        desc.abi_version = UInt32(PMK_V4_ABI_VERSION)
        desc.m = manifest.m; desc.n = manifest.n; desc.k = manifest.k; desc.r = manifest.rank
        desc.a.clean_values = ns[0].bytes.assumingMemoryBound(to: Int8.self)
        desc.a.clean_value_count = UInt64(aClean.count)
        desc.a.noise_e_codes = ns[1].bytes.assumingMemoryBound(to: UInt8.self)
        desc.a.noise_e_count = UInt64(aNoiseE.count)
        desc.a.noise_f_codes = ns[2].bytes.assumingMemoryBound(to: UInt8.self)
        desc.a.noise_f_count = UInt64(aNoiseF.count)
        desc.a.alpha_bf16 = ns[3].bytes.assumingMemoryBound(to: UInt16.self)
        desc.a.beta_bf16 = ns[4].bytes.assumingMemoryBound(to: UInt16.self)
        desc.a.scale_count = UInt64(manifest.m)
        desc.bt.clean_values = ns[5].bytes.assumingMemoryBound(to: Int8.self)
        desc.bt.clean_value_count = UInt64(bClean.count)
        desc.bt.noise_e_codes = ns[6].bytes.assumingMemoryBound(to: UInt8.self)
        desc.bt.noise_e_count = UInt64(bNoiseE.count)
        desc.bt.noise_f_codes = ns[7].bytes.assumingMemoryBound(to: UInt8.self)
        desc.bt.noise_f_count = UInt64(bNoiseF.count)
        desc.bt.alpha_bf16 = ns[8].bytes.assumingMemoryBound(to: UInt16.self)
        desc.bt.beta_bf16 = ns[9].bytes.assumingMemoryBound(to: UInt16.self)
        desc.bt.scale_count = UInt64(manifest.n)
        desc.jackpot_key = try v4Words8FromHex32(manifest.jackpot_key)
        desc.block_bound = try v4Words8FromHex32(manifest.block_bound)
        desc.share_bound = try v4Words8FromHex32(manifest.share_bound)
        desc.block_capacity = 64; desc.share_capacity = 64; desc.job_id = 0x56345f50524f4245
        guard let job = V4Job(context: self, desc: desc), let cb = queue.makeCommandBuffer() else {
            throw PMKError("v4 startup probe allocation failed")
        }
        try job.encode(cb)
        cb.commit(); cb.waitUntilCompleted(); job.complete(cb, markContextUnhealthy: false)
        job.lock.lock(); let result = job.result; job.lock.unlock()
        guard result.status == PMK_SUCCESS else {
            throw PMKError("v4 startup probe GPU status \(result.status): \(job.diagnostic)")
        }
        let qa = Data(bytes: job.aCodes.contents(), count: expectedA.count)
        let qb = Data(bytes: job.bCodes.contents(), count: expectedB.count)
        guard qa == expectedA, qb == expectedB else { throw PMKError("v4 startup probe quantized code mismatch") }
        guard let cPtr = result.c_bits else { throw PMKError("v4 startup probe missing C readback") }
        let gotC = Data(bytes: cPtr, count: expectedC.count)
        guard gotC == expectedC else { throw PMKError("v4 startup probe C mismatch") }
        guard result.block_count == manifest.blocks_expected, result.share_count == manifest.shares_expected,
              result.block_stored == manifest.blocks_expected, result.share_stored == manifest.shares_expected,
              result.stats.layout_failures == 0,
              result.stats.quantized_a == UInt64(manifest.m) * UInt64(manifest.k),
              result.stats.quantized_b == UInt64(manifest.n) * UInt64(manifest.k),
              result.stats.quant_saturated_a == 0, result.stats.quant_saturated_b == 0,
              result.stats.quant_nan_a == 0, result.stats.quant_nan_b == 0 else {
            throw PMKError("v4 startup probe counters mismatch")
        }
        var expectedByKind: [String: [String: V4StartupProbeTile]] = ["block": [:], "share": [:]]
        for tile in manifest.tiles { expectedByKind[tile.kind]?["\(tile.row):\(tile.col)"] = tile }
        func checkSlots(_ ptr: UnsafePointer<pmk_slot>?, _ count: UInt32, _ kind: String) throws {
            guard let ptr else { throw PMKError("v4 startup probe missing \(kind) slots") }
            var seen = Set<String>()
            for i in 0..<Int(count) {
                let slot = ptr[i]
                let key = "\(slot.t_rows):\(slot.t_cols)"
                guard let expected = expectedByKind[kind]?[key] else {
                    throw PMKError("v4 startup probe unexpected \(kind) slot \(key)")
                }
                let message = v4HexFromWords(words16(slot.transcript))
                let hash = v4HexFromWords(words8(slot.hash))
                guard message == expected.message, hash == expected.hash else {
                    throw PMKError("v4 startup probe \(kind) slot mismatch \(key)")
                }
                seen.insert(key)
            }
            guard seen.count == expectedByKind[kind]?.count else { throw PMKError("v4 startup probe missing \(kind) slots") }
        }
        try checkSlots(result.blocks, result.block_stored, "block")
        try checkSlots(result.shares, result.share_stored, "share")
    }

    func refreshProbe() throws {
        guard jobs == 0 else { throw PMKError("v4 probe refresh busy") }
        guard try currentOSBuild() == osBuild else { throw PMKError("OS build changed since v4 G3 admission") }
        try runV4FP8Selftest(device: device, queue: queue, pipeline: try unwrap(pipelines["v4_fp8_roundtrip"], "Missing v4_fp8_roundtrip"))
        try runStartupProbe()
        if requireAdmission {
            let loadedLibrary = try v4LoadedLibraryInfo()
            guard loadedLibrary.path == libraryPath, loadedLibrary.sha256 == librarySHA256 else {
                throw PMKError("DO NOT MINE: loaded libpmk.dylib changed since v4 init")
            }
            try requireV4Admission(deviceName: device.name, deviceClass: deviceClass, cacheKey: cacheKey,
                                   osBuild: osBuild, librarySHA256: librarySHA256, now: Date())
        }
        lastProbeUptime = ProcessInfo.processInfo.systemUptime
        healthy = true
    }

    func refreshProbeIfIdleDue(nowUptime: TimeInterval = ProcessInfo.processInfo.systemUptime) throws {
        guard jobs == 0 else { return }
        guard try currentOSBuild() == osBuild else { throw PMKError("OS build changed since v4 G3 admission") }
        if shouldRefreshProbe(last: lastProbeUptime, now: nowUptime, interval: probeRefreshInterval) {
            try refreshProbe()
        }
    }

    func recordDiagnostic(_ value: String) {
        lastDiagnostic = diagnosticString(value)
    }

    func admissionMetadataJSON(exactCells: UInt64? = nil) -> String {
        var fields: [String] = [
            "\"schema\":\"pmk-v4-admission-v1\"",
            "\"gpu_name\":\(jsonString(device.name))",
            "\"device_class\":\(jsonString(deviceClass))",
            "\"os_build\":\(jsonString(osBuild))",
            "\"cache_key\":\(jsonString(cacheKey))",
            "\"library_sha256\":\(jsonString(librarySHA256))",
            "\"kernel\":\"E\"",
            "\"metal_language\":\"3.1\"",
            "\"vendor_pearl_fp8\":\"f696760b259500ecb608469ea3953aeabbe78948\"",
        ]
        if let exactCells {
            fields.append("\"v4_g3_passed\":true")
            fields.append("\"exact_cells\":\(exactCells)")
            let now = Date().timeIntervalSince1970
            fields.append("\"last_probe_unix\":\(now)")
            fields.append("\"valid_until_unix\":\(now + v4AdmissionValidSeconds)")
        }
        return "{\(fields.joined(separator: ","))}"
    }
}

private func v4FP8DecodeBitsHost(_ code: UInt8) -> UInt32 {
    let sign = UInt32(code & 0x80) << 24
    let exp = UInt32((code >> 3) & 0x0f)
    let man = UInt32(code & 7)
    var mag: UInt32 = 0
    if exp == 0 {
        if man != 0 {
            let lz = UInt32(man.leadingZeroBitCount - 28)
            mag = ((121 - lz) << 23) | (((man << lz) & 7) << 20)
        }
    } else {
        mag = ((exp + 120) << 23) | (man << 20)
    }
    return sign | mag
}

private func runV4FP8Selftest(device: MTLDevice, queue: MTLCommandQueue, pipeline: MTLComputePipelineState) throws {
    let codes = (0..<256).map { UInt8($0) }
    let codeBuffer = codes.withUnsafeBytes { raw in
        device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
    }
    guard let codeBuffer,
          let decoded = device.makeBuffer(length: 256 * 4, options: .storageModeShared),
          let recoded = device.makeBuffer(length: 256, options: .storageModeShared),
          let cb = queue.makeCommandBuffer(),
          let e = cb.makeComputeCommandEncoder() else {
        throw PMKError("v4 FP8 selftest resource failure")
    }
    e.setComputePipelineState(pipeline)
    e.setBuffer(codeBuffer, offset: 0, index: 0)
    e.setBuffer(decoded, offset: 0, index: 1)
    e.setBuffer(recoded, offset: 0, index: 2)
    e.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                      threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    guard cb.status == .completed else {
        throw PMKError("v4 FP8 selftest GPU failure: \(cb.error.map { String(describing: $0) } ?? "unknown")")
    }
    let gotBits = decoded.contents().bindMemory(to: UInt32.self, capacity: 256)
    let gotCodes = recoded.contents().bindMemory(to: UInt8.self, capacity: 256)
    for i in 0..<256 {
        if i == 0x7f || i == 0xff { continue }
        let code = UInt8(i)
        let expectedBits = v4FP8DecodeBitsHost(code)
        guard gotBits[i] == expectedBits else {
            throw PMKError(String(format: "v4 FP8 decode mismatch code=0x%02x gpu=0x%08x host=0x%08x", i, gotBits[i], expectedBits))
        }
        guard gotCodes[i] == code else {
            throw PMKError(String(format: "v4 FP8 roundtrip mismatch code=0x%02x gpu=0x%02x", i, gotCodes[i]))
        }
    }
}

private func jsonUInt64(_ value: Any?) -> UInt64? {
    guard let number = value as? NSNumber else { return nil }
    let type = String(cString: number.objCType)
    guard type != "c" && type != "B" else { return nil }
    let double = number.doubleValue
    // Admission counts are cell counts. Bound well below Double's exact-integer cliff and UInt64.max
    // so malicious JSON such as 2^64 cannot round then trap during conversion.
    guard double.isFinite, double >= 0, double <= 9_000_000_000_000_000, floor(double) == double else { return nil }
    return UInt64(double)
}

private func requireV4Admission(deviceName: String, deviceClass: String, cacheKey: String,
                                osBuild: String, librarySHA256: String, now: Date,
                                path: String? = ProcessInfo.processInfo.environment["PMK_V4_G3_ADMISSION_FILE"]) throws {
    guard let path, !path.isEmpty else {
        throw PMKError("DO NOT MINE: missing v4 G3 admission file")
    }
    let root = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [String: Any]
    let record = (root?["devices"] as? [[String: Any]])?.first { ($0["gpu_name"] as? String) == deviceName } ?? root
    guard let record,
          (record["v4_g3_passed"] as? Bool) == true,
          (record["gpu_name"] as? String) == deviceName,
          (record["device_class"] as? String) == deviceClass,
          (record["schema"] as? String) == "pmk-v4-admission-v1",
          (record["cache_key"] as? String) == cacheKey,
          (record["library_sha256"] as? String) == librarySHA256,
          (record["os_build"] as? String) == osBuild,
          (record["kernel"] as? String) == "E",
          (record["metal_language"] as? String) == "3.1",
          (record["vendor_pearl_fp8"] as? String) == v4VendorPin,
          (jsonUInt64(record["exact_cells"]) ?? 0) >= v4ExactCellsGate else {
        throw PMKError("DO NOT MINE: stale or mismatched v4 G3 admission")
    }
    let nowValue = now.timeIntervalSince1970
    guard let lastProbe = (record["last_probe_unix"] as? NSNumber)?.doubleValue,
          let validUntil = (record["valid_until_unix"] as? NSNumber)?.doubleValue,
          lastProbe.isFinite, validUntil.isFinite,
          lastProbe > 0, lastProbe <= nowValue, validUntil >= nowValue, validUntil - lastProbe <= v4AdmissionValidSeconds + 60 else {
        throw PMKError("DO NOT MINE: stale or mismatched v4 G3 admission")
    }
}

final class V4Job {
    let context: V4Context
    let jobID: UInt64
    let m: UInt32, n: UInt32, k: UInt32
    let blockCapacity: UInt32, shareCapacity: UInt32
    let params: V4Params
    let mode: String
    let aCodes: MTLBuffer
    let bCodes: MTLBuffer
    let aFloats: MTLBuffer
    let bFloatsT: MTLBuffer
    let cBits: MTLBuffer
    let stats: MTLBuffer
    let qStatsA: MTLBuffer
    let qStatsB: MTLBuffer
    let ctr: MTLBuffer
    let blocks: MTLBuffer
    let shares: MTLBuffer
    let lock = NSLock()
    let callbackBarrier = CallbackBarrier()
    var result = pmk_v4_result()
    var diagnostic = ""
    var done = false
    var releaseClaimed = false
    var deferredRelease = false

    init?(context: V4Context, codes d: pmk_v4_codes_desc) {
        self.context = context
        jobID = d.job_id
        m = d.m; n = d.n; k = d.k
        blockCapacity = d.block_capacity; shareCapacity = d.share_capacity
        mode = "codes"
        params = V4Params(m: d.m, n: d.n, k: d.k, r: 0, blockCap: d.block_capacity, shareCap: d.share_capacity,
                          key: words8(d.jackpot_key), block: words8(d.block_bound), share: words8(d.share_bound))
        let dev = context.device
        guard let ac = copyBuffer(dev, d.a_codes, Int(d.a_code_count)),
              let bc = copyBuffer(dev, d.bt_codes, Int(d.bt_code_count)),
              let af = emptyBuffer(dev, Int(d.m) * Int(d.k) * 4),
              let bf = emptyBuffer(dev, Int(d.n) * Int(d.k) * 4),
              let c = emptyBuffer(dev, Int(d.m) * Int(d.n) * 4),
              let st = emptyBuffer(dev, ((Int(d.m) / 32) * (Int(d.n) / 32) * 8) * 4),
              let qa = emptyBuffer(dev, 12),
              let qb = emptyBuffer(dev, 12),
              let counter = emptyBuffer(dev, 8),
              let bl = emptyBuffer(dev, (Int(d.block_capacity) * v4SlotWords + v4GuardWords) * 4),
              let sh = emptyBuffer(dev, (Int(d.share_capacity) * v4SlotWords + v4GuardWords) * 4) else { return nil }
        aCodes = ac; bCodes = bc; aFloats = af; bFloatsT = bf; cBits = c; stats = st
        qStatsA = qa; qStatsB = qb; ctr = counter; blocks = bl; shares = sh
        initialize()
    }

    init?(context: V4Context, desc d: pmk_v4_job_desc) {
        self.context = context
        jobID = d.job_id
        m = d.m; n = d.n; k = d.k
        blockCapacity = d.block_capacity; shareCapacity = d.share_capacity
        mode = "fused"
        params = V4Params(m: d.m, n: d.n, k: d.k, r: d.r, blockCap: d.block_capacity, shareCap: d.share_capacity,
                          key: words8(d.jackpot_key), block: words8(d.block_bound), share: words8(d.share_bound))
        let dev = context.device
        guard let ac = emptyBuffer(dev, Int(d.m) * Int(d.k)),
              let bc = emptyBuffer(dev, Int(d.n) * Int(d.k)),
              let af = emptyBuffer(dev, Int(d.m) * Int(d.k) * 4),
              let bf = emptyBuffer(dev, Int(d.n) * Int(d.k) * 4),
              let c = emptyBuffer(dev, Int(d.m) * Int(d.n) * 4),
              let st = emptyBuffer(dev, ((Int(d.m) / 32) * (Int(d.n) / 32) * 8) * 4),
              let qa = emptyBuffer(dev, 12),
              let qb = emptyBuffer(dev, 12),
              let counter = emptyBuffer(dev, 8),
              let bl = emptyBuffer(dev, (Int(d.block_capacity) * v4SlotWords + v4GuardWords) * 4),
              let sh = emptyBuffer(dev, (Int(d.share_capacity) * v4SlotWords + v4GuardWords) * 4) else { return nil }
        aCodes = ac; bCodes = bc; aFloats = af; bFloatsT = bf; cBits = c; stats = st
        qStatsA = qa; qStatsB = qb; ctr = counter; blocks = bl; shares = sh
        guard let aClean = copyBuffer(dev, d.a.clean_values, Int(d.a.clean_value_count)),
              let aE = copyBuffer(dev, d.a.noise_e_codes, Int(d.a.noise_e_count)),
              let aF = copyBuffer(dev, d.a.noise_f_codes, Int(d.a.noise_f_count)),
              let aAlpha = copyBuffer(dev, d.a.alpha_bf16, Int(d.a.scale_count)),
              let aBeta = copyBuffer(dev, d.a.beta_bf16, Int(d.a.scale_count)),
              let bClean = copyBuffer(dev, d.bt.clean_values, Int(d.bt.clean_value_count)),
              let bE = copyBuffer(dev, d.bt.noise_e_codes, Int(d.bt.noise_e_count)),
              let bF = copyBuffer(dev, d.bt.noise_f_codes, Int(d.bt.noise_f_count)),
              let bAlpha = copyBuffer(dev, d.bt.alpha_bf16, Int(d.bt.scale_count)),
              let bBeta = copyBuffer(dev, d.bt.beta_bf16, Int(d.bt.scale_count)) else { return nil }
        fusedInputs = [aClean, aE, aF, aAlpha, aBeta, bClean, bE, bF, bAlpha, bBeta]
        initialize()
    }

    private var fusedInputs: [MTLBuffer] = []

    private func initialize() {
        memset(stats.contents(), 0, stats.length)
        memset(qStatsA.contents(), 0, qStatsA.length)
        memset(qStatsB.contents(), 0, qStatsB.length)
        memset(ctr.contents(), 0, ctr.length)
        for (buf, capacity) in [(blocks, blockCapacity), (shares, shareCapacity)] {
            let ptr = buf.contents().bindMemory(to: UInt32.self, capacity: buf.length / 4)
            ptr.advanced(by: Int(capacity) * v4SlotWords).initialize(repeating: v4Canary, count: v4GuardWords)
        }
        result.abi_version = UInt32(PMK_V4_ABI_VERSION)
        result.status = Int32(PMK_PENDING)
        result.job_id = jobID
    }

    func encode(_ cb: MTLCommandBuffer) throws {
        try encodePrep(cb)
        try encodeKernelOnly(cb)
        try encodePost(cb)
    }

    func encodePrep(_ cb: MTLCommandBuffer) throws {
        if mode == "fused" {
            try encodeQuant(cb, sideA: true)
            try encodeQuant(cb, sideA: false)
            try encodeDecodeCodes(cb)
        } else {
            try encodeDecodeCodes(cb)
        }
        try encodeTileMax(cb)
    }

    func encodeKernelOnly(_ cb: MTLCommandBuffer) throws {
        try encodeGemm(cb)
    }

    func encodePost(_ cb: MTLCommandBuffer) throws {
        try encodeLottery(cb)
    }

    private func setParams(_ encoder: MTLComputeCommandEncoder, index: Int, rowOverride: UInt32? = nil) {
        var p = params.words
        if let rowOverride { p[0] = rowOverride }
        encoder.setBytes(&p, length: 128, index: index)
    }

    private func encodeQuant(_ cb: MTLCommandBuffer, sideA: Bool) throws {
        guard let e = cb.makeComputeCommandEncoder(), let p = context.pipelines["v4_quantize_operand"] else {
            throw PMKError("No v4 quant encoder")
        }
        let base = sideA ? 0 : 5
        let rows = sideA ? Int(m) : Int(n)
        e.setComputePipelineState(p)
        e.setBuffer(fusedInputs[base + 0], offset: 0, index: 0)
        e.setBuffer(fusedInputs[base + 1], offset: 0, index: 1)
        e.setBuffer(fusedInputs[base + 2], offset: 0, index: 2)
        e.setBuffer(fusedInputs[base + 3], offset: 0, index: 3)
        e.setBuffer(fusedInputs[base + 4], offset: 0, index: 4)
        e.setBuffer(sideA ? aCodes : bCodes, offset: 0, index: 5)
        e.setBuffer(sideA ? aFloats : bFloatsT, offset: 0, index: 6)
        e.setBuffer(sideA ? qStatsA : qStatsB, offset: 0, index: 7)
        setParams(e, index: 8, rowOverride: UInt32(rows))
        e.dispatchThreadgroups(MTLSize(width: Int(k), height: rows, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        e.endEncoding()
    }

    private func encodeDecodeCodes(_ cb: MTLCommandBuffer) throws {
        guard let p = context.pipelines["v4_decode_codes"] else { throw PMKError("No v4 decode pipeline") }
        func decode(_ codes: MTLBuffer, rows: UInt32, cols: UInt32, out: MTLBuffer) throws {
            guard let e = cb.makeComputeCommandEncoder() else { throw PMKError("No v4 decode encoder") }
            var dims: [UInt32] = [rows, cols]
            e.setComputePipelineState(p)
            e.setBuffer(codes, offset: 0, index: 0)
            e.setBuffer(out, offset: 0, index: 1)
            e.setBytes(&dims, length: 8, index: 2)
            e.dispatchThreads(MTLSize(width: Int(cols), height: Int(rows), depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            e.endEncoding()
        }
        try decode(aCodes, rows: m, cols: k, out: aFloats)
        let bRowMajor = try unwrap(context.device.makeBuffer(length: Int(n) * Int(k) * 4, options: .storageModeShared),
                                   "No v4 B decode buffer")
        bFloatsScratch = bRowMajor
        try decode(bCodes, rows: n, cols: k, out: bRowMajor)
        try encodeTransposeB(cb)
    }

    private func encodeTransposeB(_ cb: MTLCommandBuffer) throws {
        guard let e = cb.makeComputeCommandEncoder(), let p = context.pipelines["v4_transpose_b"] else {
            throw PMKError("No v4 transpose encoder")
        }
        let bRowMajorFloats = bFloatsScratch ?? bFloatsT
        let transposed = context.device.makeBuffer(length: Int(n) * Int(k) * 4, options: .storageModeShared)
        guard let transposed else { throw PMKError("No v4 transpose buffer") }
        bFloatsScratch = transposed
        e.setComputePipelineState(p)
        e.setBuffer(bRowMajorFloats, offset: 0, index: 0)
        e.setBuffer(transposed, offset: 0, index: 1)
        setParams(e, index: 2)
        e.dispatchThreads(MTLSize(width: Int(k), height: Int(n), depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        e.endEncoding()
    }

    private var bFloatsScratch: MTLBuffer?

    private func encodeTileMax(_ cb: MTLCommandBuffer) throws {
        let aTiles = Int(m) / 32
        let bTiles = Int(n) / 32
        guard let tA = emptyBuffer(context.device, aTiles * (Int(k) / 32) * 4),
              let tB = emptyBuffer(context.device, bTiles * (Int(k) / 32) * 4),
              let tile = context.pipelines["v4_tilemax"] else { throw PMKError("No v4 tilemax buffers") }
        tileBuffers = [tA, tB]
        func tileMax(_ codes: MTLBuffer, rows: UInt32, out: MTLBuffer) throws {
            guard let e = cb.makeComputeCommandEncoder() else { throw PMKError("No v4 tilemax encoder") }
            var dims: [UInt32] = [rows / 32, rows, k, 0]
            e.setComputePipelineState(tile)
            e.setBuffer(codes, offset: 0, index: 0)
            e.setBuffer(out, offset: 0, index: 1)
            e.setBytes(&dims, length: 16, index: 2)
            e.dispatchThreads(MTLSize(width: Int(rows) / 32, height: Int(k) / 32, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            e.endEncoding()
        }
        try tileMax(aCodes, rows: m, out: tA)
        try tileMax(bCodes, rows: n, out: tB)
    }

    private func encodeGemm(_ cb: MTLCommandBuffer) throws {
        guard tileBuffers.count == 2, let gemm = context.pipelines["v4_kernel_e"],
              let e = cb.makeComputeCommandEncoder() else { throw PMKError("No v4 E encoder") }
        e.setComputePipelineState(gemm)
        e.setBuffer(aFloats, offset: 0, index: 0)
        e.setBuffer(bFloatsScratch ?? bFloatsT, offset: 0, index: 1)
        e.setBuffer(tileBuffers[0], offset: 0, index: 2)
        e.setBuffer(tileBuffers[1], offset: 0, index: 3)
        e.setBuffer(aCodes, offset: 0, index: 4)
        e.setBuffer(bCodes, offset: 0, index: 5)
        e.setBuffer(cBits, offset: 0, index: 6)
        e.setBuffer(stats, offset: 0, index: 7)
        setParams(e, index: 8)
        e.dispatchThreadgroups(MTLSize(width: Int(n) / 32, height: Int(m) / 32, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        e.endEncoding()
    }

    private var tileBuffers: [MTLBuffer] = []

    private func encodeLottery(_ cb: MTLCommandBuffer) throws {
        guard let e = cb.makeComputeCommandEncoder(), let p = context.pipelines["v4_lottery"] else {
            throw PMKError("No v4 lottery encoder")
        }
        e.setComputePipelineState(p)
        e.setBuffer(cBits, offset: 0, index: 0)
        e.setBuffer(ctr, offset: 0, index: 1)
        e.setBuffer(blocks, offset: 0, index: 2)
        e.setBuffer(shares, offset: 0, index: 3)
        setParams(e, index: 4)
        e.dispatchThreadgroups(MTLSize(width: Int(n) / 16, height: Int(m) / 16, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        e.endEncoding()
    }

    private func guardsIntact(_ buffer: MTLBuffer, capacity: UInt32) -> Bool {
        let words = buffer.contents().bindMemory(to: UInt32.self, capacity: buffer.length / 4)
        let start = Int(capacity) * v4SlotWords
        guard start + v4GuardWords <= buffer.length / 4 else { return false }
        for i in 0..<v4GuardWords where words[start + i] != v4Canary { return false }
        return true
    }

    func complete(_ cb: MTLCommandBuffer, markContextUnhealthy: Bool = true) {
        complete(statusBuffers: [cb], timingBuffer: cb, markContextUnhealthy: markContextUnhealthy)
    }

    func complete(statusBuffers: [MTLCommandBuffer], timingBuffer: MTLCommandBuffer, markContextUnhealthy: Bool = true) {
        var r = result
        r.gpu_start_time = timingBuffer.gpuStartTime
        r.gpu_end_time = timingBuffer.gpuEndTime
        let failed = statusBuffers.first { $0.status != .completed }
        r.status = failed == nil ? 0 : Int32(PMK_GPU_FAILED)
        if let failed {
            diagnostic = diagnosticString("v4 command buffer status=\(failed.status.rawValue): \(failed.error.map { String(describing: $0) } ?? "no Metal error")")
        } else {
            let counts = ctr.contents().bindMemory(to: UInt32.self, capacity: 2)
            let totalLotteryTiles = (m / 16) * (n / 16)
            let canariesOK = guardsIntact(blocks, capacity: blockCapacity) && guardsIntact(shares, capacity: shareCapacity)
            guard counts[0] <= totalLotteryTiles, counts[1] <= totalLotteryTiles, canariesOK else {
                r.status = Int32(PMK_GPU_FAILED)
                r.block_count = 0; r.share_count = 0
                r.block_stored = 0; r.share_stored = 0
                r.overflow = 0
                r.blocks = nil; r.shares = nil
                r.c_bits = nil; r.c_count = 0
                diagnostic = diagnosticString("v4 result integrity check failed")
                lock.lock(); result = r; done = true; lock.unlock()
                if markContextUnhealthy { context.lock.lock(); context.healthy = false; context.lock.unlock() }
                return
            }
            r.block_count = counts[0]; r.share_count = counts[1]
            r.block_stored = min(counts[0], blockCapacity)
            r.share_stored = min(counts[1], shareCapacity)
            r.overflow = (counts[0] > blockCapacity ? 1 : 0) | (counts[1] > shareCapacity ? 2 : 0)
            r.blocks = UnsafePointer(blocks.contents().assumingMemoryBound(to: pmk_slot.self))
            r.shares = UnsafePointer(shares.contents().assumingMemoryBound(to: pmk_slot.self))
            r.c_bits = UnsafePointer(cBits.contents().assumingMemoryBound(to: UInt32.self))
            r.c_count = UInt64(m) * UInt64(n)
            let statWords = stats.contents().bindMemory(to: UInt32.self, capacity: stats.length / 4)
            let tileStats = (Int(m) / 32) * (Int(n) / 32)
            var fallbackGroups: UInt64 = 0
            var layoutFailures: UInt32 = 0
            for i in 0..<tileStats {
                let base = i * 8
                for j in 0..<4 { fallbackGroups += UInt64(statWords[base + j]) }
                for j in 0..<4 { layoutFailures += statWords[base + 4 + j] }
            }
            let qa = qStatsA.contents().bindMemory(to: UInt32.self, capacity: 3)
            let qb = qStatsB.contents().bindMemory(to: UInt32.self, capacity: 3)
            r.stats.abi_version = UInt32(PMK_V4_ABI_VERSION)
            r.stats.fallback_groups = fallbackGroups
            r.stats.total_groups = UInt64(m) * UInt64(n) * UInt64(k / 32)
            r.stats.layout_failures = layoutFailures
            r.stats.quantized_a = UInt64(qa[0]); r.stats.quant_saturated_a = UInt64(qa[1]); r.stats.quant_nan_a = UInt64(qa[2])
            r.stats.quantized_b = UInt64(qb[0]); r.stats.quant_saturated_b = UInt64(qb[1]); r.stats.quant_nan_b = UInt64(qb[2])
            let fallbackRate = r.stats.total_groups == 0 ? 0 : Double(r.stats.fallback_groups) / Double(r.stats.total_groups)
            r.stats.fallback_alert = (k <= 4096 && fallbackRate > 0.01) ? 1 : 0
            if r.stats.layout_failures != 0 {
                r.status = Int32(PMK_GPU_FAILED)
                diagnostic = diagnosticString("v4 simdgroup layout probe failed")
            }
        }
        lock.lock(); result = r; done = true; lock.unlock()
        if r.status != 0, markContextUnhealthy { context.lock.lock(); context.healthy = false; context.lock.unlock() }
    }

    func claimRelease() -> (Int32, Bool) {
        lock.lock()
        guard done else { lock.unlock(); return (Int32(PMK_BUSY), false) }
        guard !releaseClaimed else { lock.unlock(); return (Int32(PMK_BUSY), false) }
        releaseClaimed = true
        let deferRelease = callbackBarrier.isCallbackThread()
        if deferRelease { deferredRelease = true }
        lock.unlock()
        return (0, !deferRelease)
    }

    func takeDeferredRelease() -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard deferredRelease else { return false }
        deferredRelease = false
        return true
    }

    func finishRelease(_ handle: UnsafeMutableRawPointer) {
        context.lock.lock(); context.jobs -= 1; context.lock.unlock()
        Unmanaged<V4Job>.fromOpaque(handle).release()
    }
}

@_cdecl("pmk_v4_init")
public func pmkV4Init(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>?, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    pmkV4InitImpl(out, errorBuffer, capacity, requireAdmission: true)
}

@_cdecl("pmk_v4_init_diagnostic")
public func pmkV4InitDiagnostic(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>?, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    pmkV4InitImpl(out, errorBuffer, capacity, requireAdmission: false)
}

private func pmkV4InitImpl(_ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>?, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ capacity: UInt64,
                           requireAdmission: Bool) -> Int32 {
    guard let out else { return Int32(PMK_INVALID) }
    out.pointee = nil
    do {
        let c = try V4Context(requireAdmission: requireAdmission)
        out.pointee = Unmanaged.passRetained(c).toOpaque()
        putString("", errorBuffer, capacity)
        return 0
    } catch let resourceError as PMKResourceError {
        putString("DO NOT MINE: \(resourceError)", errorBuffer, capacity)
        return Int32(PMK_RESOURCE)
    } catch {
        putString("DO NOT MINE: \(error)", errorBuffer, capacity)
        return Int32(PMK_PROBE_FAILED)
    }
}

@_cdecl("pmk_v4_probe")
public func pmkV4Probe(_ handle: UnsafeMutableRawPointer?, _ key: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    guard let handle else { return Int32(PMK_INVALID) }
    let c = v4Context(handle)
    c.lock.lock(); defer { c.lock.unlock() }
    do {
        try c.refreshProbe()
    } catch {
        if c.jobs == 0 {
            c.healthy = false
            c.recordDiagnostic("v4 probe failed: \(error)")
            return Int32(PMK_PROBE_FAILED)
        }
        return Int32(PMK_BUSY)
    }
    guard c.healthy else { return Int32(PMK_PROBE_FAILED) }
    putString(c.cacheKey, key, capacity)
    return 0
}

@_cdecl("pmk_v4_admission_metadata")
public func pmkV4AdmissionMetadata(_ handle: UnsafeMutableRawPointer?, _ json: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    guard let handle else { return Int32(PMK_INVALID) }
    putString(v4Context(handle).admissionMetadataJSON(), json, capacity)
    return 0
}


@_cdecl("pmk_v4_fp8_selftest")
public func pmkV4FP8Selftest(_ handle: UnsafeMutableRawPointer?, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    guard let handle else { return Int32(PMK_INVALID) }
    let c = v4Context(handle)
    do {
        try runV4FP8Selftest(device: c.device, queue: c.queue, pipeline: try unwrap(c.pipelines["v4_fp8_roundtrip"], "Missing v4_fp8_roundtrip"))
        putString("", errorBuffer, capacity)
        return 0
    } catch {
        c.lock.lock(); c.healthy = false; c.recordDiagnostic("v4 FP8 selftest failed: \(error)"); c.lock.unlock()
        putString("DO NOT MINE: \(error)", errorBuffer, capacity)
        return Int32(PMK_PROBE_FAILED)
    }
}

@_cdecl("pmk_v4_write_admission_record")
public func pmkV4WriteAdmissionRecord(_ handle: UnsafeMutableRawPointer?, _ path: UnsafePointer<CChar>?,
                                      _ exactCells: UInt64, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    guard let handle, let path, exactCells >= v4ExactCellsGate else { return Int32(PMK_INVALID) }
    do {
        let c = v4Context(handle)
        let json = "{\"devices\":[\(c.admissionMetadataJSON(exactCells: exactCells))]}\n"
        try json.write(toFile: String(cString: path), atomically: true, encoding: .utf8)
        try requireV4Admission(deviceName: c.device.name, deviceClass: c.deviceClass, cacheKey: c.cacheKey,
                               osBuild: c.osBuild, librarySHA256: c.librarySHA256, now: Date(), path: String(cString: path))
        putString("", errorBuffer, capacity)
        return 0
    } catch {
        putString("DO NOT MINE: \(error)", errorBuffer, capacity)
        return Int32(PMK_PROBE_FAILED)
    }
}

@_cdecl("pmk_v4_destroy")
public func pmkV4Destroy(_ handle: UnsafeMutableRawPointer?) {
    if let handle { Unmanaged<V4Context>.fromOpaque(handle).release() }
}

@_cdecl("pmk_v4_run_codes_diagnostic")
public func pmkV4RunCodesDiagnostic(_ handle: UnsafeMutableRawPointer?, _ descriptor: UnsafePointer<pmk_v4_codes_desc>?,
                                    _ callback: (@convention(c) (UnsafeMutableRawPointer?, UnsafeMutableRawPointer?) -> Void)?,
                                    _ user: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>?) -> Int32 {
    guard let handle, let descriptor, let out else { return Int32(PMK_INVALID) }
    out.pointee = nil
    let d = descriptor.pointee
    let expectedACodes = product(UInt64(d.m), UInt64(d.k))
    let expectedBCodes = product(UInt64(d.n), UInt64(d.k))
    guard let expectedACodes, let expectedBCodes,
          d.abi_version == PMK_V4_ABI_VERSION,
          d.m >= 16, d.n >= 16, d.k >= 1024, d.k <= 65_536,
          d.m % 32 == 0, d.n % 32 == 0, d.k % 32 == 0,
          productFitsBuffer([UInt64(d.m), UInt64(d.k)], stride: 1, device: v4Context(handle).device),
          productFitsBuffer([UInt64(d.n), UInt64(d.k)], stride: 1, device: v4Context(handle).device),
          productFitsBuffer([UInt64(d.m), UInt64(d.n)], stride: 4, device: v4Context(handle).device),
          v4ResourceBudgetOK(m: d.m, n: d.n, k: d.k, r: 0, blockCap: d.block_capacity, shareCap: d.share_capacity, fused: false),
          d.a_code_count == expectedACodes,
          d.bt_code_count == expectedBCodes,
          d.block_capacity >= 4, d.share_capacity >= 64 else { return Int32(PMK_INVALID) }
    return pmkV4Submit(handle: handle, out: out, callback: callback, user: user) { V4Job(context: $0, codes: d) }
}

@_cdecl("pmk_v4_run_job")
public func pmkV4RunJob(_ handle: UnsafeMutableRawPointer?, _ descriptor: UnsafePointer<pmk_v4_job_desc>?,
                        _ callback: (@convention(c) (UnsafeMutableRawPointer?, UnsafeMutableRawPointer?) -> Void)?,
                        _ user: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<UnsafeMutableRawPointer?>?) -> Int32 {
    guard let handle, let descriptor, let out else { return Int32(PMK_INVALID) }
    out.pointee = nil
    let d = descriptor.pointee
    let expectedAClean = product(UInt64(d.m), UInt64(d.k))
    let expectedBClean = product(UInt64(d.n), UInt64(d.k))
    let expectedAE = product(UInt64(d.m), UInt64(d.r))
    let expectedBE = product(UInt64(d.n), UInt64(d.r))
    let expectedAF = product(UInt64(d.k), UInt64(d.r))
    let expectedBF = product(UInt64(d.k), UInt64(d.r))
    guard let expectedAClean, let expectedBClean, let expectedAE, let expectedBE, let expectedAF, let expectedBF,
          d.abi_version == PMK_V4_ABI_VERSION,
          d.m >= 16, d.n >= 16, d.m <= 8192, d.n <= 8192, [1024, 4096, 16384].contains(d.k), d.r == 32,
          d.m % 32 == 0, d.n % 32 == 0, d.k % 32 == 0,
          productFitsBuffer([UInt64(d.m), UInt64(d.k)], stride: 1, device: v4Context(handle).device),
          productFitsBuffer([UInt64(d.n), UInt64(d.k)], stride: 1, device: v4Context(handle).device),
          productFitsBuffer([UInt64(d.m), UInt64(d.n)], stride: 4, device: v4Context(handle).device),
          v4ResourceBudgetOK(m: d.m, n: d.n, k: d.k, r: d.r, blockCap: d.block_capacity, shareCap: d.share_capacity, fused: true),
          d.a.clean_value_count == expectedAClean,
          d.bt.clean_value_count == expectedBClean,
          d.a.noise_e_count == expectedAE,
          d.bt.noise_e_count == expectedBE,
          d.a.noise_f_count == expectedAF,
          d.bt.noise_f_count == expectedBF,
          d.a.scale_count == UInt64(d.m), d.bt.scale_count == UInt64(d.n),
          d.block_capacity >= 4, d.share_capacity >= 64 else { return Int32(PMK_INVALID) }
    return pmkV4Submit(handle: handle, out: out, callback: callback, user: user) { V4Job(context: $0, desc: d) }
}

private func pmkV4Submit(handle: UnsafeMutableRawPointer, out: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
                         callback: (@convention(c) (UnsafeMutableRawPointer?, UnsafeMutableRawPointer?) -> Void)?,
                         user: UnsafeMutableRawPointer?, make: (V4Context) -> V4Job?) -> Int32 {
    let c = v4Context(handle)
    c.lock.lock()
    c.recordDiagnostic("")
    do {
        try c.refreshProbeIfIdleDue()
    } catch {
        c.healthy = false
        c.recordDiagnostic("v4 probe refresh failed: \(error)")
        c.lock.unlock()
        return Int32(PMK_PROBE_FAILED)
    }
    guard c.healthy else { c.lock.unlock(); return Int32(PMK_PROBE_FAILED) }
    guard c.jobs < 3 else { c.lock.unlock(); return Int32(PMK_BUSY) }
    c.jobs += 1
    c.lock.unlock()
    guard let j = make(c), let cb = c.queue.makeCommandBuffer() else {
        c.lock.lock(); c.jobs -= 1; c.recordDiagnostic("v4 job allocation failed"); c.lock.unlock()
        return Int32(PMK_RESOURCE)
    }
    var retainedJob: UnsafeMutableRawPointer?
    do {
        j.callbackBarrier.arm()
        let opaque = Unmanaged.passRetained(j).toOpaque()
        retainedJob = opaque
        if j.mode == "codes" {
            guard let prep = c.queue.makeCommandBuffer(), let kernel = c.queue.makeCommandBuffer(), let post = c.queue.makeCommandBuffer() else {
                Unmanaged<V4Job>.fromOpaque(opaque).release()
                c.lock.lock(); c.jobs -= 1; c.recordDiagnostic("v4 command buffer allocation failed"); c.lock.unlock()
                return Int32(PMK_RESOURCE)
            }
            try j.encodePrep(prep)
            try j.encodeKernelOnly(kernel)
            try j.encodePost(post)
            out.pointee = opaque
            post.addCompletedHandler { completed in
                c.completionQueue.async {
                    j.complete(statusBuffers: [prep, kernel, completed], timingBuffer: kernel)
                    j.callbackBarrier.beginCallback()
                    callback?(opaque, user)
                    j.callbackBarrier.endCallback()
                    if j.takeDeferredRelease() { j.finishRelease(opaque) }
                }
            }
            prep.commit(); kernel.commit(); post.commit()
        } else {
            try j.encode(cb)
            out.pointee = opaque
            cb.addCompletedHandler { completed in
                c.completionQueue.async {
                    j.complete(completed)
                    j.callbackBarrier.beginCallback()
                    callback?(opaque, user)
                    j.callbackBarrier.endCallback()
                    if j.takeDeferredRelease() { j.finishRelease(opaque) }
                }
            }
            cb.commit()
        }
        retainedJob = nil
        return 0
    } catch {
        if let retainedJob { Unmanaged<V4Job>.fromOpaque(retainedJob).release() }
        c.lock.lock(); c.jobs -= 1; c.recordDiagnostic("v4 encode failed: \(error)"); c.lock.unlock()
        return Int32(PMK_RESOURCE)
    }
}

@_cdecl("pmk_v4_poll")
public func pmkV4Poll(_ handle: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<pmk_v4_result>?) -> Int32 {
    guard let handle, let out else { return Int32(PMK_INVALID) }
    let j = v4Job(handle)
    j.lock.lock(); defer { j.lock.unlock() }
    guard j.done else { return Int32(PMK_PENDING) }
    out.pointee = j.result
    return j.result.status
}


@_cdecl("pmk_v4_job_quantized_codes")
public func pmkV4JobQuantizedCodes(_ handle: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<pmk_v4_quantized_codes>?) -> Int32 {
    guard let handle, let out else { return Int32(PMK_INVALID) }
    let j = v4Job(handle)
    j.lock.lock(); defer { j.lock.unlock() }
    guard j.done, j.result.status == PMK_SUCCESS else { return j.done ? j.result.status : Int32(PMK_PENDING) }
    var q = pmk_v4_quantized_codes()
    q.abi_version = UInt32(PMK_V4_ABI_VERSION)
    q.a_codes = UnsafePointer(j.aCodes.contents().assumingMemoryBound(to: UInt8.self))
    q.bt_codes = UnsafePointer(j.bCodes.contents().assumingMemoryBound(to: UInt8.self))
    q.a_code_count = UInt64(j.m) * UInt64(j.k)
    q.bt_code_count = UInt64(j.n) * UInt64(j.k)
    out.pointee = q
    return 0
}

@_cdecl("pmk_v4_context_error")
public func pmkV4ContextError(_ handle: UnsafeMutableRawPointer?, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    guard let handle else { return Int32(PMK_INVALID) }
    putString(v4Context(handle).lastDiagnostic, errorBuffer, capacity)
    return 0
}

@_cdecl("pmk_v4_job_error")
public func pmkV4JobError(_ handle: UnsafeMutableRawPointer?, _ errorBuffer: UnsafeMutablePointer<CChar>?, _ capacity: UInt64) -> Int32 {
    guard let handle else { return Int32(PMK_INVALID) }
    putString(v4Job(handle).diagnostic, errorBuffer, capacity)
    return 0
}

@_cdecl("pmk_v4_job_wait_callback")
public func pmkV4JobWaitCallback(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard let handle else { return Int32(PMK_INVALID) }
    return v4Job(handle).callbackBarrier.wait() ? 0 : Int32(PMK_BUSY)
}

@_cdecl("pmk_v4_job_release")
public func pmkV4JobRelease(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard let handle else { return Int32(PMK_INVALID) }
    let j = v4Job(handle)
    let (rc, releaseNow) = j.claimRelease()
    guard rc == 0 else { return rc }
    guard releaseNow else { return 0 }
    guard j.callbackBarrier.wait() else { return Int32(PMK_BUSY) }
    j.finishRelease(handle)
    return 0
}
