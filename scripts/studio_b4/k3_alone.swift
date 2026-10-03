import Foundation
import Metal

func fail(_ message: String) -> Never {
    fputs("{\"pass\":false,\"error\":\"\(jsonEscape(message))\"}\n", stderr)
    exit(1)
}

func jsonEscape(_ value: String) -> String {
    var out = ""
    for scalar in value.unicodeScalars {
        switch scalar {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default:
            if scalar.value < 0x20 {
                out += String(format: "\\u%04x", scalar.value)
            } else {
                out.unicodeScalars.append(scalar)
            }
        }
    }
    return out
}

func argValue(_ name: String, default defaultValue: String) -> String {
    let args = CommandLine.arguments
    for i in 0..<args.count - 1 where args[i] == name {
        return args[i + 1]
    }
    return defaultValue
}

func intArg(_ name: String, default defaultValue: String) -> Int {
    let text = argValue(name, default: defaultValue)
    guard let value = Int(text) else {
        fail("invalid \(name)")
    }
    return value
}

func findSource() -> URL {
    let env = ProcessInfo.processInfo.environment
    let root = env["PMK_B4_ROOT"] ?? FileManager.default.currentDirectoryPath
    let candidates = [
        "\(root)/libpmk/metal/k3sg.metal",
        "\(root)/resources/metal/k3sg.metal",
        "\(root)/metal/k3sg.metal",
        "\(root)/../libpmk/metal/k3sg.metal",
    ]
    for path in candidates where FileManager.default.fileExists(atPath: path) {
        return URL(fileURLWithPath: path)
    }
    fail("could not find k3sg.metal under PMK_B4_ROOT or bundle resources")
}

final class Stats: @unchecked Sendable {
    let lock = NSLock()
    var gpuSeconds: [Double] = []
    var error: Error?

    func add(_ commandBuffer: MTLCommandBuffer) {
        lock.lock()
        if let e = commandBuffer.error {
            error = e
        }
        gpuSeconds.append(max(0.0, commandBuffer.gpuEndTime - commandBuffer.gpuStartTime))
        lock.unlock()
    }
}

func percentile(_ values: [Double], _ q: Double) -> Double {
    if values.isEmpty { return 0.0 }
    let sorted = values.sorted()
    if sorted.count == 1 { return sorted[0] }
    let pos = Double(sorted.count - 1) * q
    let lo = Int(floor(pos))
    let hi = Int(ceil(pos))
    if lo == hi { return sorted[lo] }
    return sorted[lo] * (Double(hi) - pos) + sorted[hi] * (pos - Double(lo))
}

let m = intArg("--m", default: "8192")
let n = intArg("--n", default: "8192")
let k = intArg("--k", default: "4096")
let jobs = intArg("--jobs", default: "200")
let inflightLimit = intArg("--inflight", default: "2")

guard m > 0, n > 0, k > 0, m % 64 == 0, n % 64 == 0, k % 128 == 0 else {
    fail("shape must satisfy m,n % 64 == 0 and k % 128 == 0")
}
guard let device = MTLCreateSystemDefaultDevice(), device.hasUnifiedMemory, device.supportsFamily(.apple7),
      let queue = device.makeCommandQueue() else {
    fail("DO NOT MINE: Apple7+ unified-memory Metal device required")
}

let source: String
do {
    source = try String(contentsOf: findSource(), encoding: .utf8)
} catch {
    fail("could not read k3sg.metal: \(error)")
}
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
let pipeline: MTLComputePipelineState
do {
    let library = try device.makeLibrary(source: source, options: options)
    guard let function = library.makeFunction(name: "k3sg") else {
        fail("missing k3sg function")
    }
    pipeline = try device.makeComputePipelineState(function: function)
} catch {
    fail("K3-SG compile failed: \(error)")
}

let aBytes = m * k
let bBytes = n * k
let blockCapacity = Swift.max(4, Swift.min(m * n / 32, 256))
let shareCapacity = 64
let slotWords = 26
let guardWords = 8 * slotWords
let blockBytes = (blockCapacity * slotWords + guardWords) * 4
let shareBytes = (shareCapacity * slotWords + guardWords) * 4
guard let a = device.makeBuffer(length: aBytes, options: .storageModeShared),
      let b = device.makeBuffer(length: bBytes, options: .storageModeShared) else {
    fail("could not allocate K3-SG benchmark buffers")
}
memset(a.contents(), 1, aBytes)
memset(b.contents(), 2, bBytes)

struct Flight {
    let ctr: MTLBuffer
    let blocks: MTLBuffer
    let shares: MTLBuffer
    let sink: MTLBuffer
}

var flights: [Flight] = []
for _ in 0..<jobs {
    guard let ctr = device.makeBuffer(length: 8, options: .storageModeShared),
          let blocks = device.makeBuffer(length: blockBytes, options: .storageModeShared),
          let shares = device.makeBuffer(length: shareBytes, options: .storageModeShared),
          let sink = device.makeBuffer(length: 16, options: .storageModeShared) else {
        fail("could not allocate K3-SG benchmark output buffers")
    }
    flights.append(Flight(ctr: ctr, blocks: blocks, shares: shares, sink: sink))
}

func encode(_ commandBuffer: MTLCommandBuffer, flight: Flight) {
    memset(flight.ctr.contents(), 0, 8)
    var params = [UInt32](repeating: 0, count: 32)
    params[0] = UInt32(m)
    params[1] = UInt32(n)
    params[2] = UInt32(k)
    params[3] = UInt32(blockCapacity)
    params[4] = UInt32(shareCapacity)
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        fail("could not create compute encoder")
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(a, offset: 0, index: 0)
    encoder.setBuffer(b, offset: 0, index: 1)
    encoder.setBytes(&params, length: 128, index: 2)
    encoder.setBuffer(flight.ctr, offset: 0, index: 3)
    encoder.setBuffer(flight.blocks, offset: 0, index: 4)
    encoder.setBuffer(flight.shares, offset: 0, index: 5)
    encoder.setBuffer(flight.sink, offset: 0, index: 6)
    encoder.dispatchThreadgroups(
        MTLSize(width: n / 64, height: m / 64, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
    )
    encoder.endEncoding()
}

let semaphore = DispatchSemaphore(value: max(1, inflightLimit))
let group = DispatchGroup()
let stats = Stats()
let started = ProcessInfo.processInfo.systemUptime
for job in 0..<jobs {
    semaphore.wait()
    group.enter()
    guard let commandBuffer = queue.makeCommandBuffer() else {
        fail("could not create command buffer")
    }
    encode(commandBuffer, flight: flights[job])
    commandBuffer.addCompletedHandler { cb in
        stats.add(cb)
        semaphore.signal()
        group.leave()
    }
    commandBuffer.commit()
}
group.wait()
if let error = stats.error {
    fail("GPU command failed: \(error)")
}
let wallSeconds = ProcessInfo.processInfo.systemUptime - started
let opsPerJob = 2.0 * Double(m) * Double(n) * Double(k)
let ops = Double(jobs) * opsPerJob
let tops = ops / wallSeconds / 1e12
let gpuMedian = percentile(stats.gpuSeconds, 0.5) * 1e3
let gpuP99 = percentile(stats.gpuSeconds, 0.99) * 1e3
let json = """
{"mode":"k3_alone","jobs":\(jobs),"shape":{"m":\(m),"n":\(n),"k":\(k)},"wall_seconds":\(wallSeconds),"ops":\(UInt64(ops)),"ops_per_second":\(ops / wallSeconds),"tops":\(tops),"gpu_ms_median":\(gpuMedian),"gpu_ms_p99":\(gpuP99),"device":"\(jsonEscape(device.name))","pass":true}
"""
print(json)
