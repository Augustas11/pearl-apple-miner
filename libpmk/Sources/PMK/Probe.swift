import Foundation
import Metal

public struct PMKProbeReport: Sendable {
    public let deviceName: String
    public let deviceClass: String
    public let kernel: String
    public let checkedCases: Int
}

public enum PMKProbeError: Error, CustomStringConvertible {
    case missingFunction(String)
    case resource(String)
    case gpu(String)
    case layout(String)
    case knownAnswer(String)

    public var description: String {
        switch self {
        case .missingFunction(let name): return "missing Metal function \(name)"
        case .resource(let message): return "probe resource error: \(message)"
        case .gpu(let message): return "probe GPU error: \(message)"
        case .layout(let message): return "K3-SG layout probe failed: \(message)"
        case .knownAnswer(let message): return "K3 known-answer probe failed: \(message)"
        }
    }
}

private let pmkSlotWords = 26
private let pmkGuardSlots = 8
private let pmkCanary: UInt32 = 0xA5A5A5A5
private let pmkRowsPattern = [0, 8, 16, 24]
private let pmkColsPattern = [0, 1, 8, 9, 16, 17, 24, 25]

private struct PMKK3SGParams {
    var words = [UInt32](repeating: 0, count: 32)

    init(m: Int, n: Int, k: Int, capBlock: Int, capShare: Int, key: [UInt32],
         boundBlock: [UInt32], boundShare: [UInt32]) {
        words[0] = UInt32(m)
        words[1] = UInt32(n)
        words[2] = UInt32(k)
        words[3] = UInt32(capBlock)
        words[4] = UInt32(capShare)
        for i in 0..<8 {
            words[8 + i] = key[i]
            words[16 + i] = boundBlock[i]
            words[24 + i] = boundShare[i]
        }
    }
}

public func runProbe(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary,
                     pipeline: MTLComputePipelineState, kernel: K3Kernel = .sg) throws -> PMKProbeReport {
    try PMKK3Probe(device: device, queue: queue, library: library, pipeline: pipeline, kernel: kernel).run()
}

private final class PMKK3Probe {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    let pipeline: MTLComputePipelineState
    let kernel: K3Kernel

    init(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary, pipeline: MTLComputePipelineState, kernel: K3Kernel) {
        self.device = device
        self.queue = queue
        self.library = library
        self.pipeline = pipeline
        self.kernel = kernel
    }

    func run() throws -> PMKProbeReport {
        if kernel == .sg {
            try runLayoutProbe()
        } else {
            try runNALayoutProbe()
        }
        let cases = try runKnownAnswerJob()
        return PMKProbeReport(deviceName: device.name, deviceClass: metalDeviceClass(device), kernel: kernel.rawValue, checkedCases: cases)
    }

    private func runNALayoutProbe() throws {
        guard let function = library.makeFunction(name: "na_probe") else {
            throw PMKProbeError.missingFunction("na_probe")
        }
        let state: MTLComputePipelineState
        do {
            state = try device.makeComputePipelineState(function: function)
        } catch {
            throw PMKProbeError.gpu("na_probe pipeline compile failed: \(error)")
        }
        let wordsPerThread = 130
        let threadCount = 128
        guard let outputBuffer = device.makeBuffer(length: wordsPerThread * threadCount * MemoryLayout<UInt32>.stride) else {
            throw PMKProbeError.resource("could not allocate NA layout probe buffer")
        }
        memset(outputBuffer.contents(), 0, outputBuffer.length)
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PMKProbeError.gpu("could not create NA layout probe command buffer")
        }
        encoder.setComputePipelineState(state)
        encoder.setBuffer(outputBuffer, offset: 0, index: 0)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
        encoder.endEncoding()
        try commitAndBlock(commandBuffer, label: "na_probe")

        let output = readUInt32s(outputBuffer, words: wordsPerThread * threadCount)
        var failures: [String] = []
        func check(_ condition: Bool, _ message: String) {
            if !condition { failures.append(message) }
        }
        guard let rowShape = patternShape(CPUOracle.naRowsPattern),
              let colShape = patternShape(CPUOracle.naColsPattern) else {
            throw PMKProbeError.layout("committed NA patterns are not legal Pearl PeriodicPatterns")
        }
        let expectedOrigins = Set((0..<128).filter { offsetIsValid(rowShape, $0) }.flatMap { row in
            (0..<64).filter { offsetIsValid(colShape, $0) }.map { col in [row, col] }
        })
        var origins = Set<[Int]>()
        var cover = Set<[Int]>()
        for tid in 0..<threadCount {
            let base = tid * wordsPerThread
            let count = Int(output[base])
            check(count == CPUOracle.naRowsPattern.count * CPUOracle.naColsPattern.count,
                  "thread \(tid) owned \(count) valid elements")
            check(Int(output[base + 1]) >= count, "thread \(tid) capacity < valid count")
            var points: [[Int]] = []
            for i in 0..<min(count, 64) {
                let row = Int(output[base + 2 + 2 * i])
                let col = Int(output[base + 3 + 2 * i])
                points.append([row, col])
                cover.insert([row, col])
            }
            guard let r0 = points.map({ $0[0] }).min(),
                  let c0 = points.map({ $0[1] }).min() else {
                failures.append("thread \(tid) reported no coordinates")
                continue
            }
            origins.insert([r0, c0])
            let rows = Set(points.map { $0[0] }).sorted()
            let cols = Set(points.map { $0[1] }).sorted()
            check(rows.map { $0 - r0 } == CPUOracle.naRowsPattern,
                  "thread \(tid) rows \(rows.map { $0 - r0 }) != NA rows")
            check(cols.map { $0 - c0 } == CPUOracle.naColsPattern,
                  "thread \(tid) cols \(cols.map { $0 - c0 }) != NA cols")
            check(points.count == Set(points).count, "thread \(tid) contains duplicate coordinates")
        }
        check(origins == expectedOrigins, "NA origins did not match Pearl valid offsets")
        check(cover.count == 128 * 64, "NA thread sets did not partition 128x64 tile")
        if !failures.isEmpty {
            throw PMKProbeError.layout(failures.prefix(8).joined(separator: "; "))
        }
    }

    private func runLayoutProbe() throws {
        guard let function = library.makeFunction(name: "sg_probe") else {
            throw PMKProbeError.missingFunction("sg_probe")
        }
        let state: MTLComputePipelineState
        do {
            state = try device.makeComputePipelineState(function: function)
        } catch {
            throw PMKProbeError.gpu("sg_probe pipeline compile failed: \(error)")
        }

        var input = [Float](repeating: 0, count: 128)
        var bq = [[Float]](repeating: [Float](repeating: 0, count: 8), count: 8)
        for r in 0..<8 {
            for c in 0..<8 {
                input[r * 8 + c] = Float(r * 8 + c)
                bq[r][c] = Float((r * 3 + c * 5) % 7 - 3)
                input[64 + r * 8 + c] = bq[r][c]
            }
        }
        var input8 = [Int8](repeating: 0, count: 256)
        for i in 0..<256 {
            input8[i] = Int8(truncatingIfNeeded: i - 128)
        }

        guard let inputBuffer = device.makeBuffer(bytes: input, length: input.count * MemoryLayout<Float>.stride),
              let outputBuffer = device.makeBuffer(length: 448 * MemoryLayout<Float>.stride),
              let int8Buffer = device.makeBuffer(bytes: input8, length: input8.count) else {
            throw PMKProbeError.resource("could not allocate layout probe buffers")
        }

        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw PMKProbeError.gpu("could not create layout probe command buffer")
        }
        encoder.setComputePipelineState(state)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBuffer(int8Buffer, offset: 0, index: 2)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        try commitAndBlock(commandBuffer, label: "sg_probe")

        let output = Array(UnsafeBufferPointer(start: outputBuffer.contents().bindMemory(to: Float.self, capacity: 448),
                                               count: 448))
        var failures: [String] = []
        func check(_ condition: Bool, _ message: String) {
            if !condition { failures.append(message) }
        }

        guard output.prefix(64).allSatisfy({ $0.isFinite && $0 >= 0 && $0 < 64 && $0.rounded(.towardZero) == $0 }) else {
            throw PMKProbeError.layout("invalid or non-integral fragment coordinates")
        }
        var coordinates = [[(Int, Int)]](repeating: [(0, 0), (0, 0)], count: 32)
        var seen = Set<Int>()
        for lane in 0..<32 {
            for element in 0..<2 {
                let value = Int(output[lane * 2 + element])
                coordinates[lane][element] = (value / 8, value % 8)
                seen.insert(value)
            }
        }
        check(seen.count == 64 && (0..<64).allSatisfy { output[$0] == Float(Int(output[$0])) },
              "simdgroup_load/thread_elements did not cover the 8x8 fragment exactly once")

        var inverseOK = true
        for lane in 0..<32 {
            for element in 0..<2 {
                let (row, col) = coordinates[lane][element]
                if output[64 + row * 8 + col] != Float(1000 + 2 * lane + element) {
                    inverseOK = false
                }
            }
        }
        check(inverseOK, "thread_elements injection did not round-trip through simdgroup_store")

        var mmaOK = true
        for lane in 0..<32 {
            for element in 0..<2 {
                let (row, col) = coordinates[lane][element]
                var sum: Float = 0
                for t in 0..<8 {
                    sum += Float(row * 8 + t) * bq[t][col]
                }
                if output[128 + lane * 2 + element] != sum {
                    mmaOK = false
                }
            }
        }
        check(mmaOK, "fp32 MMA accumulator layout differed from the load layout")
        check((0..<256).allSatisfy { output[192 + $0] == Float($0 - 128) },
              "char to float conversion was not exact for all 256 int8 values")

        var mlxOK = true
        for lane in 0..<32 {
            let qid = lane / 4
            let fm = (qid & 4) + ((lane / 2) % 4)
            let fn = (qid & 2) * 2 + (lane % 2) * 2
            if coordinates[lane][0] != (fm, fn) || coordinates[lane][1] != (fm, fn + 1) {
                mlxOK = false
            }
        }
        check(mlxOK, "layout did not match the committed MLX get_coord map")

        var rowsPattern: [Int]?
        var colsPattern: [Int]?
        var origins = Set<[Int]>()
        var cover = [Int](repeating: 0, count: 1024)
        var productOK = true
        var samePatternOK = true
        for lane in 0..<32 {
            var owned = Set<[Int]>()
            for i in 0..<4 {
                for j in 0..<4 {
                    for element in 0..<2 {
                        owned.insert([8 * i + coordinates[lane][element].0,
                                      8 * j + coordinates[lane][element].1])
                    }
                }
            }
            let rows = Set(owned.map { $0[0] }).sorted()
            let cols = Set(owned.map { $0[1] }).sorted()
            if owned.count != rows.count * cols.count || owned.count != 32 {
                productOK = false
            }
            for xy in owned {
                cover[xy[0] * 32 + xy[1]] += 1
            }
            let rowPattern = rows.map { $0 - rows[0] }
            let colPattern = cols.map { $0 - cols[0] }
            if rowsPattern == nil {
                rowsPattern = rowPattern
                colsPattern = colPattern
            } else if rowsPattern != rowPattern || colsPattern != colPattern {
                samePatternOK = false
            }
            origins.insert([rows[0], cols[0]])
        }
        check(productOK, "a lane's 32-element set was not a rows x cols product")
        check(samePatternOK, "lanes did not share one normalized pattern")
        check(cover.allSatisfy { $0 == 1 }, "lane sets did not partition the 32x32 simdgroup tile")

        let rows = rowsPattern ?? []
        let cols = colsPattern ?? []
        guard let rowShape = patternShape(rows), let colShape = patternShape(cols) else {
            throw PMKProbeError.layout("measured patterns are not legal Pearl PeriodicPatterns")
        }
        let h = rows.count
        let w = cols.count
        check(h % 2 == 0 && w % 2 == 0 && h * w >= 32 && h * w <= 256,
              "measured h=\(h), w=\(w) violated Pearl pattern size rules")
        let rowPeriod = rowShape[2].0 * rowShape[2].1
        let colPeriod = colShape[2].0 * colShape[2].1
        check(rowPeriod == 32 && colPeriod == 32, "measured pattern periods were \(rowPeriod), \(colPeriod), not 32")
        let validRows = (0..<32).filter { offsetIsValid(rowShape, $0) }
        let validCols = (0..<32).filter { offsetIsValid(colShape, $0) }
        var expectedOrigins = Set<[Int]>()
        for row in validRows {
            for col in validCols {
                expectedOrigins.insert([row, col])
            }
        }
        check(origins == expectedOrigins, "lane origins did not equal Pearl valid offsets in [0,32)^2")
        check(rows == pmkRowsPattern && cols == pmkColsPattern,
              "measured pattern rows \(rows) cols \(cols) did not match committed SG constants")

        if !failures.isEmpty {
            throw PMKProbeError.layout(failures.joined(separator: "; "))
        }
    }

    private func runKnownAnswerJob() throws -> Int {
        let vector = try ProbeVector.load()
        guard vector.a.count == vector.m * vector.k,
              vector.b.count == vector.k * vector.n,
              vector.tiles(for: kernel).count == (vector.m * vector.n / Int(kernel.tileElements)) * pmkSlotWords else {
            throw PMKProbeError.resource("malformed K3 \(kernel.rawValue) known-answer vector dimensions")
        }
        guard vector.m == 256, vector.n == 256, vector.k == 4096 else {
            throw PMKProbeError.resource("unexpected K3 probe vector shape \(vector.m)x\(vector.n)x\(vector.k)")
        }
        guard vector.m % kernel.tileM == 0, vector.n % kernel.tileN == 0, vector.k % 128 == 0 else {
            throw PMKProbeError.resource("K3 \(kernel.rawValue) probe vector shape violates production dispatch constraints")
        }

        guard let aBuffer = vector.a.withUnsafeBytes({ raw in
            raw.baseAddress.map { device.makeBuffer(bytes: $0, length: vector.a.count) } ?? nil
        }), let bBuffer = vector.b.withUnsafeBytes({ raw in
            raw.baseAddress.map { device.makeBuffer(bytes: $0, length: vector.b.count) } ?? nil
        }), let sinkBuffer = device.makeBuffer(length: 16) else {
            throw PMKProbeError.resource("could not allocate known-answer buffers")
        }

        let tileWords = vector.tiles(for: kernel)
        let tiles = (0..<(vector.m * vector.n / Int(kernel.tileElements))).map {
            Array(tileWords[($0 * pmkSlotWords)..<(($0 + 1) * pmkSlotWords)])
        }
        var checked = 0
        for testCase in vector.cases {
            checked += 1
            let params = PMKK3SGParams(m: vector.m, n: vector.n, k: vector.k,
                                       capBlock: testCase.capBlock, capShare: testCase.capShare,
                                       key: vector.key, boundBlock: testCase.boundBlock,
                                       boundShare: testCase.boundShare)
            let blockWords = (testCase.capBlock + pmkGuardSlots) * pmkSlotWords
            let shareWords = (testCase.capShare + pmkGuardSlots) * pmkSlotWords
            guard let counters = device.makeBuffer(length: 8),
                  let blocks = device.makeBuffer(length: blockWords * MemoryLayout<UInt32>.stride),
                  let shares = device.makeBuffer(length: shareWords * MemoryLayout<UInt32>.stride) else {
                throw PMKProbeError.resource("could not allocate slot buffers for case \(testCase.name)")
            }
            memset(counters.contents(), 0, 8)
            fillCanary(blocks, words: blockWords)
            fillCanary(shares, words: shareWords)

            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw PMKProbeError.gpu("could not create K3 \(kernel.rawValue) known-answer command buffer")
            }
            var parameterWords = params.words
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(aBuffer, offset: 0, index: 0)
            encoder.setBuffer(bBuffer, offset: 0, index: 1)
            encoder.setBytes(&parameterWords, length: 128, index: 2)
            encoder.setBuffer(counters, offset: 0, index: 3)
            encoder.setBuffer(blocks, offset: 0, index: 4)
            encoder.setBuffer(shares, offset: 0, index: 5)
            encoder.setBuffer(sinkBuffer, offset: 0, index: 6)
            encoder.dispatchThreadgroups(MTLSize(width: vector.n / kernel.tileN, height: vector.m / kernel.tileM, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: kernel.threadsPerThreadgroup, height: 1, depth: 1))
            encoder.endEncoding()
            try commitAndBlock(commandBuffer, label: "K3 \(kernel.rawValue) known-answer \(testCase.name)")

            var output = [UInt32]()
            output.append(contentsOf: readUInt32s(counters, words: 2))
            output.append(contentsOf: readUInt32s(blocks, words: blockWords))
            output.append(contentsOf: readUInt32s(shares, words: shareWords))
            if let failure = checkCase(tiles: tiles, blockBound: testCase.boundBlock,
                                       shareBound: testCase.boundShare,
                                       blockCapacity: testCase.capBlock,
                                       shareCapacity: testCase.capShare,
                                       output: output) {
                throw PMKProbeError.knownAnswer("case \(testCase.name): \(failure)")
            }
        }
        return checked
    }

    private func commitAndBlock(_ commandBuffer: MTLCommandBuffer, label: String) throws {
        let semaphore = DispatchSemaphore(value: 0)
        commandBuffer.addCompletedHandler { _ in semaphore.signal() }
        commandBuffer.commit()
        semaphore.wait()
        guard commandBuffer.status == .completed else {
            throw PMKProbeError.gpu("\(label): \(String(describing: commandBuffer.error))")
        }
    }
}

private struct ProbeCase {
    let name: String
    let capBlock: Int
    let capShare: Int
    let boundBlock: [UInt32]
    let boundShare: [UInt32]
}

private struct ProbeVector {
    let m: Int
    let n: Int
    let k: Int
    let key: [UInt32]
    let cases: [ProbeCase]
    let a: Data
    let b: Data
    let tiles: [UInt32]
    let tilesNA: [UInt32]

    func tiles(for kernel: K3Kernel) -> [UInt32] {
        kernel == .na ? tilesNA : tiles
    }

    static func load() throws -> ProbeVector {
        let base = try pmkResourceURL("probe/v1_256x256x4096")
        let jobURL = base.appendingPathComponent("job.json")
        let jobData: Data
        do {
            jobData = try Data(contentsOf: jobURL)
        } catch {
            throw PMKProbeError.resource("could not read \(jobURL.path): \(error)")
        }
        let json: [String: Any]
        do {
            guard let parsed = try JSONSerialization.jsonObject(with: jobData) as? [String: Any] else {
                throw PMKProbeError.resource("job.json is not an object")
            }
            json = parsed
        } catch let probeError as PMKProbeError {
            throw probeError
        } catch {
            throw PMKProbeError.resource("could not parse job.json: \(error)")
        }
        func int(_ key: String) throws -> Int {
            guard let value = json[key] as? NSNumber else { throw PMKProbeError.resource("job.json missing \(key)") }
            return value.intValue
        }
        let casesJSON = json["cases"] as? [[String: Any]] ?? []
        guard !casesJSON.isEmpty else {
            throw PMKProbeError.resource("job.json has no probe cases")
        }
        let cases: [ProbeCase] = try casesJSON.map { entry in
            guard let name = entry["name"] as? String,
                  let capBlock = entry["cap_block"] as? NSNumber,
                  let capShare = entry["cap_share"] as? NSNumber else {
                throw PMKProbeError.resource("malformed probe case metadata")
            }
            return ProbeCase(name: name, capBlock: capBlock.intValue, capShare: capShare.intValue,
                             boundBlock: try u32Array(entry["bound_block"], label: "\(name).bound_block"),
                             boundShare: try u32Array(entry["bound_share"], label: "\(name).bound_share"))
        }
        let aURL = base.appendingPathComponent("A.bin")
        let bURL = base.appendingPathComponent("B.bin")
        let tilesURL = base.appendingPathComponent("tiles.bin")
        let tilesNAURL = base.appendingPathComponent("tiles_na.bin")
        do {
            return ProbeVector(m: try int("m"), n: try int("n"), k: try int("k"),
                               key: try u32Array(json["key"], label: "key"),
                               cases: cases,
                               a: try Data(contentsOf: aURL),
                               b: try Data(contentsOf: bURL),
                               tiles: try readU32File(tilesURL),
                               tilesNA: try readU32File(tilesNAURL))
        } catch let probeError as PMKProbeError {
            throw probeError
        } catch {
            throw PMKProbeError.resource("could not read vector payload: \(error)")
        }
    }
}

private func u32Array(_ value: Any?, label: String) throws -> [UInt32] {
    guard let numbers = value as? [NSNumber], numbers.count == 8 else {
        throw PMKProbeError.resource("job.json \(label) must be 8 u32 words")
    }
    return numbers.map { $0.uint32Value }
}

private func readU32File(_ url: URL) throws -> [UInt32] {
    let data = try Data(contentsOf: url)
    guard data.count % MemoryLayout<UInt32>.stride == 0 else {
        throw PMKProbeError.resource("\(url.lastPathComponent) length is not a multiple of u32")
    }
    var out: [UInt32] = []
    out.reserveCapacity(data.count / 4)
    var offset = 0
    while offset < data.count {
        let word = UInt32(data[offset])
            | (UInt32(data[offset + 1]) << 8)
            | (UInt32(data[offset + 2]) << 16)
            | (UInt32(data[offset + 3]) << 24)
        out.append(word)
        offset += 4
    }
    return out
}

private func readUInt32s(_ buffer: MTLBuffer, words: Int) -> [UInt32] {
    Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: UInt32.self, capacity: words), count: words))
}

private func fillCanary(_ buffer: MTLBuffer, words: Int) {
    let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: words)
    for i in 0..<words {
        ptr[i] = pmkCanary
    }
}

private func checkCase(tiles: [[UInt32]], blockBound: [UInt32], shareBound: [UInt32],
                       blockCapacity: Int, shareCapacity: Int, output: [UInt32]) -> String? {
    var byXY: [[UInt32]: [UInt32]] = [:]
    for tile in tiles {
        byXY[[tile[0], tile[1]]] = tile
    }
    let blockWords = (blockCapacity + pmkGuardSlots) * pmkSlotWords
    let parts: [(label: String, counter: Int, capacity: Int, start: Int, bound: [UInt32])] = [
        ("block", Int(output[0]), blockCapacity, 2, blockBound),
        ("share", Int(output[1]), shareCapacity, 2 + blockWords, shareBound)
    ]
    var errors: [String] = []
    for part in parts {
        let expected = Set(tiles.filter { u256LE($0[18..<26], part.bound) }.map { [$0[0], $0[1]] })
        if part.counter != expected.count {
            errors.append("\(part.label) counter \(part.counter) != oracle \(expected.count)")
        }
        let written = min(part.counter, part.capacity)
        var seen = Set<[UInt32]>()
        for slotIndex in 0..<written {
            let lo = part.start + slotIndex * pmkSlotWords
            let slot = Array(output[lo..<(lo + pmkSlotWords)])
            let xy = [slot[0], slot[1]]
            guard let tile = byXY[xy] else {
                errors.append("\(part.label) slot \(slotIndex): \(xy) is not a Pearl tile origin")
                continue
            }
            if seen.contains(xy) {
                errors.append("\(part.label) slot \(slotIndex): duplicate tile \(xy)")
            }
            seen.insert(xy)
            if tile != slot {
                errors.append("\(part.label) slot \(slotIndex) tile \(xy): transcript/hash mismatch")
            }
            if !expected.contains(xy) {
                errors.append("\(part.label) slot \(slotIndex) tile \(xy): not a find per oracle")
            }
        }
        if part.counter <= part.capacity && seen != expected {
            errors.append("\(part.label) slot set != oracle find set (\(seen.count) vs \(expected.count))")
        }
        let guardStart = part.start + written * pmkSlotWords
        let guardEnd = part.start + (part.capacity + pmkGuardSlots) * pmkSlotWords
        if guardStart < guardEnd && !output[guardStart..<guardEnd].allSatisfy({ $0 == pmkCanary }) {
            errors.append("\(part.label): write outside the \(written) valid slots")
        }
    }
    return errors.isEmpty ? nil : errors.prefix(8).joined(separator: "; ")
}

private func u256LE(_ hash: ArraySlice<UInt32>, _ bound: [UInt32]) -> Bool {
    let hashWords = Array(hash)
    for index in stride(from: 7, through: 0, by: -1) {
        if hashWords[index] < bound[index] { return true }
        if hashWords[index] > bound[index] { return false }
    }
    return true
}

private func patternShape(_ pattern: [Int]) -> [(Int, Int)]? {
    guard pattern.first == 0, zip(pattern, pattern.dropFirst()).allSatisfy({ $0 < $1 }) else {
        return nil
    }
    var p = pattern
    var shape: [(Int, Int)] = []
    while p.count > 1 {
        var found = false
        for period in 1..<p.count where p.count % period == 0 {
            let stride = p[period]
            if (0..<(p.count - period)).allSatisfy({ p[$0] + stride == p[$0 + period] }) {
                shape.append((stride, p.count / period))
                p = Array(p[0..<period])
                found = true
                break
            }
        }
        if !found { return nil }
    }
    shape.reverse()
    let period = shape.last.map { $0.0 * $0.1 } ?? 1
    if shape.count > 3 { return nil }
    while shape.count < 3 {
        shape.append((period, 1))
    }
    return shape
}

private func offsetIsValid(_ shape: [(Int, Int)], _ offset: Int) -> Bool {
    var value = offset
    for (stride, length) in shape.reversed() {
        value %= stride * length
        if value >= stride {
            return false
        }
    }
    return true
}
