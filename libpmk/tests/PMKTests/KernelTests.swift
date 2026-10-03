import XCTest
import Metal
@testable import PMK
import CPMK

final class KernelTests: XCTestCase {
    func testK3SGBenchVectorsMatchOracleSlotsAndCounters() {
        do {
            print("G3: initializing production context")
            let context = try Context(requireAdmission: false)
            print("G3: context ready")
            for name in ["v1_256x256x4096", "v2_256x256x4096_pm127", "v3_128x128x65536", "v4_pearl_c64"] {
                print("G3: loading \(name)")
                let vector = try K3SGVector.load(name)
                try runGPUCases(vector, context: context)
                print("G3: PASS \(name) \(vector.cases.count) cases")
            }
        } catch {
            XCTFail("G3: \(error)")
        }
    }

    func testCPUOracleMatchesFullJobTilesForReleaseVectors() throws {
        for name in ["v1_256x256x4096", "v3_128x128x65536"] {
            let vector = try K3SGVector.load(name)
            vector.a.withUnsafeBytes { aRaw in
                vector.b.withUnsafeBytes { bRaw in
                    guard let aBase = aRaw.baseAddress?.assumingMemoryBound(to: Int8.self),
                          let bBase = bRaw.baseAddress?.assumingMemoryBound(to: Int8.self) else {
                        XCTFail("empty vector buffers for \(name)")
                        return
                    }
                    let maxBlock = vector.maxBound(\.boundBlock)
                    let maxShare = vector.maxBound(\.boundShare)
                    let oracle = CPUOracle.tiles(m: vector.m, n: vector.n, k: vector.k,
                                                 a: aBase, b: bBase,
                                                 key: vector.key,
                                                 block: maxBlock,
                                                 share: maxShare)
                    for testCase in vector.cases {
                        XCTAssertEqual(oracle.block.filter { CPUOracle.lessEqual(Array($0[18..<26]), testCase.boundBlock) },
                                       vector.expectedSlots(bound: testCase.boundBlock),
                                       "\(name) \(testCase.name) block oracle")
                        XCTAssertEqual(oracle.share.filter { CPUOracle.lessEqual(Array($0[18..<26]), testCase.boundShare) },
                                       vector.expectedSlots(bound: testCase.boundShare),
                                       "\(name) \(testCase.name) share oracle")
                    }
                }
            }
        }
    }

    func testInjectedEncoderFailureSurfacesDiagnosticText() throws {
        let (job, _) = try makeVectorJob()
        try job.prepareNoise()
        XCTAssertThrowsError(try job.encodeNoiseStages { _ in
            throw PMKError("injected command-buffer factory failure")
        }) { error in
            XCTAssertTrue(String(describing: error).contains("injected command-buffer factory failure"))
        }
    }

    func testCompletionFailureRecordsJobDiagnostic() throws {
        let (job, context) = try makeVectorJob()
        let commandBuffer = try XCTUnwrap(context.queue.makeCommandBuffer())
        job.complete(commandBuffer)
        XCTAssertEqual(job.result.status, Int32(PMK_GPU_FAILED))
        XCTAssertTrue(job.diagnostic.contains("command buffer status"))
        var error = [CChar](repeating: 0, count: 256)
        let opaque = Unmanaged.passUnretained(job).toOpaque()
        XCTAssertEqual(pmkJobError(opaque, &error, UInt64(error.count)), 0)
        XCTAssertTrue(String(cString: error).contains("command buffer status"))
    }

    private func runGPUCases(_ vector: K3SGVector, context: Context) throws {
        guard let rawA = context.device.makeBuffer(length: vector.a.count, options: .storageModeShared),
              let rawBt = context.device.makeBuffer(length: vector.b.count, options: .storageModeShared) else {
            XCTFail("could not allocate raw vector buffers for \(vector.name)")
            return
        }
        vector.a.withUnsafeBytes { raw in
            _ = memcpy(rawA.contents(), raw.baseAddress!, vector.a.count)
        }
        vector.b.withUnsafeBytes { raw in
            _ = memcpy(rawBt.contents(), raw.baseAddress!, vector.b.count)
        }

        for testCase in vector.cases {
            var desc = pmk_job_desc()
            desc.abi_version = 1
            desc.m = UInt32(vector.m)
            desc.n = UInt32(vector.n)
            desc.k = UInt32(vector.k)
            desc.a = UnsafePointer(rawA.contents().assumingMemoryBound(to: Int8.self))
            desc.bt = UnsafePointer(rawBt.contents().assumingMemoryBound(to: Int8.self))
            desc.a_bytes = UInt64(vector.a.count)
            desc.bt_bytes = UInt64(vector.b.count)
            desc.a_seed = tuple8(vector.key)
            desc.b_seed = tuple8([UInt32](repeating: 0, count: 8))
            desc.block_bound = tuple8(testCase.boundBlock)
            desc.share_bound = tuple8(testCase.boundShare)
            desc.block_capacity = UInt32(testCase.capBlock)
            desc.share_capacity = UInt32(testCase.capShare)
            desc.cert_version = 3
            desc.rank = 128
            desc.job_id = UInt64(bitPattern: Int64(testCase.name.hashValue))

            let job = try XCTUnwrap(Job(context: context, desc: desc, rawA: rawA, rawBt: rawBt, reservation: 0))
            _ = memcpy(job.a.contents(), rawA.contents(), vector.a.count)
            _ = memcpy(job.b.contents(), rawBt.contents(), vector.b.count)
            guard let commandBuffer = context.queue.makeCommandBuffer() else {
                XCTFail("could not allocate command buffer for \(vector.name) \(testCase.name)")
                return
            }
            try job.encodeK3(commandBuffer)
            try syncComplete(commandBuffer)

            let counters = readU32(job.ctr, count: 2)
            let expectedBlocks = vector.expectedSlots(bound: testCase.boundBlock)
            let expectedShares = vector.expectedSlots(bound: testCase.boundShare)
            XCTAssertEqual(counters[0], UInt32(expectedBlocks.count), "\(vector.name) \(testCase.name) block counter")
            XCTAssertEqual(counters[1], UInt32(expectedShares.count), "\(vector.name) \(testCase.name) share counter")

            try assertStoredSlots(buffer: job.blocks, capacity: testCase.capBlock, counter: Int(counters[0]),
                                  expected: expectedBlocks, label: "\(vector.name) \(testCase.name) block")
            try assertStoredSlots(buffer: job.shares, capacity: testCase.capShare, counter: Int(counters[1]),
                                  expected: expectedShares, label: "\(vector.name) \(testCase.name) share")
        }
    }

    private func makeVectorJob() throws -> (Job, Context) {
        let context = try Context(requireAdmission: false)
        let m = 64, n = 64, k = 2048
        let rawBytes = m * k
        let btBytes = n * k
        let rawA = try XCTUnwrap(context.device.makeBuffer(length: rawBytes, options: .storageModeShared))
        let rawBt = try XCTUnwrap(context.device.makeBuffer(length: btBytes, options: .storageModeShared))
        memset(rawA.contents(), 0, rawBytes)
        memset(rawBt.contents(), 0, btBytes)
        var desc = pmk_job_desc()
        desc.abi_version = 1
        desc.m = UInt32(m)
        desc.n = UInt32(n)
        desc.k = UInt32(k)
        desc.a = UnsafePointer(rawA.contents().assumingMemoryBound(to: Int8.self))
        desc.bt = UnsafePointer(rawBt.contents().assumingMemoryBound(to: Int8.self))
        desc.a_bytes = UInt64(rawBytes)
        desc.bt_bytes = UInt64(btBytes)
        desc.a_seed = tuple8([UInt32](repeating: 1, count: 8))
        desc.b_seed = tuple8([UInt32](repeating: 2, count: 8))
        desc.block_bound = tuple8([UInt32](repeating: 0, count: 8))
        desc.share_bound = tuple8([UInt32](repeating: 0, count: 8))
        desc.block_capacity = 4
        desc.share_capacity = 64
        desc.cert_version = 3
        desc.rank = 128
        desc.job_id = 999
        let reservation = rawBytes + btBytes + (m + n) * 128 + k * 16 + ((4 + 64) * 104) + guardWords * 8 + 24
        return (try XCTUnwrap(Job(context: context, desc: desc, rawA: rawA, rawBt: rawBt,
                                  reservation: reservation)), context)
    }

    private func assertStoredSlots(buffer: MTLBuffer, capacity: Int, counter: Int,
                                   expected: [[UInt32]], label: String) throws {
        let words = readU32(buffer, count: buffer.length / 4)
        let stored = min(counter, capacity)
        let expectedByXY = Dictionary(uniqueKeysWithValues: expected.map { (slotXY($0), $0) })
        var actual: [[UInt32]] = []
        for slot in 0..<stored {
            let start = slot * slotWords
            actual.append(Array(words[start..<(start + slotWords)]))
        }
        var seen = Set<[UInt32]>()
        for (index, slot) in actual.enumerated() {
            let xy = slotXY(slot)
            XCTAssertTrue(seen.insert(xy).inserted, "\(label) duplicate stored slot \(index) at \(xy)")
            XCTAssertEqual(slot, expectedByXY[xy], "\(label) stored slot \(index) \(xy)")
        }
        if counter <= capacity {
            XCTAssertEqual(seen, Set(expected.map(slotXY)), "\(label) complete non-overflow slot set")
        } else {
            XCTAssertTrue(seen.isSubset(of: Set(expected.map(slotXY))), "\(label) overflow slot subset")
        }

        let guardStart = capacity * slotWords
        let guardEnd = guardStart + guardWords
        XCTAssertTrue(words[guardStart..<guardEnd].allSatisfy { $0 == canary }, "\(label) guard canary")
    }
}

private struct K3SGCase {
    let name: String
    let capBlock: Int
    let capShare: Int
    let boundBlock: [UInt32]
    let boundShare: [UInt32]
}

private struct K3SGVector {
    let name: String
    let m: Int
    let n: Int
    let k: Int
    let key: [UInt32]
    let cases: [K3SGCase]
    let a: Data
    let b: Data
    let tiles: [[UInt32]]

    static func load(_ name: String) throws -> K3SGVector {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let dir = root.appendingPathComponent("bench/k3sg/studio/vectors/\(name)")
        let jobURL = dir.appendingPathComponent("job.json")
        let job = try JSONSerialization.jsonObject(with: Data(contentsOf: jobURL)) as? [String: Any]
        guard let job else { throw PMKError("malformed \(jobURL.path)") }
        guard let casesJSON = job["cases"] as? [[String: Any]] else {
            throw PMKError("\(name) missing cases")
        }
        let m = try int(job, "m")
        let n = try int(job, "n")
        let k = try int(job, "k")
        let tileWords = try readU32File(dir.appendingPathComponent("tiles.bin"))
        let tileCount = m * n / 32
        guard tileWords.count == tileCount * slotWords else {
            throw PMKError("\(name) tiles.bin word count \(tileWords.count) != \(tileCount * slotWords)")
        }
        let tiles = (0..<tileCount).map { index in
            Array(tileWords[(index * slotWords)..<((index + 1) * slotWords)])
        }
        return K3SGVector(
            name: name,
            m: m,
            n: n,
            k: k,
            key: try u32Array(job["key"], label: "\(name).key"),
            cases: try casesJSON.map { entry in
                K3SGCase(name: try string(entry, "name"),
                         capBlock: try int(entry, "cap_block"),
                         capShare: try int(entry, "cap_share"),
                         boundBlock: try u32Array(entry["bound_block"], label: "\(name).bound_block"),
                         boundShare: try u32Array(entry["bound_share"], label: "\(name).bound_share"))
            },
            a: try Data(contentsOf: dir.appendingPathComponent("A.bin")),
            b: try Data(contentsOf: dir.appendingPathComponent("B.bin")),
            tiles: tiles
        )
    }

    func expectedSlots(bound: [UInt32]) -> [[UInt32]] {
        tiles.filter { CPUOracle.lessEqual(Array($0[18..<26]), bound) }
    }

    func maxBound(_ keyPath: KeyPath<K3SGCase, [UInt32]>) -> [UInt32] {
        cases.map { $0[keyPath: keyPath] }.max(by: u256LessThan) ?? [UInt32](repeating: 0, count: 8)
    }
}

private func slotXY(_ slot: [UInt32]) -> [UInt32] {
    [slot[0], slot[1]]
}

private func u256LessThan(_ lhs: [UInt32], _ rhs: [UInt32]) -> Bool {
    for index in stride(from: 7, through: 0, by: -1) {
        if lhs[index] < rhs[index] { return true }
        if lhs[index] > rhs[index] { return false }
    }
    return false
}

private func readU32(_ buffer: MTLBuffer, count: Int) -> [UInt32] {
    Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: UInt32.self, capacity: count), count: count))
}

private func readU32File(_ url: URL) throws -> [UInt32] {
    let data = try Data(contentsOf: url)
    guard data.count % 4 == 0 else { throw PMKError("\(url.path) is not u32-aligned") }
    var words: [UInt32] = []
    words.reserveCapacity(data.count / 4)
    var offset = 0
    while offset < data.count {
        words.append(UInt32(data[offset])
            | (UInt32(data[offset + 1]) << 8)
            | (UInt32(data[offset + 2]) << 16)
            | (UInt32(data[offset + 3]) << 24))
        offset += 4
    }
    return words
}

private func int(_ object: [String: Any], _ key: String) throws -> Int {
    guard let value = object[key] as? NSNumber else { throw PMKError("missing int \(key)") }
    return value.intValue
}

private func string(_ object: [String: Any], _ key: String) throws -> String {
    guard let value = object[key] as? String else { throw PMKError("missing string \(key)") }
    return value
}

private func u32Array(_ value: Any?, label: String) throws -> [UInt32] {
    guard let numbers = value as? [NSNumber], numbers.count == 8 else {
        throw PMKError("\(label) must be 8 u32 words")
    }
    return numbers.map { $0.uint32Value }
}

private func tuple8(_ words: [UInt32]) -> (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) {
    precondition(words.count == 8)
    return (words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7])
}
