import CryptoKit
import Foundation
import Metal
import XCTest
@testable import PMK
import CPMK

final class NoiseTests: XCTestCase {
    func testGPUK1K2NoiseMatchesReferenceVectors() throws {
        let vectors = try NoiseVector.loadAll()
        XCTAssertEqual(vectors.count, 21)
        let context = try Context(requireAdmission: false)

        for vector in vectors {
            var rng = XorShift64Star(seed: vector.rngSeed)
            let aSeed = rng.seed32()
            let bSeed = rng.seed32()
            XCTAssertEqual(hex(aSeed), vector.aNoiseSeedHex, "\(vector.id) A seed")
            XCTAssertEqual(hex(bSeed), vector.bNoiseSeedHex, "\(vector.id) B seed")

            let rawA = rng.signalBytes(count: vector.m * vector.k)
            let rawBt = rng.signalBytes(count: vector.n * vector.k)
            XCTAssertEqual(sha256Hex(rawA), vector.rawASHA256, "\(vector.id) raw A")
            XCTAssertEqual(sha256Hex(rawBt), vector.rawBtSHA256, "\(vector.id) raw B^T")
            XCTAssertEqual(hex(rawA.prefix(256)), vector.rawAPrefixHex, "\(vector.id) raw A prefix")
            XCTAssertEqual(hex(rawBt.prefix(256)), vector.rawBtPrefixHex, "\(vector.id) raw B^T prefix")

            let rawABuffer = try makeBuffer(context: context, bytes: rawA, label: "\(vector.id) raw A")
            let rawBtBuffer = try makeBuffer(context: context, bytes: rawBt, label: "\(vector.id) raw B^T")
            var desc = pmk_job_desc()
            desc.abi_version = 1
            desc.m = UInt32(vector.m)
            desc.n = UInt32(vector.n)
            desc.k = UInt32(vector.k)
            desc.a = UnsafePointer(rawABuffer.contents().assumingMemoryBound(to: Int8.self))
            desc.bt = UnsafePointer(rawBtBuffer.contents().assumingMemoryBound(to: Int8.self))
            desc.a_bytes = UInt64(rawA.count)
            desc.bt_bytes = UInt64(rawBt.count)
            desc.a_seed = tuple8(wordsLE(aSeed))
            desc.b_seed = tuple8(wordsLE(bSeed))
            desc.block_capacity = 4
            desc.share_capacity = 64
            desc.cert_version = 3
            desc.rank = 128

            let job = try XCTUnwrap(Job(context: context, desc: desc, rawA: rawABuffer, rawBt: rawBtBuffer, reservation: 0))
            guard let commandBuffer = context.queue.makeCommandBuffer() else {
                XCTFail("could not allocate command buffer for \(vector.id)")
                continue
            }
            try job.encodeNoise(commandBuffer)
            try syncComplete(commandBuffer)

            let noisedA = Data(bytes: job.a.contents(), count: vector.m * vector.k)
            let noisedB = Data(bytes: job.b.contents(), count: vector.k * vector.n)
            XCTAssertEqual(sha256Hex(noisedA), vector.noisedASHA256, "\(vector.id) noised A")
            XCTAssertEqual(sha256Hex(noisedB), vector.noisedBSHA256, "\(vector.id) noised B")
            XCTAssertEqual(hex(noisedA.prefix(256)), vector.noisedAPrefixHex, "\(vector.id) noised A prefix")
            XCTAssertEqual(hex(noisedB.prefix(256)), vector.noisedBPrefixHex, "\(vector.id) noised B prefix")
        }
    }

    func testCPUAndGPUBlake3SingleBlockKeyedVectors() throws {
        let vectors = try Blake3Vector.loadAll()
        XCTAssertEqual(vectors.count, 16)
        let context = try Context(requireAdmission: false)
        let ps = try context.pipeline("pmk_hash_vector")
        for vector in vectors {
            let key = try Data(hexString: vector.keyHex)
            let block = try Data(hexString: vector.blockHex)
            XCTAssertEqual(block.count, 64, vector.id)
            let digest = CPUOracle.hash(words: wordsLE(block), key: wordsLE(key))
            XCTAssertEqual(hex(wordsBytesLE(digest)), vector.crateDigestHex, vector.id)
            XCTAssertEqual(vector.pearlDigestHex, vector.crateDigestHex, vector.id)
            let output = try XCTUnwrap(context.buffer(32))
            let cb = try XCTUnwrap(context.queue.makeCommandBuffer())
            let e = try XCTUnwrap(cb.makeComputeCommandEncoder())
            e.setComputePipelineState(ps)
            var messageWords = wordsLE(block), keyWords = wordsLE(key)
            e.setBytes(&messageWords, length: 64, index: 0)
            e.setBytes(&keyWords, length: 32, index: 1)
            e.setBuffer(output, offset: 0, index: 2)
            e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            e.endEncoding()
            try syncComplete(cb)
            XCTAssertEqual(hex(Data(bytes: output.contents(), count: 32)), vector.crateDigestHex, "GPU \(vector.id)")
        }
    }
}

private struct NoiseVector {
    let id: String
    let m: Int
    let n: Int
    let k: Int
    let rank: Int
    let rngSeed: UInt64
    let aNoiseSeedHex: String
    let bNoiseSeedHex: String
    let rawASHA256: String
    let rawBtSHA256: String
    let noisedASHA256: String
    let noisedBSHA256: String
    let rawAPrefixHex: String
    let rawBtPrefixHex: String
    let noisedAPrefixHex: String
    let noisedBPrefixHex: String

    static func loadAll() throws -> [NoiseVector] {
        try resourceLines("noise_vectors.jsonl").map { line in
            let object = try jsonObject(line)
            let vector = NoiseVector(
                id: try string(object, "id"),
                m: try int(object, "m"),
                n: try int(object, "n"),
                k: try int(object, "k"),
                rank: try int(object, "rank"),
                rngSeed: try uint64Hex(object, "rng_seed"),
                aNoiseSeedHex: try string(object, "a_noise_seed"),
                bNoiseSeedHex: try string(object, "b_noise_seed"),
                rawASHA256: try string(object, "raw_a_sha256"),
                rawBtSHA256: try string(object, "raw_bt_sha256"),
                noisedASHA256: try string(object, "noised_a_sha256"),
                noisedBSHA256: try string(object, "noised_b_sha256"),
                rawAPrefixHex: try string(object, "raw_a_prefix_hex"),
                rawBtPrefixHex: try string(object, "raw_bt_prefix_hex"),
                noisedAPrefixHex: try string(object, "noised_a_prefix_hex"),
                noisedBPrefixHex: try string(object, "noised_b_prefix_hex")
            )
            XCTAssertEqual(vector.rank, 128, vector.id)
            return vector
        }
    }
}

private struct Blake3Vector {
    let id: String
    let keyHex: String
    let blockHex: String
    let pearlDigestHex: String
    let crateDigestHex: String

    static func loadAll() throws -> [Blake3Vector] {
        let data = try Data(contentsOf: resourceURL("blake3_single_block_keyed.json"))
        guard let array = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw PMKError("malformed blake3_single_block_keyed.json")
        }
        return try array.map { object in
            Blake3Vector(id: try string(object, "id"),
                         keyHex: try string(object, "key_hex"),
                         blockHex: try string(object, "block_hex"),
                         pearlDigestHex: try string(object, "pearl_digest_hex"),
                         crateDigestHex: try string(object, "crate_digest_hex"))
        }
    }
}

private struct XorShift64Star {
    var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        var x = state
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        state = x
        return x &* 0x2545_F491_4F6C_DD1D
    }

    mutating func byte() -> UInt8 {
        UInt8(truncatingIfNeeded: next() >> 56)
    }

    mutating func signalByte() -> UInt8 {
        UInt8(bitPattern: Int8(truncatingIfNeeded: Int(next() % 129) - 64))
    }

    mutating func seed32() -> Data {
        var out = Data()
        out.reserveCapacity(32)
        for _ in 0..<32 {
            out.append(byte())
        }
        return out
    }

    mutating func signalBytes(count: Int) -> Data {
        var out = Data()
        out.reserveCapacity(count)
        for _ in 0..<count {
            out.append(signalByte())
        }
        return out
    }
}

private func makeBuffer(context: Context, bytes: Data, label: String) throws -> MTLBuffer {
    guard let buffer = context.device.makeBuffer(length: bytes.count, options: .storageModeShared) else {
        throw PMKError("could not allocate \(label)")
    }
    _ = bytes.withUnsafeBytes { raw in
        memcpy(buffer.contents(), raw.baseAddress!, bytes.count)
    }
    return buffer
}

private func sha256Hex(_ data: Data) -> String {
    hex(Data(SHA256.hash(data: data)))
}

private func wordsLE(_ data: Data) -> [UInt32] {
    precondition(data.count % 4 == 0)
    var out: [UInt32] = []
    out.reserveCapacity(data.count / 4)
    for offset in stride(from: 0, to: data.count, by: 4) {
        out.append(UInt32(data[offset])
            | (UInt32(data[offset + 1]) << 8)
            | (UInt32(data[offset + 2]) << 16)
            | (UInt32(data[offset + 3]) << 24))
    }
    return out
}

private func wordsBytesLE(_ words: [UInt32]) -> Data {
    var out = Data()
    out.reserveCapacity(words.count * 4)
    for word in words {
        out.append(UInt8(truncatingIfNeeded: word))
        out.append(UInt8(truncatingIfNeeded: word >> 8))
        out.append(UInt8(truncatingIfNeeded: word >> 16))
        out.append(UInt8(truncatingIfNeeded: word >> 24))
    }
    return out
}

private func tuple8(_ words: [UInt32]) -> (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) {
    precondition(words.count == 8)
    return (words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7])
}

private func resourceLines(_ name: String) throws -> [String] {
    let text = try String(contentsOf: resourceURL(name), encoding: .utf8)
    return text.split(separator: "\n").map(String.init)
}

private func resourceURL(_ name: String) throws -> URL {
    let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("reference/vectors")
        .appendingPathComponent(name)
    guard FileManager.default.fileExists(atPath: root.path) else {
        throw PMKError("missing reference vector \(root.path)")
    }
    return root
}

private func jsonObject(_ line: String) throws -> [String: Any] {
    let data = Data(line.utf8)
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw PMKError("malformed JSON line")
    }
    return object
}

private func int(_ object: [String: Any], _ key: String) throws -> Int {
    guard let value = object[key] as? NSNumber else { throw PMKError("missing int \(key)") }
    return value.intValue
}

private func string(_ object: [String: Any], _ key: String) throws -> String {
    guard let value = object[key] as? String else { throw PMKError("missing string \(key)") }
    return value
}

private func uint64Hex(_ object: [String: Any], _ key: String) throws -> UInt64 {
    let value = try string(object, key)
    guard let parsed = UInt64(value, radix: 16) else { throw PMKError("invalid hex u64 \(key)") }
    return parsed
}

private func hex<S: Sequence>(_ bytes: S) -> String where S.Element == UInt8 {
    bytes.map { String(format: "%02x", $0) }.joined()
}

private extension Data {
    init(hexString: String) throws {
        guard hexString.count % 2 == 0 else { throw PMKError("odd-length hex string") }
        var bytes = Data()
        bytes.reserveCapacity(hexString.count / 2)
        var index = hexString.startIndex
        while index < hexString.endIndex {
            let next = hexString.index(index, offsetBy: 2)
            guard let byte = UInt8(hexString[index..<next], radix: 16) else {
                throw PMKError("invalid hex byte")
            }
            bytes.append(byte)
            index = next
        }
        self = bytes
    }
}
