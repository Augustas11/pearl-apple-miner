import Foundation

public enum CPUOracle {
    public static let rank = 128
    public static let slotWords = 26
    public static let rowsPattern = [0, 8, 16, 24]
    public static let colsPattern = [0, 1, 8, 9, 16, 17, 24, 25]

    private static let iv: [UInt32] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
    ]
    private static let messagePermutation = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
    private static let keyedHash: UInt32 = 1 << 4
    private static let chunkStart: UInt32 = 1 << 0
    private static let chunkEnd: UInt32 = 1 << 1
    private static let root: UInt32 = 1 << 3

    public static func tiles(
        m: Int,
        n: Int,
        k: Int,
        a: UnsafePointer<Int8>,
        b: UnsafePointer<Int8>,
        key: [UInt32],
        block: [UInt32],
        share: [UInt32],
        blockCapacity: Int = Int.max,
        shareCapacity: Int = Int.max
    ) -> (block: [[UInt32]], share: [[UInt32]]) {
        precondition(m > 0 && n > 0 && k > 0)
        precondition(k % rank == 0, "K3-SG oracle requires k % 128 == 0")
        precondition(key.count == 8 && block.count == 8 && share.count == 8)
        precondition(m % 32 == 0 && n % 32 == 0, "K3-SG pattern period is 32")

        let rowOrigins = validOffsets(for: rowsPattern, total: m)
        let colOrigins = validOffsets(for: colsPattern, total: n)

        var blockSlots: [[UInt32]] = []
        var shareSlots: [[UInt32]] = []
        blockSlots.reserveCapacity(min(rowOrigins.count * colOrigins.count, Swift.max(0, blockCapacity)))
        shareSlots.reserveCapacity(min(rowOrigins.count * colOrigins.count, Swift.max(0, shareCapacity)))

        for rowOrigin in rowOrigins {
            for colOrigin in colOrigins {
                let state = transcript(m: m, n: n, k: k, a: a, b: b, rowOrigin: rowOrigin, colOrigin: colOrigin)
                let h = hash(words: state.transcript, key: key)
                let slot = [UInt32(rowOrigin), UInt32(colOrigin)] + state.transcript + h
                if lessEqual(h, block), blockSlots.count < blockCapacity {
                    blockSlots.append(slot)
                }
                if lessEqual(h, share), shareSlots.count < shareCapacity {
                    shareSlots.append(slot)
                }
            }
        }
        return (blockSlots, shareSlots)
    }

    public static func hash(words: [UInt32], key: [UInt32]) -> [UInt32] {
        precondition(words.count == 16)
        precondition(key.count == 8)
        return compress(block: words, chainingValue: key)
    }

    public static func lessEqual(_ hash: [UInt32], _ bound: [UInt32]) -> Bool {
        precondition(hash.count == 8 && bound.count == 8)
        for i in stride(from: 7, through: 0, by: -1) {
            if hash[i] < bound[i] { return true }
            if hash[i] > bound[i] { return false }
        }
        return true
    }

    private struct TileState {
        var acc = [Int32](repeating: 0, count: 32)
        var transcript = [UInt32](repeating: 0, count: 16)
    }

    private static func transcript(
        m: Int,
        n: Int,
        k: Int,
        a: UnsafePointer<Int8>,
        b: UnsafePointer<Int8>,
        rowOrigin: Int,
        colOrigin: Int
    ) -> TileState {
        var state = TileState()
        for chunkStart in stride(from: 0, to: k, by: rank) {
            let transcriptIndex = (chunkStart / rank) & 15
            var x: UInt32 = 0
            for (rowSlot, dr) in rowsPattern.enumerated() {
                let row = rowOrigin + dr
                let aBase = row * k + chunkStart
                for dc in colsPattern {
                    let col = colOrigin + dc
                    var sum = Int32(0)
                    var ai = aBase
                    var bi = chunkStart * n + col
                    for _ in 0..<rank {
                        sum &+= Int32(a[ai]) * Int32(b[bi])
                        ai += 1
                        bi += n
                    }
                    let flat = rowSlot * 8 + columnPatternIndex(dc)
                    state.acc[flat] &+= sum
                    x ^= UInt32(bitPattern: state.acc[flat])
                }
            }
            state.transcript[transcriptIndex] =
                rotateLeft(state.transcript[transcriptIndex], by: 13) ^ x
        }
        return state
    }

    private static func patternShape(_ pattern: [Int]) -> [(stride: Int, length: Int)] {
        precondition(pattern.first == 0)
        precondition(zip(pattern, pattern.dropFirst()).allSatisfy { $0 < $1 })
        var remaining = pattern
        var shape: [(stride: Int, length: Int)] = []
        while remaining.count > 1 {
            var found: (stride: Int, length: Int)?
            for period in 1..<remaining.count where remaining.count % period == 0 {
                let stride = remaining[period]
                var ok = true
                for i in 0..<(remaining.count - period) where remaining[i] + stride != remaining[i + period] {
                    ok = false
                    break
                }
                if ok {
                    found = (stride, remaining.count / period)
                    remaining = Array(remaining[..<period])
                    break
                }
            }
            guard let dim = found else { preconditionFailure("pattern is not periodic") }
            shape.append(dim)
        }
        shape.reverse()
        precondition(shape.count <= 3)
        let period = shape.last.map { $0.stride * $0.length } ?? 1
        while shape.count < 3 {
            shape.append((period, 1))
        }
        return shape
    }

    private static func offsetIsValid(_ shape: [(stride: Int, length: Int)], _ offset: Int) -> Bool {
        var value = offset
        for dim in shape.reversed() {
            value %= dim.stride * dim.length
            if value >= dim.stride { return false }
        }
        return true
    }

    private static func validOffsets(for pattern: [Int], total: Int) -> [Int] {
        let shape = patternShape(pattern)
        let patternPeriod = shape[2].stride * shape[2].length
        precondition(total % patternPeriod == 0)
        return (0..<total).filter { offsetIsValid(shape, $0) }
    }

    private static func columnPatternIndex(_ colDelta: Int) -> Int {
        switch colDelta {
        case 0: return 0
        case 1: return 1
        case 8: return 2
        case 9: return 3
        case 16: return 4
        case 17: return 5
        case 24: return 6
        case 25: return 7
        default: preconditionFailure("invalid SG column pattern delta \(colDelta)")
        }
    }

    private static func compress(block: [UInt32], chainingValue: [UInt32]) -> [UInt32] {
        var state = [UInt32](repeating: 0, count: 16)
        for i in 0..<8 { state[i] = chainingValue[i] }
        for i in 0..<4 { state[8 + i] = iv[i] }
        state[12] = 0
        state[13] = 0
        state[14] = 64
        state[15] = keyedHash | chunkStart | chunkEnd | root

        var message = block
        for _ in 0..<6 {
            round(&state, message)
            message = messagePermutation.map { message[$0] }
        }
        round(&state, message)

        var out = [UInt32](repeating: 0, count: 8)
        for i in 0..<8 {
            out[i] = state[i] ^ state[i + 8]
        }
        return out
    }

    private static func round(_ v: inout [UInt32], _ m: [UInt32]) {
        g(&v, 0, 4, 8, 12, m[0], m[1])
        g(&v, 1, 5, 9, 13, m[2], m[3])
        g(&v, 2, 6, 10, 14, m[4], m[5])
        g(&v, 3, 7, 11, 15, m[6], m[7])
        g(&v, 0, 5, 10, 15, m[8], m[9])
        g(&v, 1, 6, 11, 12, m[10], m[11])
        g(&v, 2, 7, 8, 13, m[12], m[13])
        g(&v, 3, 4, 9, 14, m[14], m[15])
    }

    private static func g(
        _ v: inout [UInt32],
        _ a: Int,
        _ b: Int,
        _ c: Int,
        _ d: Int,
        _ x: UInt32,
        _ y: UInt32
    ) {
        v[a] = v[a] &+ v[b] &+ x
        v[d] = rotateRight(v[d] ^ v[a], by: 16)
        v[c] = v[c] &+ v[d]
        v[b] = rotateRight(v[b] ^ v[c], by: 12)
        v[a] = v[a] &+ v[b] &+ y
        v[d] = rotateRight(v[d] ^ v[a], by: 8)
        v[c] = v[c] &+ v[d]
        v[b] = rotateRight(v[b] ^ v[c], by: 7)
    }

    private static func rotateLeft(_ x: UInt32, by n: UInt32) -> UInt32 {
        (x << n) | (x >> (32 - n))
    }

    private static func rotateRight(_ x: UInt32, by n: UInt32) -> UInt32 {
        (x >> n) | (x << (32 - n))
    }
}

extension CPUOracle {
    /// Independent Pearl noise reconstruction for overflow recovery from retained raw inputs.
    static func noised(m: Int, n: Int, k: Int, rawA: UnsafePointer<Int8>, rawBt: UnsafePointer<Int8>,
                       aKey: [UInt32], bKey: [UInt32]) -> (a: [Int8], b: [Int8]) {
        func matrices(_ rows: Int, _ key: [UInt32], _ aSide: Bool) -> ([Int8], [(Int, Int)]) {
            var message = [UInt32](repeating: 0, count: 16)
            message[8] = aSide ? 0x65745f41 : 0x65745f42; message[9] = 0x726f736e
            var dense = [Int8](repeating: 0, count: rows * 128)
            for block in 0..<(dense.count / 32) {
                message[0] = UInt32(block + 1)
                let digest = hash(words: message, key: key)
                for i in 0..<32 { dense[block * 32 + i] = Int8((digest[i / 4] >> (8 * (i % 4))) & 63) - 32 }
            }
            message[0] = 0
            var sparse = [(Int, Int)](); sparse.reserveCapacity(k)
            for block in 0..<(k / 8) {
                message[1] = UInt32(block + 1)
                for u in hash(words: message, key: key) {
                    let first = u & 127, second = first ^ (1 + UInt32((UInt64(127) * UInt64(u)) >> 32))
                    sparse.append((Int(first), Int(second)))
                }
            }
            return (dense, sparse)
        }
        let (al, ar) = matrices(m, aKey, true), (br, bl) = matrices(n, bKey, false)
        var a = [Int8](repeating: 0, count: m * k), b = [Int8](repeating: 0, count: n * k)
        for row in 0..<m { for col in 0..<k {
            a[row * k + col] = Int8(Int(rawA[row * k + col]) + Int(al[row * 128 + ar[col].0]) - Int(al[row * 128 + ar[col].1]))
        } }
        for row in 0..<k { for col in 0..<n {
            b[row * n + col] = Int8(Int(rawBt[col * k + row]) + Int(br[col * 128 + bl[row].0]) - Int(br[col * 128 + bl[row].1]))
        } }
        return (a, b)
    }
}
