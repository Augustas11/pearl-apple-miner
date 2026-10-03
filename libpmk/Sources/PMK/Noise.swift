import Metal

extension Job {
    func encodeNoise(_ cb: MTLCommandBuffer) throws {
        try prepareNoise()
        try encodeNoiseStages { name in
            _ = name
            return cb
        }
    }

    func prepareNoise() throws {
        let m = Int(desc.m), n = Int(desc.n), k = Int(desc.k)
        guard let al = context.buffer(m * 128), let br = context.buffer(n * 128),
              let ar = context.buffer(k * 8), let bl = context.buffer(k * 8) else {
            throw PMKError("Noise buffer allocation failed")
        }
        scratch = [al, br, ar, bl]
    }

    func encodeNoiseStages(commandBufferForStage: (String) throws -> MTLCommandBuffer) throws {
        let m = Int(desc.m), n = Int(desc.n), k = Int(desc.k)
        guard scratch.count == 4 else { throw PMKError("Noise buffers missing") }
        let al = scratch[0], br = scratch[1], ar = scratch[2], bl = scratch[3]
        var params: [UInt32] = [desc.m, desc.n, desc.k, 128]
        params += words(desc.a_seed); params += words(desc.b_seed)
        func encode(_ name: String, _ buffers: [MTLBuffer], _ total: Int) throws {
            let ps = try context.pipeline(name)
            let cb = try commandBufferForStage(name)
            guard let e = cb.makeComputeCommandEncoder() else { throw PMKError("No noise encoder") }
            e.setComputePipelineState(ps)
            for (i, buf) in buffers.enumerated() { e.setBuffer(buf, offset: 0, index: i) }
            e.setBytes(&params, length: 80, index: buffers.count)
            e.dispatchThreads(MTLSize(width: total, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, ps.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            e.endEncoding()
        }
        func encodeApplyBTiled() throws {
            let name = "noise_apply_b_tiled"
            let ps = try context.pipeline(name)
            let cb = try commandBufferForStage(name)
            guard let e = cb.makeComputeCommandEncoder() else { throw PMKError("No noise encoder") }
            e.setComputePipelineState(ps)
            for (i, buf) in [rawBt, br, bl, b].enumerated() { e.setBuffer(buf, offset: 0, index: i) }
            e.setBytes(&params, length: 80, index: 4)
            e.dispatchThreadgroups(MTLSize(width: (n + 31) / 32, height: (k + 31) / 32, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
            e.endEncoding()
        }
        try encode("noise_dense_a", [al], (m * 128 + 31) / 32)
        try encode("noise_dense_b", [br], (n * 128 + 31) / 32)
        try encode("noise_sparse_a", [ar], (k + 7) / 8)
        try encode("noise_sparse_b", [bl], (k + 7) / 8)
        try encode("noise_apply_a", [rawA, al, ar, a], m * k)
        try encodeApplyBTiled()
    }
}
