import Metal
let src = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp::tensor_ops;
kernel void probe(device int8_t* a [[buffer(0)]], device int8_t* b [[buffer(1)]], device int* o [[buffer(2)]],
                  uint tid [[thread_index_in_threadgroup]]) {
  tensor<device int8_t, dextents<int32_t,2>, tensor_inline> A(a, dextents<int32_t,2>(32, 64));
  tensor<device int8_t, dextents<int32_t,2>, tensor_inline> B(b, dextents<int32_t,2>(32, 32));
  constexpr auto d = matmul2d_descriptor(64, 32, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d, execution_simdgroups<4>> op;
  auto sA = A.slice<32, 64>(0, 0); auto sB = B.slice<32, 32>(0, 0);
  auto c = op.get_destination_cooperative_tensor<decltype(sA), decltype(sB), int32_t>();
  for (ushort i = 0; i < c.get_capacity(); ++i) {
    auto idx = c.get_multidimensional_index(i);
    o[(tid * 16 + i) * 3 + 0] = c.is_valid_element(i);
    o[(tid * 16 + i) * 3 + 1] = idx[0];
    o[(tid * 16 + i) * 3 + 2] = idx[1];
  }
  if (tid == 0) o[128*16*3] = c.get_capacity();
}
"""
let dev = MTLCreateSystemDefaultDevice()!
let opts = MTLCompileOptions(); opts.languageVersion = .version4_0
let lib = try! dev.makeLibrary(source: src, options: opts)
let ps = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "probe")!)
let a = dev.makeBuffer(length: 4096)!, b = dev.makeBuffer(length: 4096)!, o = dev.makeBuffer(length: (128*16*3+1)*4)!
let q = dev.makeCommandQueue()!, cb = q.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
e.setComputePipelineState(ps); e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1); e.setBuffer(o, offset: 0, index: 2)
e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
let p = o.contents().bindMemory(to: Int32.self, capacity: 128*16*3+1)
print("capacity", p[128*16*3])
for t in [0, 1, 2, 3, 4, 31, 32, 33, 64, 96] {
  var s = "tid \(t):"
  for i in 0..<16 { s += " (\(p[(t*16+i)*3+2]),\(p[(t*16+i)*3+1]))\(p[(t*16+i)*3] == 1 ? "" : "x")" }
  print(s)
}
