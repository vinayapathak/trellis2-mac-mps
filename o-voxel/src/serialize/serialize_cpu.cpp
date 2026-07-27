// CPU-only implementation of the z-order/Hilbert (de)serialize functions declared in api.h.
// Not written upstream -- z_order.cu/hilbert.cu implement CPU::z_order_encode/decode and
// CPU::hilbert_encode/decode (plain __host__ C++, no CUDA-only syntax) in the same file as
// CUDA-only __global__ kernels, and z_order.h/hilbert.h jointly declare both the CPU and CUDA
// sides -- so a CPU-only build can't include those headers (no cuda_runtime.h available) or
// compile those .cu files (need nvcc). This file re-implements only the CPU logic, ported
// verbatim from z_order.cu's/hilbert.cu's CPU:: function bodies with CUDA decorators removed, as
// free functions binding api.h's z_order_encode_cpu/z_order_decode_cpu/hilbert_encode_cpu/
// hilbert_decode_cpu declarations directly -- no new algorithms.
#include <torch/extension.h>
#include "api.h"

namespace {

uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

uint32_t extract_bits(uint32_t v) {
    v = v & 0x49249249;
    v = (v ^ (v >>  2)) & 0x030C30C3u;
    v = (v ^ (v >>  4)) & 0x0300F00Fu;
    v = (v ^ (v >>  8)) & 0x030000FFu;
    v = (v ^ (v >> 16)) & 0x000003FFu;
    return v;
}

} // namespace


torch::Tensor
z_order_encode_cpu(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& z
) {
    torch::Tensor codes = torch::empty_like(x, torch::dtype(torch::kInt32));
    size_t N = x.size(0);
    // Hold the contiguous copies as named locals -- x.contiguous() returns a NEW temporary
    // tensor (with its own storage) whenever x is a non-contiguous view (e.g. a strided column
    // slice, which is exactly what serialize.py's coords[:, i] callers pass in). Taking .data_ptr()
    // straight off that temporary and discarding the tensor immediately leaves a dangling pointer
    // once the temporary is destroyed at the end of the statement -- confirmed via a real
    // round-trip test producing garbage for trivial all-zero input before this fix.
    torch::Tensor x_c = x.contiguous();
    torch::Tensor y_c = y.contiguous();
    torch::Tensor z_c = z.contiguous();
    const uint32_t* xp = reinterpret_cast<const uint32_t*>(x_c.data_ptr<int>());
    const uint32_t* yp = reinterpret_cast<const uint32_t*>(y_c.data_ptr<int>());
    const uint32_t* zp = reinterpret_cast<const uint32_t*>(z_c.data_ptr<int>());
    uint32_t* out = reinterpret_cast<uint32_t*>(codes.data_ptr<int>());
    for (size_t i = 0; i < N; i++) {
        uint32_t xx = expand_bits(xp[i]);
        uint32_t yy = expand_bits(yp[i]);
        uint32_t zz = expand_bits(zp[i]);
        out[i] = xx * 4 + yy * 2 + zz;
    }
    return codes;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
z_order_decode_cpu(
    const torch::Tensor& codes
) {
    size_t N = codes.size(0);
    torch::Tensor x = torch::empty_like(codes, torch::dtype(torch::kInt32));
    torch::Tensor y = torch::empty_like(codes, torch::dtype(torch::kInt32));
    torch::Tensor z = torch::empty_like(codes, torch::dtype(torch::kInt32));
    torch::Tensor codes_c = codes.contiguous();
    const uint32_t* cp = reinterpret_cast<const uint32_t*>(codes_c.data_ptr<int>());
    uint32_t* xp = reinterpret_cast<uint32_t*>(x.data_ptr<int>());
    uint32_t* yp = reinterpret_cast<uint32_t*>(y.data_ptr<int>());
    uint32_t* zp = reinterpret_cast<uint32_t*>(z.data_ptr<int>());
    for (size_t i = 0; i < N; i++) {
        xp[i] = extract_bits(cp[i] >> 2);
        yp[i] = extract_bits(cp[i] >> 1);
        zp[i] = extract_bits(cp[i]);
    }
    return std::make_tuple(x, y, z);
}


torch::Tensor
hilbert_encode_cpu(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& z
) {
    torch::Tensor codes = torch::empty_like(x);
    size_t N = x.size(0);
    torch::Tensor x_c = x.contiguous();
    torch::Tensor y_c = y.contiguous();
    torch::Tensor z_c = z.contiguous();
    const uint32_t* xp = reinterpret_cast<const uint32_t*>(x_c.data_ptr<int>());
    const uint32_t* yp = reinterpret_cast<const uint32_t*>(y_c.data_ptr<int>());
    const uint32_t* zp = reinterpret_cast<const uint32_t*>(z_c.data_ptr<int>());
    uint32_t* out = reinterpret_cast<uint32_t*>(codes.data_ptr<int>());
    for (size_t thread_id = 0; thread_id < N; thread_id++) {
        uint32_t point[3] = {xp[thread_id], yp[thread_id], zp[thread_id]};

        uint32_t m = 1 << 9, q, p, t;

        q = m;
        while (q > 1) {
            p = q - 1;
            for (int i = 0; i < 3; i++) {
                if (point[i] & q) {
                    point[0] ^= p;
                } else {
                    t = (point[0] ^ point[i]) & p;
                    point[0] ^= t;
                    point[i] ^= t;
                }
            }
            q >>= 1;
        }

        for (int i = 1; i < 3; i++) {
            point[i] ^= point[i - 1];
        }
        t = 0;
        q = m;
        while (q > 1) {
            if (point[2] & q) {
                t ^= q - 1;
            }
            q >>= 1;
        }
        for (int i = 0; i < 3; i++) {
            point[i] ^= t;
        }

        uint32_t xx = expand_bits(point[0]);
        uint32_t yy = expand_bits(point[1]);
        uint32_t zz = expand_bits(point[2]);

        out[thread_id] = xx * 4 + yy * 2 + zz;
    }
    return codes;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
hilbert_decode_cpu(
    const torch::Tensor& codes
) {
    size_t N = codes.size(0);
    torch::Tensor x = torch::empty_like(codes);
    torch::Tensor y = torch::empty_like(codes);
    torch::Tensor z = torch::empty_like(codes);
    torch::Tensor codes_c = codes.contiguous();
    const uint32_t* cp = reinterpret_cast<const uint32_t*>(codes_c.data_ptr<int>());
    uint32_t* xp = reinterpret_cast<uint32_t*>(x.data_ptr<int>());
    uint32_t* yp = reinterpret_cast<uint32_t*>(y.data_ptr<int>());
    uint32_t* zp = reinterpret_cast<uint32_t*>(z.data_ptr<int>());
    for (size_t thread_id = 0; thread_id < N; thread_id++) {
        uint32_t point[3];
        point[0] = extract_bits(cp[thread_id] >> 2);
        point[1] = extract_bits(cp[thread_id] >> 1);
        point[2] = extract_bits(cp[thread_id]);

        uint32_t m = 2 << 9, q, p, t;

        t = point[2] >> 1;
        for (int i = 2; i > 0; i--) {
            point[i] ^= point[i - 1];
        }
        point[0] ^= t;

        q = 2;
        while (q != m) {
            p = q - 1;
            for (int i = 2; i >= 0; i--) {
                if (point[i] & q) {
                    point[0] ^= p;
                } else {
                    t = (point[0] ^ point[i]) & p;
                    point[0] ^= t;
                    point[i] ^= t;
                }
            }
            q <<= 1;
        }

        xp[thread_id] = point[0];
        yp[thread_id] = point[1];
        zp[thread_id] = point[2];
    }
    return std::make_tuple(x, y, z);
}
