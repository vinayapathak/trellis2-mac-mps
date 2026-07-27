// CPU-only pybind11 entry point for o_voxel._C, built when BUILD_TARGET=cpu (setup.py's own
// branch for this, used on macOS/Apple Silicon since there's no CUDA). Not written upstream --
// ext.cpp (the CUDA-era entry point) already declares/binds every one of these same _cpu
// functions alongside CUDA-only ones (hashmap_*_cuda, z_order_*_cuda, hilbert_*_cuda,
// rasterize_voxels_cuda), so a CPU build can't compile ext.cpp directly (undefined symbols with
// no CUDA toolchain). This file is ext.cpp trimmed to only the functions setup.py's own
// `cpu_sources` list (flexible_dual_grid.cpp, volumetic_attr.cpp, svo.cpp, filter_parent.cpp,
// filter_neighbor.cpp) actually implements -- no new algorithms written here, purely a binding
// file for CPU implementations that already exist in this vendored o-voxel package.
#include <torch/extension.h>
#include "convert/api.h"
#include "io/api.h"
#include "serialize/api.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Convert functions
    m.def("mesh_to_flexible_dual_grid_cpu", &mesh_to_flexible_dual_grid_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("textured_mesh_to_volumetric_attr_cpu", &textured_mesh_to_volumetric_attr_cpu, py::call_guard<py::gil_scoped_release>());
    // Serialization functions
    m.def("z_order_encode_cpu", &z_order_encode_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("z_order_decode_cpu", &z_order_decode_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("hilbert_encode_cpu", &hilbert_encode_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("hilbert_decode_cpu", &hilbert_decode_cpu, py::call_guard<py::gil_scoped_release>());
    // IO functions
    m.def("encode_sparse_voxel_octree_cpu", &encode_sparse_voxel_octree_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("decode_sparse_voxel_octree_cpu", &decode_sparse_voxel_octree_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("encode_sparse_voxel_octree_attr_parent_cpu", &encode_sparse_voxel_octree_attr_parent_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("decode_sparse_voxel_octree_attr_parent_cpu", &decode_sparse_voxel_octree_attr_parent_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("encode_sparse_voxel_octree_attr_neighbor_cpu", &encode_sparse_voxel_octree_attr_neighbor_cpu, py::call_guard<py::gil_scoped_release>());
    m.def("decode_sparse_voxel_octree_attr_neighbor_cpu", &decode_sparse_voxel_octree_attr_neighbor_cpu, py::call_guard<py::gil_scoped_release>());
}
