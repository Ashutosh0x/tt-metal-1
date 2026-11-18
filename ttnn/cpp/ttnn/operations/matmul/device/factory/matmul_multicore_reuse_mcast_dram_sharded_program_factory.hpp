// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::prim {

// One program for the whole mesh, built with the reference device's DRAM-bank reader cores.
struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

// A mesh whose devices are harvested differently has different optimal DRAM-bank reader cores per device; one
// program with the reference device's cores makes the other devices read DRAM through far cores (Blackhole Galaxy
// 4-chip lines: +35-38 % weight-streaming time on the mismatched chips). This factory builds each device's program
// with its own reader cores. Selected only for such meshes (dram_sharded_helpers::mesh_has_uniform_dram_bank_workers);
// a uniform mesh and a single device keep the factory above.
struct MatmulMultiCoreReuseMultiCastDRAMShardedPerDeviceProgramFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

}  // namespace ttnn::prim
