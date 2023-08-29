/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"
#include "cutlass/arch/arch.h"

#include <vector>
#include "mscclpp/sm_channel.hpp"
#include "mscclpp/proxy.hpp"
#include "mscclpp/fifo.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool SplitKSerial               ///! If true, code supporting split-K via serial reduction is enabled.
>
struct Gemm {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kSplitKSerial = SplitKSerial;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorB::TensorRef ref_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;
    int *semaphore;
    int gemm_k_size;
    // For gather+scatter operations
    mscclpp::SmChannel::DeviceHandle* smChannels;
    int channel_size;
    // mscclpp::DeviceProxyFifo fifo;
    // mscclpp::Host2DeviceSemaphore::DeviceHandle* handles;
    int rank;
    int kernel_case;
    int* atmoic_counter;
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), semaphore(0), gemm_k_size(0) { }

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D,
      typename OutputOp::Params output_op = typename OutputOp::Params(),
      int *workspace = nullptr,
      mscclpp::SmChannel::DeviceHandle* smChannels_ = nullptr,
      int channel_size_ = 0,
      // mscclpp::DeviceProxyFifo fifo_ = mscclpp::DeviceProxyFifo(),
      // mscclpp::Host2DeviceSemaphore::DeviceHandle* handles_ = nullptr,
      int rank_ = 0,
      int kernel_case_ = -1,
      int* atmoic_counter_ = nullptr,
      int const *gather_A_indices = nullptr,
      int const *gather_B_indices = nullptr,
      int const *scatter_D_indices = nullptr
    ):
      problem_size(problem_size),
      grid_tiled_shape(grid_tiled_shape),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A(ref_A.layout()),
      ref_A(ref_A),
      params_B(ref_B.layout()),
      ref_B(ref_B),
      params_C(ref_C.layout()),
      ref_C(ref_C),
      params_D(ref_D.layout()),
      ref_D(ref_D),
      output_op(output_op),
      smChannels(smChannels_),
      channel_size(channel_size_),
      // fifo(fifo_),
      // handles(handles_),
      rank(rank_),
      kernel_case(kernel_case_),
      atmoic_counter(atmoic_counter_),
      gather_A_indices(gather_A_indices),
      gather_B_indices(gather_B_indices),
      scatter_D_indices(scatter_D_indices) {

      // printf("handles inside Params %p\n", handles);

      int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();

      gemm_k_size = gemm_k_iterations * Mma::Shape::kK;

    semaphore = workspace;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
    // half_t allGatherCache[Mma::Shape::kN * (128/(Mma::Shape::kN*sizeof(half)/sizeof(long)))];
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Gemm() { }

  /// Determines whether kernel satisfies alignment
  CUTLASS_HOST_DEVICE
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size,
    typename Mma::IteratorA::TensorRef ref_A,
    typename Mma::IteratorB::TensorRef ref_B,
    typename Epilogue::OutputTileIterator::TensorRef ref_C,
    typename Epilogue::OutputTileIterator::TensorRef ref_D) {

    static int const kAlignmentA = (platform::is_same<typename Mma::IteratorA::Layout,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorA::Layout,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB =  (platform::is_same<typename Mma::IteratorB::Layout,
                                                       layout::RowMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorB::Layout,
                                                        layout::RowMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Epilogue::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(ref_A, kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.k() * params.gemm_k_size,
    };

    cutlass::MatrixCoord tb_offset_B{
      threadblock_tile_offset.k() * params.gemm_k_size,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k = min(
      params.problem_size.k(), 
      (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // ----------- AllGather starts ---------------
    if (params.kernel_case == 2 && params.channel_size > 0) {
      const int prev_kN = 128; // tile size for column
      int startRowIndex = threadblock_tile_offset.m() * Mma::Shape::kM;

      const int total_columns = 96; // 12k / 128

      const int row_offset = threadblock_tile_offset.m() * total_columns;
      volatile int* done = params.atmoic_counter + 8 + row_offset;
      volatile int* done2 = params.atmoic_counter + 2048 + row_offset;

      int num_gpus = params.channel_size + 1;
      int blockId = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z + 1;
      for (int cur_column = 0; cur_column < total_columns; cur_column++)
      // for (int cur_column = blockIdx.y * 2; cur_column < blockIdx.y * 2 + 2; cur_column++) // this hangs
      {
        int owner = cur_column % num_gpus;
        if (owner == params.rank) // already the owner
        {
          continue;
        }
        volatile int* state = done + cur_column;
        volatile int* responsible = done2 + cur_column;
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        {
          int preval = atomicAdd((int*) state, 1);
          if (preval == 0)
          {
            *responsible = blockId;
          }
          // else
          // {
          //   if (params.rank == 0 && threadIdx.x == 0)
          //     printf("nores %d %d: %d %p, blockIdx %d %d %d\n", (int) threadblock_tile_offset.m(), (int) cur_column, (int) preval, state, (int) blockIdx.x, (int) blockIdx.y, (int) blockIdx.z);
          // }
        }
        __syncthreads();
        if (*responsible == blockId)
        // invoke get
        {
          int channel_idx = owner > params.rank ? (owner - 1) : owner;

          const int ColCopyThreads = Mma::Shape::kN * sizeof(half) / 16;
          const int RowCopyGroup = blockDim.x / ColCopyThreads; // blockDim.x (= WarpShape / InstructionShape * 32) / ColCopyThreads
          const int RowCopyGroupIdx = threadIdx.x / ColCopyThreads;
          for (int rowIndex = startRowIndex + RowCopyGroupIdx;
              rowIndex < startRowIndex + Mma::Shape::kM && rowIndex < params.problem_size.m();
              rowIndex += RowCopyGroup)
          {
            int row_skip = rowIndex * params.problem_size.k();
            int column_skip = cur_column * prev_kN;
            int src_offset =  (row_skip + column_skip) + params.rank * (params.problem_size.m() * params.problem_size.k());
            int dest_offset =  (row_skip + column_skip) + owner * (params.problem_size.m() * params.problem_size.k());
            params.smChannels[channel_idx].get(
                          sizeof(cutlass::half_t) * dest_offset,
                          sizeof(cutlass::half_t) * src_offset,
                          prev_kN * sizeof(cutlass::half_t),
                          threadIdx.x % ColCopyThreads, ColCopyThreads);
          }
          __syncthreads();
#define STATE_MAGIC 0x12345
          if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
          {
            (*responsible) = STATE_MAGIC;
          }
        }
      }
      __syncthreads();
      for (int cur_column = threadIdx.x; cur_column < total_columns; cur_column += blockDim.x)
      {
        int owner = cur_column % num_gpus;
        if (owner == params.rank) // already the owner
          continue;
        volatile int* state = done + cur_column;
        volatile int* responsible = done2 + cur_column;
        while ((*responsible) != STATE_MAGIC) {}
      }
      
      __syncthreads();
    }
    // ----------- AllGather ends ---------------

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A,
      params.gather_A_indices);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      params.ref_B.data(),
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B,
      params.gather_B_indices);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx();
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();
#define USEGEMM
    if (!kSplitKSerial || gemm_k_iterations > 0) {
      // Compute threadblock-scoped matrix multiply-add
#ifdef USEGEMM
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
#endif
    }

    //
    // Epilogue
    //

    OutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
      
      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      params.ref_C.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      params.scatter_D_indices
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      params.ref_D.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset,
      params.scatter_D_indices
    );

    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {

      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_offset.k());

    }

#ifdef USEGEMM
    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op, iterator_D, accumulators, iterator_C);
#endif

    //
    // Release the semaphore
    //

    if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }

      semaphore.release(lock);
    }
    if (params.channel_size == 0)
      return;
    // __syncthreads();
    if (kSplitKSerial && params.grid_tiled_shape.k() > 1 && (params.grid_tiled_shape.k() != threadblock_tile_offset.k() + 1))
      return;
    int startRowIndex = threadblock_tile_offset.m() * Mma::Shape::kM;
    int startColIndex = threadblock_tile_offset.n() * Mma::Shape::kN;
    // __syncthreads();
// #define DEBUG_CUDA
#ifdef DEBUG_CUDA
   
    if (threadIdx.x == 0 && blockIdx.x == 0) printf("kM,kN,kK %d %d %d | blockDim %d %d %d | gridDim %d %d %d\n", 
      (int) Mma::Shape::kM, (int) Mma::Shape::kN, (int) Mma::Shape::kK, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    if (threadIdx.x == 0)
    {
      printf("DEBUG startColIndex = %d\n", startColIndex);
      printf("DEBUG  threadblock_tile_offset.m()=%d, n=%d, k=%d\n", threadblock_tile_offset.m(), threadblock_tile_offset.n(), threadblock_tile_offset.k());
    }
    if (threadIdx.x == 0) {
      for (auto idx = startColIndex;  idx < startColIndex + Mma::Shape::kN; idx++)
      {
        auto val = params.ref_D.data()[idx];
        if ((float)val != 8192 * 4) {
              printf("blockIdx.x %d blockIdx.y %d *it == %f at %d\n", blockIdx.x, blockIdx.y, (float)val, idx);
            break;
        }
      }
    }
    for (int rowIndex = startRowIndex; rowIndex < startRowIndex + Mma::Shape::kM && rowIndex < params.problem_size.m(); rowIndex++)
    {
      if (threadIdx.x == 0)
      {
        printf("blockIdx.y %d startColIndex %d, params.rank %d index sum: %ld\n", blockIdx.y, (int)startColIndex, (int)params.rank,
                              rowIndex * params.problem_size.n() * (params.channel_size+1) * sizeof(cutlass::half_t)
                              + startColIndex * sizeof(cutlass::half_t) + params.rank *  params.problem_size.n() * sizeof(cutlass::half_t),
                                min(params.problem_size.n(), Mma::Shape::kN) * sizeof(cutlass::half_t));
        printf("offset %d, rank = %d\n", params.rank * 16 * 16, params.rank);
        printf("comparison %d %d\n", params.problem_size.n(), Mma::Shape::kN);
        printf("rowIndex=%d, params.problem_size.n()=%d, startColIndex=%d, sizeof=%d\n",
              (int) rowIndex, (int) params.problem_size.n(), (int) startColIndex,
              (int) sizeof(cutlass::half_t));
      }
    }
#endif
    if (params.kernel_case == 1)
    {
      // kM,kN,kK 128 128 32 | blockDim 128 1 1 | gridDim 16 96 1 (splitK) (GemmIdentityThreadblockSwizzle)
      // kM,kN,kK 128 128 32 | blockDim 128 1 1 | gridDim 16 48 8 (second GEMM, splitK=8)
      int owner = blockIdx.y % 8;
      if (owner != params.rank) {
        int channelIdx = owner;
        if (channelIdx > params.rank)
          channelIdx--;
        const int ColCopyThreads = Mma::Shape::kN * sizeof(half) / 16;
        const int RowCopyGroup = blockDim.x / ColCopyThreads; // blockDim.x (= WarpShape / InstructionShape * 32) / ColCopyThreads
        const int RowCopyGroupIdx = threadIdx.x / ColCopyThreads;
        for (int rowIndex = startRowIndex + RowCopyGroupIdx;
             rowIndex < startRowIndex + Mma::Shape::kM && rowIndex < params.problem_size.m();
             rowIndex += RowCopyGroup)
        {
          int row_skip = rowIndex * params.problem_size.n();
          int column_skip = startColIndex;
          int src_offset =  (row_skip + column_skip) + params.rank * (params.problem_size.m() * params.problem_size.n());
          params.smChannels[channelIdx].put(
                        sizeof(cutlass::half_t) * src_offset,
                        min(params.problem_size.n(), Mma::Shape::kN) * sizeof(cutlass::half_t),
                        threadIdx.x % ColCopyThreads, ColCopyThreads);
        }
      }
      __syncthreads();
      bool lastBlock = false;
      if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
      {
        __threadfence();
        int old_value = atomicAdd(params.atmoic_counter, 1);
        if (old_value + 1 == gridDim.x * gridDim.y)
        {
          // if (params.rank == 0) printf("before signal %d, rank %d threadblock.n() %d threadblock.m() %d k() %d\n", *params.atmoic_counter, params.rank, threadblock_tile_offset.n(), threadblock_tile_offset.m(), threadblock_tile_offset.k());
          *params.atmoic_counter = 0;
          lastBlock = true;
        }
      }
      if (lastBlock) {
        if (params.kernel_case == 1) // scatter
        {
          if (threadIdx.x == 0)
          {
            // printf("signal+wait\n");
            for (int i = 0; i < params.channel_size; i++)
            {
              params.smChannels[i].signal();
            }
            // __syncthreads();
            // __threadfence_system();
            for (int i = 0; i < params.channel_size; i++)
            {
              params.smChannels[i].wait();
            }
          }
        }
      }
    }
    else // allgather, kernel_case == 2
    {
      if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
      {
        __threadfence();
        int old_value = atomicAdd(params.atmoic_counter, 1);
        if (old_value + 1 == gridDim.x * gridDim.y)
        {
          // if (params.rank == 0) printf("before signal %d, rank %d threadblock.n() %d threadblock.m() %d k() %d\n", *params.atmoic_counter, params.rank, threadblock_tile_offset.n(), threadblock_tile_offset.m(), threadblock_tile_offset.k());
          *params.atmoic_counter = 0;
          *(params.atmoic_counter + 1) = 1;
        }
      }
      __syncthreads();
      if (*(params.atmoic_counter + 1) == 1) // last thread block
      {
        const int total_row_grid = 16; // 2k / 128
        const int total_col_grid = 96; // 12k / 128

        for (int i = 0; i < total_row_grid; i++)
        {
          // 128 threads > 96 total_col_grid
          int row_offset = i * total_col_grid;
          volatile int* done = params.atmoic_counter + 8 + row_offset;
          volatile int* done2 = params.atmoic_counter + 2048 + row_offset;
          *(done + int(threadIdx.x)) = 0;
          *(done2 + int(threadIdx.x)) = 0;
        }
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        {
          *(params.atmoic_counter + 1) = 0;
        }
      }
    }
  }


};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

