import cutlass
import cutlass.cute as cute


THREADS = 256
VALUES_PER_THREAD = 4
VALUES_PER_BLOCK = THREADS * VALUES_PER_THREAD


@cute.kernel
def vector_add_vec_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    N: cute.Uint32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Each CTA covers 1024 values.
    block_start = bidx * VALUES_PER_BLOCK

   
    idx0 = block_start + tidx

    for v in range(VALUES_PER_THREAD):
        idx = idx0 + v * THREADS

        if idx < N:
            C[idx] = A[idx] + B[idx]


@cute.jit
def solve(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    N: cute.Uint32,
):
    num_blocks = cute.ceil_div(N, VALUES_PER_BLOCK)

    vector_add_vec_kernel(
        A,
        B,
        C,
        N,
    ).launch(
        grid=(num_blocks, 1, 1),
        block=(THREADS, 1, 1),
    )
