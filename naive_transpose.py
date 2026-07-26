import cutlass
import cutlass.cute as cute

TILE_M = 32
TILE_N = 32
THREADS = 256
num_items_per_thread = 4

@cute.kernel
def transpose_kernel(gx:cute.Tensor, gy:cute.Tensor, m:cute.Int32, n:cute.Int32):
  thread_id, _, _ = cute.arch.thread_idx()
  tile_column, tile_row, _ = cute.arch.block_idx()
  current_tile = gx[(None, tile_row, tile_column)]
  output_tile= gy[(None, tile_column, tile_row)]
  for offset in cutlass.range_constexp(4):
    global_index = offset*THREADS + thread_id

    tile_row=global_index//TILE_N
    tile_col=global_index%TILE_N

    # Convert position inside the tile into position in the full matrix.
    global_row = tile_row * TILE_M + tile_row
    global_column = tile_column * TILE_N+ tile_column

    if global_row<m and global_column<n:
      val = current_tile[(tile_row, tile_col)]
      output_tile[(tile_col, tile_row)] = val





@cute.jit
def solve(mx:cute.Tensor, my:cute.Tensor, m:cute.Int32,n:cute.Int32):
  tiled_gx=cute.tiled_divide(mx,(TILE_M,TILE_N))
  tiled_gy=cute.tiled_divide(my,(TILE_M,TILE_N))
  row_tiles = cute.ceil_div(n,TILE_N)
  col_tiles = cute.ceil_div(m,TILE_M)
  transpose_kernel(tiled_gx, tiled_gy, m, n).launch(block=(THREADS,1,1), grid=(row_tiles, col_tiles,1))






