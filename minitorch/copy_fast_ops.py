from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit
from numba import types


from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
            # Calculate total number of elements
        size = 1
        for s in out_shape:
            size *= s

        # Check if the input and output tensors are stride-aligned
        same_strides = True
        if len(out_strides) != len(in_strides):
            same_strides = False
        else:
            for i in range(len(out_strides)):
                if out_strides[i] != in_strides[i]:
                    same_strides = False
                    break

        # Check if shapes are the same
        same_shape = True
        if len(out_shape) != len(in_shape):
            same_shape = False
        else:
            for i in range(len(out_shape)):
                if out_shape[i] != in_shape[i]:
                    same_shape = False
                    break

        if same_strides and same_shape:
            # Directly loop over storage arrays
            for i in prange(size):
                out[i] = fn(in_storage[i])
        else:
            # General case: need to compute indices
            for i in prange(size):
                out_index = np.zeros(len(out_shape), dtype=np.int32)
                in_index = np.zeros(len(in_shape), dtype=np.int32)

                # Convert linear index to multi-dimensional index for output tensor
                to_index(i, out_shape, out_index)

                # Compute position in output storage
                out_pos = index_to_position(out_index, out_strides)

                # Broadcast index to input tensor shape
                broadcast_index(out_index, out_shape, in_shape, in_index)

                # Compute position in input storage
                in_pos = index_to_position(in_index, in_strides)

                # Apply the function
                out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
            # Calculate total number of elements
        size = 1
        for s in out_shape:
            size *= s

        # Check if all tensors are stride-aligned
        same_strides = True
        if len(out_strides) != len(a_strides) or len(out_strides) != len(b_strides):
            same_strides = False
        else:
            for i in range(len(out_strides)):
                if (
                    out_strides[i] != a_strides[i]
                    or out_strides[i] != b_strides[i]
                    or a_strides[i] != b_strides[i]
                ):
                    same_strides = False
                    break

        # Check if shapes are the same
        same_shape = True
        if len(out_shape) != len(a_shape) or len(out_shape) != len(b_shape):
            same_shape = False
        else:
            for i in range(len(out_shape)):
                if (
                    out_shape[i] != a_shape[i]
                    or out_shape[i] != b_shape[i]
                    or a_shape[i] != b_shape[i]
                ):
                    same_shape = False
                    break

        if same_strides and same_shape:
            # Directly loop over storage arrays
            for i in prange(size):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            # General case
            for i in prange(size):
                out_index = np.zeros(len(out_shape), dtype=np.int32)
                a_index = np.zeros(len(a_shape), dtype=np.int32)
                b_index = np.zeros(len(b_shape), dtype=np.int32)

                # Convert linear index to multi-dimensional index for output tensor
                to_index(i, out_shape, out_index)

                # Compute position in output storage
                out_pos = index_to_position(out_index, out_strides)

                # Broadcast index to input tensor A
                broadcast_index(out_index, out_shape, a_shape, a_index)
                a_pos = index_to_position(a_index, a_strides)

                # Broadcast index to input tensor B
                broadcast_index(out_index, out_shape, b_shape, b_index)
                b_pos = index_to_position(b_index, b_strides)

                # Apply the function
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
          # Calculate total number of elements in the output tensor
        size = 1
        for s in out_shape:
            size *= s

        # For each position in the output tensor
        for i in prange(size):
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            a_index = np.zeros(len(a_shape), dtype=np.int32)

            # Convert linear index to multi-dimensional index
            to_index(i, out_shape, out_index)

            # Compute position in output storage
            out_pos = index_to_position(out_index, out_strides)

            # Initialize the accumulator with the starting value (e.g., 0.0)
            acc = out[out_pos]

            # Copy the output index to input index
            for j in range(len(out_index)):
                a_index[j] = out_index[j]

            # Iterate over the reduced dimension
            for s in range(a_shape[reduce_dim]):
                a_index[reduce_dim] = s
                a_pos = index_to_position(a_index, a_strides)
                acc = fn(acc, a_storage[a_pos])

            # Store the result
            out[out_pos] = acc

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # TODO: Implement for Task 3.2.

    # Determine dimensions and ensure they are integers
    if len(out_shape) == 3:
        batch_size = int(out_shape[0])
        M = int(out_shape[1])
        N = int(out_shape[2])
        out_batch_stride = int(out_strides[0])
        out_i_stride = int(out_strides[1])
        out_j_stride = int(out_strides[2])
    else:
        batch_size = 1
        M = int(out_shape[0])
        N = int(out_shape[1])
        out_batch_stride = 0
        out_i_stride = int(out_strides[0])
        out_j_stride = int(out_strides[1])

    K = int(a_shape[-1])  # Since a_shape[-1] == b_shape[-2]

    # Get strides for 'a' tensor
    if len(a_shape) == 3:
        a_i_stride = int(a_strides[1])
        a_k_stride = int(a_strides[2])
    else:
        a_i_stride = int(a_strides[0])
        a_k_stride = int(a_strides[1])

    # Get strides for 'b' tensor
    if len(b_shape) == 3:
        b_k_stride = int(b_strides[1])
        b_j_stride = int(b_strides[2])
    else:
        b_k_stride = int(b_strides[0])
        b_j_stride = int(b_strides[1])

    # Outer loop in parallel over the batch dimension
    for n in prange(batch_size):
        # Handle broadcasting in the batch dimension
        a_n = n if len(a_shape) == 3 and a_shape[0] > 1 else 0
        b_n = n if len(b_shape) == 3 and b_shape[0] > 1 else 0

        for i in range(M):
            for j in range(N):
                sum = 0.0  # Local accumulator
                for k in range(K):
                    # Compute positions in 'a' and 'b' storage
                    a_pos = int(a_n * a_batch_stride + i * a_i_stride + k * a_k_stride)
                    b_pos = int(b_n * b_batch_stride + k * b_k_stride + j * b_j_stride)

                    # Accumulate the product
                    sum += a_storage[a_pos] * b_storage[b_pos]

                # Compute position in 'out' storage
                out_pos = int(n * out_batch_stride + i * out_i_stride + j * out_j_stride)

                # Write the accumulated sum to the output storage
                out[out_pos] = sum

tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
