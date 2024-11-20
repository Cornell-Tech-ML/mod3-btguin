# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

- Docs: https://minitorch.github.io/

- Overview: https://minitorch.github.io/module3.html

You will need to modify `tensor_functions.py` slightly in this assignment.

- Tests:

```
python run_tests.py
```

Module 3.2

(.venv) bryanguin@Bryans-MacBook-Air mod3-btguin % python project/parallel_check.py

MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/bryanguin/Documents/torch-workspace/mod3-btguin/minitorch/fast_ops.py 
(165)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/bryanguin/Documents/torch-workspace/mod3-btguin/minitorch/fast_ops.py (165) 
---------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                          | 
        out: Storage,                                                                  | 
        out_shape: Shape,                                                              | 
        out_strides: Strides,                                                          | 
        in_storage: Storage,                                                           | 
        in_shape: Shape,                                                               | 
        in_strides: Strides,                                                           | 
    ) -> None:                                                                         | 
        # TODO: Implement for Task 3.1.                                                | 
        size = 1                                                                       | 
        for s in out_shape:                                                            | 
            size *= s                                                                  | 
                                                                                       | 
        # Check if the input and output tensors are stride-aligned                     | 
        same_strides = True                                                            | 
        if len(out_strides) != len(in_strides):                                        | 
            same_strides = False                                                       | 
        else:                                                                          | 
            for i in range(len(out_strides)):                                          | 
                if out_strides[i] != in_strides[i]:                                    | 
                    same_strides = False                                               | 
                    break                                                              | 
                                                                                       | 
        # Check if shapes are the same                                                 | 
        same_shape = True                                                              | 
        if len(out_shape) != len(in_shape):                                            | 
            same_shape = False                                                         | 
        else:                                                                          | 
            for i in range(len(out_shape)):                                            | 
                if out_shape[i] != in_shape[i]:                                        | 
                    same_shape = False                                                 | 
                    break                                                              | 
                                                                                       | 
        if same_strides and same_shape:                                                | 
            # Directly loop over storage arrays                                        | 
            for i in prange(size):-----------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                             | 
        else:                                                                          | 
            # General case: need to compute indices                                    | 
            for i in prange(size):-----------------------------------------------------| #1
                out_index = np.empty(MAX_DIMS, dtype=np.int32)                         | 
                in_index = np.empty(MAX_DIMS, dtype=np.int32)                          | 
                                                                                       | 
                # Convert linear index to multi-dimensional index for output tensor    | 
                to_index(i, out_shape, out_index)                                      | 
                                                                                       | 
                # Compute position in output storage                                   | 
                out_pos = index_to_position(out_index, out_strides)                    | 
                                                                                       | 
                # Broadcast index to input tensor shape                                | 
                broadcast_index(out_index, out_shape, in_shape, in_index)              | 
                                                                                       | 
                # Compute position in input storage                                    | 
                in_pos = index_to_position(in_index, in_strides)                       | 
                                                                                       | 
                # Apply the function                                                   | 
                out[out_pos] = fn(in_storage[in_pos])                                  | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/bryanguin/Documents/torch-workspace/mod3-btguin/minitorch/fast_ops.py 
(205) is hoisted out of the parallel loop labelled #1 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/bryanguin/Documents/torch-workspace/mod3-btguin/minitorch/fast_ops.py 
(206) is hoisted out of the parallel loop labelled #1 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/bryanguin/Documents/torch-workspace/mod3-btguin/minitorch/fast_ops.py 
(249)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/bryanguin/Documents/torch-workspace/mod3-btguin/minitorch/fast_ops.py (249) 
----------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                           | 
        out: Storage,                                                                   | 
        out_shape: Shape,                                                               | 
        out_strides: Strides,                                                           | 
        a_storage: Storage,                                                             | 
        a_shape: Shape,                                                                 | 
        a_strides: Strides,                                                             | 
        b_storage: Storage,                                                             | 
        b_shape: Shape,                                                                 | 
        b_strides: Strides,                                                             | 
    ) -> None:                                                                          | 
        # TODO: Implement for Task 3.1.                                                 | 
        # Calculate total number of elements                                            | 
        size = 1                                                                        | 
        for s in out_shape:                                                             | 
            size *= s                                                                   | 
                                                                                        | 
        # Check if all tensors are stride-aligned                                       | 
        same_strides = True                                                             | 
        if len(out_strides) != len(a_strides) or len(out_strides) != len(b_strides):    | 
            same_strides = False                                                        | 
        else:                                                                           | 
            for i in range(len(out_strides)):                                           | 
                if (                                                                    | 
                    out_strides[i] != a_strides[i]                                      | 
                    or out_strides[i] != b_strides[i]                                   | 
                    or a_strides[i] != b_strides[i]                                     | 
                ):                                                                      | 
                    same_strides = False                                                | 
                    break                                                               | 
                                                                                        | 
        # Check if shapes are the same                                                  | 
        same_shape = True                                                               | 
        if len(out_shape) != len(a_shape) or len(out_shape) != len(b_shape):            | 
            same_shape = False                                                          | 
        else:                                                                           | 
            for i in range(len(out_shape)):                                             | 
                if (                                                                    | 
                    out_shape[i] != a_shape[i]                                          | 
                    or out_shape[i] != b_shape[i]                                       | 
                    or a_shape[i] != b_shape[i]                                         | 
                ):                                                                      | 
                    same_shape = False                                                  | 
                    break                                                               | 
                                                                                        | 
        if same_strides and same_shape:                                                 | 
            # Directly loop over storage arrays                                         | 
            for i in prange(size):------------------------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                                 | 
        else:                                                                           | 
            # General case                                                              | 
            for i in prange(size):------------------------------------------------------| #3
                out_index = np.empty(MAX_DIMS, dtype=np.int32)                          | 
                a_index = np.empty(MAX_DIMS, dtype=np.int32)                            | 
                b_index = np.empty(MAX_DIMS, dtype=np.int32)                            | 
                                                                                        | 
                # Convert linear index to multi-dimensional index for output tensor     | 
                to_index(i, out_shape, out_index)                                       | 
                                                                                        | 
                # Compute position in output storage                                    | 
                out_pos = index_to_position(out_index, out_strides)                     | 
                                                                                        | 
                # Broadcast index to input tensor A                                     | 
                broadcast_index(out_index, out_shape, a_shape, a_index)                 | 
                a_pos = index_to_position(a_index, a_strides)                           | 
                                                                                        | 
                # Broadcast index to input tensor B                                     | 
                broadcast_index(out_index, out_shape, b_shape, b_index)                 | 
                b_pos = index_to_position(b_index, b_strides)                           | 
                                                                                        | 
                # Apply the function                                                    | 
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/bryanguin/Documents/torch-workspace/mod3-btguin/minitorch/fast_ops.py 
(301) is hoisted out of the parallel loop labelled #3 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/bryanguin/Documents/torch-workspace/mod3-btguin/minitorch/fast_ops.py 
(302) is hoisted out of the parallel loop labelled #3 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/bryanguin/Documents/torch-workspace/mod3-btguin/minitorch/fast_ops.py 
(303) is hoisted out of the parallel loop labelled #3 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/bryanguin/Documents/torch-workspace/mod3-btguin/minitorch/fast_ops.py 
(346)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/bryanguin/Documents/torch-workspace/mod3-btguin/minitorch/fast_ops.py (346) 
--------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                | 
        out: Storage,                                                           | 
        out_shape: Shape,                                                       | 
        out_strides: Strides,                                                   | 
        a_storage: Storage,                                                     | 
        a_shape: Shape,                                                         | 
        a_strides: Strides,                                                     | 
        reduce_dim: int,                                                        | 
    ) -> None:                                                                  | 
        # TODO: Implement for Task 3.1.                                         | 
        # Calculate total number of elements in the output tensor               | 
        size = 1                                                                | 
        for s in out_shape:                                                     | 
            size *= s                                                           | 
                                                                                | 
        # For each position in the output tensor                                | 
        for i in prange(size):--------------------------------------------------| #4
            out_index = np.empty(MAX_DIMS, dtype=np.int32)                      | 
            a_index = np.empty(MAX_DIMS, dtype=np.int32)                        | 
                                                                                | 
            # Convert linear index to multi-dimensional index                   | 
            to_index(i, out_shape, out_index)                                   | 
                                                                                | 
            # Compute position in output storage                                | 
            out_pos = index_to_position(out_index, out_strides)                 | 
                                                                                | 
            # Initialize the accumulator with the starting value (e.g., 0.0)    | 
            acc = out[out_pos]                                                  | 
                                                                                | 
            # Copy the output index to input index                              | 
            for j in range(len(out_shape)):                                     | 
                a_index[j] = out_index[j]                                       | 
                                                                                | 
            # Iterate over the reduced dimension                                | 
            for s in range(a_shape[reduce_dim]):                                | 
                a_index[reduce_dim] = s                                         | 
                a_pos = index_to_position(a_index, a_strides)                   | 
                acc = fn(acc, a_storage[a_pos])                                 | 
                                                                                | 
            # Store the result                                                  | 
            out[out_pos] = acc                                                  | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/bryanguin/Documents/torch-workspace/mod3-btguin/minitorch/fast_ops.py 
(363) is hoisted out of the parallel loop labelled #4 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/bryanguin/Documents/torch-workspace/mod3-btguin/minitorch/fast_ops.py 
(364) is hoisted out of the parallel loop labelled #4 (it will be performed 
before the loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/bryanguin/Documents/torch-workspace/mod3-btguin/minitorch/fast_ops.py 
(391)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/bryanguin/Documents/torch-workspace/mod3-btguin/minitorch/fast_ops.py (391) 
-------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                               | 
    out: Storage,                                                                          | 
    out_shape: Shape,                                                                      | 
    out_strides: Strides,                                                                  | 
    a_storage: Storage,                                                                    | 
    a_shape: Shape,                                                                        | 
    a_strides: Strides,                                                                    | 
    b_storage: Storage,                                                                    | 
    b_shape: Shape,                                                                        | 
    b_strides: Strides,                                                                    | 
) -> None:                                                                                 | 
    """NUMBA tensor matrix multiply function.                                              | 
                                                                                           | 
    Should work for any tensor shapes that broadcast as long as                            | 
                                                                                           | 
    ```                                                                                    | 
    assert a_shape[-1] == b_shape[-2]                                                      | 
    ```                                                                                    | 
                                                                                           | 
    Optimizations:                                                                         | 
                                                                                           | 
    * Outer loop in parallel                                                               | 
    * No index buffers or function calls                                                   | 
    * Inner loop should have no global writes, 1 multiply.                                 | 
                                                                                           | 
                                                                                           | 
    Args:                                                                                  | 
    ----                                                                                   | 
        out (Storage): storage for `out` tensor                                            | 
        out_shape (Shape): shape for `out` tensor                                          | 
        out_strides (Strides): strides for `out` tensor                                    | 
        a_storage (Storage): storage for `a` tensor                                        | 
        a_shape (Shape): shape for `a` tensor                                              | 
        a_strides (Strides): strides for `a` tensor                                        | 
        b_storage (Storage): storage for `b` tensor                                        | 
        b_shape (Shape): shape for `b` tensor                                              | 
        b_strides (Strides): strides for `b` tensor                                        | 
                                                                                           | 
    Returns:                                                                               | 
    -------                                                                                | 
        None : Fills in `out`                                                              | 
                                                                                           | 
    """                                                                                    | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                 | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                 | 
                                                                                           | 
    # TODO: Implement for Task 3.2.                                                        | 
                                                                                           | 
    # Determine dimensions and ensure they are integers                                    | 
    if len(out_shape) == 3:                                                                | 
        batch_size = int(out_shape[0])                                                     | 
        M = int(out_shape[1])                                                              | 
        N = int(out_shape[2])                                                              | 
        out_batch_stride = int(out_strides[0])                                             | 
        out_i_stride = int(out_strides[1])                                                 | 
        out_j_stride = int(out_strides[2])                                                 | 
    else:                                                                                  | 
        batch_size = 1                                                                     | 
        M = int(out_shape[0])                                                              | 
        N = int(out_shape[1])                                                              | 
        out_batch_stride = 0                                                               | 
        out_i_stride = int(out_strides[0])                                                 | 
        out_j_stride = int(out_strides[1])                                                 | 
                                                                                           | 
    K = int(a_shape[-1])  # Since a_shape[-1] == b_shape[-2]                               | 
                                                                                           | 
    # Get strides for 'a' tensor                                                           | 
    if len(a_shape) == 3:                                                                  | 
        a_i_stride = int(a_strides[1])                                                     | 
        a_k_stride = int(a_strides[2])                                                     | 
    else:                                                                                  | 
        a_i_stride = int(a_strides[0])                                                     | 
        a_k_stride = int(a_strides[1])                                                     | 
                                                                                           | 
    # Get strides for 'b' tensor                                                           | 
    if len(b_shape) == 3:                                                                  | 
        b_k_stride = int(b_strides[1])                                                     | 
        b_j_stride = int(b_strides[2])                                                     | 
    else:                                                                                  | 
        b_k_stride = int(b_strides[0])                                                     | 
        b_j_stride = int(b_strides[1])                                                     | 
                                                                                           | 
    # Outer loop in parallel over the batch dimension                                      | 
    for n in prange(batch_size):-----------------------------------------------------------| #5
        # Handle broadcasting in the batch dimension                                       | 
        a_n = n if len(a_shape) == 3 and a_shape[0] > 1 else 0                             | 
        b_n = n if len(b_shape) == 3 and b_shape[0] > 1 else 0                             | 
                                                                                           | 
        for i in range(M):                                                                 | 
            for j in range(N):                                                             | 
                sum = 0.0  # Local accumulator                                             | 
                for k in range(K):                                                         | 
                    # Compute positions in 'a' and 'b' storage                             | 
                    a_pos = int(a_n * a_batch_stride + i * a_i_stride + k * a_k_stride)    | 
                    b_pos = int(b_n * b_batch_stride + k * b_k_stride + j * b_j_stride)    | 
                                                                                           | 
                    # Accumulate the product                                               | 
                    sum += a_storage[a_pos] * b_storage[b_pos]                             | 
                                                                                           | 
                # Compute position in 'out' storage                                        | 
                out_pos = int(                                                             | 
                    n * out_batch_stride + i * out_i_stride + j * out_j_stride             | 
                )                                                                          | 
                                                                                           | 
                # Write the accumulated sum to the output storage                          | 
                out[out_pos] = sum                                                         | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
(.venv) bryanguin@Bryans-MacBook-Air mod3-btguin % 


Module 3.5

run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05

Epoch 0 | Loss: 4.7287 | Correct: 38 | Time: 18.6289 sec
Epoch 10 | Loss: 3.2748 | Correct: 50 | Time: 0.1212 sec
Epoch 20 | Loss: 1.5829 | Correct: 50 | Time: 0.1203 sec
Epoch 30 | Loss: 1.1938 | Correct: 49 | Time: 0.1227 sec
Epoch 40 | Loss: 1.5169 | Correct: 50 | Time: 0.2538 sec
Epoch 50 | Loss: 1.1646 | Correct: 50 | Time: 0.1217 sec
Epoch 60 | Loss: 1.2597 | Correct: 50 | Time: 0.1225 sec
Epoch 70 | Loss: 0.7410 | Correct: 50 | Time: 0.1196 sec
Epoch 80 | Loss: 0.7944 | Correct: 49 | Time: 0.1221 sec
Epoch 90 | Loss: 0.4095 | Correct: 49 | Time: 0.1196 sec
Epoch 100 | Loss: 2.1673 | Correct: 48 | Time: 0.1202 sec
Epoch 110 | Loss: 1.4308 | Correct: 49 | Time: 0.1211 sec
Epoch 120 | Loss: 2.1069 | Correct: 48 | Time: 0.1227 sec
Epoch 130 | Loss: 1.0551 | Correct: 49 | Time: 0.2723 sec
Epoch 140 | Loss: 0.2304 | Correct: 50 | Time: 0.1206 sec
Epoch 150 | Loss: 0.4501 | Correct: 49 | Time: 0.1205 sec
Epoch 160 | Loss: 1.1133 | Correct: 50 | Time: 0.1218 sec
Epoch 170 | Loss: 1.2943 | Correct: 49 | Time: 0.1216 sec
Epoch 180 | Loss: 0.4548 | Correct: 50 | Time: 0.1236 sec
Epoch 190 | Loss: 0.0185 | Correct: 50 | Time: 0.1249 sec
Epoch 200 | Loss: 1.2354 | Correct: 50 | Time: 0.1215 sec
Epoch 210 | Loss: 0.1752 | Correct: 49 | Time: 0.1225 sec
Epoch 220 | Loss: 0.0549 | Correct: 49 | Time: 0.2753 sec
Epoch 230 | Loss: 0.1330 | Correct: 49 | Time: 0.1221 sec
Epoch 240 | Loss: 0.2283 | Correct: 49 | Time: 0.1299 sec
Epoch 250 | Loss: 0.5202 | Correct: 50 | Time: 0.1208 sec
Epoch 260 | Loss: 0.0368 | Correct: 50 | Time: 0.1196 sec
Epoch 270 | Loss: 0.0103 | Correct: 49 | Time: 0.1229 sec
Epoch 280 | Loss: 0.3557 | Correct: 50 | Time: 0.1195 sec
Epoch 290 | Loss: 0.1873 | Correct: 49 | Time: 0.1240 sec
Epoch 300 | Loss: 0.0005 | Correct: 49 | Time: 0.1202 sec
Epoch 310 | Loss: 1.0919 | Correct: 49 | Time: 0.1638 sec
Epoch 320 | Loss: 0.0634 | Correct: 50 | Time: 0.1237 sec
Epoch 330 | Loss: 0.0425 | Correct: 50 | Time: 0.1185 sec
Epoch 340 | Loss: 0.0165 | Correct: 50 | Time: 0.1193 sec
Epoch 350 | Loss: 0.0004 | Correct: 49 | Time: 0.1239 sec
Epoch 360 | Loss: 0.0030 | Correct: 50 | Time: 0.1201 sec
Epoch 370 | Loss: 0.9524 | Correct: 50 | Time: 0.1212 sec
Epoch 380 | Loss: 0.0049 | Correct: 50 | Time: 0.1276 sec
Epoch 390 | Loss: 0.2564 | Correct: 50 | Time: 0.1334 sec
Epoch 400 | Loss: 0.9309 | Correct: 50 | Time: 0.2339 sec
Epoch 410 | Loss: 0.0508 | Correct: 49 | Time: 0.1198 sec
Epoch 420 | Loss: 0.0283 | Correct: 50 | Time: 0.1227 sec
Epoch 430 | Loss: 0.0663 | Correct: 49 | Time: 0.1279 sec
Epoch 440 | Loss: 0.3928 | Correct: 49 | Time: 0.1321 sec
Epoch 450 | Loss: 0.0402 | Correct: 50 | Time: 0.1180 sec
Epoch 460 | Loss: 0.1009 | Correct: 49 | Time: 0.1185 sec
Epoch 470 | Loss: 0.0128 | Correct: 49 | Time: 0.1173 sec
Epoch 480 | Loss: 0.0228 | Correct: 50 | Time: 0.1243 sec
Epoch 490 | Loss: 1.0617 | Correct: 49 | Time: 0.2715 sec

Average Time per Epoch: 0.1725 sec

run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05

Epoch 0 | Loss: 4.8170 | Correct: 43 | Time: 4.5099 sec
Epoch 10 | Loss: 2.0487 | Correct: 48 | Time: 1.8207 sec
Epoch 20 | Loss: 1.2795 | Correct: 50 | Time: 1.8515 sec
Epoch 30 | Loss: 0.6334 | Correct: 50 | Time: 1.9442 sec
Epoch 40 | Loss: 0.3698 | Correct: 49 | Time: 1.8434 sec
Epoch 50 | Loss: 1.0350 | Correct: 50 | Time: 2.1429 sec
Epoch 60 | Loss: 0.6137 | Correct: 49 | Time: 1.8242 sec
Epoch 70 | Loss: 0.1426 | Correct: 50 | Time: 1.8249 sec
Epoch 80 | Loss: 0.4869 | Correct: 50 | Time: 2.5171 sec
Epoch 90 | Loss: 0.0559 | Correct: 49 | Time: 1.8039 sec
Epoch 100 | Loss: 0.2238 | Correct: 50 | Time: 1.7964 sec
Epoch 110 | Loss: 0.0359 | Correct: 50 | Time: 1.7835 sec
Epoch 120 | Loss: 0.2065 | Correct: 50 | Time: 1.8542 sec
Epoch 130 | Loss: 0.0363 | Correct: 50 | Time: 2.2410 sec
Epoch 140 | Loss: 0.0116 | Correct: 50 | Time: 1.8079 sec
Epoch 150 | Loss: 0.1036 | Correct: 50 | Time: 1.8660 sec
Epoch 160 | Loss: 0.7722 | Correct: 50 | Time: 1.9574 sec
Epoch 170 | Loss: 0.5213 | Correct: 50 | Time: 1.7815 sec
Epoch 180 | Loss: 0.3142 | Correct: 50 | Time: 1.9841 sec
Epoch 190 | Loss: 0.0297 | Correct: 50 | Time: 1.8662 sec
Epoch 200 | Loss: 0.0084 | Correct: 50 | Time: 1.7822 sec
Epoch 210 | Loss: 0.1091 | Correct: 50 | Time: 2.3075 sec
Epoch 220 | Loss: 0.1772 | Correct: 50 | Time: 1.8812 sec
Epoch 230 | Loss: 0.0592 | Correct: 50 | Time: 1.7799 sec
Epoch 240 | Loss: 0.0355 | Correct: 50 | Time: 1.8070 sec
Epoch 250 | Loss: 0.5768 | Correct: 50 | Time: 1.8522 sec
Epoch 260 | Loss: 0.0730 | Correct: 50 | Time: 2.4953 sec
Epoch 270 | Loss: 0.0004 | Correct: 50 | Time: 1.7818 sec
Epoch 280 | Loss: 0.5480 | Correct: 50 | Time: 1.7774 sec
Epoch 290 | Loss: 0.0245 | Correct: 50 | Time: 1.8785 sec
Epoch 300 | Loss: 0.2321 | Correct: 50 | Time: 1.8171 sec
Epoch 310 | Loss: 0.4330 | Correct: 50 | Time: 2.2226 sec
Epoch 320 | Loss: 0.2840 | Correct: 50 | Time: 1.8490 sec
Epoch 330 | Loss: 0.1778 | Correct: 50 | Time: 1.7696 sec
Epoch 340 | Loss: 0.1532 | Correct: 50 | Time: 2.0991 sec
Epoch 350 | Loss: 0.0072 | Correct: 50 | Time: 1.8675 sec
Epoch 360 | Loss: 0.0498 | Correct: 50 | Time: 1.8182 sec
Epoch 370 | Loss: 0.0239 | Correct: 50 | Time: 1.7812 sec
Epoch 380 | Loss: 0.0841 | Correct: 50 | Time: 1.7928 sec
Epoch 390 | Loss: 0.1261 | Correct: 50 | Time: 2.4735 sec
Epoch 400 | Loss: 0.0129 | Correct: 50 | Time: 1.7887 sec
Epoch 410 | Loss: 0.3125 | Correct: 50 | Time: 1.7707 sec
Epoch 420 | Loss: 0.0046 | Correct: 50 | Time: 1.8354 sec
Epoch 430 | Loss: 0.4708 | Correct: 50 | Time: 1.7837 sec
Epoch 440 | Loss: 0.1972 | Correct: 50 | Time: 2.5111 sec
Epoch 450 | Loss: 0.1705 | Correct: 50 | Time: 1.8138 sec
Epoch 460 | Loss: 0.1672 | Correct: 50 | Time: 1.7810 sec
Epoch 470 | Loss: 0.0326 | Correct: 50 | Time: 1.7826 sec
Epoch 480 | Loss: 0.0126 | Correct: 50 | Time: 1.7747 sec
Epoch 490 | Loss: 0.0068 | Correct: 50 | Time: 2.5729 sec

Average Time per Epoch: 1.9364 sec

run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05

Epoch 0 | Loss: 6.3923 | Correct: 35 | Time: 17.9210 sec
Epoch 10 | Loss: 4.7978 | Correct: 42 | Time: 0.1177 sec
Epoch 20 | Loss: 4.2802 | Correct: 45 | Time: 0.1192 sec
Epoch 30 | Loss: 3.4706 | Correct: 45 | Time: 0.1181 sec
Epoch 40 | Loss: 3.6975 | Correct: 49 | Time: 0.1197 sec
Epoch 50 | Loss: 2.5009 | Correct: 47 | Time: 0.1178 sec
Epoch 60 | Loss: 1.0466 | Correct: 49 | Time: 0.2615 sec
Epoch 70 | Loss: 2.1830 | Correct: 49 | Time: 0.1192 sec
Epoch 80 | Loss: 2.6067 | Correct: 50 | Time: 0.1204 sec
Epoch 90 | Loss: 1.2065 | Correct: 49 | Time: 0.1180 sec
Epoch 100 | Loss: 1.1130 | Correct: 49 | Time: 0.1187 sec
Epoch 110 | Loss: 0.9603 | Correct: 50 | Time: 0.1307 sec
Epoch 120 | Loss: 1.1474 | Correct: 49 | Time: 0.1176 sec
Epoch 130 | Loss: 0.7451 | Correct: 50 | Time: 0.1198 sec
Epoch 140 | Loss: 2.6213 | Correct: 48 | Time: 0.1179 sec
Epoch 150 | Loss: 1.5133 | Correct: 50 | Time: 0.2427 sec
Epoch 160 | Loss: 0.3118 | Correct: 50 | Time: 0.1357 sec
Epoch 170 | Loss: 1.1785 | Correct: 50 | Time: 0.1217 sec
Epoch 180 | Loss: 1.1634 | Correct: 50 | Time: 0.1242 sec
Epoch 190 | Loss: 0.1444 | Correct: 49 | Time: 0.1182 sec
Epoch 200 | Loss: 0.5683 | Correct: 49 | Time: 0.1188 sec
Epoch 210 | Loss: 0.8015 | Correct: 49 | Time: 0.1319 sec
Epoch 220 | Loss: 1.0367 | Correct: 50 | Time: 0.1197 sec
Epoch 230 | Loss: 1.2520 | Correct: 50 | Time: 0.1175 sec
Epoch 240 | Loss: 0.4528 | Correct: 50 | Time: 0.1851 sec
Epoch 250 | Loss: 1.1055 | Correct: 48 | Time: 0.2410 sec
Epoch 260 | Loss: 0.9739 | Correct: 50 | Time: 0.1194 sec
Epoch 270 | Loss: 0.0847 | Correct: 50 | Time: 0.1247 sec
Epoch 280 | Loss: 0.0696 | Correct: 50 | Time: 0.1217 sec
Epoch 290 | Loss: 1.2662 | Correct: 50 | Time: 0.1176 sec
Epoch 300 | Loss: 0.0444 | Correct: 48 | Time: 0.1212 sec
Epoch 310 | Loss: 0.1317 | Correct: 50 | Time: 0.1195 sec
Epoch 320 | Loss: 0.7023 | Correct: 50 | Time: 0.1234 sec
Epoch 330 | Loss: 1.3013 | Correct: 50 | Time: 0.1197 sec
Epoch 340 | Loss: 0.6825 | Correct: 50 | Time: 0.2252 sec
Epoch 350 | Loss: 2.1005 | Correct: 49 | Time: 0.1225 sec
Epoch 360 | Loss: 0.2255 | Correct: 50 | Time: 0.1184 sec
Epoch 370 | Loss: 1.7860 | Correct: 50 | Time: 0.1182 sec
Epoch 380 | Loss: 0.3665 | Correct: 50 | Time: 0.1201 sec
Epoch 390 | Loss: 0.9009 | Correct: 50 | Time: 0.1206 sec
Epoch 400 | Loss: 1.2645 | Correct: 50 | Time: 0.1250 sec
Epoch 410 | Loss: 0.8058 | Correct: 50 | Time: 0.1209 sec
Epoch 420 | Loss: 0.9719 | Correct: 50 | Time: 0.1222 sec
Epoch 430 | Loss: 0.0542 | Correct: 48 | Time: 0.2279 sec
Epoch 440 | Loss: 0.7039 | Correct: 50 | Time: 0.1179 sec
Epoch 450 | Loss: 1.2312 | Correct: 50 | Time: 0.1198 sec
Epoch 460 | Loss: 0.0837 | Correct: 50 | Time: 0.1314 sec
Epoch 470 | Loss: 1.1866 | Correct: 50 | Time: 0.1186 sec
Epoch 480 | Loss: 0.5599 | Correct: 50 | Time: 0.1201 sec
Epoch 490 | Loss: 0.1663 | Correct: 50 | Time: 0.1170 sec

Average Time per Epoch: 0.1684 sec

run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05

Epoch 0 | Loss: 11.3328 | Correct: 16 | Time: 3.8078 sec
Epoch 10 | Loss: 5.8084 | Correct: 35 | Time: 1.7690 sec
Epoch 20 | Loss: 5.1229 | Correct: 42 | Time: 1.8730 sec
Epoch 30 | Loss: 3.9053 | Correct: 44 | Time: 1.7921 sec
Epoch 40 | Loss: 3.4643 | Correct: 36 | Time: 2.6099 sec
Epoch 50 | Loss: 2.3427 | Correct: 45 | Time: 1.8317 sec
Epoch 60 | Loss: 3.5000 | Correct: 45 | Time: 1.8277 sec
Epoch 70 | Loss: 3.1191 | Correct: 45 | Time: 1.8035 sec
Epoch 80 | Loss: 2.3624 | Correct: 48 | Time: 1.8035 sec
Epoch 90 | Loss: 0.9151 | Correct: 48 | Time: 2.5225 sec
Epoch 100 | Loss: 2.1125 | Correct: 48 | Time: 1.7884 sec
Epoch 110 | Loss: 1.1786 | Correct: 48 | Time: 1.8011 sec
Epoch 120 | Loss: 1.6335 | Correct: 48 | Time: 1.8657 sec
Epoch 130 | Loss: 5.5614 | Correct: 42 | Time: 1.7941 sec
Epoch 140 | Loss: 0.5925 | Correct: 48 | Time: 2.1126 sec
Epoch 150 | Loss: 2.3857 | Correct: 47 | Time: 1.8864 sec
Epoch 160 | Loss: 2.0540 | Correct: 48 | Time: 2.0490 sec
Epoch 170 | Loss: 0.3783 | Correct: 48 | Time: 1.7812 sec
Epoch 180 | Loss: 2.6565 | Correct: 45 | Time: 1.8216 sec
Epoch 190 | Loss: 5.2728 | Correct: 45 | Time: 1.9528 sec
Epoch 200 | Loss: 1.9427 | Correct: 49 | Time: 1.7969 sec
Epoch 210 | Loss: 0.7015 | Correct: 48 | Time: 2.4984 sec
Epoch 220 | Loss: 1.0378 | Correct: 48 | Time: 1.8364 sec
Epoch 230 | Loss: 4.4882 | Correct: 43 | Time: 1.7826 sec
Epoch 240 | Loss: 1.0920 | Correct: 48 | Time: 1.7884 sec
Epoch 250 | Loss: 1.7993 | Correct: 48 | Time: 1.8364 sec
Epoch 260 | Loss: 0.4591 | Correct: 47 | Time: 2.0780 sec
Epoch 270 | Loss: 0.0918 | Correct: 49 | Time: 1.7729 sec
Epoch 280 | Loss: 2.1070 | Correct: 48 | Time: 2.1969 sec
Epoch 290 | Loss: 0.8128 | Correct: 49 | Time: 1.8452 sec
Epoch 300 | Loss: 1.5073 | Correct: 49 | Time: 1.7975 sec
Epoch 310 | Loss: 1.6647 | Correct: 49 | Time: 1.8575 sec
Epoch 320 | Loss: 0.6020 | Correct: 48 | Time: 1.8353 sec
Epoch 330 | Loss: 0.7949 | Correct: 48 | Time: 2.4592 sec
Epoch 340 | Loss: 0.9963 | Correct: 49 | Time: 1.7754 sec
Epoch 350 | Loss: 1.2014 | Correct: 49 | Time: 1.8669 sec
Epoch 360 | Loss: 0.2287 | Correct: 49 | Time: 1.9636 sec
Epoch 370 | Loss: 0.7848 | Correct: 48 | Time: 1.7878 sec
Epoch 380 | Loss: 0.2306 | Correct: 47 | Time: 2.4215 sec
Epoch 390 | Loss: 0.2283 | Correct: 49 | Time: 1.8497 sec
Epoch 400 | Loss: 0.2398 | Correct: 49 | Time: 1.9563 sec
Epoch 410 | Loss: 0.8410 | Correct: 50 | Time: 1.7626 sec
Epoch 420 | Loss: 2.4972 | Correct: 49 | Time: 1.8563 sec
Epoch 430 | Loss: 1.5309 | Correct: 50 | Time: 1.8960 sec
Epoch 440 | Loss: 1.9338 | Correct: 49 | Time: 1.7762 sec
Epoch 450 | Loss: 0.3020 | Correct: 47 | Time: 2.4941 sec
Epoch 460 | Loss: 1.8237 | Correct: 49 | Time: 1.7706 sec
Epoch 470 | Loss: 0.5916 | Correct: 48 | Time: 1.7811 sec
Epoch 480 | Loss: 0.4016 | Correct: 47 | Time: 1.7771 sec
Epoch 490 | Loss: 0.3680 | Correct: 49 | Time: 1.8509 sec

Average Time per Epoch: 1.9389 sec

run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05

Epoch 0 | Loss: 6.6991 | Correct: 31 | Time: 18.2590 sec
Epoch 10 | Loss: 4.8904 | Correct: 38 | Time: 0.1631 sec
Epoch 20 | Loss: 5.2682 | Correct: 41 | Time: 0.1194 sec
Epoch 30 | Loss: 2.8961 | Correct: 33 | Time: 0.1163 sec
Epoch 40 | Loss: 3.9500 | Correct: 43 | Time: 0.1181 sec
Epoch 50 | Loss: 4.3993 | Correct: 41 | Time: 0.2684 sec
Epoch 60 | Loss: 4.8651 | Correct: 45 | Time: 0.1184 sec
Epoch 70 | Loss: 1.7469 | Correct: 42 | Time: 0.1168 sec
Epoch 80 | Loss: 2.7370 | Correct: 46 | Time: 0.1180 sec
Epoch 90 | Loss: 2.7303 | Correct: 42 | Time: 0.1183 sec
Epoch 100 | Loss: 2.5274 | Correct: 48 | Time: 0.1192 sec
Epoch 110 | Loss: 1.7728 | Correct: 37 | Time: 0.1167 sec
Epoch 120 | Loss: 2.0864 | Correct: 49 | Time: 0.1170 sec
Epoch 130 | Loss: 3.2368 | Correct: 47 | Time: 0.1155 sec
Epoch 140 | Loss: 1.7823 | Correct: 49 | Time: 0.2346 sec
Epoch 150 | Loss: 1.7613 | Correct: 50 | Time: 0.1159 sec
Epoch 160 | Loss: 2.1494 | Correct: 50 | Time: 0.1187 sec
Epoch 170 | Loss: 1.6472 | Correct: 49 | Time: 0.1162 sec
Epoch 180 | Loss: 1.4431 | Correct: 50 | Time: 0.1162 sec
Epoch 190 | Loss: 1.6761 | Correct: 50 | Time: 0.1222 sec
Epoch 200 | Loss: 1.6900 | Correct: 50 | Time: 0.1179 sec
Epoch 210 | Loss: 0.6199 | Correct: 50 | Time: 0.1171 sec
Epoch 220 | Loss: 1.2927 | Correct: 50 | Time: 0.1267 sec
Epoch 230 | Loss: 1.6622 | Correct: 50 | Time: 0.1732 sec
Epoch 240 | Loss: 1.4860 | Correct: 50 | Time: 0.1166 sec
Epoch 250 | Loss: 1.6066 | Correct: 50 | Time: 0.1198 sec
Epoch 260 | Loss: 0.3864 | Correct: 50 | Time: 0.1176 sec
Epoch 270 | Loss: 0.9170 | Correct: 50 | Time: 0.1166 sec
Epoch 280 | Loss: 0.9561 | Correct: 48 | Time: 0.1298 sec
Epoch 290 | Loss: 0.2932 | Correct: 50 | Time: 0.1155 sec
Epoch 300 | Loss: 0.9962 | Correct: 50 | Time: 0.1184 sec
Epoch 310 | Loss: 0.8629 | Correct: 50 | Time: 0.1214 sec
Epoch 320 | Loss: 1.6279 | Correct: 50 | Time: 0.1189 sec
Epoch 330 | Loss: 0.3198 | Correct: 50 | Time: 0.2626 sec
Epoch 340 | Loss: 0.8099 | Correct: 50 | Time: 0.1188 sec
Epoch 350 | Loss: 0.6008 | Correct: 50 | Time: 0.1165 sec
Epoch 360 | Loss: 0.3869 | Correct: 49 | Time: 0.1169 sec
Epoch 370 | Loss: 0.2182 | Correct: 50 | Time: 0.1169 sec
Epoch 380 | Loss: 0.5601 | Correct: 50 | Time: 0.1180 sec
Epoch 390 | Loss: 0.2131 | Correct: 50 | Time: 0.1286 sec
Epoch 400 | Loss: 0.7508 | Correct: 50 | Time: 0.1175 sec
Epoch 410 | Loss: 0.1556 | Correct: 50 | Time: 0.1164 sec
Epoch 420 | Loss: 0.5523 | Correct: 50 | Time: 0.2363 sec
Epoch 430 | Loss: 0.2122 | Correct: 50 | Time: 0.1174 sec
Epoch 440 | Loss: 0.4961 | Correct: 50 | Time: 0.1196 sec
Epoch 450 | Loss: 0.5428 | Correct: 50 | Time: 0.1248 sec
Epoch 460 | Loss: 0.0877 | Correct: 50 | Time: 0.1195 sec
Epoch 470 | Loss: 0.6391 | Correct: 50 | Time: 0.1175 sec
Epoch 480 | Loss: 0.4859 | Correct: 50 | Time: 0.1189 sec
Epoch 490 | Loss: 0.4545 | Correct: 50 | Time: 0.1181 sec

Average Time per Epoch: 0.1670 sec

run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05

Epoch 0 | Loss: 7.9066 | Correct: 34 | Time: 5.9283 sec
Epoch 10 | Loss: 3.6088 | Correct: 43 | Time: 1.9675 sec
Epoch 20 | Loss: 3.9282 | Correct: 46 | Time: 2.0248 sec
Epoch 30 | Loss: 2.3512 | Correct: 44 | Time: 1.9409 sec
Epoch 40 | Loss: 2.5080 | Correct: 44 | Time: 2.3820 sec
Epoch 50 | Loss: 1.2392 | Correct: 47 | Time: 1.9223 sec
Epoch 60 | Loss: 3.3710 | Correct: 45 | Time: 2.0135 sec
Epoch 70 | Loss: 1.9978 | Correct: 46 | Time: 2.8091 sec
Epoch 80 | Loss: 1.6462 | Correct: 45 | Time: 1.9328 sec
Epoch 90 | Loss: 2.1702 | Correct: 47 | Time: 2.0091 sec
Epoch 100 | Loss: 1.4440 | Correct: 47 | Time: 2.1641 sec
Epoch 110 | Loss: 1.7380 | Correct: 47 | Time: 1.9537 sec
Epoch 120 | Loss: 1.6918 | Correct: 49 | Time: 1.9604 sec
Epoch 130 | Loss: 1.0133 | Correct: 49 | Time: 1.9455 sec
Epoch 140 | Loss: 0.9405 | Correct: 49 | Time: 2.2802 sec
Epoch 150 | Loss: 1.5311 | Correct: 49 | Time: 1.9091 sec
Epoch 160 | Loss: 1.7180 | Correct: 50 | Time: 2.0224 sec
Epoch 170 | Loss: 2.1870 | Correct: 49 | Time: 2.8024 sec
Epoch 180 | Loss: 0.4196 | Correct: 48 | Time: 1.9211 sec
Epoch 190 | Loss: 1.3121 | Correct: 50 | Time: 1.9681 sec
Epoch 200 | Loss: 1.2823 | Correct: 48 | Time: 2.6465 sec
Epoch 210 | Loss: 1.3368 | Correct: 50 | Time: 1.9592 sec
Epoch 220 | Loss: 0.5458 | Correct: 49 | Time: 1.9615 sec
Epoch 230 | Loss: 0.6405 | Correct: 49 | Time: 2.2184 sec
Epoch 240 | Loss: 1.3284 | Correct: 50 | Time: 1.9246 sec
Epoch 250 | Loss: 0.7977 | Correct: 50 | Time: 1.8979 sec
Epoch 260 | Loss: 0.6079 | Correct: 50 | Time: 2.0630 sec
Epoch 270 | Loss: 0.6308 | Correct: 50 | Time: 2.2016 sec
Epoch 280 | Loss: 0.1824 | Correct: 50 | Time: 1.9249 sec
Epoch 290 | Loss: 1.3831 | Correct: 50 | Time: 2.0003 sec
Epoch 300 | Loss: 1.0043 | Correct: 50 | Time: 2.5448 sec
Epoch 310 | Loss: 0.3774 | Correct: 50 | Time: 1.9416 sec
Epoch 320 | Loss: 0.7609 | Correct: 50 | Time: 2.0002 sec
Epoch 330 | Loss: 0.2917 | Correct: 50 | Time: 2.6407 sec
Epoch 340 | Loss: 0.6809 | Correct: 50 | Time: 1.9307 sec
Epoch 350 | Loss: 0.2147 | Correct: 50 | Time: 1.9256 sec
Epoch 360 | Loss: 0.4211 | Correct: 50 | Time: 2.1508 sec
Epoch 370 | Loss: 0.5039 | Correct: 50 | Time: 1.9297 sec
Epoch 380 | Loss: 0.0763 | Correct: 50 | Time: 1.9502 sec
Epoch 390 | Loss: 0.2939 | Correct: 50 | Time: 2.0074 sec
Epoch 400 | Loss: 0.9525 | Correct: 50 | Time: 2.2260 sec
Epoch 410 | Loss: 0.9780 | Correct: 50 | Time: 1.9004 sec
Epoch 420 | Loss: 0.2731 | Correct: 50 | Time: 2.0097 sec
Epoch 430 | Loss: 0.1331 | Correct: 50 | Time: 2.4568 sec
Epoch 440 | Loss: 0.2044 | Correct: 50 | Time: 1.9187 sec
Epoch 450 | Loss: 0.7483 | Correct: 50 | Time: 1.9850 sec
Epoch 460 | Loss: 0.5027 | Correct: 50 | Time: 2.7719 sec
Epoch 470 | Loss: 0.3186 | Correct: 50 | Time: 1.9116 sec
Epoch 480 | Loss: 0.1204 | Correct: 50 | Time: 1.9899 sec
Epoch 490 | Loss: 0.0472 | Correct: 50 | Time: 2.3947 sec

Average Time per Epoch: 2.1084 sec

Bigger models

run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET simple --RATE 0.05

Epoch 0 | Loss: 3.4191 | Correct: 45 | Time: 20.6011 sec
Epoch 10 | Loss: 0.7426 | Correct: 50 | Time: 0.2978 sec
Epoch 20 | Loss: 0.8394 | Correct: 50 | Time: 0.5665 sec
Epoch 30 | Loss: 0.7098 | Correct: 50 | Time: 0.2886 sec
Epoch 40 | Loss: 0.0589 | Correct: 50 | Time: 0.2807 sec
Epoch 50 | Loss: 0.5642 | Correct: 50 | Time: 0.2952 sec
Epoch 60 | Loss: 0.2042 | Correct: 50 | Time: 0.3007 sec
Epoch 70 | Loss: 0.1508 | Correct: 50 | Time: 0.2948 sec
Epoch 80 | Loss: 0.4018 | Correct: 50 | Time: 0.2864 sec
Epoch 90 | Loss: 0.2298 | Correct: 50 | Time: 0.2868 sec
Epoch 100 | Loss: 0.0107 | Correct: 50 | Time: 0.2820 sec
Epoch 110 | Loss: 0.1088 | Correct: 50 | Time: 0.2839 sec
Epoch 120 | Loss: 0.0259 | Correct: 50 | Time: 0.2865 sec
Epoch 130 | Loss: 0.2226 | Correct: 50 | Time: 0.2839 sec
Epoch 140 | Loss: 0.0139 | Correct: 50 | Time: 0.2837 sec
Epoch 150 | Loss: 0.0524 | Correct: 50 | Time: 0.2988 sec
Epoch 160 | Loss: 0.0362 | Correct: 50 | Time: 0.2919 sec
Epoch 170 | Loss: 0.0363 | Correct: 50 | Time: 0.3659 sec
Epoch 180 | Loss: 0.1745 | Correct: 50 | Time: 0.2815 sec
Epoch 190 | Loss: 0.0437 | Correct: 50 | Time: 0.2799 sec
Epoch 200 | Loss: 0.0277 | Correct: 50 | Time: 0.2842 sec
Epoch 210 | Loss: 0.0109 | Correct: 50 | Time: 0.5908 sec
Epoch 220 | Loss: 0.0370 | Correct: 50 | Time: 0.2840 sec
Epoch 230 | Loss: 0.0838 | Correct: 50 | Time: 0.2962 sec
Epoch 240 | Loss: 0.0157 | Correct: 50 | Time: 0.2909 sec
Epoch 250 | Loss: 0.0727 | Correct: 50 | Time: 0.5752 sec
Epoch 260 | Loss: 0.0213 | Correct: 50 | Time: 0.2824 sec
Epoch 270 | Loss: 0.1119 | Correct: 50 | Time: 0.2878 sec
Epoch 280 | Loss: 0.0311 | Correct: 50 | Time: 0.2818 sec
Epoch 290 | Loss: 0.0221 | Correct: 50 | Time: 0.4724 sec
Epoch 300 | Loss: 0.0053 | Correct: 50 | Time: 0.2980 sec
Epoch 310 | Loss: 0.0070 | Correct: 50 | Time: 0.2965 sec
Epoch 320 | Loss: 0.0447 | Correct: 50 | Time: 0.2839 sec
Epoch 330 | Loss: 0.0695 | Correct: 50 | Time: 0.5131 sec
Epoch 340 | Loss: 0.0186 | Correct: 50 | Time: 0.2851 sec
Epoch 350 | Loss: 0.0562 | Correct: 50 | Time: 0.2870 sec
Epoch 360 | Loss: 0.0563 | Correct: 50 | Time: 0.2880 sec
Epoch 370 | Loss: 0.0038 | Correct: 50 | Time: 0.2857 sec
Epoch 380 | Loss: 0.0027 | Correct: 50 | Time: 0.2984 sec
Epoch 390 | Loss: 0.0045 | Correct: 50 | Time: 0.2971 sec
Epoch 400 | Loss: 0.0881 | Correct: 50 | Time: 0.2961 sec
Epoch 410 | Loss: 0.0051 | Correct: 50 | Time: 0.2979 sec
Epoch 420 | Loss: 0.0215 | Correct: 50 | Time: 0.2846 sec
Epoch 430 | Loss: 0.0009 | Correct: 50 | Time: 0.2806 sec
Epoch 440 | Loss: 0.0038 | Correct: 50 | Time: 0.2822 sec
Epoch 450 | Loss: 0.0223 | Correct: 50 | Time: 0.2877 sec
Epoch 460 | Loss: 0.0030 | Correct: 50 | Time: 0.2861 sec
Epoch 470 | Loss: 0.0169 | Correct: 50 | Time: 0.2962 sec
Epoch 480 | Loss: 0.0306 | Correct: 50 | Time: 0.2814 sec
Epoch 490 | Loss: 0.0284 | Correct: 50 | Time: 0.2961 sec

Average Time per Epoch: 0.3633 sec

run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET simple --RATE 0.05

Epoch 0 | Loss: 8.1230 | Correct: 22 | Time: 5.0145 sec
Epoch 10 | Loss: 0.5206 | Correct: 48 | Time: 1.9990 sec
Epoch 20 | Loss: 0.5556 | Correct: 49 | Time: 2.0865 sec
Epoch 30 | Loss: 0.7699 | Correct: 49 | Time: 2.4500 sec
Epoch 40 | Loss: 1.2071 | Correct: 50 | Time: 2.0857 sec
Epoch 50 | Loss: 1.1223 | Correct: 50 | Time: 2.0899 sec
Epoch 60 | Loss: 0.1752 | Correct: 50 | Time: 2.0412 sec
Epoch 70 | Loss: 0.0239 | Correct: 49 | Time: 2.0905 sec
Epoch 80 | Loss: 0.0883 | Correct: 49 | Time: 2.4220 sec
Epoch 90 | Loss: 0.1437 | Correct: 49 | Time: 2.1121 sec
Epoch 100 | Loss: 0.0349 | Correct: 50 | Time: 2.0185 sec
Epoch 110 | Loss: 0.0678 | Correct: 50 | Time: 2.0158 sec
Epoch 120 | Loss: 0.0507 | Correct: 50 | Time: 2.9160 sec
Epoch 130 | Loss: 0.0250 | Correct: 50 | Time: 2.0225 sec
Epoch 140 | Loss: 0.0556 | Correct: 50 | Time: 2.0207 sec
Epoch 150 | Loss: 0.2490 | Correct: 50 | Time: 2.0882 sec
Epoch 160 | Loss: 0.0277 | Correct: 50 | Time: 2.7499 sec
Epoch 170 | Loss: 0.0172 | Correct: 50 | Time: 2.0383 sec
Epoch 180 | Loss: 0.4215 | Correct: 50 | Time: 2.0181 sec
Epoch 190 | Loss: 0.0729 | Correct: 50 | Time: 2.0933 sec
Epoch 200 | Loss: 0.0009 | Correct: 50 | Time: 2.1508 sec
Epoch 210 | Loss: 0.1317 | Correct: 50 | Time: 2.3677 sec
Epoch 220 | Loss: 0.1539 | Correct: 50 | Time: 2.0990 sec
Epoch 230 | Loss: 0.0343 | Correct: 50 | Time: 2.0401 sec
Epoch 240 | Loss: 0.0342 | Correct: 50 | Time: 2.0349 sec
Epoch 250 | Loss: 0.0026 | Correct: 50 | Time: 2.9170 sec
Epoch 260 | Loss: 0.1987 | Correct: 50 | Time: 2.0052 sec
Epoch 270 | Loss: 0.1841 | Correct: 50 | Time: 2.0139 sec
Epoch 280 | Loss: 0.3283 | Correct: 50 | Time: 2.0993 sec
Epoch 290 | Loss: 0.0670 | Correct: 50 | Time: 2.6477 sec
Epoch 300 | Loss: 0.0065 | Correct: 50 | Time: 2.0421 sec
Epoch 310 | Loss: 0.0164 | Correct: 50 | Time: 2.0136 sec
Epoch 320 | Loss: 0.2884 | Correct: 50 | Time: 2.0854 sec
Epoch 330 | Loss: 0.0081 | Correct: 50 | Time: 2.1263 sec
Epoch 340 | Loss: 0.1606 | Correct: 50 | Time: 2.4957 sec
Epoch 350 | Loss: 0.3290 | Correct: 50 | Time: 2.0851 sec
Epoch 360 | Loss: 0.0208 | Correct: 50 | Time: 2.0106 sec
Epoch 370 | Loss: 0.1243 | Correct: 50 | Time: 2.0040 sec
Epoch 380 | Loss: 0.0253 | Correct: 50 | Time: 2.9113 sec
Epoch 390 | Loss: 0.0006 | Correct: 50 | Time: 2.0031 sec
Epoch 400 | Loss: 0.0033 | Correct: 50 | Time: 2.0187 sec
Epoch 410 | Loss: 0.2500 | Correct: 50 | Time: 2.1016 sec
Epoch 420 | Loss: 0.2291 | Correct: 50 | Time: 2.9083 sec
Epoch 430 | Loss: 0.0041 | Correct: 50 | Time: 1.9865 sec
Epoch 440 | Loss: 0.0065 | Correct: 50 | Time: 2.0068 sec
Epoch 450 | Loss: 0.0643 | Correct: 50 | Time: 2.1218 sec
Epoch 460 | Loss: 0.0033 | Correct: 50 | Time: 2.7559 sec
Epoch 470 | Loss: 0.0051 | Correct: 50 | Time: 2.0137 sec
Epoch 480 | Loss: 0.0207 | Correct: 50 | Time: 2.0908 sec
Epoch 490 | Loss: 0.0380 | Correct: 50 | Time: 1.9972 sec

Average Time per Epoch: 2.2007 sec

- Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py
