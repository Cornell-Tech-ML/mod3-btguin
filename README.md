# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

Module 3.5

run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05

Epoch 0 | Loss: 3.8241 | Correct: 35 | Time: 22.6667 sec
Epoch 10 | Loss: 2.1797 | Correct: 42 | Time: 0.1511 sec
Epoch 20 | Loss: 3.1374 | Correct: 43 | Time: 0.1443 sec
Epoch 30 | Loss: 3.4568 | Correct: 43 | Time: 0.2583 sec
Epoch 40 | Loss: 1.9533 | Correct: 45 | Time: 0.1437 sec
Epoch 50 | Loss: 2.8088 | Correct: 46 | Time: 0.1443 sec
Epoch 60 | Loss: 4.2475 | Correct: 46 | Time: 0.1449 sec
Epoch 70 | Loss: 2.0763 | Correct: 47 | Time: 0.1439 sec
Epoch 80 | Loss: 2.2727 | Correct: 47 | Time: 0.1474 sec
Epoch 90 | Loss: 2.3051 | Correct: 49 | Time: 0.1426 sec
Epoch 100 | Loss: 3.2368 | Correct: 47 | Time: 0.1408 sec
Epoch 110 | Loss: 2.7135 | Correct: 48 | Time: 0.2613 sec
Epoch 120 | Loss: 1.1568 | Correct: 49 | Time: 0.1570 sec
Epoch 130 | Loss: 1.1126 | Correct: 48 | Time: 0.1463 sec
Epoch 140 | Loss: 0.3403 | Correct: 48 | Time: 0.1431 sec
Epoch 150 | Loss: 0.5745 | Correct: 49 | Time: 0.1479 sec
Epoch 160 | Loss: 1.3071 | Correct: 48 | Time: 0.1458 sec
Epoch 170 | Loss: 2.5429 | Correct: 50 | Time: 0.1442 sec
Epoch 180 | Loss: 0.3596 | Correct: 50 | Time: 0.2301 sec
Epoch 190 | Loss: 1.3401 | Correct: 50 | Time: 0.2907 sec
Epoch 200 | Loss: 0.3300 | Correct: 48 | Time: 0.1445 sec
Epoch 210 | Loss: 1.1947 | Correct: 50 | Time: 0.1439 sec
Epoch 220 | Loss: 0.1687 | Correct: 49 | Time: 0.1483 sec
Epoch 230 | Loss: 1.3715 | Correct: 50 | Time: 0.1546 sec
Epoch 240 | Loss: 0.7741 | Correct: 49 | Time: 0.1486 sec
Epoch 250 | Loss: 1.4937 | Correct: 50 | Time: 0.1459 sec
Epoch 260 | Loss: 0.5271 | Correct: 50 | Time: 0.1965 sec
Epoch 270 | Loss: 0.5823 | Correct: 48 | Time: 0.1663 sec
Epoch 280 | Loss: 0.9818 | Correct: 50 | Time: 0.1480 sec
Epoch 290 | Loss: 0.5397 | Correct: 50 | Time: 0.1578 sec
Epoch 300 | Loss: 0.7686 | Correct: 50 | Time: 0.1441 sec
Epoch 310 | Loss: 1.0223 | Correct: 49 | Time: 0.1433 sec
Epoch 320 | Loss: 0.6936 | Correct: 50 | Time: 0.1460 sec
Epoch 330 | Loss: 0.3653 | Correct: 50 | Time: 0.1485 sec
Epoch 340 | Loss: 0.5186 | Correct: 50 | Time: 0.1948 sec
Epoch 350 | Loss: 1.2706 | Correct: 50 | Time: 0.1496 sec
Epoch 360 | Loss: 0.0170 | Correct: 50 | Time: 0.1478 sec
Epoch 370 | Loss: 0.6041 | Correct: 50 | Time: 0.1449 sec
Epoch 380 | Loss: 0.7466 | Correct: 50 | Time: 0.1473 sec
Epoch 390 | Loss: 0.8108 | Correct: 50 | Time: 0.1524 sec
Epoch 400 | Loss: 0.6568 | Correct: 50 | Time: 0.1440 sec
Epoch 410 | Loss: 0.8269 | Correct: 50 | Time: 0.1503 sec
Epoch 420 | Loss: 0.3830 | Correct: 50 | Time: 0.2976 sec
Epoch 430 | Loss: 0.3452 | Correct: 50 | Time: 0.1568 sec
Epoch 440 | Loss: 0.2391 | Correct: 50 | Time: 0.1500 sec
Epoch 450 | Loss: 0.9628 | Correct: 50 | Time: 0.1473 sec
Epoch 460 | Loss: 0.9735 | Correct: 50 | Time: 0.1556 sec
Epoch 470 | Loss: 0.3555 | Correct: 50 | Time: 0.1423 sec
Epoch 480 | Loss: 0.2435 | Correct: 50 | Time: 0.1498 sec
Epoch 490 | Loss: 0.5958 | Correct: 50 | Time: 0.2705 sec

Average Time per Epoch: 0.2125 sec

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py