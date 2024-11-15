# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

- Docs: https://minitorch.github.io/

- Overview: https://minitorch.github.io/module3.html

You will need to modify `tensor_functions.py` slightly in this assignment.

- Tests:

```
python run_tests.py
```

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
