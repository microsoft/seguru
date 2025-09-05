### Matrix Multiply

### Runtime costs (TODO: update the table)

| Setup arg(2048, 16)           | Time(s)       |Relative overhead|
|-------------------------------|---------------|--------|
|v1 + no-bound-check	        |1.557330589    |baseline|
|v1 + no-bound-check + fastmath |1.544111987    |--      |
|v1 + select-bound-check	    |1.574405571	|1%      |
|v1 + if-else-trap-bound-check	|6.143659647	|294%    |
|v1 + arith-immed-bound-check	|4.360865864	|180%    |
|v2 + no-bound-check	        |1.64882646	    |6%      |
|v2 + select-bound-check	    |1.648791268	|6%      |
|v2 + if-else-trap-bound-check	|6.081392385	|291%    |
|v2 + arith-immed-bound-check	|9.079779658	|483%    |
|v3 + no-bound-check	        |1.621740153    |3%      |
|v3 + select-bound-check	    |1.578538085	|1%      |
|v3 + if-else-trap-bound-check	|6.155008352	|295%    |
|v3 + arith-immed-bound-check	|3.802663921	|144%    |

### if-else-trap vs arith-immed vs select 

* Both if-else-trap and arith-immed are far more expensive than select-based inplace bound-checking. 

* `if-else-trap` one has a relative stable overhead for both v1 and v2.
This might due to that if-else-trap logic is easier to optimize out and so inner loop does not have bound checks when using if-else-trap.

* arith-immed bound check has a unstable overhead depending on the implementation.
arith-immed-bound-check is never optimized out in PTX optimization. When using arith-immed-bound-check v3 outperforms v2 and v1 since v3 will skip the inner bound check since Rust will skip bound check for the inner loop.


### inner_product_kernel(v1) vs inner_product_kernel(v2)

In theory, if we change inner_product_kernel(v1) to inner_product_kernel(v2), Rust will skip bound checking and so we might get a better performance. However, in practice, due to PTX optimization, both versions will skip bound checking for inner loop. inner_product_kernel(v2) is even more expensive than v1.

v2 will generate less-optimized ptx code, while v1 will generate highly-optimized ptx code.

Though ptx1 uses more registers, the parallelism benefits often outweigh the cost, ptx1 unrolls the loop partially (processing 4 elements per iteration).

In both cases, we did not see the bound-checking for inner loop when calculating `sum` and they are generated in MLIR but erased after some optimization.

### Wierd optimization notes

At commit [commit
61e9c91](https://github.com/MSRSSP/rs-gpu/tree/61e9c91e33fd306d8ec7fbea0e9ced1d4a4eb0f8)
and earlier, we noticed that creating a chunk from host and then pass it to GPU
is (1.58s vs 2.1s in release mode) and sometimes the debug mode is even after
then release mode.


At commit [commit 031f110](https://github.com/MSRSSP/rs-gpu/tree/031f110469e654b62d94b3d566bf85bf295e255b) However, after
we optimize the dim lib to replace the dim selection from enum value to trait
for zero-cost selection, the above wierd performance disappeared. release
(1.56s) is always faster than debug(2.1s). passing chunk from host vs creating
chunk inside gpu device is similar.