# xeval
X-Plane evaluation kit

## DATA Structure

```
Data   
│
└──
│  │
│  └─Airport
│    │
│    └─flight228
│      │
│      └─ X.csv
|      |  data_0.jpg
|      | ...
│      │
│      └─turbulence
│        └─ X.csv
│        │  target.csv
│        │  data_0.jpg
|        |  ...
│    │  ...
│
└──splits.json
```

## Before Evaluation
Launch `scripts/preparator.py` to:
* get dataset feature size
* get supervised tasks names
* get action inds
* update configs:
  * update dataset path
  * generate benchmarks config

## Splits
* Train - 45%
* Validation - 15%
* Test - 40%:
  * Train benchmarks 66%
  * Validation benchmarks 33%
