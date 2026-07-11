# QR Code Detection

This repository contains two implementations of the QR code detection algorithm:

* **Python implementation**
* **C++ implementation**

## Requirements

### Python

* Python (see the `python/` folder)
* `uv` installed

### C++

* CMake
* A C++ compiler supporting the project

---

## Python Implementation

Navigate to the `python` directory:

```bash
cd python
```

Run the evaluation script:

```bash
uv run evaluation.py
```

Before running the script, update the input folder path in `evaluation.py`:

Results available in `python/results`

* **Line 238:** modify the input folder to match your dataset location.

---

## C++ Implementation

Navigate to the C++ implementation directory:

```bash
cd c++_implementation
```

Configure and build the project:

```bash
cmake -S . -B build
cmake --build build
```

Run the executable:

```bash
./build/qrcode_detection
```
Results available in `c++_implementation/results`

---

## Performance

Measured on a dataset of **17 images**:

| Implementation | Execution Time |
| -------------- | -------------: |
| Python         | **1 min 11 s** |
| C++            | **1 min 00 s** |
