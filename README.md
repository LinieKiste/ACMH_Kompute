# ACMH\_Kompute
Reimplementation of [ACMH](https://github.com/GhiXu/ACMH) with the Kompute framework

To get ACMH running, add `-ccbin /usr/bin/g++-12` and remove `-gencode arch=compute_30,code=sm_30` in the `CUDA_NVCC_FLAGS`.

Installing SAIL:
```
git clone --recursive https://github.com/HappySeaFox/sail.git
cd sail
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make install
```
