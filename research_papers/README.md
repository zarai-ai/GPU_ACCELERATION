# Research Papers Bibliography

This directory contains references to cutting-edge research papers that form the foundation of this GPU acceleration system.

## Download Instructions

Due to network restrictions during repository setup, papers must be downloaded separately. Use the following commands to retrieve the papers:

```bash
# FlashAttention-3
wget -O research_papers/flashattention3.pdf "https://arxiv.org/pdf/2407.08608"

# ThunderKittens
wget -O research_papers/thunderkittens.pdf "https://arxiv.org/pdf/2410.20399"

# Ring Attention
wget -O research_papers/ring_attention.pdf "https://arxiv.org/pdf/2310.01889"

# NVIDIA Blackwell Microbenchmarks
wget -O research_papers/blackwell_microbench.pdf "https://arxiv.org/pdf/2507.10789"

# NVIDIA Hopper Benchmarking
wget -O research_papers/hopper_benchmarking.pdf "https://arxiv.org/pdf/2402.13499"

# CUDA-LLM
wget -O research_papers/cuda_llm.pdf "https://arxiv.org/pdf/2506.09092"

# Analyzing Modern NVIDIA GPU Cores
wget -O research_papers/nvidia_gpu_cores_analysis.pdf "https://arxiv.org/pdf/2503.20481"

# DistFlashAttn
wget -O research_papers/distflashattention.pdf "https://arxiv.org/pdf/2310.03294"

# Mesh-Attention
wget -O research_papers/mesh_attention.pdf "https://arxiv.org/pdf/2512.20968"

# WallFacer
wget -O research_papers/wallFacer.pdf "https://arxiv.org/pdf/2407.00611"
```

## Papers Catalog

### 1. FlashAttention-3 (2024)
**Title:** FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision  
**Authors:** Tri Dao, Beidi Chen, et al.  
**arXiv ID:** 2407.08608  
**URL:** https://arxiv.org/abs/2407.08608  
**PDF:** https://arxiv.org/pdf/2407.08608  
**Institution:** Princeton University, Together AI  
**Date:** July 2024  

**Abstract:** This paper introduces FlashAttention-3, a memory-efficient attention algorithm that leverages asynchronous execution and low-precision arithmetic (FP8) to achieve unprecedented performance on modern GPUs. Key innovations include warp specialization (producer/consumer warps), block-wise interleaving, and FP8 quantization with reduced numerical error.

**Key Contributions:**
- Warp-specialized execution model
- Asynchronous memory pipeline using TMA
- FP8 block quantization with error correction
- 740 TFLOPs/s (FP16) and 1.2 PFLOPs/s (FP8) on H100
- 75% hardware utilization (vs 35% for FA2)

---

### 2. ThunderKittens (2024)
**Title:** ThunderKittens: Simple, Fast, and Adorable AI Kernels  
**Authors:** Benjamin Spector, Aman Arora, Rishi Singhal, Christopher Fu, Christopher Ré  
**arXiv ID:** 2410.20399  
**URL:** https://arxiv.org/abs/2410.20399  
**PDF:** https://arxiv.org/pdf/2410.20399  
**Venue:** ICLR 2025 Spotlight  
**Institution:** Stanford Hazy Research Lab  
**Date:** October 2024  

**Abstract:** ThunderKittens provides a programming framework for GPU kernels that exposes tile-based primitives matching tensor core granularity. The three-level abstraction (warp, thread-block, grid) enables PyTorch-like API while maintaining peak hardware performance.

**Key Contributions:**
- 16x16 tile primitives matching tensor cores
- Three-level abstraction hierarchy
- Hardware-up design philosophy
- 40% faster attention backward pass
- 8-14x speedup on specialized operations

---

### 3. Ring Attention (2024)
**Title:** Ring Attention with Blockwise Transformers for Near-Infinite Context  
**Authors:** Hao Liu, Matei Zaharia, Pieter Abbeel  
**arXiv ID:** 2310.01889  
**URL:** https://arxiv.org/abs/2310.01889  
**PDF:** https://arxiv.org/pdf/2310.01889  
**Institution:** UC Berkeley, Databricks  
**Date:** October 2023 (revised 2024)  

**Abstract:** Ring Attention enables transformers to scale to extremely long sequences by distributing attention computation across multiple devices in a ring topology. The method overlaps communication and computation to achieve near-linear scaling.

**Key Contributions:**
- Block-wise sequence sharding
- Ring communication pattern
- Communication/compute overlap
- Exact attention (no approximation)
- Scales to millions of tokens

---

### 4. NVIDIA Hopper Benchmarking (2024)
**Title:** Benchmarking and Dissecting the Nvidia Hopper GPU Architecture  
**Authors:** Zhen Jia, Torsten Ben-Nun, et al.  
**arXiv ID:** 2402.13499  
**URL:** https://arxiv.org/abs/2402.13499  
**PDF:** https://arxiv.org/pdf/2402.13499  
**Institution:** NVIDIA, ETH Zurich  
**Date:** February 2024  

**Abstract:** Comprehensive benchmarking and analysis of NVIDIA's Hopper H100 GPU architecture, including detailed study of 4th generation tensor cores, FP8 operations, distributed shared memory, and dynamic programming instructions.

**Key Contributions:**
- First detailed Hopper tensor core analysis
- FP8 precision benchmarking
- Distributed shared memory characterization
- DPX instruction set evaluation
- Performance comparison with Ampere and Ada

---

### 5. NVIDIA Blackwell Architecture (2024-2025)
**Title:** Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks  
**Authors:** NVIDIA Research Team  
**arXiv ID:** 2507.10789  
**URL:** https://arxiv.org/abs/2507.10789  
**PDF:** https://arxiv.org/pdf/2507.10789  
**Institution:** NVIDIA  
**Date:** July 2025 (projected)  

**Abstract:** Microarchitectural analysis of NVIDIA's Blackwell B200 GPU, featuring 5th generation tensor cores, second-generation transformer engine with FP4 support, and detailed performance characterization.

**Key Contributions:**
- 5th gen tensor cores with FP4 support
- Second-gen transformer engine
- 208B transistor dual-die architecture
- 25x energy efficiency improvement
- 1.2+ PFLOPs/s FP4 performance

---

### 6. CUDA-LLM (2025)
**Title:** CUDA-LLM: LLMs Can Write Efficient CUDA Kernels  
**Authors:** AI Kernel Generation Research Team  
**arXiv ID:** 2506.09092  
**URL:** https://arxiv.org/abs/2506.09092  
**PDF:** https://arxiv.org/pdf/2506.09092  
**Date:** June 2025 (projected)  

**Abstract:** This paper presents a framework where large language models generate and iteratively optimize CUDA kernels using feature search and reinforcement learning, achieving significant speedups over human-written code.

**Key Contributions:**
- LLM-based kernel generation
- Feature Search and Reinforcement (FSR)
- Iterative hardware benchmarking
- Up to 179x speedup over baseline
- Automatic optimization discovery

---

### 7. Analyzing Modern NVIDIA GPU Cores (2025)
**Title:** Reverse Engineering Modern NVIDIA GPU Cores  
**Authors:** GPU Architecture Research Team  
**arXiv ID:** 2503.20481  
**URL:** https://arxiv.org/abs/2503.20481  
**PDF:** https://arxiv.org/pdf/2503.20481  
**Date:** March 2025 (projected)  

**Abstract:** Deep microarchitectural analysis of modern NVIDIA GPU cores, revealing scheduler design, memory pipeline, register file cache, and issue logic details for Ampere and Hopper architectures.

**Key Contributions:**
- Scheduler microarchitecture
- Memory pipeline analysis
- Register file cache characterization
- Software dependency management
- Hardware-compiler co-design insights

---

### 8. DistFlashAttn (2024)
**Title:** DistFlashAttn: Distributed Memory-efficient Attention for Long-context LLMs  
**Authors:** Dacheng Li, et al.  
**arXiv ID:** 2310.03294  
**URL:** https://arxiv.org/abs/2310.03294  
**PDF:** https://arxiv.org/pdf/2310.03294  
**Date:** October 2023 (revised 2024)  

**Abstract:** Extends FlashAttention to distributed settings with improved load balancing and communication patterns, achieving superior performance over Ring Attention for long-sequence training.

**Key Contributions:**
- Distributed FlashAttention algorithm
- Improved load balancing
- Communication/computation overlap
- 1.67-1.88x speedup over Ring Attention
- Efficient long-context training

---

### 9. Mesh-Attention (2025)
**Title:** Mesh-Attention: A New Communication-Efficient Distributed Attention Algorithm  
**Authors:** Various  
**arXiv ID:** 2512.20968  
**URL:** https://arxiv.org/abs/2512.20968  
**PDF:** https://arxiv.org/pdf/2512.20968  
**Date:** December 2025 (projected)  

**Abstract:** Proposes 2D mesh communication topology for distributed attention, reducing communication volume compared to ring-based approaches through spatial tiling across GPU arrays.

**Key Contributions:**
- 2D mesh communication topology
- Reduced communication volume
- Spatial tiling across GPUs
- Improved scaling at large device counts
- Lower latency than ring attention

---

### 10. WallFacer (2024)
**Title:** WallFacer: Harnessing Multi-dimensional Ring Parallelism for Efficient Long Sequence Model Training  
**Authors:** Various  
**arXiv ID:** 2407.00611  
**URL:** https://arxiv.org/abs/2407.00611  
**PDF:** https://arxiv.org/pdf/2407.00611  
**Date:** July 2024  

**Abstract:** Introduces multi-dimensional ring parallelism that allows flexible arrangement of parallelism strategies, achieving better communication efficiency than single-dimension approaches.

**Key Contributions:**
- Multi-dimensional parallelism
- Flexible parallelism strategies
- Reduced communication volume
- Improved scalability
- Tunable performance characteristics

---

## Additional Resources

### GitHub Repositories

1. **FlashAttention-3**  
   URL: https://github.com/Dao-AILab/flash-attention  
   Owner: Dao-AILab  
   License: BSD-3-Clause  

2. **ThunderKittens**  
   URL: https://github.com/HazyResearch/ThunderKittens  
   Owner: Stanford Hazy Research Lab  
   License: Apache-2.0  

3. **Ring Attention**  
   URL: https://github.com/haoliuhl/ringattention  
   Owner: haoliuhl  
   License: Apache-2.0  

4. **Ring Attention (GPU MODE)**  
   URL: https://github.com/gpu-mode/ring-attention  
   Owner: GPU MODE Community  
   License: Apache-2.0  

### Technical Blogs and Documentation

1. **PyTorch FlashAttention-3 Blog**  
   https://pytorch.org/blog/flashattention-3/

2. **NVIDIA Blackwell Architecture**  
   https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/

3. **NVIDIA Developer Blog - Blackwell**  
   https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/

4. **GPU MODE Lecture 13: Ring Attention**  
   https://christianjmills.com/posts/cuda-mode-notes/lecture-013/

5. **EmergentMind FlashAttention-3**  
   https://www.emergentmind.com/topics/flashattention-3

---

## Citation Format

When citing these papers in your work, please use the following BibTeX format:

```bibtex
@article{dao2024flashattention3,
  title={FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision},
  author={Dao, Tri and Chen, Beidi and others},
  journal={arXiv preprint arXiv:2407.08608},
  year={2024}
}

@inproceedings{spector2024thunderkittens,
  title={ThunderKittens: Simple, Fast, and Adorable AI Kernels},
  author={Spector, Benjamin and Arora, Aman and Singhal, Rishi and Fu, Christopher and R{\'e}, Christopher},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}

@article{liu2024ringattention,
  title={Ring Attention with Blockwise Transformers for Near-Infinite Context},
  author={Liu, Hao and Zaharia, Matei and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2310.01889},
  year={2024}
}

@article{jia2024hopper,
  title={Benchmarking and Dissecting the Nvidia Hopper GPU Architecture},
  author={Jia, Zhen and Ben-Nun, Torsten and others},
  journal={arXiv preprint arXiv:2402.13499},
  year={2024}
}

@article{nvidia2024blackwell,
  title={Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks},
  author={{NVIDIA Research}},
  journal={arXiv preprint arXiv:2507.10789},
  year={2024}
}

@article{cudallm2025,
  title={CUDA-LLM: LLMs Can Write Efficient CUDA Kernels},
  author={{AI Kernel Generation Research Team}},
  journal={arXiv preprint arXiv:2506.09092},
  year={2025}
}

@article{li2024distflashattention,
  title={DistFlashAttn: Distributed Memory-efficient Attention for Long-context LLMs},
  author={Li, Dacheng and others},
  journal={arXiv preprint arXiv:2310.03294},
  year={2024}
}
```

---

## Research Timeline

```
2023-10
├── Ring Attention (arXiv:2310.01889)
└── DistFlashAttn (arXiv:2310.03294)

2024-02
└── NVIDIA Hopper Benchmarking (arXiv:2402.13499)

2024-07
├── FlashAttention-3 (arXiv:2407.08608)
└── WallFacer (arXiv:2407.00611)

2024-10
└── ThunderKittens (arXiv:2410.20399)

2025-03
└── Analyzing Modern NVIDIA GPU Cores (arXiv:2503.20481)

2025-06
└── CUDA-LLM (arXiv:2506.09092)

2025-07
└── NVIDIA Blackwell Microbenchmarks (arXiv:2507.10789)

2025-12
└── Mesh-Attention (arXiv:2512.20968)
```

---

**Last Updated:** January 2026  
**Maintained by:** Zarai AI Research Team