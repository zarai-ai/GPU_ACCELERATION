# AGENTS.md - GPU_ACCELERATION Development Guidelines for AI Agents

## Document Purpose

This document provides specific guidance for AI coding agents (GitHub Copilot, Claude, GPT, etc.) working on the GPU_ACCELERATION repository. It complements `.github/copilot-instructions.md` with agent-specific workflows and expectations.

---

## Mission Statement

**Build the most advanced GPU acceleration system on planet Earth** by implementing cutting-edge research from 2024-2025 with production-grade quality, zero compromises, and full research fidelity.

---

## Agent Operating Principles

### 1. Research-First Development

**Before writing any code:**

```bash
# Step 1: Locate the research paper
# Find the arXiv paper, read the methodology

# Step 2: Find the official implementation
# Search GitHub for the official repository from the research team

# Step 3: Clone and study the implementation
git clone https://github.com/[research-team]/[official-repo]
cd [official-repo]
# Read their code, understand their approach

# Step 4: Integrate, don't reimplement
# Use their code via FFI, bindings, or direct incorporation
# Do NOT rebuild a simplified version
```

**Example - FlashAttention-3:**
```bash
# ✅ CORRECT APPROACH
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
# Study csrc/flash_attn/ for CUDA kernels
# Create Rust FFI bindings to their kernels
# Test with their provided benchmarks

# ❌ WRONG APPROACH
# Writing your own "flash_attention.rs" from scratch
# Creating a "simplified" version
# Implementing "basic attention" as a placeholder
```

### 2. Zero Fallback Policy

**Unacceptable patterns:**

```rust
// ❌ NEVER DO THIS
pub fn advanced_operation(data: &Tensor) -> Result<Tensor> {
    match try_optimized_path(data) {
        Ok(result) => Ok(result),
        Err(_) => {
            // Fallback to simple version
            Ok(simple_operation(data))
        }
    }
}

// ✅ CORRECT APPROACH
pub fn advanced_operation(data: &Tensor) -> Result<Tensor> {
    // Only one path - the correct, optimized, research-grade path
    optimized_operation(data)
        .context("Failed to execute optimized operation")
}
```

### 3. Full E2E Testing

**Test philosophy:**

```rust
// ❌ WRONG: Mock testing
#[test]
fn test_with_mock_data() {
    let mock_tensor = Tensor::zeros((1, 1));
    assert!(process(mock_tensor).is_ok());
}

// ✅ CORRECT: Real-world simulation
#[test]
fn test_full_workload() {
    // Real data sizes
    let batch_size = 32;
    let seq_len = 8192;
    let hidden_dim = 4096;
    
    // Real tensor data
    let q = Tensor::randn((batch_size, seq_len, hidden_dim));
    let k = Tensor::randn((batch_size, seq_len, hidden_dim));
    let v = Tensor::randn((batch_size, seq_len, hidden_dim));
    
    // Real GPU execution
    let result = flash_attention_3(&q, &k, &v).expect("Must succeed");
    
    // Validate correctness
    assert_eq!(result.shape(), &[batch_size, seq_len, hidden_dim]);
    
    // Validate performance
    let throughput = measure_throughput();
    assert!(throughput > TARGET_TFLOPS, "Did not meet performance target");
}
```

### 4. Edge Case Coverage

**Required edge cases for every function:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let empty = Vec::new();
        assert!(process_batch(empty).is_ok());
    }

    #[test]
    fn test_single_element() {
        let single = vec![create_tensor()];
        assert!(process_batch(single).is_ok());
    }

    #[test]
    fn test_maximum_capacity() {
        let max_batch = vec![create_tensor(); MAX_BATCH_SIZE];
        assert!(process_batch(max_batch).is_ok());
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;
        
        let service = Arc::new(GPUService::new());
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let service = Arc::clone(&service);
                thread::spawn(move || service.process(data))
            })
            .collect();
        
        for handle in handles {
            assert!(handle.join().unwrap().is_ok());
        }
    }

    #[test]
    fn test_out_of_memory_handling() {
        // Test graceful handling when GPU memory exhausted
        let huge_tensor = Tensor::zeros((1000000, 1000000));
        match process(huge_tensor) {
            Err(GPUError::OutOfMemory) => { /* expected */ }
            _ => panic!("Should handle OOM gracefully"),
        }
    }
}
```

---

## Development Workflow

### Phase 1: Research & Planning

**Before coding, create a plan:**

```markdown
## Task: Implement FlashAttention-3

### Research Review
- [ ] Read arXiv:2407.08608
- [ ] Clone https://github.com/Dao-AILab/flash-attention
- [ ] Study CUDA kernel implementation
- [ ] Understand warp specialization approach
- [ ] Document FP8 quantization scheme

### Integration Strategy
- [ ] Determine FFI requirements
- [ ] Plan Rust wrapper API
- [ ] Design error handling
- [ ] Plan testing approach

### Performance Targets
- Target: 740 TFLOPs/s (FP16) on H100
- Target: 1.2 PFLOPs/s (FP8) on H100
- Target: >70% hardware utilization
```

### Phase 2: Implementation

**Development checklist:**

```bash
# 1. Set up the research code
git submodule add https://github.com/Dao-AILab/flash-attention external/flash-attention
cd external/flash-attention && git checkout v3.0.0  # Use specific version

# 2. Create FFI bindings
# In src/ffi/flash_attention.rs
# Use bindgen or write manual bindings

# 3. Implement safe wrapper
# In src/attention/flash_v3.rs
# Provide safe Rust API

# 4. Write comprehensive tests
# In tests/attention_tests.rs
# Cover all edge cases

# 5. Benchmark performance
# In benches/attention_bench.rs
# Validate against targets
```

### Phase 3: Validation

**Pre-commit checklist:**

```bash
# Code quality
cargo fmt --check          # ✅ Must pass
cargo clippy -- -D warnings # ✅ Must pass, zero warnings
cargo check                # ✅ Must compile

# Testing
cargo test                 # ✅ All tests pass
cargo test --release       # ✅ Release mode tests pass
cargo test -- --ignored    # ✅ Even ignored tests pass

# Benchmarks
cargo bench               # ✅ Meet performance targets

# Documentation
cargo doc --no-deps       # ✅ Docs build cleanly
# Verify all public APIs documented

# Memory safety
cargo test --features=valgrind  # ✅ No leaks (if using valgrind)
# Or use other memory checking tools
```

---

## Architecture Patterns

### Microservice Structure

```rust
// src/lib.rs - Root module
pub mod core {
    pub mod gpu_service;      // Main service orchestration
    pub mod device_manager;   // GPU device management
}

pub mod kernels {
    pub mod flash_attention;  // FlashAttention-3 kernels
    pub mod thunder_kittens;  // ThunderKittens primitives
    pub mod ring_attention;   // Distributed attention
}

pub mod ffi {
    pub mod cuda_bindings;    // CUDA FFI
    pub mod flash_attn_ffi;   // FlashAttention FFI
    pub mod tk_ffi;           // ThunderKittens FFI
}

pub mod ai {
    pub mod rwkv_host;        // RWKV AI host system
    pub mod monitoring;       // Intelligent monitoring
    pub mod injection;        // Code injection system
}

pub mod utils {
    pub mod tensor;           // Tensor utilities
    pub mod error;            // Error types
    pub mod metrics;          // Performance metrics
}

// Public API
pub use core::gpu_service::GPUAccelerationService;
pub use kernels::flash_attention::FlashAttention3;
```

### Error Handling Pattern

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GPUError {
    #[error("CUDA error: {0}")]
    CudaError(String),
    
    #[error("Out of GPU memory: requested {requested}MB, available {available}MB")]
    OutOfMemory { requested: usize, available: usize },
    
    #[error("Invalid tensor dimensions: expected {expected:?}, got {actual:?}")]
    InvalidDimensions { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error("Hardware not supported: requires compute capability {required}, got {actual}")]
    UnsupportedHardware { required: String, actual: String },
    
    #[error("Research implementation error: {0}")]
    ResearchImplError(String),
}

pub type Result<T> = std::result::Result<T, GPUError>;

// Usage
pub fn flash_attention_3(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    validate_tensors(q, k, v)?;
    allocate_gpu_memory()?;
    execute_kernel(q, k, v)
        .map_err(|e| GPUError::ResearchImplError(e.to_string()))
}
```

### Concurrency Pattern

```rust
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct GPUAccelerationService {
    device_manager: Arc<RwLock<DeviceManager>>,
    kernel_cache: Arc<RwLock<KernelCache>>,
    rwkv_host: Arc<RwLock<RWKVHost>>,
}

impl GPUAccelerationService {
    pub async fn process_concurrent(&self, batches: Vec<Batch>) -> Result<Vec<Output>> {
        use futures::stream::{self, StreamExt};
        
        // Process batches concurrently with proper error handling
        let results: Vec<Result<Output>> = stream::iter(batches)
            .map(|batch| async {
                let device = self.device_manager.read().await;
                device.process(batch).await
            })
            .buffer_unordered(MAX_CONCURRENT_BATCHES)
            .collect()
            .await;
        
        // Collect results, fail fast on error
        results.into_iter().collect()
    }
}
```

---

## AI Integration Specifics

### RWKV Host System

```rust
// src/ai/rwkv_host.rs

use safetensors::SafeTensors;
use tokio::sync::Semaphore;

pub struct RWKVHost {
    model: SafeTensors,
    gpu_semaphore: Arc<Semaphore>,
    injection_system: IntelligentInjector,
    monitor: SystemMonitor,
}

impl RWKVHost {
    /// Initialize RWKV host with HuggingFace model
    pub async fn new(model_path: &Path) -> Result<Self> {
        // Load .safetensors model file
        let model = SafeTensors::load(model_path)?;
        
        // Only one GPU operation at a time (prevents conflicts)
        let gpu_semaphore = Arc::new(Semaphore::new(1));
        
        Ok(Self {
            model,
            gpu_semaphore,
            injection_system: IntelligentInjector::new(),
            monitor: SystemMonitor::new(),
        })
    }
    
    /// Run inference only when main GPU operations idle
    pub async fn inference_when_idle(&self, prompt: &str) -> Result<String> {
        // Wait for GPU to be idle
        let _permit = self.gpu_semaphore.acquire().await?;
        
        // Run RWKV inference
        self.run_inference(prompt).await
    }
    
    /// Intelligent monitoring and reporting
    pub async fn generate_report(&self) -> Result<Report> {
        let metrics = self.monitor.collect_metrics();
        let analysis = self.inference_when_idle(&format!(
            "Analyze these GPU metrics and generate a brief: {:?}",
            metrics
        )).await?;
        
        Ok(Report::from_analysis(analysis))
    }
}
```

### Code Injection System

```rust
// src/ai/injection.rs

use syn::{parse_file, Item};
use std::path::PathBuf;

pub struct IntelligentInjector {
    scanner: ArchitectureScanner,
    semantic_analyzer: SemanticAnalyzer,
}

impl IntelligentInjector {
    /// Scan importing file to discover architecture
    pub fn scan_importer(&self, file_path: &Path) -> Result<ArchitectureInfo> {
        let source = std::fs::read_to_string(file_path)?;
        let syntax_tree = parse_file(&source)?;
        
        let mut info = ArchitectureInfo::default();
        
        // Analyze imports/exports
        for item in syntax_tree.items {
            match item {
                Item::Use(use_item) => {
                    info.add_import(use_item);
                }
                Item::Fn(fn_item) => {
                    if is_main_function(&fn_item) {
                        info.main_function = Some(fn_item);
                    }
                }
                _ => {}
            }
        }
        
        Ok(info)
    }
    
    /// Intelligently determine injection points
    pub fn resolve_injection_points(&self, arch: &ArchitectureInfo) -> Vec<InjectionPoint> {
        self.semantic_analyzer.analyze(arch)
    }
    
    /// Inject usage features at determined points
    pub fn inject_features(&self, points: Vec<InjectionPoint>) -> Result<()> {
        for point in points {
            self.inject_at_point(point)?;
        }
        Ok(())
    }
}
```

---

## Performance Optimization

### GPU Memory Management

```rust
use cuda_runtime_sys::*;

pub struct GPUMemoryPool {
    allocated: Arc<RwLock<HashMap<usize, *mut c_void>>>,
    total_capacity: usize,
}

impl GPUMemoryPool {
    pub unsafe fn allocate(&self, size: usize) -> Result<*mut c_void> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        
        let status = cudaMalloc(&mut ptr as *mut *mut c_void, size);
        if status != cudaError_t::cudaSuccess {
            return Err(GPUError::OutOfMemory {
                requested: size / (1024 * 1024),
                available: self.query_available_memory()?,
            });
        }
        
        self.allocated.write().await.insert(size, ptr);
        Ok(ptr)
    }
    
    pub unsafe fn deallocate(&self, ptr: *mut c_void) -> Result<()> {
        let status = cudaFree(ptr);
        if status != cudaError_t::cudaSuccess {
            return Err(GPUError::CudaError("Failed to free memory".into()));
        }
        Ok(())
    }
}

// RAII wrapper for automatic cleanup
pub struct GPUTensor {
    ptr: *mut c_void,
    pool: Arc<GPUMemoryPool>,
}

impl Drop for GPUTensor {
    fn drop(&mut self) {
        unsafe {
            let _ = self.pool.deallocate(self.ptr);
        }
    }
}
```

### Async Kernel Execution

```rust
pub struct AsyncKernelExecutor {
    stream: cudaStream_t,
}

impl AsyncKernelExecutor {
    pub async fn execute_flash_attention(
        &self,
        q: &GPUTensor,
        k: &GPUTensor,
        v: &GPUTensor,
    ) -> Result<GPUTensor> {
        // Launch kernel asynchronously
        unsafe {
            launch_flash_attn_kernel(
                self.stream,
                q.ptr, k.ptr, v.ptr,
                /* ... params ... */
            )?;
        }
        
        // Wait for completion
        self.synchronize().await?;
        
        Ok(output)
    }
    
    async fn synchronize(&self) -> Result<()> {
        tokio::task::spawn_blocking(move || {
            unsafe {
                cudaStreamSynchronize(self.stream)
            }
        }).await??;
        Ok(())
    }
}
```

---

## Documentation Standards

### Module Documentation

```rust
//! FlashAttention-3 Implementation
//!
//! This module provides Rust bindings to the FlashAttention-3 CUDA kernels
//! from the official Dao-AILab implementation.
//!
//! # Research Paper
//!
//! **"FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"**
//! - Authors: Tri Dao, Beidi Chen, et al.
//! - Institution: Princeton University, Together AI
//! - Source: arXiv:2407.08608
//! - Link: <https://arxiv.org/abs/2407.08608>
//!
//! # Key Features
//!
//! - Warp-specialized asynchronous execution
//! - FP8 quantization support
//! - O(N) memory complexity
//! - 75%+ hardware utilization on H100
//!
//! # Performance
//!
//! - H100 FP16: 740 TFLOPs/s
//! - H100 FP8: 1.2 PFLOPs/s
//! - RTX 2080 Ti: ~120 TFLOPs/s (estimated)
//!
//! # Example
//!
//! ```rust
//! use gpu_acceleration::FlashAttention3;
//!
//! let attn = FlashAttention3::new()?;
//! let output = attn.forward(&q, &k, &v, true)?;
//! ```
//!
//! # Safety
//!
//! This module uses FFI to CUDA kernels. All unsafe operations are
//! encapsulated in safe wrappers with proper validation.
```

### Function Documentation

```rust
/// Performs forward pass of FlashAttention-3.
///
/// # Implementation Details
///
/// Uses warp-specialized producer-consumer execution model:
/// - Producer warps: Load K/V tiles asynchronously via TMA
/// - Consumer warps: Compute attention on Q using tensor cores
/// - Overlaps memory and compute operations for maximum throughput
///
/// # Arguments
///
/// * `q` - Query tensor [batch, num_heads, seq_len, head_dim]
/// * `k` - Key tensor [batch, num_heads, seq_len, head_dim]
/// * `v` - Value tensor [batch, num_heads, seq_len, head_dim]
/// * `causal` - Whether to use causal masking
///
/// # Returns
///
/// Attention output tensor [batch, num_heads, seq_len, head_dim]
///
/// # Errors
///
/// Returns `GPUError::InvalidDimensions` if tensor dimensions don't match.
/// Returns `GPUError::OutOfMemory` if GPU memory exhausted.
/// Returns `GPUError::CudaError` for CUDA runtime errors.
///
/// # Performance
///
/// - Complexity: O(N) memory, O(N²) compute
/// - Target: 740 TFLOPs/s (FP16) on H100
/// - Utilization: 75%+ hardware utilization
///
/// # Example
///
/// ```rust
/// let q = Tensor::randn(&[32, 8, 8192, 64]);
/// let k = Tensor::randn(&[32, 8, 8192, 64]);
/// let v = Tensor::randn(&[32, 8, 8192, 64]);
///
/// let output = flash_attention_3_forward(&q, &k, &v, true)?;
/// assert_eq!(output.shape(), q.shape());
/// ```
pub fn flash_attention_3_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
) -> Result<Tensor> {
    // Implementation
}
```

---

## Quality Checklist

Before submitting any code, verify:

### ✅ Research Fidelity
- [ ] Official research paper read and understood
- [ ] Official implementation cloned and studied
- [ ] Algorithm implemented with 100% fidelity
- [ ] Performance targets from paper met or exceeded
- [ ] No simplified or fallback versions

### ✅ Code Quality
- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy -- -D warnings` passes (ZERO warnings)
- [ ] `cargo check` compiles successfully
- [ ] All unsafe code has SAFETY comments
- [ ] Proper error handling (no unwrap/expect in library)

### ✅ Testing
- [ ] Unit tests for all public APIs
- [ ] Integration tests with real workloads
- [ ] All edge cases covered
- [ ] Concurrent access tested
- [ ] Performance benchmarks pass
- [ ] Memory leak tests pass

### ✅ Documentation
- [ ] All public items documented
- [ ] Module-level docs with research citations
- [ ] Examples provided
- [ ] Performance characteristics documented
- [ ] Safety requirements documented

### ✅ Performance
- [ ] Benchmarked against targets
- [ ] GPU utilization >70%
- [ ] Memory usage optimal
- [ ] Concurrency working correctly
- [ ] No race conditions

---

## Common Pitfalls to Avoid

### ❌ Pitfall 1: Simplified Reimplementation

**Wrong:**
```rust
// "I'll create my own version of FlashAttention"
pub fn my_flash_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    // Simplified algorithm that doesn't actually match the paper
}
```

**Correct:**
```rust
// Use official implementation via FFI
extern "C" {
    fn flash_attn_fwd_cuda(/* official params */);
}

pub fn flash_attention_3(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    unsafe {
        // SAFETY: Tensors validated, pointers valid for call duration
        flash_attn_fwd_cuda(/* params */);
    }
    Ok(result)
}
```

### ❌ Pitfall 2: Mock Testing

**Wrong:**
```rust
#[test]
fn test_flash_attention() {
    let tiny = Tensor::zeros((1, 1, 1, 1));
    assert!(flash_attention_3(&tiny, &tiny, &tiny, false).is_ok());
    // ❌ This doesn't test real usage!
}
```

**Correct:**
```rust
#[test]
fn test_flash_attention_real_workload() {
    // Real-world tensor sizes
    let batch = 32;
    let heads = 8;
    let seq_len = 8192;
    let head_dim = 64;
    
    let q = Tensor::randn(&[batch, heads, seq_len, head_dim]);
    let k = Tensor::randn(&[batch, heads, seq_len, head_dim]);
    let v = Tensor::randn(&[batch, heads, seq_len, head_dim]);
    
    let result = flash_attention_3(&q, &k, &v, true).expect("Must succeed");
    
    // Validate correctness
    assert_eq!(result.shape(), q.shape());
    validate_attention_output(&result, &q, &k, &v);
    
    // Validate performance
    let throughput = benchmark_throughput();
    assert!(throughput > TARGET_TFLOPS);
}
```

### ❌ Pitfall 3: Ignoring Edge Cases

**Wrong:**
```rust
pub fn process_batch(data: Vec<Tensor>) -> Vec<Tensor> {
    data.iter().map(|t| process(t).unwrap()).collect()
    // ❌ Panics on error, doesn't handle empty vec
}
```

**Correct:**
```rust
pub fn process_batch(data: Vec<Tensor>) -> Result<Vec<Tensor>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    
    data.iter()
        .map(|t| process(t))
        .collect::<Result<Vec<_>>>()
        .context("Failed to process batch")
}
```

---

## Conclusion

Remember:
- **Research fidelity**: Use official implementations
- **Zero compromises**: No fallbacks or simplified versions  
- **Production quality**: Handle all edge cases
- **Real testing**: Full live usage simulation
- **Documentation**: Cite research, explain decisions

When in doubt: **Ask, don't guess**. It's better to request clarification than to implement the wrong thing.

**Goal**: The most advanced GPU acceleration code on planet Earth. Anything less is unacceptable.

---

**Last Updated**: January 2026  
**Maintained by**: Zarai AI / EchoAI Labs  
**Contact**: https://www.zarai.ai
