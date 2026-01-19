---
description: "GPU_ACCELERATION Repository - Research-Grade Development Standards"
applyTo: "**/*"
---

# GPU_ACCELERATION - Copilot Instructions

## Project Identity

**Zarai AI / EchoAI Labs** - "The most advanced code on planet Earth"

This is a **research-grade GPU acceleration system** that implements cutting-edge algorithms from 2024-2025 academic research papers. Every line of code must meet flight-critical, DO-178C compliant, formally-verified defense standards.

## Core Principles

### 🎯 Absolute Standards (NON-NEGOTIABLE)

1. **ZERO FALLBACKS** - There is only 1 spec, and it will work. No degraded modes, no simplified alternatives.
2. **REAL RESEARCH IMPLEMENTATION** - Clone actual research repositories, use their code directly. Do not rebuild simplified versions.
3. **100% PRODUCTION READY** - No juvenile amateur code. All edge cases, concurrency, memory, and cache issues resolved.
4. **FULL E2E INTEGRATION** - All tests must be full live usage simulations in real-world conditions.
5. **0 ERRORS, 0 WARNINGS** - Code must pass `cargo check`, `cargo test`, and `cargo clippy` with zero issues.

### ❌ What NOT To Do

- **DO NOT** create simplified versions of research algorithms
- **DO NOT** implement fallback mechanisms or degraded operation modes
- **DO NOT** skip edge case handling
- **DO NOT** use stub implementations or mock functionality
- **DO NOT** create 20-25k LoC without actual research implementation
- **DO NOT** pass tests via fallbacks instead of real functionality
- **DO NOT** rebuild vital files from cloned repos - use them directly via FFI or linking

## Technology Stack

### Primary Language: Rust
- **Rust monorepo** with complete compiled universal microservice architecture
- All features architected service-side with intelligent injection architecture
- Advanced research-grade algorithms for intelligent discovery and integration
- Full determinism, maximum concurrency support
- Asymptotically-optimal, Pareto-efficient implementations

### Target Hardware
- **Primary**: NVIDIA GeForce RTX 2080 Ti
- **Advanced**: NVIDIA H100 Hopper, NVIDIA B200 Blackwell
- GPU-specific optimizations required for each target

### Core Research Technologies
1. **FlashAttention-3** (arXiv:2407.08608) - Warp-specialized async execution with FP8
2. **ThunderKittens** (arXiv:2410.20399) - Tile-based GPU primitives (16x16)
3. **Ring Attention** (arXiv:2310.01889) - Distributed block-wise attention
4. **CUDA-LLM** (arXiv:2506.09092) - AI-generated kernel optimization
5. **RWKV with GPU acceleration** - For intelligent monitoring, code injection, reporting
6. **RAIN** - Research-grade implementation (clone from GitHub)

### AI Integration Requirements
- Clone and integrate **RWKV GPU acceleration repo** (circa 2025)
- Obtain HuggingFace-compliant directory with full `.safetensors` model files
- Implement full inference and chatting capabilities
- Smaller models for intelligent features: debris housekeeping, inquiries, reporting
- Intelligent "host" system with role recognition and system integration
- GPU resource management to prevent conflicts during acceleration operations

## Architecture Requirements

### Microservice Design
```rust
// Service-side feature architecture with intelligent injection
// Example structure (NOT prescriptive - implement research-grade patterns)

pub struct GPUAccelerationService {
    // Intelligent discovery of importing architecture
    injection_scanner: ArchitectureScanner,
    feature_linker: IntelligentLinker,
    // Full featured usage built into microservice
    usage_features: UsageArchitecture,
}
```

### Injection Architecture
- **Intelligent discovery**: Scan/crawl importing files and their imports/exports
- **Semantic resolution**: Use heuristics to determine where to inject usage linkages
- **Prescient operation**: No fallbacks, only correct first-time execution
- **Full feature integration**: All usage features built into the microservice

### Repository Integration
When integrating research repositories:
```bash
# DO: Clone official repositories and use their code
git clone https://github.com/HazyResearch/ThunderKittens
git clone https://github.com/Dao-AILab/flash-attention
git clone https://github.com/haoliuhl/ringattention

# DO: Copy vital files directly and link E2E
cp research_repo/core/*.cu ./kernels/
# Link with full usage, no simplified rebuilds

# DO NOT: Rebuild simplified non-operational versions
# DO NOT: Create stub implementations
```

### Research Implementation Fidelity
- **100% source fidelity** - Exact match to published algorithms
- **100% completeness** - All features, all edge cases
- **100% accuracy** - If research has pseudocode, transmute to Rust exactly
- Upgrade cloned repos if needed to match published research
- For research with no GitHub repo: Actualize into most advanced Rust possible

## Coding Standards

### Rust Best Practices
```rust
// Use explicit error handling - no unwrap() in production
let result = operation().map_err(|e| CustomError::from(e))?;

// Concurrent operations with proper synchronization
use tokio::sync::RwLock;
use std::sync::Arc;

// GPU memory management - explicit, no leaks
unsafe {
    // SAFETY: Proper documentation required for all unsafe blocks
    // Explain why this is safe and what invariants are maintained
}

// Maximum concurrency where applicable
use rayon::prelude::*;
data.par_iter().map(|item| process(item)).collect()
```

### Performance Requirements
- Optimize for **NVIDIA 2080 Ti** as primary target
- Hardware utilization targets: **>70%** tensor core usage
- Memory efficiency: **O(N)** complexity where possible
- Full async/await support for I/O operations
- Lock-free data structures where applicable

### Security & Safety
- All unsafe code must have SAFETY comments explaining invariants
- No buffer overflows, race conditions, or memory leaks
- Input validation on all external boundaries
- Proper error propagation (no panic in library code)
- Defense-in-depth security practices

## Testing Standards

### E2E Testing Requirements
```rust
#[test]
fn test_full_live_usage_simulation() {
    // ✅ REQUIRED: Full flag usage in real conditions
    // This is how users will actually use the code
    
    let gpu_service = GPUAccelerationService::new();
    let real_data = load_actual_workload(); // Not mock data
    
    // Test with actual GPU, actual memory constraints, actual concurrency
    let result = gpu_service.accelerate(real_data);
    
    assert!(result.is_ok());
    assert_performance_targets_met(result);
}

#[test]
fn test_edge_cases() {
    // ✅ REQUIRED: All edge cases must be tested
    test_empty_input();
    test_maximum_capacity();
    test_concurrent_access();
    test_gpu_memory_pressure();
    test_recovery_from_errors();
}
```

### Test Coverage
- **100% live simulation** - No less than real-world conditions
- **All edge cases** - Empty, maximum, concurrent, error conditions
- **Performance validation** - Must meet specified targets
- **Memory safety** - No leaks, proper cleanup
- **Concurrency** - Thread-safe, race-condition free

## Documentation Standards

### Code Documentation
```rust
/// Performs GPU-accelerated attention computation using FlashAttention-3.
///
/// # Implementation
/// Based on arXiv:2407.08608 - warp-specialized asynchronous execution.
///
/// # Arguments
/// * `q` - Query tensor [batch, seq_len, hidden]
/// * `k` - Key tensor [batch, seq_len, hidden]  
/// * `v` - Value tensor [batch, seq_len, hidden]
///
/// # Returns
/// Attention output tensor [batch, seq_len, hidden]
///
/// # Performance
/// - Target: 740 TFLOPs/s (FP16), 1.2 PFLOPs/s (FP8) on H100
/// - Achieves: 75%+ hardware utilization
///
/// # Safety
/// Requires CUDA 12.0+ and compute capability 9.0+ (Hopper)
pub fn flash_attention_3(q: Tensor, k: Tensor, v: Tensor) -> Result<Tensor> {
    // Implementation
}
```

### File Headers
```rust
//! GPU Acceleration Core - FlashAttention-3 Implementation
//!
//! Research Paper: arXiv:2407.08608
//! Authors: Tri Dao, Beidi Chen, et al.
//! Institution: Princeton University, Together AI
//!
//! This module implements warp-specialized asynchronous attention
//! with FP8 quantization support for NVIDIA Hopper architecture.
```

## Build & Validation

### Pre-commit Requirements
```bash
# All must pass before committing
cargo fmt --check          # Code formatting
cargo clippy -- -D warnings # Zero warnings allowed
cargo check                # Compilation check
cargo test                 # All tests pass
cargo test --release       # Release mode tests
```

### Performance Validation
- Benchmark against targets in README.md
- Profile GPU utilization (must be >70%)
- Memory usage within bounds
- Latency and throughput targets met

## GUI Requirements

### Interface Standards
- **Advanced minimalism** - Clean, focused, informative
- **Full-featured** - All functionality accessible
- **Intelligent design** - Adaptive to context
- Real-time metrics display:
  - GPU utilization percentage
  - Memory usage (current/max)
  - Throughput (TFLOPs/s)
  - Temperature and power
- Error reporting with actionable information

## AI Features Integration

### RWKV Integration
```rust
// Intelligent monitoring and reporting
pub struct RWKVHost {
    model: SafeTensorsModel,
    gpu_scheduler: ResourceScheduler,
    
    // Situational adaptability
    code_injection: IntelligentInjector,
    monitoring: SystemMonitor,
    reporting: ReportGenerator,
}

// Resource management - prevent GPU conflicts
impl RWKVHost {
    /// Run RWKV only when main GPU operations are idle
    pub async fn run_when_idle(&self) -> Result<()> {
        self.gpu_scheduler.wait_for_idle().await?;
        self.execute_inference().await
    }
}
```

### Host System Features
- Debris and artifact housekeeping
- General inquiries handling  
- Reporting briefs generation
- Intelligent role recognition
- System integration awareness

## References & Research

### Primary Research Papers
1. FlashAttention-3: https://arxiv.org/abs/2407.08608
2. ThunderKittens: https://arxiv.org/abs/2410.20399
3. Ring Attention: https://arxiv.org/abs/2310.01889
4. Blackwell Architecture: https://arxiv.org/abs/2507.10789
5. CUDA-LLM: https://arxiv.org/abs/2506.09092
6. GPU Cores Analysis: https://arxiv.org/abs/2503.20481

### Official Repositories
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- ThunderKittens: https://github.com/HazyResearch/ThunderKittens
- Ring Attention: https://github.com/haoliuhl/ringattention
- RWKV GPU Accel: (Search for 2025 releases)
- RAIN: (Clone from GitHub)

## Quality Gates

### Milestone: Version 1.0
**ONLY** declare v1.0 when ALL criteria met:

✅ **Compilation**
- Zero errors
- Zero warnings
- Optimized for RTX 2080 Ti

✅ **Integration**
- Full E2E implementation
- All research algorithms integrated
- No simplified stub versions

✅ **Testing**
- All `cargo test` passes
- Live simulated full-flag usage tested
- All edge cases resolved

✅ **Performance**
- Targets met (see README.md)
- GPU utilization >70%
- Memory efficiency validated

✅ **Documentation**
- README.md updated/upgraded
- AGENTS.md created/updated
- All public APIs documented

✅ **Quality**
- `cargo check` passes
- `cargo clippy` passes  
- Ready for library build

## Examples

### ✅ DO: Use Research Code Directly
```rust
// Link against actual FlashAttention kernel
extern "C" {
    fn flash_attn_fwd(/* official C API params */);
}

// Wrap with safe Rust interface
pub fn flash_attention_3(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    unsafe {
        // SAFETY: Tensors validated, GPU memory allocated, 
        // pointers valid for duration of call
        flash_attn_fwd(/* params */);
    }
    Ok(result)
}
```

### ❌ DON'T: Create Simplified Versions
```rust
// ❌ WRONG: Simplified attention without research features
pub fn simple_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    // Basic implementation missing:
    // - Warp specialization
    // - Async execution
    // - FP8 support
    // - Tile-based computation
    q.matmul(k).softmax().matmul(v)
}
```

### ✅ DO: Handle All Edge Cases
```rust
pub fn process_batch(data: Vec<Tensor>) -> Result<Vec<Tensor>> {
    // Empty input
    if data.is_empty() {
        return Ok(Vec::new());
    }
    
    // Single item
    if data.len() == 1 {
        return Ok(vec![process_single(&data[0])?]);
    }
    
    // Concurrent processing with proper error handling
    data.par_iter()
        .map(|tensor| process_single(tensor))
        .collect::<Result<Vec<_>>>()
}
```

### ❌ DON'T: Skip Edge Cases
```rust
// ❌ WRONG: Assumes non-empty, doesn't handle errors
pub fn process_batch(data: Vec<Tensor>) -> Vec<Tensor> {
    data.iter().map(|t| process_single(t).unwrap()).collect()
}
```

## Communication Protocol

When you (AI agent) encounter issues:

1. **State the problem clearly** - What exact requirement cannot be met?
2. **Explain the blocker** - What specific constraint prevents progress?
3. **Propose solution** - What would you need to proceed correctly?
4. **Request guidance** - Ask user for decision/input

Do NOT:
- Implement workarounds without approval
- Create fallback mechanisms silently
- Simplify requirements to make progress
- Deliver incomplete solutions marked as "complete"

---

## Summary

This is **real engineering for real code**. EchoAI Labs / Zarai AI produces the most advanced outcomes on Earth. Every commit must reflect this standard. When in doubt: choose the more rigorous, more correct, more research-faithful implementation.

**Token efficiency matters** - No wasted cycles on code that doesn't meet these standards.
**Time matters** - Do it right the first time.
**Money matters** - Real implementations, not 25k LoC of fallbacks.

**Goal**: Research-grade, production-ready, formally-verified GPU acceleration that represents the absolute state of the art in 2025.
