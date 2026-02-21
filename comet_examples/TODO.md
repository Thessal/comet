# Language Design Discussion: Function Overloading vs. C-ABI

Currently, the Comet language's frontend logic supports function overloading (e.g., `divide(DataFrame, Constant)`, `divide(TimeSeries, DataFrame)`). However, the backend Rust `stdlib` logic relies on strict `export_binary!` macros that generate a single `extern "C"` function (like `comet_divide_step`) which only accepts flat contiguous `f64` arrays (`*const f64`).

This limits our ability to seamlessly handle different data structures (Scalars, Strings, Matrices, DataFrames) natively. Below are the options to resolve this structural conflict:

## Options Available

### 1. Drop Function Overloading 
- **Concept**: Force the Comet language to use specific names for distinct implementations (e.g., `divide_df_df`, `divide_ts_const`). 
- **Pros**: The simplest constraint. Maps cleanly 1-to-1 to the Rust backend without any compiler magic.
- **Cons**: Extremely unergonomic for users. Code aesthetics will suffer dramatically. Mathematical expressions become deeply fragmented.

### 2. Rust Function Name Mangling (C++ Style)
- **Concept**: Keep function overloading in the `.cm` language, but let the Comet compiler mangle the names during LLVM IR generation. For example, `divide(DataFrame, Constant)` strictly calls the C-ABI function `comet_divide_step_df_const`.
- **Pros**: Clean user-facing API. Statically typed at compile time.
- **Cons**: Requires the Rust `stdlib` to bloat up with dozens of heavily duplicated, explicit implementations and export macros for every permutable variation of an operator.

### 3. Trait-Based Dynamic Dispatch (Type Tagging)
- **Concept**: Change the `*const f64` ABI to pass structured pointers containing type tags (e.g., passing a `struct CometData { dtype, void* ptr }`). The single Rust `comet_divide_step` function does internal pattern matching/dynamic-dispatch to safely cast and calculate the values.
- **Pros**: Extremely extensible. Adding Strings or Matrices in the future natively meshes with this architecture without requiring new entry-points.
- **Cons**: Introduces runtime branching overhead during execution. Complexity of memory ownership across the ABI boundary naturally increases.

### 4. Compiler-Side JIT Promotion / Broadcasting
- **Concept**: The Rust backend only implements the most general denominator case (`Array` x `Array` -> `Array`). If a user passes a `Constant` and a `DataFrame`, the Comet Compiler's LLVM codegen automatically allocates an array and broadcasts the constant into an array *before* making the raw C-ABI call to `comet_divide_step`.
- **Pros**: Keeps the Rust backend incredibly clean and DRY. Retains optimal performance. Puts the logical burden of type coercions entirely into the compiler.
- **Cons**: Does not inherently solve the addition of distinct fundamental types like `String`, which would eventually still require separate operators.

## Recommendation
**Option 4 (JIT Broadcasting) combined with Option 2 (Mangling for fundamental divergence)** is the most common approach taken by vector computation libraries (like NumPy/XLA). 

For now, the `stdlib.cm` files have been modified to condense the overloaded definitions into single `Series` implementations to strictly align with what the Rust backend currently outputs.

## Choice 
**Option 3 (Trait-Based Dynamic Dispatch) is the best long-term solution**

### Cost Analysis: Dynamic Dispatch vs Pure Computation

When evaluating the performance cost, it is critical to distinguish between **per-call** overhead and **per-element** overhead.

1. **Pure Computational Cost (The Major Cost)**:
   - For an array of 1000 * 1000 `f64` elements, you are processing 1,000,000 elements (8 megabytes of data).
   - Iterating through 1,000,000 elements doing floating-point arithmetic is an `O(N)` operation. This is heavily bottlenecked by RAM/Cache memory bandwidth and CPU vectorization (SIMD) capabilities. It takes on the order of milliseconds per operator execution.

2. **Dynamic Dispatch Cost (The Minor Cost)**:
   - If structured correctly, the dynamic dispatch (e.g., checking a `dtype` enum or using a `match` statement) only happens **once per `step` call**, *outside* of the 1,000,000 element loop.
   - Example:
     ```rust
     match (a.dtype, b.dtype) {
         (Type::F64Array, Type::F64Array) => {
             // Dispatch happens here (O(1) cost - roughly 2-5 CPU cycles)
             for i in 0..len { ... } // 1,000,000 iterations happen here
         }
         // ...
     }
     ```
   - This single `match` or vtable lookup takes less than a nanosecond (essentially `O(1)`). 
   - Across 15~20 operators, the total overhead of dynamic dispatch is still just a few dozen nanoseconds.

**Conclusion**: The pure computational and memory bandwidth cost is absolutely the dominant factor. The cost of dynamic dispatch is microscopically negligible as long as it is done *outside* the inner loops. The binary size will grow slightly due to branch implementations for different types, but for a standalone dynamic library running on modern hardware, this size increase is trivial.
