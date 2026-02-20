# LLVM IR Codegen Migration Roadmap

## Phase 1: Inkwell Setup & Definitions
- [x] Initialize `inkwell::context::Context`, `Module`, and `Builder` inside the `Codegen` structs.
- [x] Define the base LLVM types that we'll need (`f64`, `i64`, pointers, etc.). To do that, write a table about parameter and return types of each operator, in src/stdlib/TODO.md. For ambiguous cases like Bool type and Data operator, skip it and mark it in src/stdlib/TODO.md. 
- [x] Implement `declare_externals()` to inject the LLVM signatures for all the `comet_*_init`, `comet_*_step`, and `comet_*_free` C-ABI functions we wrote in Rust.
- [x] Write a test code that generates LLVM IR, and make sure it compiles.

## Phase 2: Translation & Memory Allocation
- [ ] Translate `ExecutionNode::Source` into inputs fetched from the function arguments (`double** inputs`).
- [ ] Initialize operator states via their LLVM IR `init` declarations.
- [ ] Implement serialization and deserialization of operator states.
- [ ] Allocate intermediate arrays (via `alloca` or `malloc`) to hold output references for each graph node.

## Phase 3: The Native Event Loop
- [ ] Emit the loop construct block branching (`t = 0` up to `num_timesteps`).
- [ ] Resolve graph dependencies efficiently, generating GEP (GetElementPtr) instructions into the respective memory strides.
- [ ] Execute the corresponding `comet_*_step` call for each operation in topological order.
- [ ] Write a test code that converts a call tree into LLVM IR and print it to the console.

## Phase 4: Cleanup & Export
- [ ] Add the cleanup logic to deallocate all operator states via `comet_*_free`.
- [ ] Wire the outputs to the main return array payload.
- [ ] Update `main.rs` to configure an inkwell `Context` and output the compiled IR into a `.ll` testable artifact.
