use crate::comet::ir::{ExecutionGraph, ExecutionNode};
use std::collections::HashMap;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::builder::Builder;
use inkwell::AddressSpace;
use std::process::Command;
use std::path::Path;

pub struct Codegen<'ctx> {
    pub context: &'ctx Context,
    pub module: Module<'ctx>,
    pub builder: Builder<'ctx>,
}

impl<'ctx> Codegen<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();
        Codegen { context, module, builder }
    }

    pub fn emit_library(&self, output_base: &str) -> Result<(), String> {
        let ll_path = format!("{}.ll", output_base);
        let so_path = format!("{}.so", output_base);

        // 1. Write the bitcode
        if let Err(e) = self.module.print_to_file(Path::new(&ll_path)) {
            return Err(format!("Failed to write LLVM IR: {:?}", e));
        }

        // 2. Invoke clang to build the shared library
        let status = Command::new("clang")
            .args(&[
                "-shared",
                "-fPIC",
                "-O3",
                "-o",
                &so_path,
                &ll_path,
            ])
            .status()
            .map_err(|e| format!("Failed to invoke clang: {}", e))?;

        if !status.success() {
            return Err("Clang failed to compile the shared library.".to_string());
        }

        Ok(())
    }

    pub fn generate_ir(&self, graph: &crate::comet::ir::ExecutionGraph) -> String {
        self.declare_externals(graph);

        self.generate_variant_init(graph);
        self.generate_variant_executor(graph);

        self.module.print_to_string().to_string()
    }

    fn declare_externals(&self, graph: &crate::comet::ir::ExecutionGraph) {
        let void_type = self.context.void_type();
        let i64_type = self.context.i64_type();
        let f64_ptr_type = self.context.ptr_type(AddressSpace::default());
        let opaque_ptr_type = self.context.ptr_type(AddressSpace::default());
        let i8_ptr_type = self.context.ptr_type(AddressSpace::default());

        // libc malloc / free
        let malloc_type = i8_ptr_type.fn_type(&[i64_type.into()], false);
        self.module.add_function("malloc", malloc_type, None);
        let free_type = void_type.fn_type(&[i8_ptr_type.into()], false);
        self.module.add_function("free", free_type, None);

        // llvm intrinsic memcpy
        let bool_type = self.context.bool_type();
        let memcpy_type = void_type.fn_type(&[
            i8_ptr_type.into(), // dest
            i8_ptr_type.into(), // src
            i64_type.into(),    // len
            bool_type.into()    // isvolatile
        ], false);
        self.module.add_function("llvm.memcpy.p0i8.p0i8.i64", memcpy_type, None);

        // Define CometData LLVM struct globally: { i32, double* }
        let _comet_data_type = self.context.struct_type(&[
            self.context.i32_type().into(), // dtype
            f64_ptr_type.into()             // ptr
        ], false);

        let mut required_functions = std::collections::HashMap::new();
        for node in &graph.nodes {
            if let crate::comet::ir::ExecutionNode::Operation { op, args } = node {
                let stream_args_count = args.iter().filter(|&&id| !matches!(&graph.nodes[id], crate::comet::ir::ExecutionNode::Constant { .. })).count();
                required_functions.insert(op.clone(), stream_args_count);
            }
        }

        for (func_name, arg_count) in required_functions {
            let fn_name_lower = func_name.to_lowercase();
            if fn_name_lower == "unknown" { continue; }
            
            let mut step_args = vec![opaque_ptr_type.into()];
            for _ in 0..arg_count {
                step_args.push(opaque_ptr_type.into()); // Pass CometData by pointer, not by value!
            }
            step_args.push(f64_ptr_type.into()); // output ptr
            step_args.push(i64_type.into());     // len
            let step_type = void_type.fn_type(&step_args, false);
            self.module.add_function(&format!("comet_{}_step", fn_name_lower), step_type, None);
        }
    }

    fn generate_variant_executor(&self, graph: &ExecutionGraph) {
        for (idx, ast) in graph.ast_strings.iter().enumerate() {
            let ast_cstring = std::ffi::CString::new(ast.as_str()).unwrap_or_default();
            let string_val = self.context.const_string(ast_cstring.as_bytes_with_nul(), false);
            let global_ast = self.module.add_global(string_val.get_type(), None, &format!("comet_ast_0_{}", idx));
            global_ast.set_initializer(&string_val);
            global_ast.set_linkage(inkwell::module::Linkage::External);
        }
        
        let void_type = self.context.void_type();
        let i32_type = self.context.i32_type();
        let i64_type = self.context.i64_type();
        let f64_type = self.context.f64_type();
        let f64_ptr_type = self.context.ptr_type(AddressSpace::default());
        let f64_ptr_ptr_type = self.context.ptr_type(AddressSpace::default());
        let opaque_ptr_type = self.context.ptr_type(AddressSpace::default());

        let comet_data_type = self.context.struct_type(&[
            i32_type.into(), // dtype
            f64_ptr_type.into() // ptr
        ], false);

        let fn_type = void_type.fn_type(&[
            f64_ptr_ptr_type.into(), // inputs
            f64_ptr_ptr_type.into(), // outputs
            opaque_ptr_type.into(),  // state_blob (contiguous buffer)
            i64_type.into(),         // len
            i64_type.into()          // timesteps
        ], false);

        let function_name = "execute_variant_0";
        let function = self.module.add_function(function_name, fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        let inputs_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let outputs_ptr_array = function.get_nth_param(1).unwrap().into_pointer_value();
        let state_blob_raw = function.get_nth_param(2).unwrap().into_pointer_value();
        let len_val = function.get_nth_param(3).unwrap().into_int_value();
        let timesteps_val = function.get_nth_param(4).unwrap().into_int_value();
        
        // 1. Initialization Phase & Memory Allocations
        let malloc_fn = self.module.get_function("malloc").unwrap();
        let alloc_size = self.builder.build_int_mul(len_val, i64_type.const_int(8, false), "alloc_size").unwrap();

        let mut node_outputs = HashMap::new();
        let mut node_states = HashMap::new();
        let mut source_indices = HashMap::new();
        let mut next_source_idx = 0;
        
        // Cache outputs pointers array lookups before event loop
        let mut loaded_output_ptrs = Vec::new();
        for i in 0..graph.output_nodes.len() {
            let idx_val = i64_type.const_int(i as u64, false);
            let gep = unsafe { self.builder.build_gep(f64_ptr_type, outputs_ptr_array, &[idx_val], &format!("out_gep_{}", i)).unwrap() };
            let load = self.builder.build_load(f64_ptr_type, gep, &format!("out_ptr_{}", i)).unwrap();
            loaded_output_ptrs.push(load.into_pointer_value());
        }

        // Build the state tracking struct type (each stateful op gets a designated memory block)
        let state_field_type = opaque_ptr_type; // Store pointers!
        let mut state_fields: Vec<inkwell::types::BasicTypeEnum> = Vec::new();
        let mut stateful_node_indices = HashMap::new();

        for (node_id, node) in graph.nodes.iter().enumerate() {
            if let ExecutionNode::Operation { op, .. } = node {
                if op.to_lowercase() != "unknown" {
                    stateful_node_indices.insert(node_id, state_fields.len() as u32);
                    state_fields.push(state_field_type.into());
                }
            }
        }
        let state_struct_type = self.context.struct_type(&state_fields, false);

        for (node_id, node) in graph.nodes.iter().enumerate() {
            match node {
                ExecutionNode::Source { name, .. } => {
                    let idx = *source_indices.entry(name.clone()).or_insert_with(|| {
                        let i = next_source_idx;
                        next_source_idx += 1;
                        i
                    });
                    
                    let idx_val = i64_type.const_int(idx as u64, false);
                    let gep = unsafe { self.builder.build_gep(f64_ptr_type, inputs_ptr, &[idx_val], &format!("input_gep_{}", node_id)).unwrap() };
                    let load = self.builder.build_load(f64_ptr_type, gep, &format!("var_{}_ptr", node_id)).unwrap();
                    node_outputs.insert(node_id, load.into_pointer_value());
                },
                ExecutionNode::Constant { value } => {
                    use crate::comet::ir::ConstantValue;
                    match value {
                        ConstantValue::Float(f) => {
                            let f_val = f64_type.const_float(*f);
                            let alloca = self.builder.build_alloca(f64_type, &format!("const_ptr_{}", node_id)).unwrap();
                            self.builder.build_store(alloca, f_val).unwrap();
                            node_outputs.insert(node_id, alloca);
                        },
                        ConstantValue::Integer(i) => {
                            let f_val = f64_type.const_float(*i as f64);
                            let alloca = self.builder.build_alloca(f64_type, &format!("const_ptr_{}", node_id)).unwrap();
                            self.builder.build_store(alloca, f_val).unwrap();
                            node_outputs.insert(node_id, alloca);
                        },
                        _ => {}
                    }
                },
                ExecutionNode::Operation { .. } => {
                    let mut is_intermediate = false;
                    for other_node in &graph.nodes {
                        if let ExecutionNode::Operation { args, .. } = other_node {
                            if args.contains(&node_id) { is_intermediate = true; }
                        }
                    }
                    let output_indices: Vec<usize> = graph.output_nodes.iter().enumerate()
                        .filter(|&(_, &id)| id == node_id).map(|(i, _)| i).collect();
                    
                    let needs_malloc = is_intermediate || output_indices.len() != 1;
                    
                    if needs_malloc {
                        let malloc_call = self.builder.build_call(malloc_fn, &[alloc_size.into()], &format!("malloc_out_{}", node_id)).unwrap();
                        let out_ptr = malloc_call.try_as_basic_value().unwrap_basic().into_pointer_value();
                        node_outputs.insert(node_id, out_ptr);
                    }
                    
                    if let Some(target_idx) = stateful_node_indices.get(&node_id) {
                        // Dynamically map struct offset correctly!
                        let state_k_ptr = self.builder.build_struct_gep(state_struct_type, state_blob_raw, *target_idx, &format!("state_offset_{}", node_id)).unwrap();
                        let loaded_state_ptr = self.builder.build_load(opaque_ptr_type, state_k_ptr, &format!("state_ptr_{}", node_id)).unwrap();
                        node_states.insert(node_id, loaded_state_ptr.into_pointer_value());
                    }
                }
            }
        }
        
        // 2. Event Loop Phase
        let loop_bb = self.context.append_basic_block(function, "event_loop");
        let loop_inc_bb = self.context.append_basic_block(function, "event_loop_inc");
        let loop_cond_bb = self.context.append_basic_block(function, "event_loop_cond");
        let loop_end_bb = self.context.append_basic_block(function, "event_loop_end");

        let t_ptr = self.builder.build_alloca(i64_type, "t").unwrap();
        self.builder.build_store(t_ptr, i64_type.const_int(0, false)).unwrap();
        self.builder.build_unconditional_branch(loop_cond_bb).unwrap();

        // Condition
        self.builder.position_at_end(loop_cond_bb);
        let t_val = self.builder.build_load(i64_type, t_ptr, "t_val").unwrap().into_int_value();
        let cond = self.builder.build_int_compare(inkwell::IntPredicate::ULT, t_val, timesteps_val, "loop_cond").unwrap();
        self.builder.build_conditional_branch(cond, loop_bb, loop_end_bb).unwrap();

        // Loop Body
        self.builder.position_at_end(loop_bb);
        let offset = self.builder.build_int_mul(t_val, len_val, "offset").unwrap();

        for (node_id, node) in graph.nodes.iter().enumerate() {
            if let ExecutionNode::Operation { op, args } = node {
                let func_name = op.to_lowercase();
                if let Some(step_fn) = self.module.get_function(&format!("comet_{}_step", func_name)) {
                    let mut call_args = Vec::new();
                    
                    // State Ptr
                    if let Some(state_ptr) = node_states.get(&node_id) {
                        call_args.push((*state_ptr).into());
                    } else {
                        // Passes null opaque pointer if state strictly isn't resolved properly (safety fallback)
                        call_args.push(opaque_ptr_type.const_null().into());
                    }
                    
                    // Input Args
                    for &arg_id in args {
                        if matches!(&graph.nodes[arg_id], ExecutionNode::Constant { .. }) {
                            continue;
                        }

                        let mut arg_ptr = node_outputs[&arg_id];
                        let mut node_dtype = 2; // Default to DataFrame=2
                        // If it's a Source node, we must offset its stream pointer
                        if let ExecutionNode::Source { type_name, .. } = &graph.nodes[arg_id] {
                            arg_ptr = unsafe { self.builder.build_gep(f64_type, arg_ptr, &[offset], &format!("stream_in_{}", arg_id)).unwrap() };
                            if type_name == "Constant" { node_dtype = 0; }
                            else if type_name == "TimeSeries" { node_dtype = 1; }
                        }

                        // Build CometData struct dynamically to pass to BinaryOp
                        let mut struct_val = comet_data_type.get_undef();
                        struct_val = self.builder.build_insert_value(struct_val, i32_type.const_int(node_dtype, false), 0, "insert_dtype").unwrap().into_struct_value();
                        struct_val = self.builder.build_insert_value(struct_val, arg_ptr, 1, "insert_ptr").unwrap().into_struct_value();
                        
                        let struct_ptr = self.builder.build_alloca(comet_data_type, "comet_data_ptr").unwrap();
                        self.builder.build_store(struct_ptr, struct_val).unwrap();
                        call_args.push(struct_ptr.into());
                    }
                    
                    // Output Ptr
                    let mut directly_written_output = None;
                    let out_ptr;
                    
                    let output_indices: Vec<usize> = graph.output_nodes.iter().enumerate()
                        .filter(|&(_, &id)| id == node_id).map(|(i, _)| i).collect();

                    if let Some(&malloc_ptr) = node_outputs.get(&node_id) {
                        out_ptr = malloc_ptr;
                    } else if output_indices.len() == 1 {
                        let variant_idx = output_indices[0];
                        out_ptr = unsafe { self.builder.build_gep(f64_type, loaded_output_ptrs[variant_idx], &[offset], &format!("stream_out_direct_{}", node_id)).unwrap() };
                        directly_written_output = Some(variant_idx);
                    } else {
                        // Dead code path fallback
                        out_ptr = f64_ptr_type.const_null();
                    }
                    call_args.push(out_ptr.into());
                    
                    // Len
                    call_args.push(len_val.into());
                    
                    self.builder.build_call(step_fn, &call_args, &format!("step_{}", node_id)).unwrap();
                    
                    if !output_indices.is_empty() {
                        let memcpy_fn = self.module.get_function("llvm.memcpy.p0i8.p0i8.i64").unwrap();
                        for &variant_idx in &output_indices {
                            if Some(variant_idx) == directly_written_output {
                                continue;
                            }
                            let dest_ptr = unsafe { self.builder.build_gep(f64_type, loaded_output_ptrs[variant_idx], &[offset], &format!("stream_out_copy_{}", node_id)).unwrap() };
                            
                            self.builder.build_call(memcpy_fn, &[
                                dest_ptr.into(),
                                out_ptr.into(),
                                alloc_size.into(),
                                self.context.bool_type().const_int(0, false).into() // isvolatile = false
                            ], &format!("memcpy_out_{}", node_id)).unwrap();
                        }
                    }
                }
            }
        }
        self.builder.build_unconditional_branch(loop_inc_bb).unwrap();

        // Loop Increment
        self.builder.position_at_end(loop_inc_bb);
        let t_next = self.builder.build_int_add(t_val, i64_type.const_int(1, false), "t_next").unwrap();
        self.builder.build_store(t_ptr, t_next).unwrap();
        self.builder.build_unconditional_branch(loop_cond_bb).unwrap();

        // 3. Cleanup Phase
        self.builder.position_at_end(loop_end_bb);
        let free_fn = self.module.get_function("free").unwrap();
        
        for (node_id, node) in graph.nodes.iter().enumerate() {
            // Free intermediate buffers
            if let ExecutionNode::Operation { .. } = node {
                if let Some(&malloc_ptr) = node_outputs.get(&node_id) {
                    self.builder.build_call(free_fn, &[malloc_ptr.into()], &format!("free_buf_{}", node_id)).unwrap();
                }
            }
            // ExecutionNode::Constant uses raw alloca inline memory! So NO free operations injected!
        }

        let _ = self.builder.build_return(None);
    }
    fn generate_variant_init(&self, graph: &ExecutionGraph) {
        let i32_type = self.context.i32_type();
        let i64_type = self.context.i64_type();
        let f64_type = self.context.f64_type();
        let opaque_ptr_type = self.context.ptr_type(AddressSpace::default());

        let fn_type = opaque_ptr_type.fn_type(&[i64_type.into()], false);
        let function_name = "init_variant_0";
        let function = self.module.add_function(function_name, fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        let len_val = function.get_nth_param(0).unwrap().into_int_value();
        
        let state_field_type = opaque_ptr_type;
        let mut state_fields: Vec<inkwell::types::BasicTypeEnum> = Vec::new();
        let mut stateful_node_indices = HashMap::new();

        for (node_id, node) in graph.nodes.iter().enumerate() {
            match node {
                ExecutionNode::Operation { op, .. } => {
                    if op.to_lowercase() != "unknown" {
                        stateful_node_indices.insert(node_id, state_fields.len() as u32);
                        state_fields.push(state_field_type.into());
                    }
                },
                ExecutionNode::Source { .. } => {
                    // Source nodes do not possess internal state. Their data pointers are mapped via Python inputs.
                },
                _ => {}
            }
        }
        let state_struct_type = self.context.struct_type(&state_fields, false);
        
        let malloc_fn = self.module.get_function("malloc").unwrap();
        let struct_size = self.builder.build_int_mul(
            i64_type.const_int(state_fields.len() as u64, false),
            i64_type.const_int(8, false),
            "state_size"
        ).unwrap();
        
        let malloc_call = self.builder.build_call(malloc_fn, &[struct_size.into()], "malloc_state").unwrap();
        let state_blob_raw = malloc_call.try_as_basic_value().unwrap_basic().into_pointer_value();

        for (node_id, node) in graph.nodes.iter().enumerate() {
            let mut struct_ptr = None;

            match node {
                ExecutionNode::Source { .. } => {
                    // No initialization step for Sources since Python manages the memory for the inputs buffer.
                },
                ExecutionNode::Operation { op, args } => {
                    let func_name = op.to_lowercase();
                    
                    let init_fn_name = format!("comet_{}_init", func_name);
                    let init_fn = match self.module.get_function(&init_fn_name) {
                        Some(f) => f,
                        None => {
                            let mut param_types: Vec<inkwell::types::BasicTypeEnum> = Vec::new();
                            for &arg_id in args {
                                if let ExecutionNode::Constant { value } = &graph.nodes[arg_id] {
                                    use crate::comet::ir::ConstantValue;
                                    match value {
                                        ConstantValue::Integer(_) => param_types.push(i64_type.into()),
                                        ConstantValue::Float(_) => param_types.push(f64_type.into()),
                                        ConstantValue::String => {
                                            param_types.push(opaque_ptr_type.into()); // ptr
                                            param_types.push(i64_type.into());        // len
                                        },
                                        ConstantValue::Boolean => param_types.push(i64_type.into()),
                                    }
                                }
                            }
                            
                            param_types.push(i64_type.into()); // append len
                            
                            let meta_params: Vec<inkwell::types::BasicMetadataTypeEnum> = param_types.iter().map(|&t| t.into()).collect();
                            let init_type = opaque_ptr_type.fn_type(&meta_params, false);
                            self.module.add_function(&init_fn_name, init_type, None)
                        }
                    };

                    let mut call_args = Vec::new();
                    for &arg_id in args {
                        if let ExecutionNode::Constant { value } = &graph.nodes[arg_id] {
                            use crate::comet::ir::ConstantValue;
                            match value {
                                ConstantValue::Integer(i) => call_args.push(i64_type.const_int(*i as u64, false).into()),
                                ConstantValue::Float(f) => call_args.push(f64_type.const_float(*f).into()),
                                ConstantValue::String => {
                                    call_args.push(opaque_ptr_type.const_null().into());
                                    call_args.push(i64_type.const_int(0, false).into());
                                },
                                ConstantValue::Boolean => call_args.push(i64_type.const_int(0, false).into()),
                            }
                        }
                    }
                    
                    if init_fn.count_params() as usize == call_args.len() + 2 {
                         call_args.push(i64_type.const_int(0, false).into());
                    }

                    call_args.push(len_val.into());

                    let mut final_args = Vec::new();
                    for i in 0..init_fn.count_params() {
                        if i < call_args.len() as u32 {
                             final_args.push(call_args[i as usize]);
                        }
                    }

                    let init_call = self.builder.build_call(init_fn, &final_args, &format!("init_call_{}", node_id)).unwrap();
                    struct_ptr = Some(init_call.try_as_basic_value().unwrap_basic().into_pointer_value());
                },
                _ => {}
            }

            if let Some(ptr) = struct_ptr {
                if let Some(target_idx) = stateful_node_indices.get(&node_id) {
                    let state_k_ptr = self.builder.build_struct_gep(state_struct_type, state_blob_raw, *target_idx, &format!("state_offset_{}", node_id)).unwrap();
                    self.builder.build_store(state_k_ptr, ptr).unwrap();
                }
            }
        }

        self.builder.build_return(Some(&state_blob_raw)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comet::ir::{ExecutionGraph, ExecutionNode};
    use inkwell::context::Context;

    #[test]
    fn test_llvm_ir_generation() {
        let context = Context::create();
        let codegen = Codegen::new(&context, "test_module");

        let mut graph = ExecutionGraph::new();
        graph.add_node(ExecutionNode::Source {
            name: "close".to_string(),
            type_name: "Signal".to_string(),
        });
        graph.add_node(ExecutionNode::Operation {
            op: "cs_zscore".to_string(),
            args: vec![0],
        });

        let ir = codegen.generate_ir(&vec![graph]);
        println!("Generated IR:\n{}", ir);
        assert!(ir.contains("comet_cs_zscore_step"));
    }
}