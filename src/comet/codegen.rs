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

    pub fn generate_ir(&self, contexts: &Vec<crate::comet::synthesis::Context>, symbol_table: &crate::comet::symbols::SymbolTable) -> String {
        self.declare_externals(symbol_table);

        for (i, ctx) in contexts.iter().enumerate() {
            self.generate_variant_executor(i, &ctx.graph);
        }

        self.module.print_to_string().to_string()
    }

    fn declare_externals(&self, symbol_table: &crate::comet::symbols::SymbolTable) {
        let void_type = self.context.void_type();
        let i64_type = self.context.i64_type();
        let f64_ptr_type = self.context.f64_type().ptr_type(AddressSpace::default());
        let opaque_ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
        let i8_ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());

        // libc malloc / free
        let malloc_type = i8_ptr_type.fn_type(&[i64_type.into()], false);
        self.module.add_function("malloc", malloc_type, None);
        let free_type = void_type.fn_type(&[i8_ptr_type.into()], false);
        self.module.add_function("free", free_type, None);

        // Define CometData LLVM struct globally: { i32, double* }
        let comet_data_type = self.context.struct_type(&[
            self.context.i32_type().into(), // dtype
            f64_ptr_type.into()             // ptr
        ], false);

        for (func_name, func_info) in &symbol_table.functions {
            let fn_name_lower = func_name.to_lowercase();
            
            let init_type = opaque_ptr_type.fn_type(&[i64_type.into(), i64_type.into()], false);
            self.module.add_function(&format!("comet_{}_init", fn_name_lower), init_type, None);
            
            let mut step_args = vec![opaque_ptr_type.into()];
            for _ in 0..func_info.params.len() {
                step_args.push(comet_data_type.into());
            }
            step_args.push(f64_ptr_type.into()); // output ptr
            step_args.push(i64_type.into());     // len
            let step_type = void_type.fn_type(&step_args, false);
            self.module.add_function(&format!("comet_{}_step", fn_name_lower), step_type, None);
            
            let free_type = void_type.fn_type(&[opaque_ptr_type.into()], false);
            self.module.add_function(&format!("comet_{}_free", fn_name_lower), free_type, None);
        }
    }

    fn generate_variant_executor(&self, id: usize, graph: &ExecutionGraph) {
        let void_type = self.context.void_type();
        let i32_type = self.context.i32_type();
        let i64_type = self.context.i64_type();
        let f64_ptr_type = self.context.f64_type().ptr_type(AddressSpace::default());
        let f64_ptr_ptr_type = f64_ptr_type.ptr_type(AddressSpace::default());

        let comet_data_type = self.context.struct_type(&[
            i32_type.into(), // dtype
            f64_ptr_type.into() // ptr
        ], false);

        let fn_type = void_type.fn_type(&[
            f64_ptr_ptr_type.into(), // inputs
            f64_ptr_type.into(),     // output
            i64_type.into(),         // len
            i64_type.into()          // timesteps
        ], false);

        let function_name = format!("execute_variant_{}", id);
        let function = self.module.add_function(&function_name, fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(basic_block);

        let inputs_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let len_val = function.get_nth_param(2).unwrap().into_int_value();
        
        // 1. Initialization Phase & Memory Allocations
        let malloc_fn = self.module.get_function("malloc").unwrap();
        let alloc_size = self.builder.build_int_mul(len_val, i64_type.const_int(8, false), "alloc_size").unwrap();

        let mut node_outputs = HashMap::new();
        let mut node_states = HashMap::new();
        let mut source_indices = HashMap::new();
        let mut next_source_idx = 0;

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
                ExecutionNode::Constant { .. } => {
                    let malloc_call = self.builder.build_call(malloc_fn, &[alloc_size.into()], &format!("malloc_const_{}", node_id)).unwrap();
                    let raw_ptr = malloc_call.try_as_basic_value().unwrap_basic().into_pointer_value();
                    node_outputs.insert(node_id, raw_ptr);
                },
                ExecutionNode::Operation { op, .. } => {
                    let malloc_call = self.builder.build_call(malloc_fn, &[alloc_size.into()], &format!("malloc_out_{}", node_id)).unwrap();
                    let out_ptr = malloc_call.try_as_basic_value().unwrap_basic().into_pointer_value();
                    node_outputs.insert(node_id, out_ptr);
                    
                    let func_name = op.to_lowercase();

                    if func_name != "unknown" {
                        if let Some(init_fn) = self.module.get_function(&format!("comet_{}_init", func_name)) {
                            let period = i64_type.const_int(10, false); // Example stub constraint value
                            let init_call = self.builder.build_call(init_fn, &[period.into(), len_val.into()], &format!("state_{}", node_id)).unwrap();
                            let state_ptr = init_call.try_as_basic_value().unwrap_basic().into_pointer_value();
                            node_states.insert(node_id, state_ptr);
                        }
                    }
                }
            }
        }
        
        let output_ptr = function.get_nth_param(1).unwrap().into_pointer_value();
        let timesteps_val = function.get_nth_param(3).unwrap().into_int_value();
        
        let f64_type = self.context.f64_type();

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
                    }
                    
                    // Input Args
                    for &arg_id in args {
                        let mut arg_ptr = node_outputs[&arg_id];
                        let mut node_dtype = 2; // Default to DataFrame=2
                        // If it's a Source node, we must offset its stream pointer
                        if let ExecutionNode::Source { type_name, .. } = &graph.nodes[arg_id] {
                            arg_ptr = unsafe { self.builder.build_gep(f64_type, arg_ptr, &[offset], &format!("stream_in_{}", arg_id)).unwrap() };
                            if type_name == "Constant" { node_dtype = 0; }
                            else if type_name == "TimeSeries" { node_dtype = 1; }
                        } else if let ExecutionNode::Constant { .. } = &graph.nodes[arg_id] {
                            node_dtype = 0; // Constant=0
                        }

                        // Build CometData struct dynamically to pass to BinaryOp
                        let mut struct_val = comet_data_type.get_undef();
                        struct_val = self.builder.build_insert_value(struct_val, i32_type.const_int(node_dtype, false), 0, "insert_dtype").unwrap().into_struct_value();
                        struct_val = self.builder.build_insert_value(struct_val, arg_ptr, 1, "insert_ptr").unwrap().into_struct_value();
                        
                        call_args.push(struct_val.into());
                    }
                    
                    // Output Ptr
                    let mut out_ptr = node_outputs[&node_id];
                    if node_id == graph.nodes.len() - 1 {
                        // The final node writes directly to the function's output pointer stream
                        out_ptr = unsafe { self.builder.build_gep(f64_type, output_ptr, &[offset], "stream_out").unwrap() };
                    }
                    call_args.push(out_ptr.into());
                    
                    // Len
                    call_args.push(len_val.into());
                    
                    self.builder.build_call(step_fn, &call_args, &format!("step_{}", node_id)).unwrap();
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
            if let ExecutionNode::Operation { op, .. } = node {
                if node_id != graph.nodes.len() - 1 {
                    self.builder.build_call(free_fn, &[node_outputs[&node_id].into()], &format!("free_buf_{}", node_id)).unwrap();
                }

                // Free States
                if let Some(state_ptr) = node_states.get(&node_id) {
                    let func_name = op.to_lowercase();
                    if let Some(free_fn) = self.module.get_function(&format!("comet_{}_free", func_name)) {
                        self.builder.build_call(free_fn, &[(*state_ptr).into()], &format!("free_state_{}", node_id)).unwrap();
                    }
                }
            } else if let ExecutionNode::Constant { .. } = node {
                self.builder.build_call(free_fn, &[node_outputs[&node_id].into()], &format!("free_const_{}", node_id)).unwrap();
            }
        }

        self.builder.build_return(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comet::ir::{ExecutionGraph, ExecutionNode};
    use crate::comet::synthesis::Context as SynthesisContext;
    use inkwell::context::Context;
    use std::collections::HashMap;

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

        let synth_ctx = SynthesisContext {
            variables: HashMap::new(),
            graph,
        };

        let mut st = crate::comet::symbols::SymbolTable::new();
        st.functions.insert("cs_zscore".to_string(), crate::comet::symbols::FuncInfo {
            name: "cs_zscore".to_string(),
            params: vec![crate::comet::ast::TypedArg { name: "x".to_string(), constraint: crate::comet::ast::Constraint::None }],
            return_type: crate::comet::ast::Constraint::None,
            body: crate::comet::ast::Block { stmts: vec![] }
        });

        let ir = codegen.generate_ir(&vec![synth_ctx], &st);
        println!("Generated IR:\n{}", ir);
        assert!(ir.contains("comet_cs_zscore_step"));
    }
}
