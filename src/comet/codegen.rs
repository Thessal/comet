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

    pub fn generate_ir(&self, contexts: &Vec<crate::comet::synthesis::Context>) -> String {
        self.declare_externals();

        for (i, ctx) in contexts.iter().enumerate() {
            self.generate_variant_executor(i, &ctx.graph);
        }

        self.module.print_to_string().to_string()
    }

    fn declare_externals(&self) {
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

        // Example: cs_zscore
        // declare %CsZscoreState* @comet_cs_zscore_init(i64, i64)
        let init_type = opaque_ptr_type.fn_type(&[i64_type.into(), i64_type.into()], false);
        self.module.add_function("comet_cs_zscore_init", init_type, None);

        // declare void @comet_cs_zscore_step(%CsZscoreState*, double*, double*, i64)
        let step_type = void_type.fn_type(&[
            opaque_ptr_type.into(), 
            f64_ptr_type.into(), 
            f64_ptr_type.into(), 
            i64_type.into()
        ], false);
        self.module.add_function("comet_cs_zscore_step", step_type, None);

        // declare void @comet_cs_zscore_free(%CsZscoreState*)
        let free_type = void_type.fn_type(&[opaque_ptr_type.into()], false);
        self.module.add_function("comet_cs_zscore_free", free_type, None);

        // Serialization Support
        // declare i8* @comet_cs_zscore_serialize(%CsZscoreState*)
        let serialize_type = i8_ptr_type.fn_type(&[opaque_ptr_type.into()], false);
        self.module.add_function("comet_cs_zscore_serialize", serialize_type, None);
        
        // declare %CsZscoreState* @comet_cs_zscore_deserialize(i8*)
        let deserialize_type = opaque_ptr_type.fn_type(&[i8_ptr_type.into()], false);
        self.module.add_function("comet_cs_zscore_deserialize", deserialize_type, None);

        // TODO: Declare external signatures for ALL other ops inside stdlib
    }

    fn generate_variant_executor(&self, id: usize, graph: &ExecutionGraph) {
        let void_type = self.context.void_type();
        let i64_type = self.context.i64_type();
        let f64_ptr_type = self.context.f64_type().ptr_type(AddressSpace::default());
        let f64_ptr_ptr_type = f64_ptr_type.ptr_type(AddressSpace::default());

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
                        // If it's a Source node, we must offset its stream pointer
                        if let ExecutionNode::Source { .. } = graph.nodes[arg_id] {
                            arg_ptr = unsafe { self.builder.build_gep(f64_type, arg_ptr, &[offset], &format!("stream_in_{}", arg_id)).unwrap() };
                        }
                        call_args.push(arg_ptr.into());
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

        let ir = codegen.generate_ir(&vec![synth_ctx]);
        println!("Generated IR:\n{}", ir);
        assert!(ir.contains("comet_cs_zscore_step"));
    }
}
