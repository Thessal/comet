use crate::comet::ir::{ExecutionGraph, ExecutionNode, OperatorOp};
use std::collections::HashMap;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::builder::Builder;
use inkwell::AddressSpace;

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

        // 1. Initialization Phase
        // TODO: Build states and map execution node dependencies
        
        // 2. Event Loop Phase
        // TODO: Build LLVM `loop` block handling memory iterators

        // 3. Cleanup Phase
        // TODO: Generate _free instructions

        self.builder.build_return(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comet::ir::{ExecutionGraph, ExecutionNode, OperatorOp};
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
            op: OperatorOp::ZScore,
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
