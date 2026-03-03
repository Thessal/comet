use crate::comet::ir::{ConstantValue, ExecutionGraph, ExecutionNode};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

pub struct Codegen {
    pub module_name: String,
    pub stdlib_path: Option<String>,
}

impl Codegen {
    pub fn new(module_name: &str, stdlib_path: Option<String>) -> Self {
        Codegen {
            module_name: module_name.to_string(),
            stdlib_path,
        }
    }

    pub fn emit_library(&self, output_base: &str, ir_string: &str) -> Result<(), String> {
        let out_dir = PathBuf::from(format!(".comet_tmp_{}", self.module_name));
        let _ = fs::remove_dir_all(&out_dir);
        fs::create_dir_all(out_dir.join("src")).map_err(|e| e.to_string())?;

        let current_dir = std::env::current_dir().map_err(|e| e.to_string())?;

        let stdlib_dep = if let Some(path) = &self.stdlib_path {
            let p = std::path::Path::new(path);
            let dir = if p.is_file() || path.ends_with(".so") {
                p.parent()
                    .unwrap_or(std::path::Path::new("."))
                    .to_string_lossy()
                    .to_string()
            } else {
                path.to_string()
            };
            format!(r#"stdlib = {{ path = "{}", package = "comet" }}"#, dir)
        } else {
            format!(
                r#"stdlib = {{ path = "{}", package = "comet" }}"#,
                current_dir.to_string_lossy()
            )
        };

        let cargo_toml = format!(
            r#"
[package]
name = "{}"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib"]

[dependencies]
{}
"#,
            self.module_name, stdlib_dep
        );

        fs::write(out_dir.join("Cargo.toml"), cargo_toml).map_err(|e| e.to_string())?;
        fs::write(out_dir.join("src/lib.rs"), ir_string).map_err(|e| e.to_string())?;

        let status = Command::new("cargo")
            .arg("build")
            .arg("--release")
            .env("CARGO_TARGET_DIR", current_dir.join("target"))
            .current_dir(&out_dir)
            .status()
            .map_err(|e| e.to_string())?;

        if !status.success() {
            return Err("Failed to compile generated Rust code".into());
        }

        let so_name = format!("lib{}.so", self.module_name.replace("-", "_"));
        let target_so = current_dir.join("target").join("release").join(&so_name);

        let dest_so = PathBuf::from(format!("{}.so", output_base));
        fs::copy(&target_so, &dest_so).map_err(|e| e.to_string())?;

        Ok(())
    }

    pub fn generate_ir(&self, graph: &ExecutionGraph) -> String {
        let mut ast_exports = TokenStream::new();
        for (idx, ast) in graph.ast_strings.iter().enumerate() {
            let ast_bytes = format!("{}\0", ast).into_bytes();
            let len = ast_bytes.len();
            let ident = format_ident!("comet_ast_0_{}", idx);
            let byte_str = proc_macro2::Literal::byte_string(&ast_bytes);
            ast_exports.extend(quote! {
                #[unsafe(no_mangle)]
                pub static #ident: [u8; #len] = *#byte_str;
            });
        }

        let mut stateful_nodes = HashMap::new();
        let mut num_state_fields: usize = 0;
        let mut state_struct_fields = TokenStream::new();

        let to_pascal_case = |s: &str| -> String {
            let mut result = String::new();
            let mut capitalize = true;
            for c in s.chars() {
                if c == '_' {
                    capitalize = true;
                } else if capitalize {
                    result.extend(c.to_uppercase());
                    capitalize = false;
                } else {
                    result.push(c);
                }
            }
            result
        };

        for (node_id, node) in graph.nodes.iter().enumerate() {
            if let ExecutionNode::Operation { op, args: _ } = node {
                let func_name = op.to_lowercase();
                if func_name != "unknown" {
                    stateful_nodes.insert(node_id, num_state_fields);

                    let mod_ident = format_ident!(
                        "{}",
                        if func_name == "const" {
                            "r#const".to_string()
                        } else if func_name == "where" {
                            "r#where".to_string()
                        } else {
                            func_name.clone()
                        }
                    );

                    let struct_name_str = if func_name == "consume" {
                        "ConsumeFloatState".to_string()
                    } else if func_name == "cs_zscore" {
                        "CsZscoreState".to_string()
                    } else {
                        format!("{}State", to_pascal_case(&func_name))
                    };

                    let struct_ident = format_ident!("{}", struct_name_str);
                    let field_ident = format_ident!("s{}", num_state_fields);

                    state_struct_fields.extend(quote! {
                        pub #field_ident: stdlib::#mod_ident::#struct_ident,
                    });

                    num_state_fields += 1;
                }
            }
        }

        // --- Init Function ---
        let mut init_calls = TokenStream::new();
        for (node_id, node) in graph.nodes.iter().enumerate() {
            if let ExecutionNode::Operation { op, args } = node {
                let func_name = op.to_lowercase();
                if func_name == "unknown" {
                    continue;
                }

                let state_idx = stateful_nodes[&node_id];
                let field_ident = format_ident!("s{}", state_idx);

                let mod_ident = format_ident!(
                    "{}",
                    if func_name == "const" {
                        "r#const".to_string()
                    } else if func_name == "where" {
                        "r#where".to_string()
                    } else {
                        func_name.clone()
                    }
                );

                let struct_name_str = if func_name == "consume" {
                    "ConsumeFloatState".to_string()
                } else if func_name == "cs_zscore" {
                    "CsZscoreState".to_string()
                } else {
                    format!("{}State", to_pascal_case(&func_name))
                };
                let struct_ident = format_ident!("{}", struct_name_str);

                let mut const_args = Vec::new();
                for &arg_id in args {
                    if let ExecutionNode::Constant { value } = &graph.nodes[arg_id] {
                        match value {
                            ConstantValue::Integer(i) => const_args.push(quote!(#i as usize)),
                            ConstantValue::Float(f) => const_args.push(quote!(#f as f64)),
                            ConstantValue::String => {
                                const_args.push(quote!(std::ptr::null()));
                                const_args.push(quote!(0));
                            }
                            ConstantValue::Boolean => const_args.push(quote!(0)),
                        }
                    }
                }

                let init_expr = match func_name.as_str() {
                    "clip" | "const" | "cs_rank_nonzero" | "data" => {
                        quote!(stdlib::#mod_ident::#struct_ident::new(#(#const_args,)* len as usize))
                    }
                    "tail_to_nan" | "where" => {
                        quote!(stdlib::#mod_ident::#struct_ident::new())
                    }
                    _ => {
                        // Standard macro functions take (period, len).
                        let period = if const_args.is_empty() {
                            quote!(0usize)
                        } else {
                            const_args[0].clone()
                        };
                        quote!(stdlib::#mod_ident::#struct_ident::new(#period, len as usize))
                    }
                };

                init_calls.extend(quote! {
                    #field_ident: #init_expr,
                });
            }
        }

        // --- Execute Function ---
        let mut execute_body = TokenStream::new();
        let alloc_vecs = quote! {
            let mut node_outputs: std::collections::HashMap<usize, Vec<f64>> = std::collections::HashMap::new();
        };
        execute_body.extend(alloc_vecs);

        let mut allocated_nodes = std::collections::HashSet::new();

        // Map initial allocations
        let mut source_idx: usize = 0;
        let mut init_allocs = TokenStream::new();
        for (node_id, node) in graph.nodes.iter().enumerate() {
            match node {
                ExecutionNode::Source { .. } => {
                    init_allocs.extend(quote! {
                        let mut source_vec = Vec::with_capacity((timesteps * len) as usize);
                        unsafe {
                            let src_ptr = *inputs.add(#source_idx);
                            source_vec.extend_from_slice(std::slice::from_raw_parts(src_ptr, (timesteps * len) as usize));
                        }
                        node_outputs.insert(#node_id, source_vec);
                    });
                    source_idx += 1;
                    allocated_nodes.insert(node_id);
                }
                ExecutionNode::Constant { value } => {
                    let c_val = match value {
                        ConstantValue::Float(f) => quote!(#f),
                        ConstantValue::Integer(i) => quote!(#i as f64),
                        _ => quote!(0.0),
                    };
                    init_allocs.extend(quote! {
                        node_outputs.insert(#node_id, vec![#c_val]);
                    });
                    allocated_nodes.insert(node_id);
                }
                ExecutionNode::Operation { op, args } => {
                    let mut is_intermediate = false;
                    for other_node in &graph.nodes {
                        if let ExecutionNode::Operation {
                            args: other_args, ..
                        } = other_node
                        {
                            if other_args.contains(&node_id) {
                                is_intermediate = true;
                            }
                        }
                    }
                    let out_len = graph
                        .output_nodes
                        .iter()
                        .filter(|&&id| id == node_id)
                        .count();
                    let needs_malloc = is_intermediate || out_len != 1;
                    if needs_malloc {
                        let func_name = op.to_lowercase();

                        let mut out_shape = stdlib::OutputShape::DataFrame;
                        for meta in inventory::iter::<stdlib::OperatorMeta> {
                            if meta.name == func_name.as_str() {
                                out_shape = meta.output_shape;
                                break;
                            }
                        }

                        let out_width = match out_shape {
                            stdlib::OutputShape::Matrix => quote! { (len * len) as usize },
                            stdlib::OutputShape::TimeSeries => quote! { 1usize },
                            _ => quote! { len as usize },
                        };
                        init_allocs.extend(quote! {
                            node_outputs.insert(#node_id, vec![0.0; #out_width]);
                        });
                        allocated_nodes.insert(node_id);
                    }
                }
            }
        }
        execute_body.extend(init_allocs);

        // Pre-create output ptr lookups
        let num_outputs = graph.output_nodes.len();
        execute_body.extend(quote! {
            let mut out_ptrs = Vec::with_capacity(#num_outputs);
            for i in 0..#num_outputs {
                out_ptrs.push(unsafe { *outputs.add(i) });
            }
        });

        // Event loop mapping
        let mut loop_body = TokenStream::new();

        for (node_id, node) in graph.nodes.iter().enumerate() {
            if let ExecutionNode::Operation { op, args } = node {
                let func_name = op.to_lowercase();
                if func_name == "unknown" {
                    continue;
                }

                let state_idx = stateful_nodes[&node_id];

                let mut step_args = TokenStream::new();
                for &arg_id in args {
                    if matches!(&graph.nodes[arg_id], ExecutionNode::Constant { .. }) {
                        continue;
                    }

                    // Source node dynamically handles DataType
                    let (dtype, is_source) =
                        if let ExecutionNode::Source { type_name, .. } = &graph.nodes[arg_id] {
                            match type_name.as_str() {
                                "Constant" => (quote!(stdlib::DataType::Constant), true),
                                "TimeSeries" => (quote!(stdlib::DataType::TimeSeries), true),
                                _ => (quote!(stdlib::DataType::DataFrame), true),
                            }
                        } else {
                            (quote!(stdlib::DataType::DataFrame), false)
                        };

                    step_args.extend(quote! {
                        {
                            let vec_ref = node_outputs.get(&#arg_id).unwrap();
                            let base_ptr = vec_ref.as_ptr();
                            let ptr = if #is_source && matches!(#dtype, stdlib::DataType::DataFrame | stdlib::DataType::TimeSeries) {
                                unsafe { base_ptr.add(offset) }
                            } else {
                                base_ptr
                            };
                            stdlib::CometData { dtype: #dtype, ptr }
                        },
                    });
                }

                let mut out_shape = stdlib::OutputShape::DataFrame;
                for meta in inventory::iter::<stdlib::OperatorMeta> {
                    if meta.name == func_name.as_str() {
                        out_shape = meta.output_shape;
                        break;
                    }
                }

                let out_width = match out_shape {
                    stdlib::OutputShape::Matrix => quote! { (len * len) as usize },
                    stdlib::OutputShape::TimeSeries => quote! { 1usize },
                    _ => quote! { len as usize },
                };

                // Output pointer handling
                let output_indices: Vec<usize> = graph
                    .output_nodes
                    .iter()
                    .enumerate()
                    .filter(|&(_, &id)| id == node_id)
                    .map(|(i, _)| i)
                    .collect();

                let out_ptr_expr = if allocated_nodes.contains(&node_id) {
                    quote! { node_outputs.get_mut(&#node_id).unwrap().as_mut_ptr() }
                } else if output_indices.len() == 1 {
                    let idx = output_indices[0];
                    quote! { unsafe { out_ptrs[#idx].add(t * (#out_width)) } }
                } else {
                    quote! { std::ptr::null_mut::<f64>() }
                };

                let mut memcpy_lines = TokenStream::new();
                for &variant_idx in &output_indices {
                    memcpy_lines.extend(quote! {
                        let final_ptr = unsafe { out_ptrs[#variant_idx].add(t * (#out_width)) };
                        if out_ptr != final_ptr && !out_ptr.is_null() {
                            unsafe { std::ptr::copy_nonoverlapping(out_ptr, final_ptr, #out_width) };
                        }
                    });
                }

                let field_ident = format_ident!("s{}", state_idx);
                loop_body.extend(quote! {
                    let out_ptr = #out_ptr_expr;
                    state.#field_ident.step(#step_args out_ptr);
                    #memcpy_lines
                });
            }
        }

        let run_loop = quote! {
            let state = unsafe { &mut *(state_blob as *mut PipelineState) };
            for t in 0..(timesteps as usize) {
                let offset = t * (len as usize);
                #loop_body
            }
        };
        execute_body.extend(run_loop);

        let code = quote! {
            extern crate stdlib;
            use stdlib::{UnaryOp, BinaryOp, TernaryOp, ZeroAryOp};

            #ast_exports

            pub struct PipelineState {
                #state_struct_fields
            }

            #[unsafe(no_mangle)]
            pub extern "C" fn init_variant_0(len: i64) -> *mut PipelineState {
                let state = Box::new(PipelineState {
                    #init_calls
                });
                Box::into_raw(state)
            }

            #[unsafe(no_mangle)]
            pub extern "C" fn execute_variant_0(
                inputs: *const *const f64,
                outputs: *const *mut f64,
                state_blob: *mut std::ffi::c_void,
                len: i64,
                timesteps: i64
            ) {
                #execute_body
            }

            #[unsafe(no_mangle)]
            pub extern "C" fn free_variant_0(state_blob: *mut std::ffi::c_void) {
                if !state_blob.is_null() {
                    unsafe {
                        let _ = Box::from_raw(state_blob as *mut PipelineState);
                    }
                }
            }
        };

        code.to_string()
    }
}
