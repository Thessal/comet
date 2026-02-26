import ctypes
import numpy as np

def run_test():
    try:
        stdlib = ctypes.CDLL('./target/debug/libstdlib.so', mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        print(f"Failed to load stdlib: {e}")
        return

    try:
        lib = ctypes.CDLL('./volume_spike_stage_0.so')
    except OSError as e:
        print(f"Failed to load library: {e}")
        return

    lib.execute_variant_0.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), # inputs
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), # outputs
        ctypes.c_void_p,                                 # state
        ctypes.c_int64,                                  # feature_len
        ctypes.c_int64                                   # timesteps
    ]
    lib.execute_variant_0.restype = None

    timesteps = 10
    feature_len = 2
    
    input1 = np.arange(0, timesteps * feature_len, dtype=np.float64)
    input2 = np.arange(100, 100 + timesteps * feature_len, dtype=np.float64)
    
    InputArrayType = ctypes.POINTER(ctypes.c_double) * 2
    inputs_arr = InputArrayType(
        input1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        input2.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    num_outputs = 12 # Based on the multiple_variants_minimal.cm stage 0 LLVM IR showing out_gep_11
    outputs = [np.zeros((timesteps * feature_len,), dtype=np.float64) for _ in range(num_outputs)]
    
    OutputArrayType = ctypes.POINTER(ctypes.c_double) * num_outputs
    outputs_arr = OutputArrayType(*[
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for out in outputs
    ])

    lib.init_variant_0.argtypes = [ctypes.c_int64]
    lib.init_variant_0.restype = ctypes.c_void_p
    
    print("Calling automated LLVM initializer")
    state_blob_ptr = lib.init_variant_0(feature_len)

    for i in range(num_outputs):
        array_type = ctypes.c_char * 1024
        ast = array_type.in_dll(lib, f"comet_ast_0_{i}").value.decode("utf8")
        print(ast)

    print(f"Executing volume_spike_stage_0.so with {num_outputs} variants for {timesteps} timesteps...")
    try:
        lib.execute_variant_0(
            inputs_arr,
            outputs_arr,
            ctypes.cast(state_blob_ptr, ctypes.c_void_p),
            feature_len,
            timesteps
        )
        print("Execution finished successfully.")
        for i, out in enumerate(outputs):
            print(f"Variant {i} Output:", out)
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == '__main__':
    run_test()