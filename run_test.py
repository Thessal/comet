import ctypes
import numpy as np

def run_test():
    # Load the standard library which implements the actual operators
    try:
        stdlib = ctypes.CDLL('./target/debug/libstdlib.so', mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        print(f"Failed to load stdlib: {e}")
        return

    # Load the synthesized flow library
    try:
        lib = ctypes.CDLL('./volume_spike.so')
    except OSError as e:
        print(f"Failed to load library: {e}")
        return

    # Extract embedded AST
    try:
        ast_bytes = (ctypes.c_char * 1024).in_dll(lib, "comet_ast_0").value
        print(f"-- Embedded AST/Equation --\n{ast_bytes.decode('utf-8')}\n---------------------------")
    except ValueError:
        print("Warning: Embedded AST string not found in the library.")

    # void execute_variant_0(double** inputs, double* output, void* state_blob, int64_t len, int64_t timesteps)
    lib.execute_variant_0.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
        ctypes.POINTER(ctypes.c_double),                 
        ctypes.c_void_p,                                 
        ctypes.c_int64,                                  
        ctypes.c_int64                                   
    ]
    lib.execute_variant_0.restype = None

    # Setup 10 timesteps, single feature per step
    timesteps = 10
    feature_len = 1
    
    # input1 (volume) and input2 (mean_vol)
    input1 = np.full((timesteps * feature_len,), 100.0, dtype=np.float64)
    input2 = np.full((timesteps * feature_len,), 50.0, dtype=np.float64) # Should output 2.0
    
    InputArrayType = ctypes.POINTER(ctypes.c_double) * 2
    inputs_arr = InputArrayType(
        input1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        input2.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    output = np.zeros((timesteps * feature_len,), dtype=np.float64)
    out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Create dummy state blob, 1MB is safe
    state_blob = ctypes.create_string_buffer(1024 * 1024)
    
    # Operator 0: data("volume")
    stdlib.comet_data_init.restype = ctypes.c_void_p
    data1_ptr = stdlib.comet_data_init(b"volume", 6, feature_len)
    ctypes.memmove(ctypes.byref(state_blob, 0), data1_ptr, 128)
    
    # Operator 1: data("volume")
    data2_ptr = stdlib.comet_data_init(b"volume", 6, feature_len)
    ctypes.memmove(ctypes.byref(state_blob, 2048), data2_ptr, 128)

    # Operator 2: ts_mean(..., lookback=10)
    stdlib.comet_ts_mean_init.restype = ctypes.c_void_p
    mean_ptr = stdlib.comet_ts_mean_init(10, feature_len)
    ctypes.memmove(ctypes.byref(state_blob, 4096), mean_ptr, 128)

    # Operator 3: divide
    stdlib.comet_divide_init.restype = ctypes.c_void_p
    div_ptr = stdlib.comet_divide_init(0, feature_len)
    # DivideState is zero sized, reading 128 bytes from its pointer causes a segfault. No memmove needed.

    print(f"Executing volume_spike.so variant 0 for {timesteps} timesteps...")
    try:
        lib.execute_variant_0(
            inputs_arr,
            out_ptr,
            ctypes.cast(state_blob, ctypes.c_void_p),
            feature_len,
            timesteps
        )
        print("Execution finished successfully.")
        print("Output slice:", output[:5])
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == '__main__':
    run_test()