import ctypes
import numpy as np

def run_test():
    try:
        stdlib = ctypes.CDLL('./target/debug/libstdlib.so', mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        print(f"Failed to load stdlib: {e}")
        return

    try:
        lib = ctypes.CDLL('./volume_spike.so')
    except OSError as e:
        print(f"Failed to load library: {e}")
        return

    lib.execute_variant_0.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), 
        ctypes.POINTER(ctypes.c_double),                 
        ctypes.c_void_p,                                 
        ctypes.c_int64,                                  
        ctypes.c_int64                                   
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

    output = np.zeros((timesteps * feature_len,), dtype=np.float64)
    out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    lib.init_variant_0.argtypes = [ctypes.c_int64]
    lib.init_variant_0.restype = ctypes.c_void_p
    
    print("Calling automated LLVM initializer")
    state_blob_ptr = lib.init_variant_0(feature_len) # It will be nice if there's easy way to visualize the state blob, with named offsets. But that's optional. 

    print(f"Executing volume_spike.so variant 0 for {timesteps} timesteps...")
    try:
        lib.execute_variant_0(
            inputs_arr,
            ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_double)),
            ctypes.cast(state_blob_ptr, ctypes.c_void_p),
            feature_len,
            timesteps
        )
        print("Execution finished successfully.")
        print("Output:", output)
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == '__main__':
    run_test()