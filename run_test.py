import ctypes
import numpy as np

def run_test(filename):
    try:
        stdlib = ctypes.CDLL('./target/debug/libstdlib.so', mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        print(f"Failed to load stdlib: {e}")
        return

    try:
        lib = ctypes.CDLL(filename)
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
    
    # Simulated data dictionary that replaces input1, input2
    data_dict = {
        "trade_count": np.arange(0, timesteps * feature_len, dtype=np.float64),
        "close": np.arange(100, 100 + timesteps * feature_len, dtype=np.float64),
        "volume": np.arange(200, 200 + timesteps * feature_len, dtype=np.float64),
    }

    try:
        array_type_meta = ctypes.c_char * 1024
        input_names_bytes = array_type_meta.in_dll(lib, "comet_input_names_0").value
        input_names_str = input_names_bytes.decode("utf8")
        input_names = [n.strip() for n in input_names_str.split(",") if n.strip()]
    except ValueError:
        print("comet_input_names_0 not found in library.")
        input_names = []

    print(f"Detected expected inputs: {input_names}")

    num_inputs = len(input_names)
    if num_inputs > 0:
        InputArrayType = ctypes.POINTER(ctypes.c_double) * num_inputs
        inputs_list = []
        for name in input_names:
            if name in data_dict:
                inputs_list.append(data_dict[name].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            else:
                print(f"Warning: expected input '{name}' not found in data_dict, using zeros.")
                zeros = np.zeros(timesteps * feature_len, dtype=np.float64)
                inputs_list.append(zeros.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                
        inputs_arr = InputArrayType(*inputs_list)
    else:
        # Fallback if there are no inputs
        InputArrayType = ctypes.POINTER(ctypes.c_double) * 0
        inputs_arr = InputArrayType()

    num_outputs = 0
    while True:
        try:
            array_type = ctypes.c_char * 1024
            array_type.in_dll(lib, f"comet_ast_0_{num_outputs}")
            num_outputs += 1
        except ValueError:
            break

    print(f"Detected {num_outputs} variants.")
    if num_outputs == 0:
        return

    outputs = [np.zeros((timesteps * feature_len,), dtype=np.float64) for _ in range(num_outputs)]
    
    OutputArrayType = ctypes.POINTER(ctypes.c_double) * num_outputs
    outputs_arr = OutputArrayType(*[
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for out in outputs
    ])

    lib.init_variant_0.argtypes = [ctypes.c_int64]
    lib.init_variant_0.restype = ctypes.c_void_p
    
    print("Calling automated Rust initializer")
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

import glob 

if __name__ == '__main__':
    for filename in sorted(glob.glob("./*.so")):
        print(filename)
        try:
            run_test(filename)
        except Exception as e:
            print(f"Error during execution: {e}")