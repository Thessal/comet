; src/stdlib/test_ts_mean.ll

%TsMeanState = type opaque

declare %TsMeanState* @comet_ts_mean_init(i64, i64)
declare void @comet_ts_mean_step(%TsMeanState*, double*, double*, i64)
declare void @comet_ts_mean_free(%TsMeanState*)
declare i32 @printf(i8*, ...)
declare i8* @malloc(i64)
declare void @free(i8*)

@.str.fmt = private unnamed_addr constant [76 x i8] c"Step %d: Input [%.1f, %.1f, %.1f, %.1f] -> Output [%.1f, %.1f, %.1f, %.1f]\0A\00", align 1

define i32 @main() {
entry:
    ; Initialize ts_mean with period = 3, len = 4
    %state = call %TsMeanState* @comet_ts_mean_init(i64 3, i64 4)

    ; Allocate memory for 4 doubles (4 * 8 bytes = 32 bytes)
    %a_void = call i8* @malloc(i64 32)
    %a_ptr = bitcast i8* %a_void to double*
    
    %out_void = call i8* @malloc(i64 32)
    %out_ptr = bitcast i8* %out_void to double*

    ; Step 1: [10.0, 20.0, 30.0, 40.0]
    %a0 = getelementptr inbounds double, double* %a_ptr, i64 0
    store double 10.0, double* %a0
    %a1 = getelementptr inbounds double, double* %a_ptr, i64 1
    store double 20.0, double* %a1
    %a2 = getelementptr inbounds double, double* %a_ptr, i64 2
    store double 30.0, double* %a2
    %a3 = getelementptr inbounds double, double* %a_ptr, i64 3
    store double 40.0, double* %a3

    call void @comet_ts_mean_step(%TsMeanState* %state, double* %a_ptr, double* %out_ptr, i64 4)
    
    %fmt_ptr = getelementptr inbounds [76 x i8], [76 x i8]* @.str.fmt, i64 0, i64 0
    
    %out0_ptr = getelementptr inbounds double, double* %out_ptr, i64 0
    %o0_1 = load double, double* %out0_ptr
    %out1_ptr = getelementptr inbounds double, double* %out_ptr, i64 1
    %o1_1 = load double, double* %out1_ptr
    %out2_ptr = getelementptr inbounds double, double* %out_ptr, i64 2
    %o2_1 = load double, double* %out2_ptr
    %out3_ptr = getelementptr inbounds double, double* %out_ptr, i64 3
    %o3_1 = load double, double* %out3_ptr
    
    call i32 (i8*, ...) @printf(i8* %fmt_ptr, i32 1, double 10.0, double 20.0, double 30.0, double 40.0, double %o0_1, double %o1_1, double %o2_1, double %o3_1)
    
    ; Step 2: [NaN, 30.0, NaN, 10.0]
    store double 0x7FF8000000000000, double* %a0
    store double 30.0, double* %a1
    store double 0x7FF8000000000000, double* %a2
    store double 10.0, double* %a3
    
    call void @comet_ts_mean_step(%TsMeanState* %state, double* %a_ptr, double* %out_ptr, i64 4)
    
    %o0_2 = load double, double* %out0_ptr
    %o1_2 = load double, double* %out1_ptr
    %o2_2 = load double, double* %out2_ptr
    %o3_2 = load double, double* %out3_ptr
    
    call i32 (i8*, ...) @printf(i8* %fmt_ptr, i32 2, double 0x7FF8000000000000, double 30.0, double 0x7FF8000000000000, double 10.0, double %o0_2, double %o1_2, double %o2_2, double %o3_2)

    ; Step 3: [20.0, 40.0, 60.0, 70.0]
    store double 20.0, double* %a0
    store double 40.0, double* %a1
    store double 60.0, double* %a2
    store double 70.0, double* %a3
    
    call void @comet_ts_mean_step(%TsMeanState* %state, double* %a_ptr, double* %out_ptr, i64 4)
    
    %o0_3 = load double, double* %out0_ptr
    %o1_3 = load double, double* %out1_ptr
    %o2_3 = load double, double* %out2_ptr
    %o3_3 = load double, double* %out3_ptr
    
    call i32 (i8*, ...) @printf(i8* %fmt_ptr, i32 3, double 20.0, double 40.0, double 60.0, double 70.0, double %o0_3, double %o1_3, double %o2_3, double %o3_3)

    ; Step 4: [NaN, NaN, NaN, NaN]
    store double 0x7FF8000000000000, double* %a0
    store double 0x7FF8000000000000, double* %a1
    store double 0x7FF8000000000000, double* %a2
    store double 0x7FF8000000000000, double* %a3
    
    call void @comet_ts_mean_step(%TsMeanState* %state, double* %a_ptr, double* %out_ptr, i64 4)
    
    %o0_4 = load double, double* %out0_ptr
    %o1_4 = load double, double* %out1_ptr
    %o2_4 = load double, double* %out2_ptr
    %o3_4 = load double, double* %out3_ptr
    
    call i32 (i8*, ...) @printf(i8* %fmt_ptr, i32 4, double 0x7FF8000000000000, double 0x7FF8000000000000, double 0x7FF8000000000000, double 0x7FF8000000000000, double %o0_4, double %o1_4, double %o2_4, double %o3_4)

    ; Cleanup
    call void @comet_ts_mean_free(%TsMeanState* %state)
    call void @free(i8* %a_void)
    call void @free(i8* %out_void)

    ret i32 0
}
