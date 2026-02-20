; src/stdlib/test_add.ll

; Declare the types and external functions for our stateful binary operator
%AddState = type opaque

declare %AddState* @comet_add_init(i64, i64)
declare void @comet_add_step(%AddState*, double*, double*, double*, i64)
declare void @comet_add_free(%AddState*)
declare i32 @printf(i8*, ...)
declare i8* @malloc(i64)
declare void @free(i8*)

@.str = private unnamed_addr constant [22 x i8] c"Result at idx %d: %f\0A\00", align 1

define i32 @main() {
entry:
    ; Initialize add state (period doesn't matter for add, so pass 0, len=3)
    %state = call %AddState* @comet_add_init(i64 0, i64 3)

    ; Allocate memory for three arrays of 3 doubles (3 * 8 bytes = 24 bytes each)
    ; Array A
    %a_void = call i8* @malloc(i64 24)
    %a_ptr = bitcast i8* %a_void to double*
    
    ; Array B 
    %b_void = call i8* @malloc(i64 24)
    %b_ptr = bitcast i8* %b_void to double*
    
    ; Array Out
    %out_void = call i8* @malloc(i64 24)
    %out_ptr = bitcast i8* %out_void to double*

    ; Initialize Array A: [1.5, 2.5, 3.5]
    %a0 = getelementptr inbounds double, double* %a_ptr, i64 0
    store double 1.5, double* %a0
    %a1 = getelementptr inbounds double, double* %a_ptr, i64 1
    store double 2.5, double* %a1
    %a2 = getelementptr inbounds double, double* %a_ptr, i64 2
    store double 3.5, double* %a2

    ; Initialize Array B: [10.0, 20.0, 30.0]
    %b0 = getelementptr inbounds double, double* %b_ptr, i64 0
    store double 10.0, double* %b0
    %b1 = getelementptr inbounds double, double* %b_ptr, i64 1
    store double 20.0, double* %b1
    %b2 = getelementptr inbounds double, double* %b_ptr, i64 2
    store double 30.0, double* %b2

    ; Execute vectorized step
    call void @comet_add_step(%AddState* %state, double* %a_ptr, double* %b_ptr, double* %out_ptr, i64 3)

    ; Print results
    %fmt_ptr = getelementptr inbounds [22 x i8], [22 x i8]* @.str, i64 0, i64 0

    %out0_ptr = getelementptr inbounds double, double* %out_ptr, i64 0
    %out0 = load double, double* %out0_ptr
    call i32 (i8*, ...) @printf(i8* %fmt_ptr, i32 0, double %out0)

    %out1_ptr = getelementptr inbounds double, double* %out_ptr, i64 1
    %out1 = load double, double* %out1_ptr
    call i32 (i8*, ...) @printf(i8* %fmt_ptr, i32 1, double %out1)

    %out2_ptr = getelementptr inbounds double, double* %out_ptr, i64 2
    %out2 = load double, double* %out2_ptr
    call i32 (i8*, ...) @printf(i8* %fmt_ptr, i32 2, double %out2)

    ; Cleanup
    call void @comet_add_free(%AddState* %state)
    call void @free(i8* %a_void)
    call void @free(i8* %b_void)
    call void @free(i8* %out_void)

    ret i32 0
}
