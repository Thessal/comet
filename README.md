# Comet

Comet is a quant algo fuzzer that synthesizes LLVM IRs.

It is more like a macro rather than a language. Without memory assignment, it is just a pure function.

So you can analyze the "behavior" or "flow" of function directly, with their metadata. And study what causes good alpha.

## Minimal Example

Here is a minimal `.cm` script formulating a ratio between a raw time-series feature and its moving average:

```comet
Fn data(symbol: String) -> DataFrame
Fn divide(signal: DataFrame, reference: DataFrame) -> DataFrame
Fn ts_mean(child: DataFrame, lookback: Integer) -> DataFrame

Flow volume_spike {
    volume = data(symbol="volume")
    mean_vol = ts_mean(child=volume, lookback=10)
    divide(signal=volume, reference=mean_vol)
}
```

## Generated LLVM IR

```llvm
; ModuleID = 'volume_spike'
source_filename = "volume_spike"

@comet_ast_0 = global [98 x i8] c"divide(signal=data(symbol=\22volume\22), reference=ts_mean(child=data(symbol=\22volume\22), lookback=10))\00"

declare ptr @malloc(i64)

declare void @free(ptr)

declare void @comet_ts_mean_step(ptr, { i32, ptr }, ptr, i64)

declare void @comet_data_step(ptr, ptr, i64)

declare void @comet_divide_step(ptr, { i32, ptr }, { i32, ptr }, ptr, i64)

define void @execute_variant_0(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4) {
entry:
  %alloc_size = mul i64 %3, 8
  %malloc_out_1 = call ptr @malloc(i64 %alloc_size)
  %state_offset_1 = getelementptr inbounds nuw { [256 x double], [256 x double], [256 x double], [256 x double] }, ptr %2, i32 0, i32 0
  %malloc_out_3 = call ptr @malloc(i64 %alloc_size)
  %state_offset_3 = getelementptr inbounds nuw { [256 x double], [256 x double], [256 x double], [256 x double] }, ptr %2, i32 0, i32 1
  %const_ptr_4 = alloca double, align 8
  store double 1.000000e+01, ptr %const_ptr_4, align 8
  %malloc_out_5 = call ptr @malloc(i64 %alloc_size)
  %state_offset_5 = getelementptr inbounds nuw { [256 x double], [256 x double], [256 x double], [256 x double] }, ptr %2, i32 0, i32 2
  %malloc_out_6 = call ptr @malloc(i64 %alloc_size)
  %state_offset_6 = getelementptr inbounds nuw { [256 x double], [256 x double], [256 x double], [256 x double] }, ptr %2, i32 0, i32 3
  %t = alloca i64, align 8
  store i64 0, ptr %t, align 4
  br label %event_loop_cond

event_loop:                                       ; preds = %event_loop_cond
  %offset = mul i64 %t_val, %3
  call void @comet_data_step(ptr %state_offset_1, ptr %malloc_out_1, i64 %3)
  call void @comet_data_step(ptr %state_offset_3, ptr %malloc_out_3, i64 %3)
  %insert_ptr = insertvalue { i32, ptr } { i32 2, ptr undef }, ptr %malloc_out_3, 1
  %comet_data_ptr = alloca { i32, ptr }, align 8
  store { i32, ptr } %insert_ptr, ptr %comet_data_ptr, align 8
  call void @comet_ts_mean_step(ptr %state_offset_5, ptr %comet_data_ptr, ptr %malloc_out_5, i64 %3)
  %insert_ptr1 = insertvalue { i32, ptr } { i32 2, ptr undef }, ptr %malloc_out_1, 1
  %comet_data_ptr2 = alloca { i32, ptr }, align 8
  store { i32, ptr } %insert_ptr1, ptr %comet_data_ptr2, align 8
  %insert_ptr3 = insertvalue { i32, ptr } { i32 2, ptr undef }, ptr %malloc_out_5, 1
  %comet_data_ptr4 = alloca { i32, ptr }, align 8
  store { i32, ptr } %insert_ptr3, ptr %comet_data_ptr4, align 8
  %stream_out = getelementptr double, ptr %1, i64 %offset
  call void @comet_divide_step(ptr %state_offset_6, ptr %comet_data_ptr2, ptr %comet_data_ptr4, ptr %stream_out, i64 %3)
  br label %event_loop_inc

event_loop_inc:                                   ; preds = %event_loop
  %t_next = add i64 %t_val, 1
  store i64 %t_next, ptr %t, align 4
  br label %event_loop_cond

event_loop_cond:                                  ; preds = %event_loop_inc, %entry
  %t_val = load i64, ptr %t, align 4
  %loop_cond = icmp ult i64 %t_val, %4
  br i1 %loop_cond, label %event_loop, label %event_loop_end

event_loop_end:                                   ; preds = %event_loop_cond
  call void @free(ptr %malloc_out_1)
  call void @free(ptr %malloc_out_3)
  call void @free(ptr %malloc_out_5)
  ret void
}
```