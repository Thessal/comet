# Comet

Comet is a quant algo fuzzer that synthesizes rust code.

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
