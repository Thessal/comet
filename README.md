# Comet

Comet is a quant algo fuzzer.


## Minimal Example

Here is a minimal `.cm` script formulating a ratio between a raw time-series feature and its moving average:

```comet
Behavior Comparator(signal: DataFrame, eps: Float, reference: DataFrame) {
    weights="behavior_1_compare.pth", train=true, supervised_epochs=100
} -> DataFrame

Flow volume_spike {
    volume = data("volume")
    adv20 = data("adv20")
    Comparator(volume, 0.1, adv20)
}
```

## Usage 

Transformer pretraining
```
cargo run -- examples/behavior_1.cm
```