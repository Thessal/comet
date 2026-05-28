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

## For non-nixos
See: 
https://github.com/Thessal/libtorch-releases/releases/tag/v2.9.0

Environment variables:
```
export LIBTORCH=/home/jongkook90/libtorch
export LIBTORCH_INCLUDE=/home/jongkook90/libtorch/
export LIBTORCH_LIB=/home/jongkook90/libtorch/
export LD_LIBRARY_PATH=/home/jongkook90/libtorch/lib
```

### System-wide Libtorch
```
sudo ln -s ~/libtorch/lib/* /usr/lib
sudo ln -s ~/libtorch/include/* /usr/include
sudo ln -s ~/libtorch/share/cmake/* /usr/share/cmake
```
it just works