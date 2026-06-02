# Comet

Comet is a quant algo fuzzer.


## Minimal Example

Here is a minimal `.cm` script formulating a ratio between a raw time-series feature and its moving average:

```comet
Behavior Comparator(signal: DataFrame, reference: DataFrame) {
    weights="behavior_1_compare.pth", train=true, supervised_epochs=100,
    operators = [add, divide],
    integers = [ ], floats = [ ], strings=[ ]
} -> DataFrame

Flow volume_spike {
    volume = data("volume")
    adv20 = data("adv20")
    Comparator(volume, adv20)
}
```

## Usage 

Random Search
```
cargo run --release -- --file ./examples/behavior_1.cm

Expr: add(data("adv20"), divide(data("adv20"), add(add(data("volume"), add(data("adv20"), add(data("adv20"), data("adv20")))), data("adv20"))))
Expr: data("volume")
Expr: data("adv20")
Expr: add(data("volume"), add(divide(add(add(divide(data("adv20"), data("adv20")), divide(data("volume"), divide(data("adv20"), data("volume")))), divide(divide(data("volume"), data("adv20")), divide(data("volume"), data("adv20")))), divide(data("volume"), data("adv20"))), add(data("adv20"), data("volume"))))
Expr: data("adv20")
Expr: data("volume")
Expr: data("volume")
Expression failed to terminate
Expr: data("volume")
Expr: divide(divide(divide(data("adv20"), data("adv20")), data("volume")), data("adv20"))
```

## For non-nixos
DGX Spark (ARM64)

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