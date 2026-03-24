# Synthesis 

The Comet synthesizer expands a semantic definition (Flow, Behavior) into a concrete AST block of function and behavior calls. 

Unlike older versions of Comet, **the compiler no longer attempts to exhaustively synthesize implementations or combinatorically search valid trees.** Instead, it constructs a sequence representing the mathematical structure, and the **Transformer model** handles the structural search, variant sampling, and validation.

The synthesizer only evaluates Flow and Literal extensions (such as looping over list choices and ranges) to generate base template variations.

## Iteration over Flow

Flow is a cartesian product space of behaviors and sets of literals.

- Integer literals expansion example : The following code yields three implementations.
```comet
Fn data(id: String) -> DataFrame
Fn ts_mean (a: DataFrame, window: Int) -> DataFrame 
Flow historical_volume {
   variousdays = [5, 21, 252] # choice(5, 21, 252)
   // or variousdays = [10..10..50] # range(10, 50, step=10).
   ts_mean(a=data(id="volume"), window=variousdays)
}
```
- Floating point literal expansion example : The following code yields three implementations.
```comet
Fn data(id: String) -> DataFrame
Fn ts_mean (a: DataFrame, window: Float) -> DataFrame 
Flow historical_volume {
   variousdays = [5.0, 21.0, 252.0] # choice(5.0, 21.0, 252.0)
   // or variousdays = [10.0..10.0..50.0] # range(10.0, 50.0, step=10.0).
   ts_mean(a=data(id="volume"), window=variousdays)
}
```

- String literal expansion example : The following code yields three implementations.
```comet
Fn data(id: String) -> DataFrame
Fn ts_mean (a: DataFrame, window: Int) -> DataFrame 
Flow historical_volume {
   variousdata = ["volume", "adv20"] # choice("volume", "adv20")
   ts_mean(a=data(id=variousdata), window=21)
}
```

