## Modifications :
* category feature will be removed for simplicity.
* integer type will be removed because of optimiztion difficulty
* series type will be removed for simplicity


## Example Code : 

```
Behavior Comparator(a: DataFrame, b: DataFrame, c: Float, d: DataFrame) -> DataFrame 
# DataFrame -> Float -> DataFrame -> DataFrame (output of behavior is always dataframe)

# Comment : currying notation
Fn data(symbol: String) -> DataFrame # DataFrame
Fn consume_float(b: Float) -> Void   # Float -> ()
Fn cs_rank(a: DataFrame) -> DataFrame  # DataFrame -> DataFrame
Fn ts_diff(a: DataFrame, days: Float) -> DataFrame  # DataFrame -> Float -> DataFrame
Fn divide(a: DataFrame, b: DataFrame) -> DataFrame  # DataFrame -> DataFrame -> DataFrame

Flow volume_spike {
    volume = data(symbol="volume")
    adv20 = data(symbol="adv20")
    variousdays = [5, 21, 252]
    Comparator(a=volume, b=adv20, c=0.1, d=adv20)
}
```

## Process logic : 
* For each behavior (in this case, 'Comparator'), new transformer is created.
  * available tokens [ consume_float, cs_rank, ts_diff, divide ] (data operator is special and excluded)
  * given constraint is a:df,b:df,c:f,d:df -> df. let's say [D, D, F, D] for simplicity. 
  * transformer should be able to generate expression like divide(divide(a,ts_rank(b,c)), cs_rank(d))
  * first step
    * transformer input [D, D, F, D] , [ ] ( input parameters and output type of the behavior) 
      * This status represent expression with type "() -> D", and consumed parameters [D, D, F, D] 
    * Available choices (transformer decides) :
      * increase order 
      * '() -> *' typed operator (introducing new internal variable is always possible)
    * Possible next step inputs (environment calculates) : 
      * increase order : [D, D, F] , [ D] 
      * '() -> *' : [D, D, F, D], [ * ]
    * Let's assume that 'increase order' was selected.
  * second step
    * transformer input [D, D, F] , [ D ] 
    * Available choices (transformer decides) :
      * increase order 
      * cs_rank (D -> D)
    * Possible next step inputs (environment calculates) : 
      * increase order : [D, D] , [ F, D ] 
      * cs_rank : [D, D, F] , [ D ] 
    * Let's assume that 'cs_rank' was selected.
  * third step
    * input : [D, D, F] , [ D ] 
    * Available choice, and next step
      * increase order : [D, D], [F, D]
      * cs_rank : [D, D, F] , [ D ] 
    * Let's assume that 'increase order' was selected.
  * fourth step
    * input : [D, D], [F, D]
    * Available choice (()->*, D->*, D->F->*), and next step
      * increase order : [D ], [D, F, D]
      * consume_float : [D, D], [D ] (becuase consume_float is F -> (), leading F was removed)
    * Let's assume that 'increase order' was selected.
  * fifth step
    * input : [D ], [D, F, D]
    * Available choice, and next step
      * increase order : [], [D, F, D, D]
      * ts_diff : [D ], [D, D] (becuase ts_diff is D -> F -> D, leading D, F was replaced to D)
      * cs_rank : [D ], [D, F, D]
    * Let's assume that 'ts_diff' was selected.
  * sixth step
    * input : [D ], [D, D]
    * Available choice, and next step
      * increase order : [], [D, D, D]
      * divide : [D ], [D ] 
      * cs_rank : [D ], [D, D]
    * Let's assume that 'divide' was selected.
  * seventh step
    * input : [D ], [D ] 
    * Available choicd, and next step
      * increase order : [], [D, D]
      * cs_rank : [D ], [D ]
    * Let's assume that 'increase order' was selected.
  * eighth step
    * input : [], [D, D]
    * Available choice, and next step
      * divide : [], [D ]
    * Let's assume that 'divide' was selected.
  * ninth step
    * input : [], [D ] (D->D matches exit condition)
    * Available choice, and next step 
      * done : (exit)
      * '() -> *' typed operator (introducing new internal variable is always possible)
    * there are no '() -> *' typed operator, so only done is available.
    