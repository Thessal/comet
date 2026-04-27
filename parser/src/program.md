## Program 

### Operators
Each operators are defined as a function with type signature. 

If we could use Lean language notation, it would be like this :
``` Lean
abbrev DataFrame := List (Array Float)
def ts_mean (lookback : { r : Real // 0 < r }) (x : DataFrame) : DataFrame := ...
-- or 
def ts_mean (lookback : { r : Nat }) (x : DataFrame) : DataFrame := ...
```

As you can see here, the type signature is quite implicit. It is not easy to explicitly write down the constraints of the arguments, because it results too complicated inheritance system.

Therefore, we use minimal type system that does not filter out semantically invalid operators such as "division by zero" or "days < 1", which often drives the symbolic regression into overfit, due to high information loss. (see program::TypeDecl)

For example, we say that "ts_mean is a order 2 operator that takes float and dataframe, and returns dataframe." For simplicity, we assume that all operators return dataframe.

The RL agent we are trying to train needs to semantically understand the data distribution in the stack, and choose only meaningful operators. 
