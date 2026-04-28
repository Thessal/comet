# RL State and Action definition

## State Representation

```
#[derive(Debug, Clone)]
pub struct SearchState {
    pub params: HashMap<Signal, bool>, // true if used. all of them need to be used.
    pub expression: PolishExpr, // Polish expression
    pub stack: Vec<(Signal, Option<Vec<Vec<f64>>>)>, // (type, data)
}
```
