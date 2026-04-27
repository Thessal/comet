## Environment for symbolic regression

### Motivation

Purpose of this RL Environment is to train a policy that can choose a) semantically proper operator and b) syntactically correct structure, by observing a) distribution of intermediate data from reverse polish notation and b) current stack status. 

The reason why we are not doing this by direct AST prediction using GCN / Graph transformer is, because transformer based approach [Kamienny 2022] showed the importance of data distribution as context in their ablation study, and the link prediction performace for AST using graph transformer is not clear.

Additionally, consideration on scaling ( I think it can be similar to pop-art approach, but I am not sure ) implies that the interpretation of the latent space, like already stated in that paper.

Here, our point is to descretize the choice of constant, instead of 

### Reset 

* `action_space` : List of avaialble operators Vec<Action>

### State (search::SearchState)

For the parser to check valid action candidates and outputs valid sequence, we need to define three lists of operators(TypeDecls).

See program.md. 

* `stack`: list of parameters that are to be matched by reduce() operation.
* `intermediate_data`: calculated dataframe for last N operators. ## need to be added to SearchState

### Action (search::Action)

The environment uses a shift-reduce approach to build the syntax tree (or reverse polish notation sequence) step-by-step.
* **Shift operations** (`Shift`, `ShiftInteger`, `ShiftFloat`, `ShiftString`): Push a parameter from `unprocessed_params` or a newly generated constant onto the `stack`. This represents adding leaf nodes or variables to the program.
* **Reduce operation** (`Reduce(Ident)`): Apply an operator. This pops the required number of arguments from the `stack`. the result is pushed into the stack.

```rust
pub enum Action {
    // Increase order (move parameter from unprocessed to stack)
    ShiftInteger(i64),
    ShiftFloat(f64),
    ShiftString(String),
    // Apply operator and reduce stack
    Reduce(Ident), 
    // If current full reduction is possible, agent can decide to stop, or continue to generate. 
    Done, 
}
```


## Implementation detail

* CometRlEnv
  * Initialization 
    * Derives and stores action space 
    * Configurable RL model (DQN, PPO, A2C)
  * Reset : they have to be emptied
    * `syntax_tree`: Vec<SyntaxTreeElem>
    * `trajectory`: Vec<(Action, SearchState, f32 (reward), bool (done) )>
  * Step : 
    * List valid actions using the parser
    * Sample action from the model (Use action masking)
    * Update `search_state`
    * When the action is reduce, append the result to the `syntax_tree`

* What the env expects the parser to do 
  * Is Action(Reduce) possible for next choice?
  * Is Action(ShiftInteger) possible for next choice?
  * ... 
  * Is Action(Done) possible for next choice?

