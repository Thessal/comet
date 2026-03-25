## 1. State Prompting (Input Tokens)

Each token of the input sequence represents the **Current Evaluation State**. 

1. `unprocessed_params` (Variables still required to be shifted/consumed. We only provide the Next Unprocessed Variable Type)
2. `stack` (AST components currently produced and waiting to be reduced/passed as arguments)

### Prompt Layout
Single input token is formatted as:
`[Current State 0] = [PREV_MODEL_OUTPUT] [NEXT_UNPROCESSED] [STACK_TOP] [STACK_TOP-1] ... [STACK_0] [0] ... [0]` ( max size = 8. deeper stack is truncated. maximum supported arity of functions is 6. )

Sequence of current state is fed into the transformer. 
`[Current State 0], [Current State 1], ..., [Current State N]`

## 2. Model Prediction (Output Tokens)

Given the prompt sequence above, the Transformer emits probabilities over the `Action` vocabulary.

### Vocabulary
The vocabulary consists of all valid environment synthesis actions:
- **Actions (example)**:
  - `Shift`
  - `Reduce_cs_rank`
  - `Reduce_add`
  - `Reduce_consume_float`
  - `Done`

**Output Step:**
Given `[Current State 0]` where `[Current State 0] = [T_Float] [SEP] [T_DataFrame]`, the model projects logits where `Shift` is the highest probability, since `Float` cannot be passed into standard binary operators with `DataFrame` without shifting it onto the stack first.

- **Target Token**: `Shift`

`[Current State 1]` can be calculated from `[Current State 0]` and `Shift` action, by the environment.
And then, `[Current State 0] [Current State 1]` can be fed into the transformer to generate next action.

Action masking logic (get_available_actions in search.rs):
* `Done` token is valid only when `unprocessed_params` is empty and `stack` has only one element which is equal to `target_return`.
* Reduce action is valid only when the stack has enough elements to reduce and the types of the elements match the types of the arguments of the function.
* Shift action is valid only when `unprocessed_params` is not empty.

The probaility distribution of invalid actions are masked to zero before sampling.

## 3. Architecture Diagram

The Transformer model (`rl/src/model.rs`) processes the 3D sequence array prompt according to the following mathematical pipeline:

```text
  [batch_size, seq_len=N, state_size=8]  (Input Tokens)
                 │
                 ├── [slice: 0..1] ────► [batch_size, seq_len=N, 1] (PREV_ACTION)
                 │
                 └── [slice: 1..8] ────► [batch_size, seq_len=N, 7] (TYPES)
                                                 │
       ┌──────────────────────┐        ┌─────────┴────────────┐
       │   Action Embedding   │        │    Type Embedding    │
       └─────────┬────────────┘        └─────────┬────────────┘
                 │                               │
 [batch_size*seq_len=N, 1, d_model]   [..., 7, d_model]  (Flattened seq directly to 2D)
                 │                               │
                 └──────────────┬────────────────┘
                                ▼
                       [Concatenate (dim=2)]
                                │
                [batch_size * seq_len=N, 8, d_model]
                                │
                                ▼
                      ┌──────────────────────┐
                      │    State Encoder     │ (Hierarchical Transformer processing the 8 tokens)
                      │   (Self-Attention)   │
                      └─────────┬────────────┘
                                │
                                ▼ [Mean Pooling over dim=1]
                                │
                   [batch_size * seq_len=N, 1, d_model]
                                │
                                ▼ [Reshape back to Sequence]
                                │
                 [batch_size, seq_len=N, d_model]  <─── (Now a single dense vector per State)
                                │
                                ▼
                      ┌──────────────────────┐        ┌──────────────────────┐
                      │       + Add          │ <───── │ Positional Embedding │
                      └─────────┬────────────┘        └──────────────────────┘
                                │
                                ▼
                      ┌──────────────────────┐
                      │  Sequence Encoder    │  (Causal Autoregressive Masking applied over N limit)
                      │  (Self-Attention)    │
                      └─────────┬────────────┘
                                │
                 [batch_size, seq_len=N, d_model]
                                │
                                ▼
                      ┌──────────────────────┐
                      │  Action Projection   │  (Linear layer -> Action Vocabulary)
                      └─────────┬────────────┘
                                │
            [batch_size, seq_len=N, action_vocab_size]  (Output Logits)
                                │
                                ▼
                      ┌──────────────────────┐
                      │   Action Masking &   │  (Invalid Actions set to -INFINITY)
                      │   Softmax            │
                      └─────────┬────────────┘
                                ▼
                          [ACTION PROBS]
```