1. Update behavior syntax 
```
Behavior <behavior name> (<parameter name> : <input type>, ...) { weights = "weights.pth", train = false } -> <output type>
```

---

2. Runtime 
For RL training, it is better to implement an iterative runtime with hash consing / memoization, (with eviction, LRU)
In the previous time, we implemented codegen. Ignore old codegen, and let's think it later.

runtime is a function that receives its state (cache), expression tree or polish notation, and returns the result of the expression. (I guess expression tree is easier for syntax error cache.)

**Common Mistakes:**
* Neglecting syntax verification: Shift-reduce parsers frequently generate invalid expressions (e.g., stack underflow or missing reductions). The runtime must gracefully catch these without crashing.
* Hash-consing Polish notation directly: Identical sub-trees can have different Polish notation sequences depending on ordering. Abstract Syntax Trees (AST) uniquely represent computation boundaries and are structurally better for caching/memoization.
* Unbounded caching: Without eviction policies, LRU cache for millions of generated trees will encounter Out-Of-Memory (OOM) errors quickly.

**Checklists:**
* [ ] Implement an LRU cache or limited-size memoization table for previously evaluated sub-trees. (maybe we can use polish notation as key)
* [ ] Create a translation/parsing step that safely converts Polish notation (shift/reduce sequence) into an AST, rejecting invalid sequences with an immediate "syntax error" flag and penalizing the reward.
* [ ] Wrap the runtime into an Environment Interface (like `Environment` trait in Burn-RL) that strictly defines `step()` and `reset()`.

---

3. Agent Implementation & Training Pipeline

Because PPO inherently struggles from a "cold start" with random weights—especially in discrete token-generation tasks—training is split into two structural phases:

**Phase 1: Supervised Pre-training (Behavioral Cloning)**
1. **Random AST Generation**: Generate thousands of random, structurally correct Abstract Syntax Trees (ASTs) by recursively selecting from `stdlib` functions (`divide`, `ts_diff`, etc.) that map correctly to the required structural return types up to a `max_depth`. Ensure terminals match the required typing constraints.
2. **Translation to Target Labels**: Calculate the corresponding sequence of Polish notation parser tokens (`shift`, `reduce(id)`) for each generated AST.
3. **Dataset Construction**: Serialize the generated sequences and context into a supervised training format.
4. **Behavioral Cloning**: Train the Transformer utilizing standard Cross-Entropy Loss to predict those exact token sequences.
*Reasoning: This guarantees the model natively learns your grammar, state representation, and basic syntactic logic with perfectly stable gradients, bypassing the massive instability of early RL.*

**Phase 2: Reinforcement Learning (PPO)**
For each behavior, load the previously pre-trained weights from Phase 1 to initialize the Transformer. (If `weights.pth` is explicitly not found, initialize randomly, though this is heavily mathematically discouraged). Then, initialize an empty episode buffer and utilize PPO to explore and fine-tune against the actual fitness reward (risk-seeking optimization).

Agent requirements (Phase 2 PPO Rollout)
* For each iteration (episode)
  * the expression tree or polish notation should be synthesized.
    * To synthesize it, the behavior in 'ast' have to be sampled.
    * Behavior is sampled using transformer.
       * unprocessed_params, stack is observed and encoded and iteratively fed into transformer.
       * transformer generates tokens (shift or reduce(id)) iteratively.
       * Use action masking (modify probability distirbution before sampling) to prevent syntax error.
    * Resulting sequence of tokens is polish notation.
  * expression tree or polish notation is fed into the runtime.
  * runtime returns the result of the behavior, from the expression tree or polish notation.
  * score function (fitness) is calculated using the result. let's simply use cross entropy between output and target for this time. (output is normalized to [0,1] using sigmoid)
    * multiobjective optimization is needed later.
    * equation complexity regulariztion will be considered later.
  * if train = true,
    * (States, Actions, Rewards, StateValues, LogProbs) are recorded.
    * PPO updates weights using recorded batch of interactions to compute Advantages.
    * weights file is saved at the end of the run.

**Important Concept: Actor-Critic Architecture**
To apply PPO successfully, your neural network system must operate two distinct components (which can either be two completely separate neural networks or two distinct output heads on the same core Transformer):

* **The Actor Network**: This is your sequence generator. It receives the state (`unprocessed_params` and `stack`) and outputs the probability distribution (logits) across the action space for the next valid parser token (`shift` or `reduce`). Its objective is to learn *how to build the best equation*.
* **The Value Critic Network**: This component calculates a single scalar value corresponding to the *total expected final fitness reward* from the current partial state. Its objective is to evaluate *how good the current partial equation is*. Because actual fitness is calculated only *after* the entire sequence is generated (a "Sparse Reward"), PPO struggles to identify which specific tokens drove the final score. The Critic provides this crucial missing link. By comparing the Critic's baseline prediction against the final achieved fitness (the "Advantage"), PPO learns precisely which specific actions the Actor Network should be encouraged to take again.


**Common Mistakes:**
* Not logging intermediate RL metrics (e.g., episode lengths, syntax error frequency, value loss vs policy loss), which makes debugging an agent that fails to learn practically impossible.
* Ignoring proper positional and structural multi-modal encodings when feeding `unprocessed_params` and the `stack` into the transformer.
* Running autoregressive generation without a maximum token length constraint, trapping the simulator in infinite shift/reduce loops.

**Checklists:**
* [ ] Design state encoder: Ensure that `unprocessed_params` and actual stack tensors are sequence-encoded with positional markers for the transformer.
* [ ] Implement an Action Masking function that evaluates the active parser state and zeros out invalid shift/shift/reduce logits.
* [ ] Add a distinct Value Network or a separate linear head on the Transformer to calculate expected return (State Value) at *every* decoding step.
* [ ] Expand the rollout buffer design to track `LogProbs`, episode markers (`Dones`), and `Values`, matching standard PPO buffer requirements.
* [ ] Define an absolute hard limit on sequence output length (e.g., max 50 generated tokens) allowing the episode to hard-terminate early if exceeded.

---

4. Fitness calculation

Ideally, simple backtester should be developed.

But for this time, let's use cross entropy between output and target.

Let's use data/returns.csv.gz for the target. (output is normalized to [0,1] using sigmoid)