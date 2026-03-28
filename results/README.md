## Symbolic regession 

### Introduction

Using sequence generation model for polish notation of equation involves two step: 
* Estimation of reward distribution conditioned on prefix of equation previous sequence.
* Selection of next sequence.
    * Expectation maximization : Token that maximizes expected value (Mean field approximation) of reward distribution averaged over future trajectory.
    * Risk-seeking choice (Petersen 2021) : Token that maximizes maximum value of reward distribution.

To calculate expected value of reward distribution averaged over future trajectory, RL algorithms use MDP approximation.
In other word, RL models transformer as matrix multiplication.

Expectation maximization causes solution diversity loss, because repeated application of transition matrix converges to the largest eigencomponent.

So the question is, how to draw all good trajectories:
* just train transformer and hope it to extrapolate?
* generate trajectory with MDP assumption but use risk-seeking choice?

### Purpose of this code

Detection of transformer's reward hacking.

* Observation : entropy of logits are high but sqeuence repeats
* Interpretation : transformer is working like a matrix multiplication and it converges to the largest eigencomponent.


## Engineering detail

### Methods

* Environment : shift-reduce sequence parser (See Astudillo 2020)
* Policy : REINFOCE
* State : shift-reduce parse stack and lookahead tokens (See Petersen 2021, Kamienny 2022)
* Actions : shift [next, 5, 21, 252, 0.1, 0.5, 0.9, "volume", "adv20"], reduce [add, divide, ts_mean, ts_diff, consume_float, cs_rank], done (See Kamienny 2022)
* Reward : Entropy adjusted position correlation
* Architecture : Transformer


### Equation space 
* Task: Finding relation between volume and return.
* Universe : ["aapl", "nvda", "tsla", "msft", "amzn"], 2024, minutely

    * Distribution embedding and bagging was uesd for solution diversity in Kamienny's work (Kamienny 2022)
    * Parent and sibling embedding was used in Deep Symbolic Regression (Petersen 2021)
    * In this code, similar approach with Petersen 2021 was used (stack and lookahead embedding)

```
Behavior Comparator(signal: DataFrame, eps: Float, reference: DataFrame) {
    weights="behavior_1_compare.pth", train=true, supervised_epochs=100,
    operators = [add, divide, ts_mean, ts_diff, consume_float, cs_rank],
    integers = [5, 21, 252], floats = [0.1, 0.5, 0.9], strings=["volume", "adv20"]
} -> DataFrame
   #  integers = [5, 21, 252], floats = [0.1, 0.5, 0.9, 5.0, 21.0, 252.0], strings=["volume", "adv20"]

Flow volume_spike {
    volume = data("volume")
    adv20 = data("adv20")
    Comparator(volume, 0.1, adv20)
}
```