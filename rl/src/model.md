### Deep symbolic regression : 
* Petersen's ICLR 2021

### Assumption : 
* They say that "sparse reward and credit assignment issues typical of reinforcement learning" (Kamienny 2023)
* But, I think Batch Diversity can be measured for intermediate reward, to solve delayed reward problem.
  * Maybe the policy is an representation of diversity, and loss of diversity is real problem.
  * Maybe importance sampling pi_new/pi_old can be replaced into diversity_new/diversity_old.

### Two variants : 
* RNN (original)
* attention mechanism