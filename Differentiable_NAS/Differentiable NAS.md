## Differentiable NAS
1. DARTS paper (Liu et al., 2019) → arxiv.org/abs/1806.09055 
2. DARTS official code (GitHub: quark0/darts) 

> Given a DAG, find the best assignment of (predefined) operations to edges (1 edge -> 1 operation)

FIXED (predefined):                                   LEARNED (by DARTS):
─────────────────────        ───────────────────
- Number of nodes                                 • α values per edge (which op wins)
- Which edges exist                                •  Network weights w (inside each op)
- Candidate op set           
- Number of cells            
- How cells stack

![](images/Pasted%20image%2020260425174658.png)
### Training - how to choose a value over categorical building blocks?
In short - 
1. Randomly initialize weight vector $\alpha_o$ 
2. Apply softmax on $\alpha_o$
3. Scale outputs of each architecture by its associated weight   -> a weighted sum of all outputs
4. Backprop with respect to $\alpha$ and $\omega$. The latter contains the weights of architectures, e.g. conv 3x3.
5. After training - argmax over $\alpha$ at each edge and the architecture is ready.
![](images/Pasted%20image%2020260425180750.png)
## Interesting subject - Zero-Cost NAS
ZERO-COST PROXIES FOR LIGHTWEIGHT NAS https://arxiv.org/pdf/2101.08134

Of course, the aforementioned approach requires a lot of compute.  This work seems to be the remedy for that.

Idea:
>"Look at the network at random initialization,
 measure something cheap,
 and predict which architecture will be best after training"
→ requires one forward/backward pass

But **what do we measure?**
- NASWOT — Activation Diversity: a good network maps different inputs to **different activation patterns**. If two inputs produce identical ReLU firing patterns, the network can't distinguish them — that's bad.
- Synflow — Gradient Flow: measures how well **gradients flow through the network**. Architectures where gradient signal dies out (vanishing gradients at init) will be hard to train.
- GradNorm: stronger gradients at init → easier to train → better final accuracy.