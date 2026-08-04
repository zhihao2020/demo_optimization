# Safe Exploration in Reinforcement Learning: A Generalized Formulation and Algorithms

Akifumi Wachi LINE Corporation akifumi.wachi@linecorp.com

Xun Shen Osaka University shenxun@eei.eng.osaka-u.ac.jp

Wataru Hashimoto Osaka University hashimoto@is.eei.eng.osaka-u.ac.jp

Kazumune Hashimoto Osaka University hashimoto@eei.eng.osaka-u.ac.jp

## Abstract

Safe exploration is essential for the practical use of reinforcement learning (RL) in many real-world scenarios. In this paper, we present a generalized safe exploration (GSE) problem as a unified formulation of common safe exploration problems. We then propose a solution of the GSE problem in the form of a meta-algorithm for safe exploration, MASE, which combines an unconstrained RL algorithm with an uncertainty quantifier to guarantee safety in the current episode while properly penalizing unsafe explorations before actual safety violation to discourage them in future episodes. The advantage of MASE is that we can optimize a policy while guaranteeing with a high probability that no safety constraint will be violated under proper assumptions. Specifically, we present two variants of MASE with different constructions of the uncertainty quantifier: one based on generalized linear models with theoretical guarantees of safety and near-optimality, and another that combines a Gaussian process to ensure safety with a deep RL algorithm to maximize the reward. Finally, we demonstrate that our proposed algorithm achieves better performance than state-of-the-art algorithms on grid-world and Safety Gym benchmarks without violating any safety constraints, even during training.

## 1 Introduction

Safe reinforcement learning (RL) is a promising paradigm that enables policy optimizations for safety-critical decision-making problems (e.g., autonomous driving, healthcare, and robotics), where it is necessary to incorporate safety requirements to prevent RL policies from posing risks to humans or objects [14]. As a result, safe exploration has received significant attention in recent years as a crucial issue for ensuring the safety of RL during both the learning and execution phases [6].

Safe exploration in RL has typically been addressed by formulating a constrained RL problem in which the policy optimization is subject to safety constraints [9, 18]. While there have been many attempts under different types of constraint representations (e.g., expected cumulative cost [3], CVaR [29]), satisfying constraints almost surely or with high probability received less attention to date. Imagine safety-critical applications such as planetary exploration where even a single constraint violation may result in catastrophic failure. NASA’s engineers hope Mars rovers to ensure safety at least with high probability [8]; thus, constraint satisfaction “on average” does not fit their purpose.

While several algorithms have addressed this problem with this stricter notion of safety, there are several formulations in terms of how the constraints are represented, including cumulative [32], state [39], and instantaneous constraints [41], which respectively correspond to Problems 1, 2, and 3 as we will discuss shortly in Section 2. Unfortunately, there has been limited discussion on the relationships between these approaches, making it challenging for researchers to acquire a systematic understanding of the field as a whole. If a generalized problem were to be formulated, then the research community could pool their efforts to develop suitable algorithms.

A closer examination of existing algorithms that span the entire theory-to-practice spectrum reveals several areas for improvement. Practical algorithms using deep RL $( \mathrm { e } . \mathrm { g } . , [ 3 2 ] , [ 3 9 ] , [ 4 0 ] )$ may provide satisfactory performance after convergence, but do not usually guarantee safety during training. In contrast, theoretical studies (e.g., [4], [41]) that guarantee safety with high probability during training often have limitations, such as relying on strong assumptions (e.g., known state transition) or experiencing decreased performance in complex environments. In summary, many algorithms have been proposed in various safe RL formulations, but the creation of a safe exploration algorithm that is both practically useful and supported by theoretical foundations remains an open problem.

Contributions. We first present a generalized safe exploration (GSE) problem and prove its generality compared with existing safe exploration problems. By taking advantage of the tractable form of the safety constraint in the GSE problem, we establish a meta-algorithm for safe exploration, MASE. This algorithm employs an uncertainty quantifier for a high-probability guarantee that the safety constraints are not violated and penalizes the agent before safety violation, under the assumption that the agent has access to an “emergency stop” authority. Our MASE is both practically useful and theoretically well-founded, which allows us to optimize a policy via an arbitrary RL algorithm under the high-probability safety guarantee, even during training. We then provide two specific variants of MASE with different uncertainty quantifiers. One is based on generalized linear models (GLMs), for which we theoretically provide high-probability guarantees of safety and near-optimality. The other is more practical, combining a Gaussian process (GP, [27]) to ensure safety with a deep RL algorithm to maximize the reward. Finally, we show that MASE performs better than state-of-the-art algorithms on the grid-world and Safety Gym [28] without violating any safety constraints, even during training.

## 2 Preliminaries

Definitions. We consider an episodic safe RL problem in a constrained Markov decision process (CMDP, [3]), $\mathcal { M } = \langle { \mathcal { S } } , { \mathcal { A } } , H , \bar { \mathcal { P } } , r , g , s _ { 1 } \rangle$ , where $s$ is a state space, A is an action space, $H \in \mathbb { Z } _ { > 0 }$ is a (fixed) length of each episode, $\mathcal { P } : \mathcal { S } \times \mathcal { A } \times \mathcal { S } \to [ \dot { 0 } , 1 ]$ is a state transition probability, $r : \mathcal { S \times A } \to [ \bar { 0 , 1 } ]$ is a reward function, $g : { \cal S } \times { \cal A }  [ 0 , 1 ]$ is a safety (cost) function, and $s _ { 1 } \in { \mathcal { S } }$ is an initial state. At each discrete time step, with a given (fully-observable) state $s ,$ the agent selects an action a with respect to its policy $\pi : { \mathcal { S } }  A ,$ , receiving the new state $s ^ { \prime } ,$ reward $r ,$ and safety cost g. Though we assume a deterministic policy, our core ideas can be extended to stochastic policy settings. Given a policy π, the value and action-value functions in a state s at time $h$ are respectively defined as

$$
V _ {r, h} ^ {\pi} (s) := \mathbb {E} _ {\pi} \left[ \sum_ {h ^ {\prime} = h} ^ {H} \gamma_ {r} ^ {h ^ {\prime}} r (s _ {h ^ {\prime}}, a _ {h ^ {\prime}})   \bigg |   s _ {h} = s \right]
$$

and $\begin{array} { r } { Q _ { r , h } ^ { \pi } ( s , a ) : = \mathbb { E } _ { \pi } \Big [ \sum _ { h ^ { \prime } = h } ^ { H } \gamma _ { r } ^ { h ^ { \prime } } r ( s _ { h ^ { \prime } } , a _ { h ^ { \prime } } ) \ | \ s _ { h } = s , a _ { h } = a \Big ] } \end{array}$ , where the expectation $\mathbb { E } _ { \pi }$ is taken over the random state-action sequence $\{ ( s _ { h ^ { \prime } } , a _ { h ^ { \prime } } ) \} _ { h ^ { \prime } = h } ^ { H }$ induced by the policy $\pi .$ . Additionally, $\gamma _ { r } \in ( 0 , 1 ]$ is a discount factor for the reward function. In the remainder of this paper, we define $\begin{array} { r } { V _ { \mathrm { m a x } } : = \frac { \dot { 1 } - \gamma _ { r } ^ { H } } { 1 - \gamma _ { r } } } \end{array}$ and let $\mathcal { T } _ { h } : ( S \times \mathcal { A } \to \mathbb { R } ) \to ( S \times \mathcal { A } \to \mathbb { R } )$ denote the Bellman update operator $\begin{array} { r } { \mathcal { T } _ { h } ( Q ) ( s , a ) : = \mathbb { E } \left[ r ( s _ { h } , a _ { h } ) + \gamma _ { r } V _ { Q } ( s _ { h + 1 } ) \ | \ s _ { h } = s , a _ { h } = a \right] } \end{array}$ , where $V _ { Q } ( s ) : = \operatorname* { m a x } _ { a \in \mathcal { A } } Q ( s , a )$

Three common safe RL problems. We tackle safe RL problems where constraints must be satisfied almost surely, even during training. While such problems have garnered attention in the research community, there are several types of formulations, and their relations are yet to be fully investigated.

One of the most popular formulations for safe RL problems involves maximizing $V _ { r } ^ { \pi } : = V _ { r , 1 } ^ { \pi } ( s _ { 1 } )$ under the constraint that the cumulative cost is less than a threshold, which is described as follows: Problem 1 (Almost surely safe RL with cumulative constraint [32]).

$$
\max _ {\pi} V _ {r} ^ {\pi} \quad \text { subject   to } \quad \operatorname * {P r} \left[ \sum_ {h = 1} ^ {H} \gamma_ {g} ^ {h} g (s _ {h}, a _ {h}) \leq \xi_ {1}   \bigg |   \mathcal {P}, \pi \right] = 1,
$$

where $\xi _ { 1 } \in \mathbb { R } _ { \geq 0 }$ is a constant representing a threshold, and $\gamma _ { g } \in ( 0 , 1 ]$ is a discount factor for $g .$

Observe that, the expectation is not taken regarding the safety constraint in Problem 1. This problem was studied in [32], which is stricter than the conventional one where the expectation is taken with respect to the cumulative safety cost function (i.e., $\begin{array} { r } { \mathbb { E } _ { \pi } [ \sum _ { h = 1 } ^ { H } \gamma _ { g } ^ { h } g ( s _ { h } , a _ { h } ) ] \le \xi _ { 1 } ) } \end{array}$

Another popular formulation involves leveraging the state constraints so that safety corresponds to avoiding visits to a set of unsafe states. This type of formulation has been widely adopted by previous studies on safe-critical robotics tasks [38–40, 45], which is written as follows:

Problem 2 (Safe RL with state constraints).

$$
\max _ {\pi} V _ {r} ^ {\pi} \quad \text { subject   to } \quad \mathbb {E} \left[ \sum_ {h = 1} ^ {H} \gamma_ {g} ^ {h} \mathbb {I} (s _ {h} \in S _ {\text { unsafe }}) \bigg | \mathcal {P}, \pi \right] \leq \xi_ {2},
$$

where $\xi _ { 2 } \in \mathbb { R } _ { \geq 0 }$ is a threshold, I(·) is the indicator function, and $S _ { \mathrm { u n s a f e } } \subset S$ is a set of unsafe states.

Finally, some existing studies formulate safe RL problems via an instantaneous constraint, attempting to ensure safety even during the learning stage while aiming for extremely safety-critical applications such as planetary exploration [42] or healthcare [36]. Such studies typically require the agent to satisfy the following instantaneous safety constraint at every time step.

Problem 3 (Safe RL with instantaneous constraints).

$$
\max _ {\pi} V _ {r} ^ {\pi} \quad \text { subject   to } \quad \operatorname * {P r} \Big [ g (s _ {h}, a _ {h}) \leq \xi_ {3} \mid \mathcal {P}, \pi \Big ] = 1, \quad \forall h \in [ 1, H ],
$$

where $\xi _ { 3 } \in \mathbb { R } _ { \geq 0 }$ is a time-invariant safety threshold.

## 3 Problem Formulation

This paper also requires an agent to optimize a policy under a safety constraint, as in the three common safe $\mathtt { R L }$ problems. We seek to find the optimal policy $\pi ^ { \star } : { \mathcal { S } }  { \mathcal { A } }$ of the following problem, which will hereinafter be referred to as the “generalized” safe exploration (GSE) problem:

Problem 4 (GSE problem). Let $b _ { h } \in \mathbb { R }$ denote a time-varying threshold.

$$
\max _ {\pi} V _ {r} ^ {\pi} \quad \text { subject   to } \quad \operatorname * {P r} \Big [ g (s _ {h}, a _ {h}) \leq b _ {h} \mid \mathcal {P}, \pi \Big ] = 1, \quad \forall h \in [ 1, H ].
$$

This constraint is instantaneous, which requires the agent to learn a policy without a single constraint violation not only after convergence but also during training. We assume that the threshold is myopically known; that is, $b _ { h }$ is known at time h, but unknown before that. Crucially, at every time step $h ,$ since $s _ { h }$ is a fully observable state and the agent’s policy is deterministic, we will use a simplified inequality represented as $g ( s _ { h } , a _ { h } ) \le b _ { h }$ in the rest of this paper. This constraint is akin to that in Problem 3, with the difference that the safety threshold is time-varying.

Importance of the GSE problem. Though our problem may not seem relevant to Problems 1 and 2, we will shortly present and prove a theorem on the relationship between the GSE problem and the three common safe RL problems.

Theorem 3.1. Problems 1, 2, and 3 can be transformed into the GSE problem (i.e., Problem 4).

See Appendix B for the proof. In other words, the feasible policy space in the GSE problem can be identical to those in the other three problems by properly defining the safety cost function g and threshold $b _ { h }$ . Crucially, Problem 1 is a special case of the GSE problem with $b _ { h } = \eta _ { h }$ for all $h ,$ where $\eta _ { h + 1 } = \gamma _ { q } ^ { - 1 } \cdot \left( \eta _ { h } - g ( s _ { h } , a _ { h } ) \right)$ with $\eta _ { 0 } = \xi _ { 1 }$ . It is particularly beneficial to convert Problems 1 and 2, which have additive constraint structures, to the GSE problem, which has an instantaneous constraint. The accurate estimation of the cumulative safety value in Problems 1 and 2 is difficult because they depend on the trajectories induced by a policy. Dealing with the instantaneous constraint in the GSE problem is easier, both theoretically and empirically. Also, especially when the environment is time-varying (e.g., there are moving obstacles), the GSE problem is more useful than Problem 3.

Typical CMDP formulations with expected cumulative (safety) cost are out of the scope of the GSE problem. In such problems, the safety notion is milder; hence, although many advanced deep RL algorithms have been actively proposed that perform well in complex environments after convergence, their performance in terms of safety during learning is usually low, as reported by Stooke et al. [33] or Wachi et al. [43]. Risk-constrained MDPs are also important safe RL problems that are not covered by the GSE problem; they have been widely studied by representing risk as a constraint on some conditional value-at-risk [11] or using chance constraints [24, 26].<sup>1</sup>

Difficulties and Assumptions. Theorem 3.1 insists that the GSE problem covers a wide range of safe RL formulations and is worth solving, but the problem is intractable without assumptions. We now discuss the difficulties in solving the GSE problem, and then list the assumptions in this paper.

The biggest difficulty with the GSE problem lies in the fact that there may be no viable safe action given the current state $s _ { h } ,$ safety cost $^ { g , }$ , and threshold $b _ { h }$ . When $b _ { h } = 0 . 1$ and $g ( s _ { h } , a ) = 0 . 5 , \forall a \in \mathcal { A }$ the agent has no viable action for ensuring safety. The agent needs to guarantee safety, even during training, where little environmental information is available; hence, it is significant for the agent to avoid such situations where there is no action that guarantees safety. Another difficulty is related to the regularity of the safety cost function and the strictness of the safety constraint. In this paper, the safety cost function is unknown a priori.; thus, when the safety cost does not exhibit any regularity, the agent can neither infer the safety of decisions nor guarantee safety almost surely.

To address the first difficulty mentioned above, we use Assumptions 3.2 and 3.3.

Assumption 3.2 (Safety margin). There exists $\zeta \in \mathbb { R } _ { > 0 }$ such that $\begin{array} { r l } { \mathrm { P r } [ g ( s _ { h } , a _ { h } ) \le b _ { h } - \zeta \mid \mathcal { P } , \pi ^ { \star } ] } & { { } = } \end{array}$ $1 , \forall h \in [ 1 , H ]$

Assumption 3.3 (Emergency stop action). Let $\widehat { a }$ be an emergency stop action such that $\mathcal { P } ( s _ { 1 } ~ |$ $s , \widehat { a } ) = 1$ for all $s \in S .$ . The agent is allowed to execute the emergency stop action and reset the environment if and only if the probability of guaranteed safety is not sufficiently high.

Assumption 3.2 is mild; this is similar to the Slater condition, which is widely adopted in the CMDP literature [13, 25]. We consider Assumption 3.3 is also natural for safety-critical applications because it is usually better to guarantee safety, even with human interventions, if the agent requires help in emergency cases. In some applications (e.g., the agent is in a hazardous environment), however, emergency stop actions should often be avoided because of the expensive cost of human intervention. In such cases, the agent needs to learn a reset policy allowing them to return to the initial state as in Eysenbach et al. [15], rather than asking for human help, which we will leave to future work.

As for the second difficulty, we assume that the safety cost function belongs to a class where uncertainty can be estimated and guarantee the satisfaction of the safety constraint with high probability. We present an assumption regarding an important notion called an uncertainty quantifier:

Assumption 3.4 (δ-uncertainty quantifier). Let $\mu : { \mathcal { S } } \times { \mathcal { A } } $ R denote the estimated mean function of safety. There exists a δ-uncertainty quantifier $\Gamma : S \times \mathcal { A } $ R such that $| g ( s , a ) - \mu ( s , a ) | \leq \Gamma ( s , a )$ for all $( s , a ) \in S \times A$ , with a probability of at least $1 - \delta$

## 4 Method

We propose MASE for the GSE problem, which combines an unconstrained RL algorithm with additional mechanisms for addressing the safety constraints. The pseudo-code is provided in Algorithm 1, and a conceptual illustration can be seen in Figure 1.

The most notable feature of MASE is that safety is guaranteed via the δ-uncertainty quantifier and the emergency stop action (lines $3 - 9 )$ . The δ-uncertainty quantifier is particularly useful because we can guarantee that the confidence bound contains the true safety cost function, that is, $g ( s , a ) \in [ \mu ( s , \bar { a ) } \pm \Gamma ( s , a ) ]$ ] for all $s \in S$ and $a \in A .$ . This means that, if the agent chooses actions such that $\mu ( s _ { h } , a _ { h } ) + \Gamma ( s _ { h } , a _ { h } ) \le b _ { h }$ , then $g ( s _ { h } , a _ { h } ) \le b _ { h }$ holds with a probability of at least $1 - \delta$ Regarding the first difficulty mentioned in Section 3, it is crucial that there is at least one safe action. Thus, at every time step $h ,$ the agent computes a set of actions that are considered to satisfy the safety constraints with a probability at least $1 - \delta$ given the state $s _ { h }$ and threshold $b _ { h }$ . This is represented as

$$
\mathcal {A} _ {h} ^ {+} := \left\{a \in \mathcal {A} \mid \min \left\{1, \mu \left(s _ {h}, a\right) + \Gamma \left(s _ {h}, a\right) \right\} \leq b _ {h} \right\}.
$$

Whenever the agent identifies that at least one action will guarantee safety, the agent is required to choose an action within $\mathcal { A } _ { h } ^ { + }$ (line 3). The emergency stop action is executed if and only if there is no viable action satisfying the safety constraint $( \mathrm { i . e . , \mathcal { A } _ { { h + 1 } } ^ { + } \neq \emptyset } )$ ; that is, the agent is allowed to execute ba and start a new episode from an initial safe state (lines 6 – 9). The safety cost is upper-bounded by 1 because $g \in [ 0 , \bar { 1 } ]$ by definition. Note that MASE proactively avoids unsafe actions by selecting the emergency stop action to take beforehand; this is in contrast to Sun et al. [37], whose method terminates the episode immediately after the agent has already violated a safety constraint.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Meta-Algorithm for Safe Exploration (MASE)

1: for episode  $t = 1, 2, \ldots, T$  do
2:    for time  $h = 1, 2, \ldots, H$  do
3:    Take “safe” action  $a_h = \pi(s_h)$  within  $A_h^+$   ▷ Execute only safe actions
4:    Receive reward  $r(s_h, a_h)$ , safety cost  $g(s_h, a_h)$ , and next state  $s_{h+1}$ 
5:    Update safety threshold  $b_{h+1}$ 
6:    if  $A_{h+1}^+ = \emptyset$  then
7:    Compute  $\widehat{r}(s_h, a_h) = -\frac{c}{\min_{a \in \mathcal{A}} \Gamma(s_{h+1}, a)}$   ▷ Penalty for the emergency stop action
8:    Append  $(s_h, a_h, \widehat{r}(s_h, a_h), s_{h+1})$  to D
9:    break (i.e., take action  $\widehat{a}$ )   ▷ Execute the emergency stop action
10:    else
11:    Append  $(s_h, a_h, r(s_h, a_h), s_{h+1})$  to D
12:    Optimize a policy  $\pi$  based on D via an (unconstrained) RL algorithm
13:    Update the uncertainty quantifier  $\Gamma$  and rewrite D
</div>

When safety is guaranteed in the manner described above, the question remains as to how to obtain a policy that maximizes the expected cumulative reward. As such, we first convert the original CMDP M to the following unconstrained MDP

$$
\widehat {\mathcal {M}} := \langle \mathcal {S}, \{\mathcal {A}, \widehat {a} \}, H, \mathcal {P}, \widehat {r}, s _ {1} \rangle .
$$

The changes from M lie in the action space and the reward function, as well as in the absence of the safety cost function. First, the action space is augmented so that the agent can execute the emergency stop action, ba. The second modification concerns the reward function. When executing the emergency stop action ba, the agent is penalized as its sacrifice so that the same situation will not occur in future episodes; hence, we modify the reward function as follows:

$$
\widehat {r} (s _ {h}, a _ {h}) = \left\{ \begin{array}{l l} - c / \min _ {a \in \mathcal {A}} \Gamma (s _ {h + 1}, a) & \text { if } \mathcal {A} _ {h + 1} ^ {+} = \emptyset , \\ r (s _ {h}, a _ {h}) & \text { otherwise }, \end{array} \right.\tag{1}
$$

where $c \in \mathbb { R } _ { > 0 }$ is a positive scalar representing a penalty for performing the emergency stop. This penalty is assigned to the state-action pair $\left( s _ { h } , a _ { h } \right)$ that placed the agent into the undesirable situation at time step $h + 1$ , represented as $\mathcal { A } _ { h + 1 } ^ { + } = \emptyset$ (see Figure 1).

To show that MASE is a reasonable safe RL algorithm, we express the following intuitions. Consider the ideal situation in which the safety cost function is accurately es timated for any state-action pairs; that is, $\Gamma ( s , a ) = 0$ for all (s, a). In this case, all emergency stop actions are properly executed, and the safety constraint will be violated at the next time step if the agent executes other actions. It is reasonable for the agent to receive a penalty of ${ \widehat { r } } ( s , a ) = - \infty$ because this state-action pair surely causes a safety violation without the emergency stop action. Unfortunately, however, the safety

![](35e1e16c0122d853b796cd8d12120eaf35c6a2fa81ec2cf78e29b1c39e5e2498.jpg)  
Figure 1: Conceptual illustration of MASE. At every time step h, the agent chooses action $a _ { h }$ within $\mathcal { A } _ { h } ^ { \dagger }$ If there is no safe action at state $s _ { h + 1 }$ satisfying the constraint, the emergency stop action ba is executed and the agent receives a large penalty for $\left( s _ { h } , a _ { h } \right)$

cost is uncertain and the agent conservatively executes ba although there are still actions satisfying the safety constraint, especially in the early phase of training; hence, we increase or reduce the penalty according to the magnitude of uncertainty in (1) to avoid an excessively large penalty.

We must carefully consider the fact that the quality of information regarding the modified reward function rb is uneven in the replay buffer <sup>D</sup>. Specifically, in the early phase of training, the δ- uncertainty quantifier is loose. Hence, the emergency stop action is likely to be executed even if viable actions remain; that is, the agent will receive unnecessary penalties. In contrast, the emergency stop actions in later phases are executed with confidence, as reflected by the tight δ-uncertainty quantifier. Thus, as in line 15, we rewrite the replay buffer D while updating Γ depending on the model in terms of the safety cost function (as for specific methods to update Γ, see Sections 5 and 6).

Connections to shielding methods. The notion of the emergency stop action is akin to shielding [2, 20] which has been actively studied in various problem settings including partially-observable environments [10] or multi-agent settings [23]. Thus, MASE can be regarded as a variant of shielding methods (especially, preemptive shielding in [2]) that is specialized for the GSE problem. On the other hand, MASE does not only block unsafe actions but also provides proper penalties for executing the emergency stop actions based on the uncertainty quantifier, which leads to rigorous theoretical guarantees presented shortly. Such theoretical advantages can be enjoyed in many safe RL problems because of the wide applicability of the GSE problem backed by Theorem 3.1.

Advantages of MASE. Though certain existing algorithms for Problem 3 (i.e., the closest problem to the GSE problem) theoretically guarantee safety during learning, several strong assumptions are needed, such as a known and deterministic state transition and regular safety function as in [41] and a known feature mapping function that is linear with respect to transition kernels, reward, and safety as in [5]. Such algorithms have little affinity with deep RL; thus, their actual performance in complex environments tends to be poor. In contrast, MASE is compatible with any advanced RL algorithms, which can also handle various constraint formulations while maintaining the safety guarantee.

Validity of MASE. We conclude this section by presenting the following two theorems to show that our MASE produces reasonable operations in solving the GSE problem.

Theorem 4.1. Under Assumption 3.4, MASE guarantees safety with a probability of at least $1 - \delta .$

Theorem 4.2. Assume that the safety costfunction is estimatedfor any state-action pairs with an accuracy of better than ${ \frac { \zeta } { 2 } } ,$ that is, $\Gamma ( s , a ) \leq \frac { \zeta } { 2 }$ for all $( s , a )$ . Set $c \in \mathbb { R } t o$ be a sufficiently large scalar such that $\begin{array} { r } { c > \frac { \zeta V _ { \mathrm { m a x } } } { 2 \gamma _ { r } ^ { H } } } \end{array}$ . Then, the optimal policy in $\widehat { \mathcal { M } }$ is identical to that in $\mathcal { M } .$

See Appendix D for the proofs. Unfortunately, obtaining a δ-uncertainty qualifier that works in general cases is highly challenging. To develop a feasible model for the uncertainty quantification, we assume that the safety cost can be modeled via a GLM in Section 5 and via a GP in Section 6.

## 5 A Provable Algorithm under Generalized Linear CMDP Assumptions

In this section, we focus on CMDPs with generalized linear structures and analyze the theoretical properties of MASE. Specifically, we provide a provable algorithm to use a class of GLMs denoted as $\mathcal { F }$ for modeling $Q _ { r , h } ^ { \star } : = Q _ { r , h } ^ { \pi ^ { \star } }$ and $^ { g , }$ and then provide theoretical results on safety and optimality.

## 5.1 Generalized Linear CMDPs

We extend the assumption in Wang et al. [44] from unconstrained MDPs to CMDPs settings. Our assumption is based on GLMs as with [44] that makes a strictly weaker assumption than their Linear MDP assumption [19, 46]. As preliminaries, we first list the necessary definitions and assumptions.

Definition 5.1 (GLMs). Let d $\in \mathbb { Z } _ { > 0 }$ be a feature dimension and let $\mathbb { B } ^ { d } : = \left\{ \mathbf { x } \in \mathbb { R } ^ { d } : \| \mathbf { x } \| _ { 2 } \leq 1 \right\}$ be the $l _ { 2 }$ ball in $\mathbb { R } ^ { d }$ . For a known feature mapping function $\phi : \mathcal { S } \times \mathcal { A }  \mathbb { B } ^ { d }$ and a known link function $f : [ - 1 , 1 ] \to [ - 1 , 1 ]$ , the class of generalized linear model is denoted as ${ \mathcal { F } } : = \{ ( s , a ) $ $f ( \langle \phi _ { s , a } , \theta \rangle ) : \theta \in \mathbb { B } ^ { d } \}$ where $\phi _ { s , a } : = \phi ( s , a )$

Assumption 5.2 (Regular link function). The link function $f ( \cdot )$ is twice differentiable and is either monotonically increasing or decreasing. Furthermore, there exist absolute constants $0 < \underline { { \kappa } } < \overline { { \kappa } } < \infty$ and $M < \infty$ such that $\underline { { \bar { \kappa } } } < | f ^ { \prime } ( \mathbf { x } ) | < \overline { { \kappa } }$ and $| f ^ { \prime \prime } ( \mathbf { x } ) | < M$ for all $| \mathbf { x } | \leq 1$

This assumption on the regular link function is standard in previous studies (e.g., [16], [21]). Linear and logistic models are the special cases of the GLM where the link functions are defined as $f ( \mathbf { x } ) = \mathbf { x }$ and $f ( { \bf { x } } ) = 1 / ( 1 + e ^ { - { \bf { x } } } )$ . In both cases, the link functions satisfy Assumption 5.2.

We finally make the assumption of generalized linear CMDPs (GL-CMDPs), which extends the notion of the optimistic closure for unconstrained MDP settings in Wang et al. [44].

Assumption 5.3 (GL-CMDP). For any $1 \leq h < H$ and $u \in \mathcal { F } _ { \mathrm { u p } } .$ , we have $\mathcal { T } _ { h } ( u ) \in \mathcal { F }$ and $g \in { \mathcal { F } } .$

Recall that $\mathcal { T } _ { h }$ is the Bellman update operator. In Assumption 5.3, with a positive semi-definite matrix $A \in \mathbb { R } ^ { d \times d } \succ ($ and a fixed positive constant $\alpha _ { \mathrm { m a x } } \in \mathbb { R } _ { > 0 }$ , we define

$$
\mathcal {F} _ {\mathrm{up}} := \{(s, a) \rightarrow \min \left\{V _ {\max}, f \left(\left\langle \phi_ {s, a}, \theta \right\rangle\right) + \alpha \| \phi_ {s, a} \| _ {A} \right\}: \theta \in \mathbb {B} ^ {d}, 0 \leq \alpha \leq \alpha_ {\max}, \| A \| _ {\mathrm{op}} \leq 1 \},
$$

where $\| \mathbf { x } \| _ { A } : = { \sqrt { \mathbf { x } ^ { \top } A \mathbf { x } } }$ is the matrix Mahalanobis seminorm, and $\| A \| _ { \mathrm { o p } }$ is the matrix operator norm. For simplicity, we suppose the same link functions for the Q-function and the safety cost function, but it is acceptable to use different link functions. Note that Assumption 5.3 is a more general assumption than Amani et al. [5] that assumes linear transition kernel, reward, and safety cost functions or Wachi et al. [43] that assumes a known transition and GLMs in terms of reward and safety cost functions.

## 5.2 GLM-MASE Algorithm

We introduce an algorithm GLM-MASE under Assumptions 5.2 and 5.3. Hereinafter, we explicitly denote the episode for each variable. For example, we let $s _ { h } ^ { ( t ) }$ or $a _ { h } ^ { ( t ) }$ denote a state or action at the time step h of episode t. We also let $\widetilde { \phi } _ { h } ^ { ( t ) } : = \phi ( s _ { h } ^ { ( t ) } , a _ { h } ^ { ( t ) } )$ ) for more concise notations.

Uncertainty quantifiers. To actualize MASE in the generalized linear CMDP settings, we first need to consider how to obtain the δ-uncertainty quantifier in terms of the safety cost function. Since we assume $g \in { \mathcal { F } }$ , we can define the δ-uncertainty quantifier based on the existing studies on GLMs, especially in the field of multi-armed bandit [22, 16]. Based on Assumptions 5.2 and 5.3, we now provide a lemma regarding the δ-uncertainty quantifier on safety.

Lemma 5.4. Suppose Assumptions 5.2 and 5.3 hold. Set $\delta \ : = \ : \frac { 1 } { T H }$ . With a universal constant $\begin{array} { r } { C \in \mathbb { R } _ { > 0 } , l e t C _ { g } : = C \overline { { \kappa } } \underline { { \kappa } } ^ { - 1 } \sqrt { 1 + M + \overline { { \kappa } } + d ^ { 2 } \ln \left( \frac { 1 + \overline { { \kappa } } + \alpha _ { \operatorname* { m a x } } } { \delta } \right) } } \end{array}$ . Define

$$
\Gamma (s, a) := C _ {g} \cdot \| \phi_ {s, a} \| _ {\Lambda_ {h, t} ^ {- 1}} \quad w i t h \quad \Lambda_ {h, t} := \sum_ {\tau \leq t} \widetilde {\phi} _ {h} ^ {(\tau)} \widetilde {\phi} _ {h} ^ {(\tau)} + I,
$$

where I is the identity matrix. Let $\widehat { \theta } _ { h , t } ^ { g } \in \mathbb { R } ^ { d }$ be the ridge estimate, which is computed by $\widehat { \theta } _ { h , t } ^ { g } : =$ arg min $\begin{array} { r } { \cdot \| \theta \| _ { 2 } { \le } 1 \sum _ { \tau \le t } \Big ( g \big ( s _ { h } ^ { ( \tau ) } , a _ { h } ^ { ( \tau ) } \big ) - f \big ( \big < \widetilde { \phi } _ { h } ^ { ( \tau ) } , \theta \big > \big ) \Big ) ^ { 2 } } \end{array}$ . Then, thefollowing inequality holds

$$
| g (s _ {h} ^ {(t)}, a _ {h} ^ {(t)}) - f (\langle \widetilde {\phi} _ {h} ^ {(t)}, \widehat {\theta} _ {h, t} ^ {g} \rangle) | \leq \Gamma (s _ {h} ^ {(t)}, a _ {h} ^ {(t)})
$$

for all $( s , a ) \in S \times A ,$ , with a probability at least $1 - \delta .$

For the purpose of qualifying uncertainty in GLMs, the weighted l<sub>2</sub>-norm of $\phi \left( \mathrm { i . e . , } \left\| \phi _ { s , a } \right\| _ { \Lambda _ { h , t } ^ { - 1 } } \right)$ plays an important role. Because we assume that the Q-function and safety cost function share the same feature, we have a similar lemma on the uncertainty quantifier regarding the Q-function as follows:

Lemma 5.5. Suppose Assumptions 5.2 and 5.3 hold. Let $\widehat { \theta } _ { h . t } ^ { Q } ~ \in ~ \mathbb { R } ^ { d }$ denote the ridge estimate; that is, $\begin{array} { r } { \widehat { \theta } _ { h , t } ^ { Q } : = \arg \operatorname* { m i n } _ { \| \theta \| _ { 2 } \leq 1 } \sum _ { \tau \leq t } \left( y _ { h } ^ { ( \tau ) } - f ( \langle \widetilde { \phi } _ { h } ^ { ( \tau ) } , \theta \rangle ) \right) ^ { 2 } } \end{array}$ , where $y _ { h } ^ { ( \tau ) } : = r ( s _ { h } ^ { ( \tau ) } , a _ { h } ^ { ( \tau ) } ) +$ ma $\mathrm { c } _ { a ^ { \prime } \in \mathcal { A } } \widehat { Q } _ { r , h + 1 } ^ { ( \tau ) } ( s _ { h + 1 } ^ { ( \tau ) } , a ^ { \prime } ) f o r { a l l } \tau \leq t$ with

$$
\widehat {Q} _ {r, h} ^ {(t)} (s, a) := \min \left\{V _ {\max}, f (\langle \phi_ {s, a}, \hat {\theta} _ {h, t} ^ {Q} \rangle) + C _ {Q / g} \Gamma (s, a) \right\}
$$

that is initialized with $\widehat { Q } _ { r , h } ^ { ( 0 ) } = 0 f o r$ all $h \leq H$ and $\widehat { Q } _ { r , H + 1 } ^ { ( t ) } = 0 f o r$ all $1 \leq t \leq T$ . Then, with a universal constant $C _ { Q / g } \in \mathbb { R } _ { > 0 }$ , the following inequalities holds

$$
| Q _ {r, h} ^ {\star} (s _ {h} ^ {(t)}, a _ {h} ^ {(t)}) - f (\langle \widetilde {\phi} _ {h} ^ {(t)}, \widehat {\theta} _ {h, t} ^ {Q} \rangle) | \leq C _ {Q / g} \cdot \Gamma (s _ {h} ^ {(t)}, a _ {h} ^ {(t)})
$$

for all $( s , a ) \in S \times A ,$ , with a probability at least $1 - \delta$

Note that $\Gamma ( s _ { h } ^ { ( t ) } , a _ { h } ^ { ( t ) } )$ is the δ-uncertainty quantifier with respect to the safety cost function. One of the biggest advantages of the generalized linear CMDPs is that the magnitude of uncertainty for the Q-function is proportional to that for the safety cost function. Hence, by exploring the Q-function based on the optimism in the face of the uncertainty principle [7, 35], the safety cost function is also explored simultaneously, which contributes to the efficient exploration of state-action spaces.

Integration into MASE. The GLM-MASE is an algorithm to integrate the δ-uncertainty quantifiers inferred by the GLM into the MASE sequence. Detailed pseudo code is presented in Appendix E.

To deal with the safety constraint, GLM-MASE leverages the upper bound inferred by the GLM; that is, for all h and t, the agent takes only actions that satisfy

$$
f \left(\langle \widetilde {\phi} _ {h} ^ {(t)}, \widehat {\theta} _ {h, t} ^ {g} \rangle\right) + \Gamma \left(s _ {h} ^ {(t)}, a _ {h} ^ {(t)}\right) \leq b _ {h}.
$$

By Lemma 5.4, such state-action pairs satisfy the safety constraint, i.e. $g ( s _ { h } ^ { ( t ) } , s _ { h } ^ { ( t ) } ) \leq b _ { h } ,$ , for all h and t, with a probability at least $1 - \delta .$ . If there is no action satisfying the safety constraint $( \mathrm { i . e . }$ $\mathcal { A } _ { h } ^ { + } = \emptyset )$ , the emergency stop action ba is taken, and then the agent receives a penalty defined in (1).

As for policy optimization, we follow the optimism in the face of the uncertainty principle. Specifically, the policy π is optimized so that the upper-confidence bound of the Q-function characterized by rb is maximized; that is, for any state $s \in S$ , the policy is computed as follows:

$$
\pi_ {h} ^ {(t)} (s) = \underset {a \in \mathcal {A}} {\arg \max} \widehat {Q} _ {\widehat {r}, h} ^ {(t)} (s, a).
$$

Intuitively, this equation enables us to 1) solve the exploration and exploitation dilemma by incorporating the optimistic estimates of the Q-function and 2) make the agent avoid generating trajectories to violate the safety constraint via the modified reward function.

Theoretical results. We now provide two theorems regarding safety and near-optimality. For both theorems, see Appendix E for the proofs.

Theorem 5.6. Suppose the assumptions in Lemma 5.4 hold. Then, the GLM-MASE satisfies $g ( s _ { h } ^ { ( t ) } , a _ { h } ^ { ( t ) } ) \leq b _ { h }$ for all $t \in [ 1 , T ]$ and $h \in [ 1 , H ]$ , with a probability at least $1 - \delta$

Theorem 5.7. Suppose the assumptions in Lemmas 5.4 and 5.5 hold. Let $C _ { 1 }$ and $C _ { 2 }$ be positive, universal constants. Also, with a sufficiently large T, let t<sup>⋆</sup> denote the smallest integer satisfying $\lambda _ { \operatorname* { m i n } } ( \Sigma ) t H - C _ { 1 } \sqrt { t H d } - C _ { 2 } \sqrt { t H \ln \delta ^ { - 1 } } \geq 2 C _ { g } \cdot \zeta ^ { - 1 }$ , where $\lambda _ { \operatorname* { m i n } } ( \Sigma )$ is the minimum eigenvalue ofthe second moment matrix Σ. Then, the policy $\pi ^ { ( t ) }$ obtained by GLM-MASE at episode t satisfies

$$
\sum_ {t = t ^ {\star}} ^ {T} \left[ V _ {r} ^ {\pi^ {\star}} - V _ {r} ^ {\pi^ {(t)}} \right] \leq \widetilde {O} \left(H \sqrt {d ^ {3} (T - t ^ {*})}\right)
$$

with probability at least $1 - \delta .$

Theorem 5.6 shows that the GLM-MASE guarantees safety with high probability for every time step and episode, which is a variant of Theorem 4.1 under the generalized linear CMDP assumption and corresponding δ-uncertainty quantifier. Theorem 5.7 demonstrates the agent’s ability to act near-optimally after a sufficiently large number of episodes. The proof is based on the following idea. $\operatorname { A f t e r } t ^ { \star }$ episodes, the safety cost function and the Q-function are estimated with an accuracy better than $\frac { \zeta } { 2 }$ . Then, based on Theorem 4.2, the optimal policy in $\widehat { \mathcal { M } }$ is identical to that in $\mathcal { M } ;$ thus, the agent achieves a near-optimal policy by leveraging the (well-estimated) optimistic Q-function.

## 6 A Practical Algorithm

Though we established an algorithm backed by theory under the generalized linear CMDP assumption in Section 5, it is often challenging to obtain proper feature mapping functions in complicated environments. Thus, in this section, we propose a more practical algorithm combining a GP-based estimator to guarantee safety with unconstrained deep RL algorithms to maximize the reward.

Guaranteeing safety via GPs. As shown in the previous sections, the δ-uncertainty quantifier plays a critical role in MASE. To qualify the uncertainty in terms of the safety cost function $g ,$ we consider modeling it as a $\mathrm { G P : } \ j ( z ) \ \overset { \cdot } { \sim } \ \mathcal { G P } ( \mu ( z ) , k ( \overset { \cdot } { z , } z ^ { \prime } ) )$ ), where $z : = [ s , \dot { a } ] , \mu ( z )$ is a mean function, and $k ( z , z ^ { \prime } )$ is a covariance function. The posterior distribution over $g ( \cdot , \cdot )$ is computed based on n $\in \mathbb { Z } _ { > 0 }$ observations at state-action pairs $( z _ { 1 } , z _ { 2 } , \ldots , z _ { n } )$ with safety measurements $\pmb { y } _ { n } : = \{ y _ { 1 } , y _ { 2 } , \dots , y _ { n } \}$ , where $y _ { n } : = g ( z _ { n } ) + N _ { n }$ and $\dot { N } _ { n } \sim \mathcal { N } ( 0 , \omega ^ { 2 } )$ is zero-mean Gaussian noise with a standard deviation of $\omega \in \mathbb { R } _ { > 0 }$ . We consider episodic RL problems, and so $n \approx t H + h$ for episode t and time step $h ,$ although the equality does not hold because of the episode cutoffs. Using the past measurements, the posterior mean, variance, and covariance are computed analytically as $\mu _ { n } ( z ) = k _ { n } ^ { \top } ( z ) ( K _ { n } + \omega ^ { 2 } \bar { I } ) ^ { - 1 } y _ { n } , \sigma _ { n } ( z ) = k _ { n } ( z , z )$ , and $k _ { n } ( z , z ^ { \prime } ) = k ( \bar { z , } z ^ { \prime } ) - k _ { n } ^ { \top } ( z ) ( \dot { K _ { n } } +$ $\omega ^ { 2 } \pmb { I } ) ^ { - 1 } \pmb { k } _ { n } ( z ^ { \prime } )$ , where $\pmb { k } _ { n } ( z ) = [ k ( z _ { 1 } , z ) , \dots , k ( z _ { n } , z ) ] ^ { \top }$ and $K _ { n }$ is the positive definite kernel matrix. We now present a theorem on the safety guarantee.

Theorem 6.1. Assume $\Vert g \Vert _ { k } ^ { 2 } \leq B a n d N _ { n } \leq \omega f o r a l l n \geq 1 . S e t \beta _ { n } ^ { 1 / 2 } : = B + 4 \omega \sqrt { \nu _ { n } + 1 + \ln ( 1 / \delta ) }$ and construct the δ-uncertainty quantifier by

$$
\Gamma (s, a) := \beta_ {n} \cdot \sigma_ {n} (s, a), \quad \forall (s, a) \in \mathcal {S} \times \mathcal {A},\tag{2}
$$

where $\nu _ { n }$ is the information capacity associated with kernel k. Then, MASE based on (2) satisfies the safety constraint $g ( s _ { h } ^ { ( t ) } , a _ { h } ^ { ( t ) } ) \leq b _ { h }$ for all t and h with a probability ofat least $1 - \delta .$

See Appendix F for the proofs. Theorem 6.1 guarantees that the safety constraint is satisfied by combining the GP-based δ-uncertainty quantifier in (2) and the emergency stop action.

Maximizing reward via deep RL. The remaining task is to optimize the policy via the modified reward function rb in (1), whereby the agent is penalized for emergency stop actions. This problem is decoupled from the safety constraint and can be solved as the following unconstrained RL problem:

$$
\pi := \operatorname * {a r g   m a x} _ {\pi} V _ {\widehat {r}} ^ {\pi}.\tag{3}
$$

There are many excellent algorithms for solving (3) such as trust region policy optimization (TRPO, [31]) and twin delayed deep deterministic policy gradient (TD3, [17]). One of the key benefits of our MASE is such compatibility with a broad range of unconstrained (deep) RL algorithms.

## 7 Experiments

We conduct two experiments. The first is on Safety Gym [28], where an agent must maximize the expected cumulative reward under a safety constraint with additive structures as in Problems 1 and 2. The safety cost function g is binary (i.e., 1 for an unsafe state-action pair and 0 otherwise), and the safety threshold is set to $\xi _ { 1 } = 2 0 .$ . The reason for choosing Safety Gym is that this benchmark is complex and elaborate, and has been used to evaluate a variety of excellent algorithms. The second is a grid world where a safety constraint is instantaneous as in Problem 3. Due to the page limit, we present the settings and results of the grid-world experiment in Appendix H.

To solve the Safety Gym tasks, we implement the practical algorithm presented in Section 6 as follows. First, we convert the problem into a GSE problem by defining $\begin{array} { r } { b _ { h } : = \gamma _ { g } ^ { - 1 } { \cdot } ( \xi _ { 1 } - \sum _ { h ^ { \prime } = 0 } ^ { h - 1 } \gamma _ { g } ^ { h ^ { \prime } } g ( s _ { h ^ { \prime } } , a _ { h ^ { \prime } } ) ) } \end{array}$ and enforcing the safety constraint represented as $g ( s _ { h } , a _ { h } ) \le b _ { h }$ for every time step h in each episode. Second, to infer the safety cost, we use deep GP to conduct training and inference, as in [30] when dealing with high-dimensional input spaces. Finally, as for the policy optimization in ${ \widehat { \mathcal { M } } } .$ , we leverage the TRPO algorithm. We delegate other details to Appendix G.

Baselines and metrics. We use the following four algorithms as baselines. The first is TRPO, which is a safety-agnostic deep RL algorithm that purely optimizes a policy without safety consideration. The second and third are CPO [1] and TRPO-Lagrangian [28], which are well-known algorithms for solving CMDPs. The final algorithm is Sauté RL [32], which is a recent, state-of-the-art algorithm for solving safe RL problems where constraints must be satisfied almost surely. We employ the following three metrics to evaluate our MASE and the aforementioned four baselines: 1) the expected cumulative reward, 2) the expected cumulative safety, and 3) the maximum cumulative safety. We execute each algorithm with five random seeds and compute the means and confidence intervals.

Results. The experimental results are summarized in Figure 2. The figures show that TRPO, TRPO-Lagrangian, and CPO successfully learn the policies, but violate the safety constraints during training and even after convergence. Sauté RL is much safer than those three algorithms, but the safety constraint is not satisfied in some episodes, and the performance of the policy significantly deteriorates in terms of the cumulative reward during training. Our MASE obtains better policies in a smaller number of samples compared with Sauté RL, while also satisfying the safety constraints with respect to both the average and the worst-case. Note that, after convergence, the policy obtained by MASE performs worse than those obtained by the baseline algorithms in terms of reward, as shown in

![](6cf7597002662a5f5e85753f2bc1053e75eb46d1fa13298dd6b30a4d226a06ec.jpg)  
(a) Average episode return.

![](5385129f6ec18ac0e35b9806a8b1444c1e66deed52d556a4ab22eccae209d3fc.jpg)  
(b) Average episode safety.

![](08be14aa9e5df5d184763d3cdc1cbd008dfe0a2f673fbdc1059fe807fa43ac01.jpg)  
(c) Maximum episode safety.

![](359019def98eda63814a27c727ef55577ae5df719d7400b3341d54a2ab3db3f1.jpg)  
(d) Average episode return.

![](bb1a4432e50d78619e19302a066863d4c7db0bc258eb38cb733b253015513fe8.jpg)  
(e) Average episode safety.

![](f1a2178742e502d8df9e39b0f3a263bce09153c00b5bea7ed2d164b8944e189c.jpg)  
(f) Maximum episode safety.

Figure 2: Experimental results on Safety Gym (Top: PointGoal1, Bottom: CarGoal1). The proposed MASE satisfies the safety constraint in every episode and achieves better performance in terms of the reward than the state-of-the-art method called Sauté RL. Conventional methods (i.e., TRPO, TRPO-Lagrangian, and CPO) repeatedly violate the safety constraint, especially in the early phase of training. Shaded areas represent 1σ confidence intervals across five different random seeds.

![](9cae75aab6bbc1a8e42ef11df1e2694a4bc4dd99729908fb1a8375fc44079a5e.jpg)  
(a) Average episode return.

![](228e30d92a87e904059506fe22874a0125f4dc9694367c1d8705390eacdc155c.jpg)  
(b) Average episode safety.

![](627d7918c7b9b976555e6893db2b33cf611e74e07e8e6962d741077ebd039a54.jpg)  
(c) Maximum episode safety.

Figure 3: Experimental results on Safety Gym (CarGoal1) with the epoch of 1000. The box plots show the converged performance. Though MASE performs worse than other baselines in terms of reward, the acquired policy is still near-optimal. As for safety, while baselines violate the safety constraint in most of the episodes, MASE guarantees the satisfaction of the severe safety constraint.

Figure 3. The emergency stop action is a variant of resetting actions that are common in episodic RL settings, which prevent the agent from exploring the state-action spaces since the uncertainty quantifier is sometimes quite conservative. We consider that this is a reason why the converged reward performance of MASE is worse than other methods. However, because we require the agent to solve difficult problems where safety is guaranteed at every time step and episode, we consider that this result is reasonable, and further performance improvements are left to future work.

## 8 Conclusion

In this article, we first introduced the GSE problem and proved that it is more general than three common safe RL problems. We then proposed MASE to optimize a policy under safety constraints that allow the agent to execute an emergency stop action at the sacrifice of a penalty based on the δ-uncertainty qualifier. As a specific instance of MASE, we first presented GLM-MASE to theoretically guarantee the near-optimality and safety of the acquired policy under generalized linear CMDP assumptions. Finally, we provided a practical MASE and empirically evaluated its performance in comparison with several baselines on the Safety Gym and grid-world.

# Acknowledgments and Disclosure of Funding

We would like to thank the anonymous reviewers for their helpful comments. This work is partially supported by JST CREST JPMJCR201 and by JSPS KAKENHI Grant 21K14184.

## References

[1] Achiam, J., Held, D., Tamar, A., and Abbeel, P. (2017). Constrained policy optimization. In International Conference on Machine Learning (ICML).

[2] Alshiekh, M., Bloem, R., Ehlers, R., Könighofer, B., Niekum, S., and Topcu, U. (2018). Safe reinforcement learning via shielding. In AAAI Conference on Artificial Intelligence (AAAI).

[3] Altman, E. (1999). Constrained Markov decision processes, volume 7. CRC Press.

[4] Amani, S., Alizadeh, M., and Thrampoulidis, C. (2019). Linear stochastic bandits under safety constraints. In Neural Information Processing Systems (NeurIPS).

[5] Amani, S., Thrampoulidis, C., and Yang, L. (2021). Safe reinforcement learning with linear function approximation. In International Conference on Machine Learning (ICML).

[6] Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., and Mané, D. (2016). Concrete problems in AI safety. arXiv preprint arXiv:1606.06565.

[7] Auer, P. and Ortner, R. (2007). Logarithmic online regret bounds for undiscounted reinforcement learning. In Neural Information Processing Systems (NeurIPS).

[8] Bajracharya, M., Maimone, M. W., and Helmick, D. (2008). Autonomy for mars rovers: Past, present, and future. Computer, 41(12):44–50.

[9] Brunke, L., Greeff, M., Hall, A. W., Yuan, Z., Zhou, S., Panerati, J., and Schoellig, A. P. (2022). Safe learning in robotics: From learning-based control to safe reinforcement learning. Annual Review ofControl, Robotics, and Autonomous Systems, 5:411–444.

[10] Carr, S., Jansen, N., Junges, S., and Topcu, U. (2023). Safe reinforcement learning via shielding under partial observability. In AAAI Conference on Artificial Intelligence.

[11] Chow, Y., Ghavamzadeh, M., Janson, L., and Pavone, M. (2017). Risk-constrained reinforcement learning with percentile risk criteria. Journal of Machine Learning Research (JMLR), 18(1):6070– 6120.

[12] Chowdhury, S. R. and Gopalan, A. (2017). On kernelized multi-armed bandits. In International Conference on Machine Learning (ICML).

[13] Ding, D., Zhang, K., Basar, T., and Jovanovic, M. (2020). Natural policy gradient primal-dual method for constrained Markov decision processes. In Neural Information Processing Systems (NeurIPS).

[14] Dulac-Arnold, G., Levine, N., Mankowitz, D. J., Li, J., Paduraru, C., Gowal, S., and Hester, T. (2021). Challenges of real-world reinforcement learning: definitions, benchmarks and analysis. Machine Learning.

[15] Eysenbach, B., Gu, S., Ibarz, J., and Levine, S. (2018). Leave no trace: Learning to reset for safe and autonomous reinforcement learning. In International Conference on Learning Representations (ICLR).

[16] Filippi, S., Cappe, O., Garivier, A., and Szepesvári, C. (2010). Parametric bandits: The generalized linear case. In Neural Information Processing Systems (NeurIPS).

[17] Fujimoto, S., van Hoof, H., and Meger, D. (2018). Addressing function approximation error in actor-critic methods. In International Conference on Machine Learning (ICML).

[18] Garcıa, J. and Fernández, F. (2015). A comprehensive survey on safe reinforcement learning. Journal ofMachine Learning Research (JMLR), 16(1):1437–1480.

[19] Jin, C., Yang, Z., Wang, Z., and Jordan, M. I. (2020). Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory (COLT).

[20] Könighofer, B., Bloem, R., Junges, S., Jansen, N., and Serban, A. (2020). Safe reinforcement learning using probabilistic shields. In International Conference on Concurrency Theory: CONCUR.

[21] Li, L., Chu, W., Langford, J., and Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. In International Conference on World Wide Web (WWW).

[22] Li, L., Lu, Y., and Zhou, D. (2017). Provably optimal algorithms for generalized linear contextual bandits. In International Conference on Machine Learning (ICML), pages 2071–2080.

[23] Melcer, D., Amato, C., and Tripakis, S. (2022). Shield decentralization for safe multi-agent reinforcement learning. In Neural Information Processing Systems (NeurIPS).

[24] Ono, M., Pavone, M., Kuwata, Y., and Balaram, J. (2015). Chance-constrained dynamic programming with application to risk-aware robotic space exploration. Autonomous Robots, 39(4):555–571.

[25] Paternain, S., Chamon, L., Calvo-Fullana, M., and Ribeiro, A. (2019). Constrained reinforcement learning has zero duality gap. Neural Information Processing Systems (NeurIPS).

[26] Pfrommer, S., Gautam, T., Zhou, A., and Sojoudi, S. (2022). Safe reinforcement learning with chance-constrained model predictive control. In Learning for Dynamics and Control Conference (L4DC), pages 291–303.

[27] Rasmussen, C. E. (2003). Gaussian processes in machine learning. In Summer school on machine learning, pages 63–71. Springer.

[28] Ray, A., Achiam, J., and Amodei, D. (2019). Benchmarking safe exploration in deep reinforcement learning. OpenAI.

[29] Rockafellar, R. T., Uryasev, S., et al. (2000). Optimization of conditional value-at-risk. Journal ofrisk, 2:21–42.

[30] Salimbeni, H. and Deisenroth, M. (2017). Doubly stochastic variational inference for deep Gaussian processes. In Neural Information Processing Systems (NeurIPS).

[31] Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. (2015). Trust region policy optimization. In International Conference on Machine Learning (ICML).

[32] Sootla, A., Cowen-Rivers, A. I., Jafferjee, T., Wang, Z., Mguni, D. H., Wang, J., and Ammar, H. (2022). Sauté RL: Almost surely safe reinforcement learning using state augmentation. In International Conference on Machine Learning (ICML).

[33] Stooke, A., Achiam, J., and Abbeel, P. (2020). Responsive safety in reinforcement learning by pid Lagrangian methods. In International Conference on Machine Learning (ICML), pages 9133–9143.

[34] Strehl, A. L. and Littman, M. L. (2005). A theoretical analysis of model-based interval estimation. In International Conference on Machine Learning (ICML).

[35] Strehl, A. L. and Littman, M. L. (2008). An analysis of model-based interval estimation for Markov decision processes. Journal ofComputer and System Sciences, 74(8):1309–1331.

[36] Sui, Y., Gotovos, A., Burdick, J. W., and Krause, A. (2015). Safe exploration for optimization with Gaussian processes. In International Conference on Machine Learning (ICML).

[37] Sun, H., Xu, Z., Fang, M., Peng, Z., Guo, J., Dai, B., and Zhou, B. (2021). Safe exploration by solving early terminated MDP. arXiv preprint arXiv:2107.04200.

[38] Thananjeyan, B., Balakrishna, A., Nair, S., Luo, M., Srinivasan, K., Hwang, M., Gonzalez, J. E., Ibarz, J., Finn, C., and Goldberg, K. (2021). Recovery RL: Safe reinforcement learning with learned recovery zones. IEEE Robotics and Automation Letters, 6(3):4915–4922.

[39] Thomas, G., Luo, Y., and Ma, T. (2021). Safe reinforcement learning by imagining the near future. In Neural Information Processing Systems (NeurIPS).

[40] Turchetta, M., Kolobov, A., Shah, S., Krause, A., and Agarwal, A. (2020). Safe reinforcement learning via curriculum induction. In Neural Information Processing Systems (NeurIPS), volume 33, pages 12151–12162.

[41] Wachi, A. and Sui, Y. (2020). Safe reinforcement learning in constrained Markov decision processes. In International Conference on Machine Learning (ICML).

[42] Wachi, A., Sui, Y., Yue, Y., and Ono, M. (2018). Safe exploration and optimization of constrained MDPs using Gaussian processes. In AAAI Conference on Artificial Intelligence (AAAI).

[43] Wachi, A., Wei, Y., and Sui, Y. (2021). Safe policy optimization with local generalized linear function approximations. In Neural Information Processing Systems (NeurIPS).

[44] Wang, Y., Wang, R., Du, S. S., and Krishnamurthy, A. (2020). Optimism in reinforcement learning with generalized linear function approximation. In International Conference on Learning Representations (ICLR).

[45] Wang, Y., Zhan, S. S., Jiao, R., Wang, Z., Jin, W., Yang, Z., Wang, Z., Huang, C., and Zhu, Q. (2023). Enforcing hard constraints with soft barriers: Safe reinforcement learning in unknown stochastic environments. In International Conference on Machine Learning, pages 36593–36604. PMLR.

[46] Yang, L. and Wang, M. (2019). Sample-optimal parametric Q-learning using linearly additive features. In International Conference on Machine Learning (ICML).

## Appendices

## A Limitations and Potential Negative Societal Impacts

We will first discuss limitations and potential negative societal impacts regarding our work.

## A.1 Limitations

One of the major limitations of this study is that emergency stop actions are allowed for agents. Emergency stop actions should often be avoided because of the expensive cost of human intervention in many applications $( \mathrm { e . g . }$ , the agent is in a hazardous or remote environment). In future work, we will investigate an algorithm that requires the agent to learn a reset policy allowing them to return to the initial state as in [15], rather than asking for human intervention via emergency stop actions.

Another limitation is how to construct the uncertainty quantifier. In our experiment, because we used a computationally inexpensive deep GP algorithm [30] and the uncertainty quantifier is updated at the end of the episode (see Line 13 in Algorithm 1), the computational time of the GP part was much smaller than the RL part in our experiment settings. However, since GP is generally a computationally expensive algorithm, GP can be a computational bottleneck in some cases.

## A.2 Potential Negative Societal Impacts

We believe that safety is an essential requirement for applying RL in many real problems. While we have not found any potential negative societal impact of our proposed method, we must remain aware that RL algorithms are vulnerable to misuse and ours is no exception.

## B Proof of Theorem 3.1

We first present lemmas regarding the relationship between the GSE problem and Problems 1, 2, and 3. After that, we present the proof for the Theorem 3.1 in Appendix B.4 by combining them.

## B.1 Relationship between the GSE problem and Problem 1

Lemma B.1. Problem 1 can be transformed into the GSE problem.

Proof. We first utilize a safety state augmentation technique presented in Sootla et al. [32] by defining a new variable $\eta _ { h }$ such that

$$
\eta_ {h} := \gamma_ {g} ^ {- h} \cdot \left(\xi_ {1} - \sum_ {h ^ {\prime} = 1} ^ {h - 1} \gamma_ {g} ^ {h ^ {\prime}} g (s _ {h ^ {\prime}}, a _ {h ^ {\prime}})\right), \quad \forall h \in [ 1, H ].\tag{4}
$$

This new variable $\eta _ { h }$ means the remaining safety budget associated with the discount factor $\gamma _ { g }$ , which is updated as follows:

$$
\eta_ {h + 1} = \gamma_ {g} ^ {- 1} \cdot (\eta_ {h} - g (s _ {h}, a _ {h})) \quad \text { with } \quad \eta_ {0} = \xi_ {1}.\tag{5}
$$

By (4), the necessary and sufficient condition for satisfying the constraint in Problem 1 is

$$
\eta_ {h} \geq 0, \quad \forall h \in [ 1, H ].\tag{6}
$$

By (5), we have

$$
\eta_ {h + 1} \geq 0, \quad \forall h \in [ 1, H ] \iff \eta_ {h} - g (s _ {h}, a _ {h}) \geq 0, \quad \forall h \in [ 1, H ].\tag{7}
$$

In summary, by introducing the new variable $\eta _ { h }$ , Problem 1 is rewritten to the following problem:

$$
\max _ {\pi} V _ {r} ^ {\pi} \quad \text { subject   to } \quad \operatorname * {P r} [ g (s _ {h}, a _ {h}) \leq \eta_ {h} \mid \mathcal {P}, \pi ] = 1, \quad \forall h \in [ 1, H ].\tag{8}
$$

The aforementioned problem (8) is a special case of the GSE problem with $b _ { h } : = \eta _ { h }$ . Therefore, we obtain the desired lemma. □

## B.2 Relationship between the GSE problem and Problem 2

Lemma B.2. Problem 2 can be transformed into the GSE problem.

Proof. The following chain of inequalities holds:

$$
\begin{array}{l} \mathbb {E} \left[ \sum_ {h = 1} ^ {H} \gamma_ {g} ^ {h} \mathbb {I} (s _ {h} \in S _ {\text {unsafe}})   \bigg |   \mathcal {P}, \pi \right] \leq \xi_ {2} \\ \Longleftrightarrow \mathbb {E} \left[ \sum_ {h ^ {\prime} = 1} ^ {h} \gamma_ {g} ^ {h ^ {\prime}} \mathbb {I} (s _ {h ^ {\prime}} \in S _ {\text {unsafe}})   \bigg |   \mathcal {P}, \pi \right] \leq \xi_ {2}, \quad \forall h \in [ 1, H ] \\ \Longleftrightarrow \mathbb {E} \left[ \gamma_ {g} ^ {h} \mathbb {I} (s _ {h} \in S _ {\text {unsafe}})   |   \mathcal {P}, \pi \right] \leq \xi_ {2} - \mathbb {E} \left[ \sum_ {h ^ {\prime} = 1} ^ {h - 1} \gamma_ {g} ^ {h ^ {\prime}} \mathbb {I} (s _ {h ^ {\prime}} \in S _ {\text {unsafe}})   \bigg |   \mathcal {P}, \pi \right], \quad \forall h \in [ 2, H ] \\ \Longleftrightarrow \mathbb {E} \left[ \mathbb {I} (s _ {h} \in S _ {\text {unsafe}})   |   \mathcal {P}, \pi \right] \leq \gamma_ {g} ^ {- h} \left\{\xi_ {2} - \mathbb {E} \left[ \sum_ {h ^ {\prime} = 1} ^ {h - 1} \gamma_ {g} ^ {h ^ {\prime}} \mathbb {I} (s _ {h ^ {\prime}} \in S _ {\text {unsafe}})   \bigg |   \mathcal {P}, \pi \right] \right\}, \quad \forall h \in [ 2, H ] \end{array}
$$

Because we assume Markov property, we simply define the safety cost function $g : { \mathcal { S } } \times { \mathcal { A } } $ R as

$$
g (s _ {h}, a) := \mathbb {E} [ \mathbb {I} (s _ {h} \in S _ {\text { unsafe }}) \mid \mathcal {P}, \pi ], \quad \forall a \in \mathcal {A}.
$$

We now set

$$
b _ {h} := \gamma_ {g} ^ {- h} \left\{\xi_ {2} - \mathbb {E} \left[ \sum_ {h ^ {\prime} = 1} ^ {h - 1} \gamma_ {g} ^ {h ^ {\prime}} \mathbb {I} (s _ {h ^ {\prime}} \in S _ {\text { unsafe }}) \biggm | \mathcal {P}, \pi \right] \right\},
$$

then the Problem 2 can be transformed into

$$
\operatorname * {P r} [ g (s, a) \leq b _ {h} \mid \mathcal {P}, \pi ] = 1.
$$

Finally, we obtained the desired lemma.

## B.3 Relationship between the GSE problem and Problem 3

Lemma B.3. Problem 3 can be transformed into the GSEproblem.

Proof. Set $b _ { h } = \xi _ { 3 }$ for all h. Then, the GSE problem is identical to Problem 3.

## B.4 Summary

Proof. Combining Lemma B.1, B.2, and B.3, we obtain the desired Theorem 3.1.

## C Connections to Safe RL Problems with Chance Constraints

As a strongly related formulation to Problem 2, policy optimization under joint chance-constraints has been studied especially in the field of control theory such as Ono et al. [24] and Pfrommer et al. [26], which is written as

Problem 5 (Safe RL with joint chance constraints). Let $\xi _ { 5 } \in \mathbb { R } _ { \geq 0 }$ be a constant representing a safety threshold. Also, let $S _ { \mathrm { u n s a f e } } \subset S$ denote a set of unsafe states. Find the optimal policy $\pi ^ { \star }$ such that

$$
\max _ {\pi} V _ {r} ^ {\pi} \quad \text { subject   to } \quad \operatorname * {P r} \left[ \bigvee_ {h = 1} ^ {H} s _ {h} \in S _ {\text { unsafe }} \mid \mathcal {P}, \pi \right] \leq \xi_ {5}.
$$

Lemma C.1. Problem 2 is a conservative approximation of Problem 5.

Proof. This lemma mostly follows from Theorem 1 in Ono et al. [24]. Regarding the constraint in Problem 5, we have the following chain of equations:

$$
\begin{array}{l} \operatorname * {P r} \left[ \bigvee_ {h = 1} ^ {H} s _ {h} \in S _ {\text { unsafe }}   \Bigg |   \mathcal {P}, \pi \right] \leq \sum_ {h = 1} ^ {H} \operatorname * {P r} \left[ s _ {h} \in S _ {\text { unsafe }}   \mid   \mathcal {P}, \pi \right] \\ = \sum_ {h = 1} ^ {H} \mathbb {E} \left[ \mathbb {I} (s _ {h} \in S _ {\text { unsafe }})   \mid   \mathcal {P}, \pi \right] \\ = \mathbb {E} \left[ \sum_ {h = 1} ^ {H} \mathbb {I} (s _ {h} \in S _ {\text { unsafe }})   \Bigg |   \mathcal {P}, \pi \right]. \end{array}\tag{9}
$$

(10)

In the first step, we used Boole’s inequality (i.e., $\operatorname* { P r } [ A \cup B ] \leq \operatorname* { P r } [ A ] + \operatorname* { P r } [ B ] )$ . The final term in (10) is the LHS of the constraint in Problem 2, which implies that Problem 2 is a conservative approximation of Problem 5. Therefore, the GSE problem is also a conservative approximation of Problem 5. □

Corollary C.2. Suppose we solve the GSE problem by properly defining the safety function $g ( \cdot , \cdot )$ and the threshold $b _ { h }$ . Then, the obtained policy is a conservative solution ofProblem 5.

Remark C.3. It is extremely challenging to directly solve the Problem 5 characterized by joint chance-constraints without approximation, as discussed in Ono et al. [24]. Practically, it would be a promising and realistic approach to solve Problem 5 by converting it into the GSE problem.

More detailed explanations for the aforementioned remark are as follows. It is extremely challenging to directly solve the Problem 5 characterized by joint chance-constraints. Most of the previous work does not directly deal with this type of constraint and uses some approximations or assumptions. For example, Pfrommer et al. [26] assume a known linear time-invariant dynamics. Also, Ono et al. [24] approximate the joint chance constraint as in the above procedure and obtain

$$
\operatorname * {P r} \left[ \bigvee_ {h = 1} ^ {H} s _ {h} \in S _ {\text { unsafe }} \mid \mathcal {P}, \pi \right] \leq \mathbb {E} \left[ \sum_ {h = 1} ^ {H} \mathbb {I} (s _ {h} \in S _ {\text { unsafe }}) \mid \mathcal {P}, \pi \right].
$$

This is a conservative approximation with an additive structure, which is easier to solve than the original joint chance constraint. Ono et al. [24] deals with the above constraints with additive safety structure. By additionally transforming the conservatively-approximated problem into the GSE problem, the problem would become easier to handle because the safety constraint is instantaneous.

## D Proof of Theorems 4.1 and 4.2

## D.1 Proof of Theorem 4.1

Proof. Recall that, at every time step $h ,$ the MASE chooses actions satisfying

$$
\mu (s _ {h}, a _ {h}) + \Gamma (s _ {h}, a _ {h}) \leq b _ {h}.\tag{11}
$$

By definition of the δ-uncertainty quantifier, the actions that are conservatively chosen based on (11) also satisfy the safety constraint, with a probability of at least $1 - \delta ;$ that is,

$$
g (s _ {h}, a _ {h}) \leq b _ {h}.\tag{12}
$$

In addition, when there is no action satisfying (11), the emergency stop action is executed; that is, safety is guaranteed with a probability of 1. Hence, the desired theorem is now obtained. □

## D.2 Proof of Theorem 4.2

Proof. Assumption 3.2 implies that there exists a policy that satisfies a more conservative safety constraint written as

$$
g (s _ {h}, a _ {h}) \leq b _ {h} - \zeta , \quad \forall h \in [ 1, H ].\tag{13}
$$

By combining (13) and the assumption $\begin{array} { r } { \Gamma ( s , a ) \leq \frac { 1 } { 2 } \zeta , \forall ( s , a ) \in S \times A _ { \sf } } \end{array}$ , we have

$$
\mu (s _ {h}, a _ {h}) + \Gamma (s _ {h}, a _ {h}) \leq g (s _ {h}, a _ {h}) + \zeta \leq b _ {h}, \quad \forall h \in [ 1, H ],\tag{14}
$$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 GLM-MASE
1: for episode $t = 1, 2, \cdots, T$ do
2:    for time $h = 1, \ldots, H$ do
3:    Take action $a_h = \pi(s_h)$ within $A_h^+$
4:    Receive reward $r(s_h, a_h)$ and next state $s_{h+1}$
5:    Receive safety cost $g(s_h, a_h)$ and update safety threshold $b_{h+1}$
6:    if $A_h^+ = \emptyset$ then
7:    Compute $\widehat{r}(s_h, a_h) = -\frac{c}{\min_{a \in A} \Gamma(s_{h+1}, a)}$
8:    Append $(s_h, a_h, \widehat{r}(s_h, a_h), s_{h+1})$ to $D$
9:    break (i.e., emergency stop action $\widehat{a}$)
10:    else
11:    Append $(s_h, a_h, r(s_h, a_h), s_{h+1})$ to $D$
12:    Update $\widehat{\theta}_{h,t}^g$ and $\widehat{\theta}_{h,t}^Q$
13:    Compute the optimistic Q-estimate
$\widehat{Q}_{\widehat{r},h}^{(t)}(s, a) = \min\{V_{\max}, f(\langle\phi_{s,a}, \widehat{\theta}_{h,t}^Q\rangle) + C_{Q/g}\Gamma(s, a)\}$
14:    Optimize the policy by
$\pi_h^{(t)}(s) = \arg\max_{a \in A} \widehat{Q}_{\widehat{r},h}^{(t)}(s, a).$
15:    Update $\Gamma(s, a) := C_g \cdot \| \phi_{s,a} \|_{\Lambda_{h,t}^{-1}}$ and then rewrite $D$
</div>

which guarantees that there exists a policy that conservatively satisfies the safety constraint via the δ-uncertainty quantifier $\Gamma ( \cdot , \cdot )$ at every time step h, with a probability of at least $\overset { \cdot } { 1 } - \delta$

When we set c to be a sufficiently large scalar such that $\begin{array} { r } { c > \frac { \zeta } { 2 \gamma _ { r } ^ { H } } V _ { \mathrm { m a x } } } \end{array}$ , the penalty $\widehat { r }$ satisfies

$$
\begin{array}{r l} \widehat {r} (s _ {h}, a _ {h}) & = \frac {- c}{\min _ {a \in \mathcal {A}} \Gamma (s _ {h + 1} , a)} \\ & <   \frac {- \frac {1}{2} \zeta \cdot \frac {1}{\gamma_ {r} ^ {H}} V _ {\max}}{\frac {1}{2} \zeta} \\ & <   - \gamma_ {r} ^ {- H} V _ {\max}. \end{array}
$$

This means that, when the constraint violation happens even a single time, the value by a policy obtained in $\widehat { \mathcal { M } }$ becomes negative because max $V _ { r } ^ { \pi } ( s ) \leq V _ { \mathrm { m a x } }$

Under Assumption 3.2, after convergence, the optimal policy in $\widehat { M }$ will not violate the safety constraint, and thus the emergency stop action ba will not be executed. In this case, the modified (unconstrained) MDP $\widehat { \mathcal { M } }$ is identical to the original CMDP M. Therefore, we now obtain the desired theorem. □

## E Supplementary materials regarding GLM-MASE

## E.1 Pseudo-code for GLM-MASE

We first present the pseudo-code for GLM-MASE in Algorithm 2.

## E.2 Proofs of Lemmas 5.4 and 5.5

Proof. See Lemma 1 (and Lemma 7) in Wang et al. [44].

## E.3 Preliminary Lemmas

Lemma E.1. Suppose the assumptions in Lemma 5.4 and 5.5 hold. Let $C _ { 1 }$ and $C _ { 2 }$ be positive, universal constants. Also, with a sufficiently large $T ,$ , let $t ^ { \star }$ denote the smallest integer satisfying

$$
\lambda_ {\mathrm{min}} (\Sigma) t H - C _ {1} \sqrt {t H d} - C _ {2} \sqrt {t H \ln \delta^ {- 1}} \geq 2 \cdot C _ {g} \cdot \zeta^ {- 1}\tag{15}
$$

where $\lambda _ { \operatorname* { m i n } } ( \Sigma )$ is the minimum eigenvalue ofthe second moment matrix Σ. Then, we have

$$
\Gamma (s _ {h} ^ {(t)}, a _ {h} ^ {(t)}) \leq \frac {1}{2} \cdot \zeta .\tag{16}
$$

Proof. By Proposition 1 of Li et al. [22],

$$
\lambda_ {\min} (\Lambda_ {h, t}) \geq \lambda_ {\min} (\Sigma) t H - C _ {1} \sqrt {t H d} - C _ {2} \sqrt {t H \ln \delta^ {- 1}}.
$$

By combining the aforementioned inequality with (15), we have

$$
\lambda_ {\mathrm{min}} (\Lambda_ {h, t}) \geq 2 \cdot C _ {g} \cdot \zeta^ {- 1}.
$$

Using the definition of $\begin{array} { r } { \lambda _ { \operatorname* { m a x } } ( \Lambda _ { h , t } ^ { - 1 } ) = \frac { 1 } { \lambda _ { \operatorname* { m i n } } \left( \Lambda _ { h , t } \right) } } \end{array}$ , the following chain of equations hold for all $t \in [ t ^ { \star } , T ]$ and $h \in [ 1 , H ]$ :

$$
\begin{array}{r l} & {\Gamma (s _ {h} ^ {(t)}, a _ {h} ^ {(t)}) = C _ {g} \cdot \| \phi_ {s, a} \| _ {\Lambda_ {h, t} ^ {- 1}}} \\ & {\qquad \leq C _ {g} \cdot \lambda_ {\max} (\Lambda_ {h, t} ^ {- 1})} \\ & {\qquad = C _ {g} \cdot \lambda_ {\min} ^ {- 1} (\Lambda_ {h, t})} \\ & {\qquad \leq \frac {1}{2} \zeta .} \end{array}
$$

Therefore, we have the desired lemma.

## E.4 Proof of Theorem 5.6

Proof. By definition, the GLM-MASE chooses actions satisfying

$$
f (\langle \widetilde {\phi} _ {h} ^ {(\tau)}, \widehat {\theta} _ {h, t} ^ {g} \rangle) + \Gamma (s _ {h} ^ {(t)}, a _ {h} ^ {(t)}) \leq b _ {h}.\tag{17}
$$

By Lemma 5.4, the actions that are conservatively chosen based on (17) also satisfy the actual safety constraint, with a probability at least $1 - \delta ;$ that is,

$$
g (s _ {h} ^ {(t)}, a _ {h} ^ {(t)}) \leq b _ {h}.\tag{18}
$$

In addition, when there is no action satisfying (17), the emergency stop action is executed where no unsafe action will be executed. Hence, the desired theorem is now obtained. □

## E.5 Proof of Theorem 5.7

By Assumption 3.2, the optimal policy $\pi ^ { \star }$ satisfies

$$
g \left(s _ {h}, \pi^ {\star} \left(s _ {h}\right)\right) \leq b _ {h} - \zeta , \quad \forall h \in [ 1, H ].\tag{19}
$$

Thus, the set of state-action pairs that are potentially visited by $\pi ^ { \star }$ are written as

$$
\{(s, a) \in \mathcal {S} \times \mathcal {A} \mid g (s, a) \leq b _ {h} - \zeta \},
$$

which satisfies the following chain of inequalities:

$$
\begin{array}{c} \{(s, a) \mid g (s, a) \leq b _ {h} - \zeta \} \subseteq \{(s, a) \mid \mu (s, a) - \Gamma (s, a) \leq b _ {h} - \zeta \} \\ \subseteq \{(s, a) \mid \mu (s, a) + \Gamma (s, a) \leq b _ {h} \}. \end{array}
$$

The state and action spaces in the last line represent the set of state-action pairs that may be visited by the policy obtained by the MASE algorithm. We used Lemma 5.4 in the first line and $\begin{array} { r } { \Gamma ( s , a ) < \frac { \zeta } { 2 } } \end{array}$ in the second line.

By Lemma 8 in Strehl and Littman [34], the total regret can be decomposed into two parts as follows:

$$
\sum_ {t = t ^ {\star}} ^ {T} \left[ V _ {r} ^ {\pi^ {\star}} - V _ {r} ^ {\pi_ {t}} \right] = \mathcal {R} (T) + \sum_ {t = t ^ {\star}} V _ {\mathrm{max}} \delta ,\tag{20}
$$

where the first term is the regret under the condition that the confidence bound based on δ-uncertainty quantifier successfully contains the true safety function. Also, the second term is the regret under the opposite condition, which occurs with a probability δ.

□

The first term in (20) is upper-bounded based on Wang et al. [44] as follows:

$$
\begin{array}{r l} & {\mathcal {R} (T) \leq O \Bigg (H \sqrt {(T - t ^ {\star}) \ln ((T - t ^ {\star}) H)}} \\ & {\qquad + H \overline {{\kappa}} _ {\underline {{\kappa}}} ^ {- 1} \sqrt {M + \overline {{\kappa}} + d ^ {2} \ln \left(\frac {\overline {{\kappa}} + \alpha_ {\max}}{(T - t ^ {\star}) H}\right) \cdot (T - t ^ {\star}) d \ln \left(1 + \frac {(T - t ^ {\star})}{d}\right)} \Bigg)} \\ & {\leq \widetilde {O} (H \sqrt {d ^ {3} (T - t ^ {\star})}).} \end{array}
$$

As for the second term in (20), set $\begin{array} { r } { \delta = \frac { 1 } { T H } } \end{array}$ and then we have

$$
\begin{array}{l} \sum_ {t = t ^ {\star}} ^ {T} V _ {\max} \cdot \delta = \sum_ {t = t ^ {\star}} ^ {T} V _ {\max} \cdot \frac {1}{T H} \\ \qquad = \sum_ {t = t ^ {\star}} ^ {T} \frac {1 - \gamma^ {H}}{1 - \gamma} \cdot \frac {1}{T H} \\ \qquad \leq \sum_ {t = t ^ {\star}} ^ {T} H \cdot \frac {1}{T H} \\ \qquad \leq O (1). \end{array}
$$

In summary, the regret can be upper bounded by $\widetilde { \cal O } ( H \sqrt { d ^ { 3 } ( T - t ^ { \star } ) } )$ . We now obtain the desired theorem.

## F Proof of Theorem 6.1

Lemma F.1 (δ-uncertainty quantifier). Assume $\| g \| _ { k } ^ { 2 } \le B$ and $N _ { n } \leq \omega$ for all $n \geq 1$ . Set

$$
\Gamma (s, a) := \beta_ {n} \cdot \sigma_ {n} (s, a), \quad \forall (s, a) \in \mathcal {S} \times \mathcal {A}\tag{21}
$$

with $\beta _ { n } ^ { 1 / 2 } : = B + 4 \omega \sqrt { \nu _ { n } + 1 + \ln \delta ^ { - 1 } }$ , where $\nu _ { n }$ is the information capacity associated with kernel k. Then, Γ is a δ-uncertainty quantifier.

Proof. Recall the assumption that $\| g \| _ { k } ^ { 2 } \le B$ and $N _ { n } \leq \omega$ , ∀n $\geq 1$ . Also, set

$$
\beta_ {n} ^ {1 / 2} := B + 4 \omega \sqrt {\nu_ {n} + 1 + \ln \delta^ {- 1}}.
$$

By Theorem 2 in Chowdhury and Gopalan [12], we have

$$
\mid g (s, a) - \mu_ {n} (s, a) \mid \leq \beta_ {n} \cdot \sigma_ {n} (s, a), \quad \forall (s, a) \in \mathcal {S} \times \mathcal {A}
$$

for all $n \geq 1$ , with probability at least 1 − δ. Now, define

$$
\Gamma (s, a) := \beta_ {n} \cdot \sigma_ {n} (s, a), \quad \forall (s, a) \in \mathcal {S} \times \mathcal {A}\tag{22}
$$

then we interpret that Γ : $S \times \mathcal { A }  \mathbb { R }$ is a δ-uncertainty quantifier based on GP.

## F.1 Proof of Theorem 6.1

Proof. Based on Lemma F.1, when there is at least one safe action $( \mathrm { i . e . , } A _ { h } ^ { + } \neq \emptyset )$ , the satisfaction of the safety constraint is guaranteed based on the δ-uncertainty quantifier with a probability at least 1 − δ. Also, if there is no safe action $( \mathrm { i . e . , \it A _ { h } ^ { + } } = \emptyset )$ , the emergency stop action is executed. In both cases, MASE guarantees the satisfaction of the safety constraint with probability at least 1 − δ.

Table 1: Hyper-parameters for Safety Gym experiments.

<table><tr><td></td><td>NAME</td><td>VALUE</td></tr><tr><td rowspan="11">COMMON PARAMETERS</td><td>NETWORK ARCHITECTURE</td><td>[64, 64]</td></tr><tr><td>ACTIVATION FUNCTION</td><td>tanh</td></tr><tr><td>LEARNING RATE (CRITIC)</td><td> $5 \times 10^{-3}$ </td></tr><tr><td>LEARNING RATE (POLICY)</td><td> $3 \times 10^{-4}$ </td></tr><tr><td>LEARNING RATE (PENALTY)</td><td> $3 \times 10^{-2}$ </td></tr><tr><td>DISCOUNT FACTOR (REWARD)</td><td>0.99</td></tr><tr><td>DISCOUNT FACTOR (SAFETY)</td><td>0.99</td></tr><tr><td>STEPS PER EPOCH</td><td>10,000</td></tr><tr><td>NUMBER OF GRADIENT STEPS</td><td>80</td></tr><tr><td>NUMBER OF EPOCHS</td><td>500</td></tr><tr><td>TARGET KL</td><td>0.01</td></tr><tr><td rowspan="4">TRPO &amp; CPO</td><td>DAMPING COEFFICIENT</td><td>0.1</td></tr><tr><td>BACKTRACK COEFFICIENT</td><td>0.8</td></tr><tr><td>BACKTRACK ITERATIONS</td><td>10</td></tr><tr><td>LEARNING MARGIN</td><td>FALSE</td></tr><tr><td rowspan="4">MASE</td><td>PENALTY FOR EMERGENCY STOP ACTIONS</td><td>-1</td></tr><tr><td>DEEP GP NETWORK ARCHITECTURE</td><td>[16, 16]</td></tr><tr><td>NUMBER OF INDUCING POINTS</td><td>128</td></tr><tr><td>KERNEL FUNCTION</td><td>RADIAL BASIS FUNCTION</td></tr></table>

## G Details of Safety-Gym Experiment

We present the details regarding our experiments using Safety-Gym. Our experimental setting is based on Sootla et al. [32], which is slightly different from the original Safety Gym in that the obstacles (i.e., unsafe region) are replaced deterministically so that the environment is solvable and there is a viable solution. In this experiment, we used a machine with Intel(R) Xeon(R) Silver 4210 CPU, 128GB RAM, and NVIDIA A100 GPU. For a fair comparison, we basically used the same hyper-parameter as in Sootla et al. [32], which is summarized in Table 1.

In our experiment, when the agent identified that there was no safe action based on the GP-based uncertainty quantifier, we simply terminated the current episode (i.e., resetting) immediately after the emergency stop action and started the new episode. The frequency of the emergency stop actions is shown in Table 2.

Table 2: Frequency of the emergency stop actions.

<table><tr><td>TASK</td><td>TOTAL</td><td>LAST 100 EPISODES</td></tr><tr><td>POINTGOAL1</td><td>154/500</td><td>24/100</td></tr><tr><td>CARGOAL1</td><td>397/500</td><td>46/100</td></tr></table>

The emergency stop action is a variant of so-called resetting actions that are common in episodic RL settings, which prevent the agent from exploring the state-action spaces since the uncertainty quantifier is sometimes quite conservative. We consider that this is the reason why the reward performance of our MASE is worse than other methods (e.g., TRPO-Lagrangian, CPO). However, because we require the agent to solve more difficult problems where safety is guaranteed at every time step and episode, we consider that this result is reasonable to some extent. Though it is better for an algorithm for such a severe safety constraint to have a comparable performance as CPO, we will leave it for future work.

We also conducted an experiment to compare the performance of MASE with the early-terminated MDP (ET-MDP, [37]) algorithm. The ET-MDP is an algorithm to execute emergency stop actions immediately after safety constraints are violated. Figure 4 shows the experimental results. The ET-MDP and MASE exhibit similar learning curves on the average episode reward and average episode safety. However, while ET-MDP violated the safety constraint in most episodes (i.e., almost all episodes are terminated after an unsafe action is executed), MASE did not violate any safety constraint.

![](97e17e2b97e083aa131d3a10c1163c903a2aec1d2ff731fc2b9f3f3ed02e0c4c.jpg)  
(a) Average episode return.

![](e99ad1161c04316f0ceca6dfd3347d140c050a1c2f4bc059a0fdf065afc0afdb.jpg)  
(b) Average episode safety.

![](bd14fb93ad3098b6f2767998a6ad61ba7b57f79b4d6bf9bead7793ada6d0fae7.jpg)  
(c) Maximum episode safety.

![](2846d1d5f49b587475e4c6441ec0e4933866acd06de4c49fb5980d379a8fcf8e.jpg)  
(d) Average episode return.

![](c3ba4bfe829cdb1cdbd4573922b2a36f7a1024b2d5f27774a5888a039b1d7c14.jpg)  
(e) Average episode safety.

![](f75c2ccf5cc6ff27f60884b014a22ccb5c7fdc590a5e07bba3ffbaae00962f9b.jpg)  
(f) Maximum episode safety.  
Figure 4: Experimental results on Safety Gym (Top: PointGoal1, Bottom: CarGoal1) with an additional implementation of the Early-terminated MDP (ET-MDP) algorithm (Sun et al. [37]).

## H Grid-world Experiment

We also conduct an experiment using the grid-world problem as in Wachi and Sui [41]. Experimental settings are based on their original implementation (https://github.com/akifumi-wachi-4/ safe\_near\_optimal\_mdp). We consider a 20×20 square grid in which reward and safety functions are randomly generated. There are two types regarding the safety threshold: one is time-invariant as in [41] and the other is time-variant as in the GSE problem.

We run SNO-MDP [41] and MASE in 100 randomly generated environments, and we compute the reward collected by the algorithms and count the number of episodes in which the safety constraint is violated at least once. The reward is normalized with respect to that by SNO-MDP.

Table 3: Experimental results for grid-world experiments.

<table><tr><td rowspan="2"></td><td colspan="2">TIME-INVARIANT SAFETY THRESHOLD</td><td colspan="2">TIME-VARIANT SAFETY THRESHOLD</td></tr><tr><td>REWARD</td><td>SAFETY VIOLATION</td><td>REWARD</td><td>SAFETY VIOLATION</td></tr><tr><td>SNO-MDP [41]</td><td> $1.0 \pm 0.0$ </td><td>0</td><td> $1.0 \pm 0.0$ </td><td>87</td></tr><tr><td>MASE (OURS)</td><td> $1.0 \pm 0.0$ </td><td>0</td><td> $2.4 \pm 1.0$ </td><td>0</td></tr></table>

The experimental results are shown in Table 3. When the safety threshold is time-invariant, MASE be haves identically with SNO-MDP; thus, the performance of the MASE is comparable with that of SNO-MDP. When the safety threshold is time-variant, SNO-MDP cannot deal with it by nature; hence, the safety constraint is not satisfied in most of the episodes. In contrast, our MASE satisfies the safety constraint in every episode, which also contributes to the larger reward.