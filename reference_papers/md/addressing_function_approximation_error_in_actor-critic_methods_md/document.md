# Addressing Function Approximation Error in Actor-Critic Methods

Scott Fujimoto <sup>1</sup> Herke van Hoof<sup>2</sup> David Meger <sup>1</sup>

## Abstract

In value-based reinforcement learning methods such as deep Q-learning, function approximation errors are known to lead to overestimated value estimates and suboptimal policies. We show that this problem persists in an actor-critic setting and propose novel mechanisms to minimize its effects on both the actor and the critic. Our algorithm builds on Double Q-learning, by taking the minimum value between a pair of critics to limit overestimation. We draw the connection between target networks and overestimation bias, and suggest delaying policy updates to reduce per-update error and further improve performance. We evaluate our method on the suite of OpenAI gym tasks, outperforming the state of the art in every environment tested.

## 1. Introduction

In reinforcement learning problems with discrete action spaces, the issue of value overestimation as a result of function approximation errors is well-studied. However, similar issues with actor-critic methods in continuous control domains have been largely left untouched. In this paper, we show overestimation bias and the accumulation of error in temporal difference methods are present in an actor-critic setting. Our proposed method addresses these issues, and greatly outperforms the current state of the art.

Overestimation bias is a property of Q-learning in which the maximization of a noisy value estimate induces a consistent overestimation (Thrun & Schwartz, 1993). In a function approximation setting, this noise is unavoidable given the imprecision of the estimator. This inaccuracy is further exaggerated by the nature of temporal difference learning (Sutton, 1988), in which an estimate of the value function is updated using the estimate of a subsequent state. This means using an imprecise estimate within each update will lead to an accumulation of error. Due to overestimation bias, this accumulated error can cause arbitrarily bad states to be estimated as high value, resulting in suboptimal policy updates and divergent behavior.

This paper begins by establishing this overestimation property is also present for deterministic policy gradients (Silver et al., 2014), in the continuous control setting. Furthermore, we find the ubiquitous solution in the discrete action setting, Double DQN (Van Hasselt et al., 2016), to be ineffective in an actor-critic setting. During training, Double DQN estimates the value of the current policy with a separate target value function, allowing actions to be evaluated without maximization bias. Unfortunately, due to the slow-changing policy in an actor-critic setting, the current and target value estimates remain too similar to avoid maximization bias. This can be dealt with by adapting an older variant, Double Q-learning (Van Hasselt, 2010), to an actor-critic format by using a pair of independently trained critics. While this allows for a less biased value estimation, even an unbiased estimate with high variance can still lead to future overestimations in local regions of state space, which in turn can negatively affect the global policy. To address this concern, we propose a clipped Double Q-learning variant which leverages the notion that a value estimate suffering from overestimation bias can be used as an approximate upper-bound to the true value estimate. This favors underestimations, which do not tend to be propagated during learning, as actions with low value estimates are avoided by the policy.

Given the connection of noise to overestimation bias, this paper contains a number of components that address variance reduction. First, we show that target networks, a common approach in deep Q-learning methods, are critical for variance reduction by reducing the accumulation of errors. Second, to address the coupling of value and policy, we propose delaying policy updates until the value estimate has converged. Finally, we introduce a novel regularization strategy, where a SARSA-style update bootstraps similar action estimates to further reduce variance.

Our modifications are applied to the state of the art actorcritic method for continuous control, Deep Deterministic Policy Gradient algorithm (DDPG) (Lillicrap et al., 2015), to form the Twin Delayed Deep Deterministic policy gradient algorithm (TD3), an actor-critic algorithm which considers the interplay between function approximation error in both policy and value updates. We evaluate our algorithm on seven continuous control domains from OpenAI gym (Brockman et al., 2016), where we outperform the state of the art by a wide margin.

Given the recent concerns in reproducibility (Henderson et al., 2017), we run our experiments across a large number of seeds with fair evaluation metrics, perform ablation studies across each contribution, and open source both our code and learning curves (https://github.com/ sfujim/TD3).

## 2. Related Work

Function approximation error and its effect on bias and variance in reinforcement learning algorithms have been studied in prior works (Pendrith et al., 1997; Mannor et al., 2007). Our work focuses on two outcomes that occur as the result of estimation error, namely overestimation bias and a high variance build-up.

Several approaches exist to reduce the effects of overestimation bias due to function approximation and policy optimization in Q-learning. Double Q-learning uses two independent estimators to make unbiased value estimates (Van Hasselt, 2010; Van Hasselt et al., 2016). Other approaches have focused directly on reducing the variance (Anschel et al., 2017), minimizing over-fitting to early high variance estimates (Fox et al., 2016), or through corrective terms (Lee et al., 2013). Further, the variance of the value estimate has been considered directly for risk-aversion (Mannor & Tsitsiklis, 2011) and exploration (O’Donoghue et al., 2017), but without connection to overestimation bias.

The concern of variance due to the accumulation of error in temporal difference learning has been largely dealt with by either minimizing the size of errors at each time step or mixing off-policy and Monte-Carlo returns. Our work shows the importance of a standard technique, target networks, for the reduction of per-update error, and develops a regularization technique for the variance reduction by averaging over value estimates. Concurrently, Nachum et al. (2018) showed smoothed value functions could be used to train stochastic policies with reduced variance and improved performance. Methods with multi-step returns offer a trade-off between accumulated estimation bias and variance induced by the policy and the environment. These methods have been shown to be an effective approach, through importance sampling (Precup et al., 2001; Munos et al., 2016), distributed methods (Mnih et al., 2016; Espeholt et al., 2018), and approximate bounds (He et al., 2016). However, rather than provide a direct solution to the accumulation of error, these methods circumvent the problem by considering a longer horizon. Another approach is a reduction in the discount factor (Petrik & Scherrer, 2009), reducing the contribution of each error.

Our method builds on the Deterministic Policy Gradient algorithm (DPG) (Silver et al., 2014), an actor-critic method which uses a learned value estimate to train a deterministic policy. An extension of DPG to deep reinforcement learning, DDPG (Lillicrap et al., 2015), has shown to produce state of the art results with an efficient number of iterations. Orthogonal to our approach, recent improvements to DDPG include distributed methods (Popov et al., 2017), along with multi-step returns and prioritized experience replay (Schaul et al., 2016; Horgan et al., 2018), and distributional methods (Bellemare et al., 2017; Barth-Maron et al., 2018).

## 3. Background

Reinforcement learning considers the paradigm of an agent interacting with its environment with the aim of learning reward-maximizing behavior. At each discrete time step t, with a given state $s \in S ,$ the agent selects actions $a \ \in \ A$ with respect to its policy $\pi : { \mathcal { S } }  A ,$ receiving a reward r and the new state of the environment $s ^ { \prime } .$ The return is defined as the discounted sum of rewards $\begin{array} { r } { R _ { t } = \sum _ { i = t } ^ { T } \gamma ^ { i - t } r ( s _ { i } , a _ { i } ) } \end{array}$ , where $\gamma$ is a discount factor determining the priority of short-term rewards.

In reinforcement learning, the objective is to find the optimal policy $\pi _ { \phi } ,$ , with parameters $\phi ,$ which maximizes the expected return $J ( \phi ) = \mathbb { E } _ { s _ { i } \sim p _ { \pi } , a _ { i } \sim \pi } \left[ R _ { 0 } \right]$ . For continuous control, parametrized policies $\pi _ { \phi }$ can be updated by taking the gradient of the expected return $\nabla _ { \phi } J ( \phi )$ . In actor-critic methods, the policy, known as the actor, can be updated through the deterministic policy gradient algorithm (Silver et al., 2014):

$$
\nabla_ {\phi} J (\phi) = \mathbb {E} _ {s \sim p _ {\pi}} \left[ \nabla_ {a} Q ^ {\pi} (s, a) | _ {a = \pi (s)} \nabla_ {\phi} \pi_ {\phi} (s) \right].\tag{1}
$$

$Q ^ { \pi } ( s , a ) \ = \ \mathbb { E } _ { s _ { i } \sim p _ { \pi } , a _ { i } \sim \pi } \left[ R _ { t } | s , a \right]$ , the expected return when performing action a in state s and following π after, is known as the critic or the value function.

In Q-learning, the value function can be learned using temporal difference learning (Sutton, 1988; Watkins, 1989), an update rule based on the Bellman equation (Bellman, 1957). The Bellman equation is a fundamental relationship between the value of a state-action pair $( s , a )$ and the value of the subsequent state-action pair $( s ^ { \prime } , a ^ { \prime } )$

$$
Q ^ {\pi} (s, a) = r + \gamma \mathbb {E} _ {s ^ {\prime}, a ^ {\prime}} \left[ Q ^ {\pi} (s ^ {\prime}, a ^ {\prime}) \right], \quad a ^ {\prime} \sim \pi (s ^ {\prime}).\tag{2}
$$

For a large state space, the value can be estimated with a differentiable function approximator $Q _ { \theta } ( s , a )$ , with parameters θ. In deep Q-learning (Mnih et al., 2015), the network is updated by using temporal difference learning with a secondary frozen target network $Q _ { \theta ^ { \prime } } ( s , a )$ to maintain a fixed objective y over multiple updates:

$$
y = r + \gamma Q _ {\theta^ {\prime}} (s ^ {\prime}, a ^ {\prime}), \quad a ^ {\prime} \sim \pi_ {\phi^ {\prime}} (s ^ {\prime}),\tag{3}
$$

where the actions are selected from a target actor network $\pi _ { \phi ^ { \prime } }$ . The weights of a target network are either updated periodically to exactly match the weights of the current network, or by some proportion τ at each time step $\theta ^ { \prime } \gets$ $\tau \theta + ( 1 - \tau ) \theta ^ { \prime }$ . This update can be applied in an off-policy fashion, sampling random mini-batches of transitions from an experience replay buffer (Lin, 1992).

## 4. Overestimation Bias

In Q-learning with discrete actions, the value estimate is updated with a greedy target $\begin{array} { r } { y = r + \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q ( s ^ { \prime } , a ^ { \prime } ) } \end{array}$ however, if the target is susceptible to error , then the maximum over the value along with its error will generally be greater than the true maximum, E<sub></sub>[max<sub>a</sub>0 $\left( Q ( s ^ { \prime } , a ^ { \prime } ) + \epsilon \right) ] \ge$ ma $\mathsf { \Pi } _ { \mathsf { L } a ^ { \prime } } Q ( s ^ { \prime } , a ^ { \prime } )$ (Thrun & Schwartz, 1993). As a result, even initially zero-mean error can cause value updates to result in a consistent overestimation bias, which is then propagated through the Bellman equation. This is problematic as errors induced by function approximation are unavoidable.

While in the discrete action setting overestimation bias is an obvious artifact from the analytical maximization, the presence and effects of overestimation bias is less clear in an actor-critic setting where the policy is updated via gradient descent. We begin by proving that the value estimate in deterministic policy gradients will be an overestimation under some basic assumptions in Section 4.1 and then propose a clipped variant of Double Q-learning in an actor-critic setting to reduce overestimation bias in Section 4.2.

## 4.1. Overestimation Bias in Actor-Critic

In actor-critic methods the policy is updated with respect to the value estimates of an approximate critic. In this section we assume the policy is updated using the deterministic policy gradient, and show that the update induces overestimation in the value estimate. Given current policy parameters $\phi ,$ let $\phi _ { \mathrm { a p p r o x } }$ define the parameters from the actor update induced by the maximization of the approximate critic $Q _ { \theta } ( s , a )$ and $\phi _ { \mathrm { t r u e } }$ the parameters from the hypothetical actor update with respect to the true underlying value function $Q ^ { \pi } ( s , a )$ (which is not known during learning):

$$
\begin{array}{r} \phi_ {\mathrm{approx}} = \phi + \frac {\alpha}{Z _ {1}} \mathbb {E} _ {s \sim p _ {\pi}} \left[ \nabla_ {\phi} \pi_ {\phi} (s) \nabla_ {a} Q _ {\theta} (s, a) | _ {a = \pi_ {\phi} (s)} \right] \\ \phi_ {\mathrm{true}} = \phi + \frac {\alpha}{Z _ {2}} \mathbb {E} _ {s \sim p _ {\pi}} \left[ \nabla_ {\phi} \pi_ {\phi} (s) \nabla_ {a} Q ^ {\pi} (s, a) | _ {a = \pi_ {\phi} (s)} \right], \end{array}\tag{4}
$$

where we assume $Z _ { 1 }$ and $Z _ { 2 }$ are chosen to normalize the gradient, i.e., such that $Z ^ { - 1 } | | \mathbb { E } [ \cdot ] | | = 1$ . Without normalized gradients, overestimation bias is still guaranteed to occur with slightly stricter conditions. We examine this case further in the supplementary material. We denote π<sub>approx</sub> and $\pi _ { \mathrm { t r u e } }$ as the policy with parameters $\phi _ { \mathrm { a p p r o x } }$ and $\phi _ { \mathrm { t r u e } }$ respectively.

![](a0c758d5349031bd9e708fcba42a107e0d32ea1cd8a5184b216eaa21572614d8.jpg)  
(a) Hopper-v1

![](70538456cdb1fb64057e6f54d0be7e61b3143a0310cb3e4e02ca85848b7df97d.jpg)  
(b) Walker2d-v1  
Figure 1. Measuring overestimation bias in the value estimates of DDPG and our proposed method, Clipped Double Q-learning (CDQ), on MuJoCo environments over 1 million time steps.

As the gradient direction is a local maximizer, there exists $\epsilon _ { 1 }$ sufficiently small such that if $\alpha \leq \epsilon _ { 1 }$ then the approximate value of $\pi _ { \mathrm { a p p r o x } }$ will be bounded below by the approximate value of $\pi _ { \mathrm { t r u e } } \colon$

$$
\mathbb {E} \left[ Q _ {\theta} (s, \pi_ {\text { approx }} (s)) \right] \geq \mathbb {E} \left[ Q _ {\theta} (s, \pi_ {\text { true }} (s)) \right].\tag{5}
$$

Conversely, there exists $\epsilon _ { 2 }$ sufficiently small such that if $\alpha \leq \epsilon _ { 2 }$ then the true value of $\pi _ { \mathrm { a p p r o x } }$ will be bounded above by the true value of $\pi _ { \mathrm { t r u e } }$ :

$$
\mathbb {E} \left[ Q ^ {\pi} (s, \pi_ {\mathrm{true}} (s)) \right] \geq \mathbb {E} \left[ Q ^ {\pi} (s, \pi_ {\mathrm{approx}} (s)) \right].\tag{6}
$$

If in expectation the value estimate is at least as large as the true value with respect to $\phi _ { t r u e } , \mathbb { E } \left[ Q _ { \theta } \left( s , \pi _ { \mathrm { t r u e } } ( s ) \right) \right] \geq$ E $\left[ Q ^ { \pi } \left( s , \pi _ { \mathrm { t r u e } } ( s ) \right) \right]$ ], then Equations (5) and (6) imply that if $\alpha < \operatorname* { m i n } ( \epsilon _ { 1 } , \epsilon _ { 2 } )$ , then the value estimate will be overestimated:

$$
\mathbb {E} \left[ Q _ {\theta} (s, \pi_ {\text { approx }} (s)) \right] \geq \mathbb {E} \left[ Q ^ {\pi} (s, \pi_ {\text { approx }} (s)) \right].\tag{7}
$$

Although this overestimation may be minimal with each update, the presence of error raises two concerns. Firstly, the overestimation may develop into a more significant bias over many updates if left unchecked. Secondly, an inaccurate value estimate may lead to poor policy updates. This is particularly problematic because a feedback loop is created, in which suboptimal actions might be highly rated by the suboptimal critic, reinforcing the suboptimal action in the next policy update.

Does this theoretical overestimation occur in practice for state-of-the-art methods? We answer this question by plotting the value estimate of DDPG (Lillicrap et al., 2015) over time while it learns on the OpenAI gym environments Hopper-v1 and Walker2d-v1 (Brockman et al., 2016). In Figure 1, we graph the average value estimate over 10000 states and compare it to an estimate of the true value. The true value is estimated using the average discounted return over 1000 episodes following the current policy, starting from states sampled from the replay buffer. A very clear overestimation bias occurs from the learning procedure, which contrasts with the novel method that we describe in the following section, Clipped Double Q-learning, which greatly reduces overestimation by the critic.

![](3b31f4732dd74651e48f6ecb1012b78db15ec04f300d69b13fabad1cda5c9981.jpg)  
(a) Hopper-v1

![](7402561a19450fcb26af871a7e4d5d8a509bbb71b80e99466af6e0daeb8d77be.jpg)  
(b) Walker2d-v1  
Figure 2. Measuring overestimation bias in the value estimates of actor critic variants of Double DQN (DDQN-AC) and Double Qlearning (DQ-AC) on MuJoCo environments over 1 million time steps.

## 4.2. Clipped Double Q-Learning for Actor-Critic

While several approaches to reducing overestimation bias have been proposed, we find them ineffective in an actorcritic setting. This section introduces a novel clipped variant of Double Q-learning (Van Hasselt, 2010), which can replace the critic in any actor-critic method.

In Double Q-learning, the greedy update is disentangled from the value function by maintaining two separate value estimates, each of which is used to update the other. If the value estimates are independent, they can be used to make unbiased estimates of the actions selected using the opposite value estimate. In Double DQN (Van Hasselt et al., 2016), the authors propose using the target network as one of the value estimates, and obtain a policy by greedy maximization of the current value network rather than the target network. In an actor-critic setting, an analogous update uses the current policy rather than the target policy in the learning target:

$$
y = r + \gamma Q _ {\theta^ {\prime}} (s ^ {\prime}, \pi_ {\phi} (s ^ {\prime})).\tag{8}
$$

In practice however, we found that with the slow-changing policy in actor-critic, the current and target networks were too similar to make an independent estimation, and offered little improvement. Instead, the original Double Q-learning formulation can be used, with a pair of actors $( \pi _ { \phi _ { 1 } } , \pi _ { \phi _ { 2 } } )$ and critics $( Q _ { \theta _ { 1 } } , Q _ { \theta _ { 2 } } )$ , where $\pi _ { \phi 1 }$ is optimized with respect to $Q _ { \theta _ { 1 } }$ and $\pi _ { \phi _ { 2 } }$ with respect to $Q _ { \theta _ { 2 } }$

$$
\begin{array}{l} y _ {1} = r + \gamma Q _ {\theta_ {2} ^ {\prime}} (s ^ {\prime}, \pi_ {\phi_ {1}} (s ^ {\prime})) \\ y _ {2} = r + \gamma Q _ {\theta_ {1} ^ {\prime}} (s ^ {\prime}, \pi_ {\phi_ {2}} (s ^ {\prime})). \end{array}\tag{9}
$$

We measure the overestimation bias in Figure 2, which demonstrates that the actor-critic Double DQN suffers from a similar overestimation as DDPG (as shown in Figure 1). While Double Q-learning is more effective, it does not entirely eliminate the overestimation. We further show this reduction is not sufficient experimentally in Section 6.1.

As $\pi _ { \phi _ { 1 } }$ optimizes with respect to $Q _ { \theta _ { 1 } }$ , using an independent estimate in the target update of $Q _ { \theta _ { 1 } }$ would avoid the bias introduced by the policy update. However the critics are not entirely independent, due to the use of the opposite critic in the learning targets, as well as the same replay buffer. As a result, for some states s we will have $Q _ { \theta _ { 2 } } ( s , \pi _ { \phi _ { 1 } } ( s ) ) > Q _ { \theta _ { 1 } } ( s , \pi _ { \phi _ { 1 } } ( s ) )$ . This is problematic because $Q _ { \theta _ { 1 } } ( s , \pi _ { \phi _ { 1 } } ( s ) )$ will generally overestimate the true value, and in certain areas of the state space the overestimation will be further exaggerated. To address this problem, we propose to simply upper-bound the less biased value estimate $Q _ { \theta _ { 2 } }$ by the biased estimate $Q _ { \theta _ { 1 } }$ . This results in taking the minimum between the two estimates, to give the target update of our Clipped Double Q-learning algorithm:

$$
y _ {1} = r + \gamma \min _ {i = 1, 2} Q _ {\theta_ {i} ^ {\prime}} (s ^ {\prime}, \pi_ {\phi_ {1}} (s ^ {\prime})).\tag{10}
$$

With Clipped Double Q-learning, the value target cannot introduce any additional overestimation over using the standard Q-learning target. While this update rule may induce an underestimation bias, this is far preferable to overestimation bias, as unlike overestimated actions, the value of underestimated actions will not be explicitly propagated through the policy update.

In implementation, computational costs can be reduced by using a single actor optimized with respect to $Q _ { \theta _ { 1 } }$ . We then use the same target $y _ { 2 } = y _ { 1 }$ for $Q _ { \theta _ { 2 } }$ . If $Q _ { \theta _ { 2 } } > Q _ { \theta _ { 1 } }$ then the update is identical to the standard update and induces no additional bias. If $Q _ { \theta _ { 2 } } < Q _ { \theta _ { 1 } }$ , this suggests overestimation has occurred and the value is reduced similar to Double Qlearning. A proof of convergence in the finite MDP setting follows from this intuition. We provide formal details and justification in the supplementary material.

A secondary benefit is that by treating the function approximation error as a random variable we can see that the minimum operator should provide higher value to states with lower variance estimation error, as the expected minimum of a set of random variables decreases as the variance of the random variables increases. This effect means that the minimization in Equation (10) will lead to a preference for states with low-variance value estimates, leading to safer policy updates with stable learning targets.

## 5. Addressing Variance

While Section 4 deals with the contribution of variance to overestimation bias, we also argue that variance itself should be directly addressed. Besides the impact on overestimation bias, high variance estimates provide a noisy gradient for the policy update. This is known to reduce learning speed (Sutton & Barto, 1998) as well as hurt performance in practice. In this section we emphasize the importance of minimizing error at each update, build the connection between target networks and estimation error and propose modifications to the learning procedure of actor-critic for variance reduction.

## 5.1. Accumulating Error

Due to the temporal difference update, where an estimate of the value function is built from an estimate of a subsequent state, there is a build up of error. While it is reasonable to expect small error for an individual update, these estimation errors can accumulate, resulting in the potential for large overestimation bias and suboptimal policy updates. This is exacerbated in a function approximation setting where the Bellman equation is never exactly satisfied, and each update leaves some amount of residual TD-error $\delta ( s , a )$

$$
Q _ {\theta} (s, a) = r + \gamma \mathbb {E} [ Q _ {\theta} (s ^ {\prime}, a ^ {\prime}) ] - \delta (s, a).\tag{11}
$$

It can then be shown that rather than learning an estimate of the expected return, the value estimate approximates the expected return minus the expected discounted sum of future TD-errors:

$$
\begin{array}{l} Q _ {\theta} (s _ {t}, a _ {t}) = r _ {t} + \gamma \mathbb {E} [ Q _ {\theta} (s _ {t + 1}, a _ {t + 1}) ] - \delta_ {t} \\ = r _ {t} + \gamma \mathbb {E} \left[ r _ {t + 1} + \gamma \mathbb {E} \left[ Q _ {\theta} (s _ {t + 2}, a _ {t + 2}) - \delta_ {t + 1} \right] \right] - \delta_ {t} \\ = \mathbb {E} _ {s _ {i} \sim p _ {\pi}, a _ {i} \sim \pi} \left[ \sum_ {i = t} ^ {T} \gamma^ {i - t} (r _ {i} - \delta_ {i}) \right]. \end{array} \tag {1}\tag{12}
$$

If the value estimate is a function of future reward and estimation error, it follows that the variance of the estimate will be proportional to the variance of future reward and estimation error. Given a large discount factor γ, the variance can grow rapidly with each update if the error from each update is not tamed. Furthermore each gradient update only reduces error with respect to a small mini-batch which gives no guarantees about the size of errors in value estimates outside the mini-batch.

## 5.2. Target Networks and Delayed Policy Updates

In this section we examine the relationship between target networks and function approximation error, and show the use of a stable target reduces the growth of error. This insight allows us to consider the interplay between high variance estimates and policy performance, when designing reinforcement learning algorithms.

Target networks are a well-known tool to achieve stability in deep reinforcement learning. As deep function approximators require multiple gradient updates to converge, target networks provide a stable objective in the learning procedure, and allow a greater coverage of the training data. Without a fixed target, each update may leave residual error which will begin to accumulate. While the accumulation of error can be detrimental in itself, when paired with a policy maximizing over the value estimate, it can result in wildly divergent values.

![](9340af5de70773b6ff40a710c2a10ae9d1cec8e0505eec194f1485404d8b9ca5.jpg)  
(a) Fixed Policy

![](0327f5ec62be50d66d9c84ee6cd218126d861b47baf8fe99c04882810abe613c.jpg)  
(b) Learned Policy  
Figure 3. Average estimated value of a randomly selected state on Hopper-v1 without target networks, (τ = 1), and with slowupdating target networks, $( \tau = 0 . 1 , 0 . 0 1 )$ , with a fixed and a learned policy.

To provide some intuition, we examine the learning behavior with and without target networks on both the critic and actor in Figure 3, where we graph the value, in a similar manner to Figure 1, in the Hopper-v1 environment. In (a) we compare the behavior with a fixed policy and in (b) we examine the value estimates with a policy that continues to learn, trained with the current value estimate. The target networks use a slow-moving update rate, parametrized by τ .

While updating the value estimate without target networks $( \tau = 1 )$ ) increases the volatility, all update rates result in similar convergent behaviors when considering a fixed policy. However, when the policy is trained with the current value estimate, the use of fast-updating target networks results in highly divergent behavior.

When do actor-critic methods fail to learn? These results suggest that the divergence that occurs without target networks is the result of policy updates with a high variance value estimate. Figure 3, as well as Section 4, suggest failure can occur due to the interplay between the actor and critic updates. Value estimates diverge through overestimation when the policy is poor, and the policy will become poor if the value estimate itself is inaccurate.

If target networks can be used to reduce the error over multiple updates, and policy updates on high-error states cause divergent behavior, then the policy network should be updated at a lower frequency than the value network, to first minimize error before introducing a policy update. We propose delaying policy updates until the value error is as small as possible. The modification is to only update the policy and target networks after a fixed number of updates d to the critic. To ensure the TD-error remains small, we update the

## target networks slowly $\theta ^ { \prime }  \tau \theta + ( 1 - \tau ) \theta ^ { \prime } .$

By sufficiently delaying the policy updates we limit the likelihood of repeating updates with respect to an unchanged critic. The less frequent policy updates that do occur will use a value estimate with lower variance, and in principle, should result in higher quality policy updates. This creates a two-timescale algorithm, as often required for convergence in the linear setting (Konda & Tsitsiklis, 2003). The effectiveness of this strategy is captured by our empirical results presented in Section 6.1, which show an improvement in performance while using fewer policy updates.

## 5.3. Target Policy Smoothing Regularization

A concern with deterministic policies is they can overfit to narrow peaks in the value estimate. When updating the critic, a learning target using a deterministic policy is highly susceptible to inaccuracies induced by function approximation error, increasing the variance of the target. This induced variance can be reduced through regularization. We introduce a regularization strategy for deep value learning, target policy smoothing, which mimics the learning update from SARSA (Sutton & Barto, 1998). Our approach enforces the notion that similar actions should have similar value. While the function approximation does this implicitly, the relationship between similar actions can be forced explicitly by modifying the training procedure. We propose that fitting the value of a small area around the target action

$$
y = r + \mathbb {E} _ {\epsilon} \left[ Q _ {\theta^ {\prime}} (s ^ {\prime}, \pi_ {\phi^ {\prime}} (s ^ {\prime}) + \epsilon) \right],\tag{13}
$$

would have the benefit of smoothing the value estimate by bootstrapping off of similar state-action value estimates. In practice, we can approximate this expectation over actions by adding a small amount of random noise to the target policy and averaging over mini-batches. This makes our modified target update:

$$
\begin{array}{c} y = r + \gamma Q _ {\theta^ {\prime}} (s ^ {\prime}, \pi_ {\phi^ {\prime}} (s ^ {\prime}) + \epsilon), \\ \epsilon \sim \mathrm{clip} (\mathcal {N} (0, \sigma), - c, c), \end{array}\tag{14}
$$

where the added noise is clipped to keep the target close to the original action. The outcome is an algorithm reminiscent of Expected SARSA (Van Seijen et al., 2009), where the value estimate is instead learned off-policy and the noise added to the target policy is chosen independently of the exploration policy. The value estimate learned is with respect to a noisy policy defined by the parameter σ.

Intuitively, it is known that policies derived from SARSA value estimates tend to be safer, as they provide higher value to actions resistant to perturbations. Thus, this style of update can additionally lead to improvement in stochastic domains with failure cases. A similar idea was introduced concurrently by Nachum et al. (2018), smoothing over $Q _ { \theta }$ rather than $Q _ { \theta ^ { \prime } }$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 TD3

Initialize critic networks $Q_{\theta_1}, Q_{\theta_2}$, and actor network $\pi_\phi$ with random parameters $\theta_1, \theta_2, \phi$

Initialize target networks $\theta'_1 \leftarrow \theta_1, \theta'_2 \leftarrow \theta_2, \phi' \leftarrow \phi$

Initialize replay buffer $\mathcal{B}$

for $t = 1$ to $T$ do

Select action with exploration noise $a \sim \pi_\phi(s) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma)$ and observe reward $r$ and new state $s'$

Store transition tuple $(s, a, r, s')$ in $\mathcal{B}$

Sample mini-batch of $N$ transitions $(s, a, r, s')$ from $\mathcal{B}$ $\tilde{a} \leftarrow \pi_{\phi'}(s') + \epsilon, \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \tilde{\sigma}), -c, c)$ $y \leftarrow r + \gamma \min_{i=1,2} Q_{\theta'_i}(s', \tilde{a})$

Update critics $\theta_i \leftarrow \arg\min_{\theta_i} N^{-1} \sum(y - Q_{\theta_i}(s, a))^2$

if $t \mod d$ then

Update $\phi$ by the deterministic policy gradient:

$\nabla_\phi J(\phi) = N^{-1} \sum \nabla_a Q_{\theta_1}(s, a)|_{a=\pi_\phi(s)} \nabla_\phi \pi_\phi(s)$

Update target networks:

$\theta'_i \leftarrow \tau\theta_i + (1 - \tau)\theta'_i$ $\phi' \leftarrow \tau\phi + (1 - \tau)\phi'$

end if

end for
</div>

(a)  
![](b722188c75ad8a577d70a66a1971b53c7980900d6c8a072cf30a547ec21b88b4.jpg)  
(b)

(c)  
![](9129744c1dfcf2edafce53412a615f27aeb9e1ea5a852c29a2f301c15f614500.jpg)  
(d)  
Figure 4. Example MuJoCo environments (a) HalfCheetah-v1, (b) Hopper-v1, (c) Walker2d-v1, (d) Ant-v1.

## 6. Experiments

We present the Twin Delayed Deep Deterministic policy gradient algorithm (TD3), which builds on the Deep Deterministic Policy Gradient algorithm (DDPG) (Lillicrap et al., 2015) by applying the modifications described in Sections 4.2, 5.2 and 5.3 to increase the stability and performance with consideration of function approximation error. TD3 maintains a pair of critics along with a single actor. For each time step, we update the pair of critics towards the minimum target value of actions selected by the target policy:

$$
\begin{array}{l} y = r + \gamma \min _ {i = 1, 2} Q _ {\theta_ {i} ^ {\prime}} (s ^ {\prime}, \pi_ {\phi^ {\prime}} (s ^ {\prime}) + \epsilon), \\ \epsilon \sim \mathrm{clip} (\mathcal {N} (0, \sigma), - c, c). \end{array}\tag{15}
$$

Every d iterations, the policy is updated with respect to $Q _ { \theta _ { 1 } }$ following the deterministic policy gradient algorithm (Silver et al., 2014). TD3 is summarized in Algorithm 1.

![](6e0a433f62100e102901cd1f6bc253e6ad30fe40455ca8d71e7cf38864a87255.jpg)  
(a) HalfCheetah-v1

![](5e27aff9e73049b89921d4524070fc8a61ec349257abe4ddafa9ef660d28df7f.jpg)  
(b) Hopper-v1

![](e972145774cdd2545175943af47f2c58b92ab92c2ecb6e693b5605b3c3bc4ca2.jpg)  
(c) Walker2d-v1

![](4c6b25ed14757c7e3e4a998c9caef6200874f98ca8d2d8357a21d9ee469f2046.jpg)  
(d) Ant-v1

![](65220b3fb680655d3f9e4633e736670b61df32ceeba49ce348dbc4d4037c1ecf.jpg)  
(e) Reacher-v1

![](f9cb01e65862591d3c1785f994a75dc84a14c3f8d0a57b017eb495472c9aba49.jpg)  
(f) InvertedPendulum-v1

![](aa2443e9d28fe379dae550d6aeadb040aa607e58bf1f6084f2b87ceaa0df3a6d.jpg)  
(g) InvertedDoublePendulum-v1  
Figure 5. Learning curves for the OpenAI gym continuous control tasks. The shaded region represents half a standard deviation of the average evaluation over 10 trials. Curves are smoothed uniformly for visual clarity.

Table 1. Max Average Return over 10 trials of 1 million time steps. Maximum value for each task is bolded. ± corresponds to a single standard deviation over trials.

<table><tr><td>Environment</td><td>TD3</td><td>DDPG</td><td>Our DDPG</td><td>PPO</td><td>TRPO</td><td>ACKTR</td><td>SAC</td></tr><tr><td>HalfCheetah</td><td>9636.95 ± 859.065</td><td>3305.60</td><td>8577.29</td><td>1795.43</td><td>-15.57</td><td>1450.46</td><td>2347.19</td></tr><tr><td>Hopper</td><td>3564.07 ± 114.74</td><td>2020.46</td><td>1860.02</td><td>2164.70</td><td>2471.30</td><td>2428.39</td><td>2996.66</td></tr><tr><td>Walker2d</td><td>4682.82 ± 539.64</td><td>1843.85</td><td>3098.11</td><td>3317.69</td><td>2321.47</td><td>1216.70</td><td>1283.67</td></tr><tr><td>Ant</td><td>4372.44 ± 1000.33</td><td>1005.30</td><td>888.77</td><td>1083.20</td><td>-75.85</td><td>1821.94</td><td>655.35</td></tr><tr><td>Reacher</td><td>-3.60 ± 0.56</td><td>-6.51</td><td>-4.01</td><td>-6.18</td><td>-111.43</td><td>-4.26</td><td>-4.44</td></tr><tr><td>InvPendulum</td><td>1000.00 ± 0.00</td><td>1000.00</td><td>1000.00</td><td>1000.00</td><td>985.40</td><td>1000.00</td><td>1000.00</td></tr><tr><td>InvDoublePendulum</td><td>9337.47 ± 14.96</td><td>9355.52</td><td>8369.95</td><td>8977.94</td><td>205.85</td><td>9081.92</td><td>8487.15</td></tr></table>

## 6.1. Evaluation

To evaluate our algorithm, we measure its performance on the suite of MuJoCo continuous control tasks (Todorov et al., 2012), interfaced through OpenAI Gym (Brockman et al., 2016) (Figure 4). To allow for reproducible comparison, we use the original set of tasks from Brockman et al. (2016) with no modifications to the environment or reward.

For our implementation of DDPG (Lillicrap et al., 2015), we use a two layer feedforward neural network of 400 and 300 hidden nodes respectively, with rectified linear units (ReLU) between each layer for both the actor and critic, and a final tanh unit following the output of the actor. Unlike the original DDPG, the critic receives both the state and action as input to the first layer. Both network parameters are updated using Adam (Kingma & Ba, 2014) with a learning rate of $1 0 ^ { - 3 }$ . After each time step, the networks are trained with a mini-batch of a 100 transitions, sampled uniformly from a replay buffer containing the entire history of the agent.

The target policy smoothing is implemented by adding  $\mathcal { N } ( 0 , 0 . 2 )$ to the actions chosen by the target actor network, clipped to $\left( - 0 . 5 , 0 . 5 \right)$ , delayed policy updates consists of only updating the actor and target critic network every d iterations, with $d = 2 .$ . While a larger d would result in a larger benefit with respect to accumulating errors, for fair comparison, the critics are only trained once per time step, and training the actor for too few iterations would cripple learning. Both target networks are updated with $\tau = 0 . 0 0 5$

To remove the dependency on the initial parameters of the policy we use a purely exploratory policy for the first 10000 time steps of stable length environments (HalfCheetah-v1 and Ant-v1) and the first 1000 time steps for the remaining environments. Afterwards, we use an off-policy exploration strategy, adding Gaussian noise $\mathcal { N } ( 0 , 0 . 1 )$ to each action. Unlike the original implementation of DDPG, we used uncorrelated noise for exploration as we found noise drawn from the Ornstein-Uhlenbeck (Uhlenbeck & Ornstein, 1930) process offered no performance benefits.

Each task is run for 1 million time steps with evaluations every 5000 time steps, where each evaluation reports the average reward over 10 episodes with no exploration noise. Our results are reported over 10 random seeds of the Gym simulator and the network initialization.

We compare our algorithm against DDPG (Lillicrap et al., 2015) as well as the state of art policy gradient algorithms: PPO (Schulman et al., 2017), ACKTR (Wu et al., 2017) and TRPO (Schulman et al., 2015), as implemented by OpenAI’s baselines repository (Dhariwal et al., 2017), and SAC (Haarnoja et al., 2018), as implemented by the author’s GitHub<sup>1</sup>. Additionally, we compare our method with our re-tuned version of DDPG, which includes all architecture and hyper-parameter modifications to DDPG without any of our proposed adjustments. A full comparison between our re-tuned version and the baselines DDPG is provided in the supplementary material.

Our results are presented in Table 1 and learning curves in Figure 5. TD3 matches or outperforms all other algorithms in both final performance and learning speed across all tasks.

## 6.2. Ablation Studies

We perform ablation studies to understand the contribution of each individual component: Clipped Double Q-learning (Section 4.2), delayed policy updates (Section 5.2) and target policy smoothing (Section 5.3). We present our results in Table 2 in which we compare the performance of removing each component from TD3 along with our modifications to the architecture and hyper-parameters. Additional learning curves can be found in the supplementary material.

The significance of each component varies task to task. While the addition of only a single component causes insignificant improvement in most cases, the addition of combinations performs at a much higher level. The full algorithm outperforms every other combination in most tasks. Although the actor is trained for only half the number of iterations, the inclusion of delayed policy update generally improves performance, while reducing training time.

We additionally compare the effectiveness of the actor-critic variants of Double Q-learning (Van Hasselt, 2010) and Double DQN (Van Hasselt et al., 2016), denoted DQ-AC and DDQN-AC respectively, in Table 2. For fairness in comparison, these methods also benefited from delayed policy updates, target policy smoothing and use our architecture and hyper-parameters. Both methods were shown to reduce overestimation bias less than Clipped Double Q-learning in Section 4. This is reflected empirically, as both methods result in insignificant improvements over TD3 - CDQ, with an exception in the Ant-v1 environment, which appears to benefit greatly from any overestimation reduction. As the inclusion of Clipped Double Q-learning into our full method outperforms both prior methods, this suggests that subduing the overestimations from the unbiased estimator is an effective measure to improve performance.

Table 2. Average return over the last 10 evaluations over 10 trials of 1 million time steps, comparing ablation over delayed policy updates (DP), target policy smoothing (TPS), Clipped Double Q-learning (CDQ) and our architecture, hyper-parameters and exploration (AHE). Maximum value for each task is bolded.

<table><tr><td>Method</td><td>HCheetah</td><td>Hopper</td><td>Walker2d</td><td>Ant</td></tr><tr><td>TD3</td><td>9532.99</td><td>3304.75</td><td>4565.24</td><td>4185.06</td></tr><tr><td>DDPG</td><td>3162.50</td><td>1731.94</td><td>1520.90</td><td>816.35</td></tr><tr><td>AHE</td><td>8401.02</td><td>1061.77</td><td>2362.13</td><td>564.07</td></tr><tr><td>AHE + DP</td><td>7588.64</td><td>1465.11</td><td>2459.53</td><td>896.13</td></tr><tr><td>AHE + TPS</td><td>9023.40</td><td>907.56</td><td>2961.36</td><td>872.17</td></tr><tr><td>AHE + CDQ</td><td>6470.20</td><td>1134.14</td><td>3979.21</td><td>3818.71</td></tr><tr><td>TD3 - DP</td><td>9590.65</td><td>2407.42</td><td>4695.50</td><td>3754.26</td></tr><tr><td>TD3 - TPS</td><td>8987.69</td><td>2392.59</td><td>4033.67</td><td>4155.24</td></tr><tr><td>TD3 - CDQ</td><td>9792.80</td><td>1837.32</td><td>2579.39</td><td>849.75</td></tr><tr><td>DQ-AC</td><td>9433.87</td><td>1773.71</td><td>3100.45</td><td>2445.97</td></tr><tr><td>DDQN-AC</td><td>10306.90</td><td>2155.75</td><td>3116.81</td><td>1092.18</td></tr></table>

## 7. Conclusion

Overestimation has been identified as a key problem in value-based methods. In this paper, we establish overestimation bias is also problematic in actor-critic methods. We find the common solutions for reducing overestimation bias in deep Q-learning with discrete actions are ineffective in an actor-critic setting, and develop a novel variant of Double Q-learning which limits possible overestimation. Our results demonstrate that mitigating overestimation can greatly improve the performance of modern algorithms.

Due to the connection between noise and overestimation, we examine the accumulation of errors from temporal difference learning. Our work investigates the importance of a standard technique in deep reinforcement learning, target networks, and examines their role in limiting errors from imprecise function approximation and stochastic optimization. Finally, we introduce a SARSA-style regularization technique which modifies the temporal difference target to bootstrap off similar state-action pairs.

Taken together, these improvements define our proposed approach, the Twin Delayed Deep Deterministic policy gradient algorithm (TD3), which greatly improves both the learning speed and performance of DDPG in a number of challenging tasks in the continuous control setting. Our algorithm exceeds the performance of numerous state of the art algorithms. As our modifications are simple to implement, they can be easily added to any other actor-critic algorithm.

## References

Anschel, O., Baram, N., and Shimkin, N. Averaged-dqn: Variance reduction and stabilization for deep reinforcement learning. In International Conference on Machine Learning, pp. 176–185, 2017.

Barth-Maron, G., Hoffman, M. W., Budden, D., Dabney, W., Horgan, D., TB, D., Muldal, A., Heess, N., and Lillicrap, T. Distributional policy gradients. International Conference on Learning Representations, 2018.

Bellemare, M. G., Dabney, W., and Munos, R. A distributional perspective on reinforcement learning. In International Conference on Machine Learning, pp. 449–458, 2017.

Bellman, R. Dynamic Programming. Princeton University Press, 1957.

Bertsekas, D. P. Dynamic programming and optimal control, volume 1. Athena scientific Belmont, MA, 1995.

Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., and Zaremba, W. Openai gym, 2016.

Dhariwal, P., Hesse, C., Plappert, M., Radford, A., Schulman, J., Sidor, S., and Wu, Y. Openai baselines. https: //github.com/openai/baselines, 2017.

Espeholt, L., Soyer, H., Munos, R., Simonyan, K., Mnih, V., Ward, T., Doron, Y., Firoiu, V., Harley, T., Dunning, I., et al. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. arXiv preprint arXiv:1802.01561, 2018.

Fox, R., Pakman, A., and Tishby, N. Taming the noise in reinforcement learning via soft updates. In Proceedings of the Thirty-Second Conference on Uncertainty in Artificial Intelligence, pp. 202–211. AUAI Press, 2016.

Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290, 2018.

He, F. S., Liu, Y., Schwing, A. G., and Peng, J. Learning to play in a day: Faster deep reinforcement learning by optimality tightening. arXiv preprint arXiv:1611.01606, 2016.

Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., and Meger, D. Deep Reinforcement Learning that Matters. arXiv preprint arXiv:1709.06560, 2017.

Horgan, D., Quan, J., Budden, D., Barth-Maron, G., Hessel, M., van Hasselt, H., and Silver, D. Distributed prioritized experience replay. International Conference on Learning Representations, 2018.

Kingma, D. and Ba, J. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Konda, V. R. and Tsitsiklis, J. N. On actor-critic algorithms. SIAMjournal on Control and Optimization, 42(4):1143– 1166, 2003.

Lee, D., Defourny, B., and Powell, W. B. Bias-corrected q-learning to control max-operator bias in q-learning. In Adaptive Dynamic Programming And Reinforcement Learning (ADPRL), 2013 IEEE Symposium on, pp. 93–99. IEEE, 2013.

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., and Wierstra, D. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.

Lin, L.-J. Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine learning, 8(3-4):293–321, 1992.

Mannor, S. and Tsitsiklis, J. N. Mean-variance optimization in markov decision processes. In International Conference on Machine Learning, pp. 177–184, 2011.

Mannor, S., Simester, D., Sun, P., and Tsitsiklis, J. N. Bias and variance approximation in value function estimates. Management Science, 53(2):308–322, 2007.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. Human-level control through deep reinforcement learning. Nature, 518(7540): 529–533, 2015.

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., and Kavukcuoglu, K. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning, pp. 1928– 1937, 2016.

Munos, R., Stepleton, T., Harutyunyan, A., and Bellemare, M. Safe and efficient off-policy reinforcement learning. In Advances in Neural Information Processing Systems, pp. 1054–1062, 2016.

Nachum, O., Norouzi, M., Tucker, G., and Schuurmans, D. Smoothed action value functions for learning gaussian policies. arXiv preprint arXiv:1803.02348, 2018.

O’Donoghue, B., Osband, I., Munos, R., and Mnih, V. The uncertainty bellman equation and exploration. arXiv preprint arXiv:1709.05380, 2017.

Pendrith, M. D., Ryan, M. R., et al. Estimator variance in reinforcement learning: Theoretical problems and practical solutions. University of New South Wales, School of Computer Science and Engineering, 1997.

Petrik, M. and Scherrer, B. Biasing approximate dynamic programming with a lower discount factor. In Advances in Neural Information Processing Systems, pp. 1265–1272, 2009.

Popov, I., Heess, N., Lillicrap, T., Hafner, R., Barth-Maron, G., Vecerik, M., Lampe, T., Tassa, Y., Erez, T., and Riedmiller, M. Data-efficient deep reinforcement learning for dexterous manipulation. arXiv preprint arXiv:1704.03073, 2017.

Precup, D., Sutton, R. S., and Dasgupta, S. Off-policy temporal-difference learning with function approximation. In International Conference on Machine Learning, pp. 417–424, 2001.

Schaul, T., Quan, J., Antonoglou, I., and Silver, D. Prioritized experience replay. In International Conference on Learning Representations, Puerto Rico, 2016.

Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. Trust region policy optimization. In International Conference on Machine Learning, pp. 1889–1897, 2015.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., and Riedmiller, M. Deterministic policy gradient algorithms. In International Conference on Machine Learning, pp. 387–395, 2014.

Singh, S., Jaakkola, T., Littman, M. L., and Szepesvari,´ C. Convergence results for single-step on-policy reinforcement-learning algorithms. Machine learning, 38(3):287–308, 2000.

Sutton, R. S. Learning to predict by the methods of temporal differences. Machine learning, 3(1):9–44, 1988.

Sutton, R. S. and Barto, A. G. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

Thrun, S. and Schwartz, A. Issues in using function approximation for reinforcement learning. In Proceedings ofthe 1993 Connectionist Models Summer School Hillsdale, NJ. Lawrence Erlbaum, 1993.

Todorov, E., Erez, T., and Tassa, Y. Mujoco: A physics engine for model-based control. In Intelligent Robots and Systems (IROS), 2012 IEEE/RSJ International Conference on, pp. 5026–5033. IEEE, 2012.

Uhlenbeck, G. E. and Ornstein, L. S. On the theory of the brownian motion. Physical review, 36(5):823, 1930.

Van Hasselt, H. Double q-learning. In Advances in Neural Information Processing Systems, pp. 2613–2621, 2010.

Van Hasselt, H., Guez, A., and Silver, D. Deep reinforcement learning with double q-learning. In AAAI, pp. 2094– 2100, 2016.

Van Seijen, H., Van Hasselt, H., Whiteson, S., and Wiering, M. A theoretical and empirical analysis of expected sarsa. In Adaptive Dynamic Programming and Reinforcement Learning, 2009. ADPRL’09. IEEE Symposium on, pp. 177–184. IEEE, 2009.

Watkins, C. J. C. H. Learningfrom delayed rewards. PhD thesis, King’s College, Cambridge, 1989.

Wu, Y., Mansimov, E., Grosse, R. B., Liao, S., and Ba, J. Scalable trust-region method for deep reinforcement learning using kronecker-factored approximation. In Advances in Neural Information Processing Systems, pp. 5285–5294, 2017.

## A. Proof of Convergence of Clipped Double Q-Learning

In a version of Clipped Double Q-learning for a finite MDP setting, we maintain two tabular value estimates $Q ^ { A } , Q ^ { B }$ . At each time step we select actions $a ^ { * } = \operatorname { a r g m a x } _ { a } Q ^ { A } ( s , a )$ and then perform an update by setting target y:

$$
\begin{array}{l} a ^ {*} = \underset {a} {\operatorname{argmax}} Q ^ {A} (s ^ {\prime}, a) \\ y = r + \gamma \min (Q ^ {A} (s ^ {\prime}, a ^ {*}), Q ^ {B} (s ^ {\prime}, a ^ {*})), \end{array}\tag{16}
$$

and update the value estimates with respect to the target and learning rate $\alpha _ { t } ( s , a )$

$$
\begin{array}{l} Q ^ {A} (s, a) = Q ^ {A} (s, a) + \alpha_ {t} (s, a) (y - Q ^ {A} (s, a)) \\ Q ^ {B} (s, a) = Q ^ {B} (s, a) + \alpha_ {t} (s, a) (y - Q ^ {B} (s, a)). \end{array}\tag{17}
$$

In a finite MDP setting, Double Q-learning is often used to deal with noise induced by random rewards or state transitions, and so either $Q ^ { A } \ \mathrm { o r } \ { \bar { Q } } ^ { B }$ is updated randomly. However, in a function approximation setting, the interest may be more towards the approximation error and thus we can update both $Q ^ { A }$ and $Q ^ { B }$ at each iteration. The proof extends naturally to updating either randomly.

The proof borrows heavily from the proof of convergence of SARSA (Singh et al., 2000) as well as Double Q-learning (Van Hasselt, 2010). The proof of lemma 1 can be found in Singh et al. (2000), building on a proposition from Bertsekas (1995).

Lemma 1. Consider a stochastic process $( \zeta _ { t } , \Delta _ { t } , F _ { t } ) , t \ge 0$ where $\zeta _ { t } , \Delta _ { t } , F _ { t } : X \to \mathbb { R }$ satisfy the equation:

$$
\Delta_ {t + 1} (x _ {t}) = (1 - \zeta_ {t} (x _ {t})) \Delta_ {t} (x _ {t}) + \zeta_ {t} (x _ {t}) F _ {t} (x _ {t}),\tag{18}
$$

where $x _ { t } \in X$ and $t = 0 , 1 , 2 , \ldots$ Let $P _ { t }$ be a sequence of increasing σ-fields such that $\zeta _ { 0 }$ and $\Delta _ { 0 }$ are $P _ { 0 }$ -measurable and $\zeta _ { t } , \Delta _ { t }$ and $F _ { t - 1 }$ are P -measurable, $t = 1 , 2 , \ldots$ Assume that the following hold:

1. The set X is finite.

2. $\begin{array} { r } { \zeta _ { t } ( x _ { t } ) \in [ 0 , 1 ] , \sum _ { t } \zeta _ { t } ( x _ { t } ) = \infty , \sum _ { t } ( \zeta _ { t } ( x _ { t } ) ) ^ { 2 } < \infty } \end{array}$ with probability 1 and x = $x _ { t } : \zeta ( x ) = 0$

3. E $[ F _ { t } | P _ { t } ] \left| \right| \leq \kappa | | \Delta _ { t } | | + c _ { t }$ where $\kappa \in [ 0 , 1 )$ and $c _ { t }$ converges to 0 with probability 1.

4. $\mathrm { V a r } [ F _ { t } ( x _ { t } ) | P _ { t } ] \leq K ( 1 + \kappa | | \Delta _ { t } | | ) ^ { 2 }$ , where K is some constant

Where $| | \cdot | |$ denotes the maximum norm. Then $\Delta _ { t }$ converges to 0 with probability 1.

## Theorem 1. Given the following conditions:

1. Each state action pair is sampled an infinite number of times.

2. The MDP is finite.

3. $\gamma \in [ 0 , 1 )$

4. Q values are stored in a lookup table.

5. Both $Q ^ { A }$ and $Q ^ { B }$ receive an infinite number of updates.

6. The learning rates satisfy $\alpha _ { t } ( s , a ) \in [ 0 , 1 ] , \sum _ { t } \alpha _ { t } ( s , a ) = \infty , \sum _ { t } ( \alpha _ { t } ( s , a ) ) ^ { 2 } < \infty$ with probability 1 and $\alpha _ { t } ( s , a ) =$ 0 $, \forall ( s , a ) \neq ( s _ { t } , a _ { t } )$

7. $\operatorname { V a r } [ r ( s , a ) ] < \infty , \forall s , a .$

Then Clipped Double Q-learning will converge to the optimal value function $Q ^ { * }$ , as defined by the Bellman optimality equation, with probability 1.

Proof of Theorem 1. We apply Lemma 1 with $P _ { t } = \{ Q _ { 0 } ^ { A } , Q _ { 0 } ^ { B } , s _ { 0 } , a _ { 0 } , \alpha _ { 0 } , r _ { 1 } , s _ { 1 } , . . . , s _ { t } , a _ { t } \} , X = S \times A , \Delta _ { t } = Q _ { t } ^ { A } -$ $Q ^ { * } , \zeta _ { t } = \alpha _ { t }$

First note that condition 1 and 4 of the lemma holds by the conditions 2 and 7 of the theorem respectively. Lemma condition 2 holds by the theorem condition 6 along with our selection of $\zeta _ { t } = \alpha _ { t }$

Defining a<sup>∗</sup> = argmax<sub>a</sub> $Q ^ { A } ( s _ { t + 1 } , a )$ we have

$$
\begin{array}{r l} & {\Delta_ {t + 1} (s _ {t}, a _ {t}) = (1 - \alpha_ {t} (s _ {t}, a _ {t})) (Q _ {t} ^ {A} (s _ {t}, a _ {t}) - Q ^ {*} (s _ {t}, a _ {t}))} \\ & {\qquad + \alpha_ {t} (s _ {t}, a _ {t}) (r _ {t} + \gamma \min (Q _ {t} ^ {A} (s _ {t + 1}, a ^ {*}), Q _ {t} ^ {B} (s _ {t + 1}, a ^ {*})) - Q ^ {*} (s _ {t}, a _ {t}))} \\ & {\qquad = (1 - \alpha_ {t} (s _ {t}, a _ {t})) \Delta_ {t} (s _ {t}, a _ {t}) + \alpha_ {t} (s _ {t}, a _ {t}) F _ {t} (s _ {t}, a _ {t})),} \end{array}\tag{19}
$$

where we have defined $F _ { t } ( s _ { t } , a _ { t } )$ as:

$$
\begin{array}{r l} & F _ {t} (s _ {t}, a _ {t}) = r _ {t} + \gamma \min (Q _ {t} ^ {A} (s _ {t + 1}, a ^ {*}), Q _ {t} ^ {B} (s _ {t + 1}, a ^ {*})) - Q _ {t} ^ {*} (s _ {t}, a _ {t}) \\ & \qquad = r _ {t} + \gamma \min (Q _ {t} ^ {A} (s _ {t + 1}, a ^ {*}), Q _ {t} ^ {B} (s _ {t + 1}, a ^ {*})) - Q _ {t} ^ {*} (s _ {t}, a _ {t}) + \gamma Q _ {t} ^ {A} (s _ {t + 1}, a ^ {*}) - \gamma Q _ {t} ^ {A} (s _ {t + 1}, a ^ {*}) \\ & \qquad = F _ {t} ^ {Q} (s _ {t}, a _ {t}) + c _ {t}, \end{array}\tag{20}
$$

where $F _ { t } ^ { Q } = r _ { t } + \gamma Q _ { t } ^ { A } ( s _ { t + 1 } , a ^ { * } ) - Q _ { t } ^ { * } ( s _ { t } , a _ { t } )$ denotes the value of $F _ { t }$ under standard Q-learning and $c _ { t } = $ $\gamma \operatorname* { m i n } ( Q _ { t } ^ { A } ( s _ { t + 1 } , a ^ { * } ) , Q _ { t } ^ { B } ( s _ { t + 1 } , a ^ { * } ) ) - \gamma Q _ { t } ^ { A } ( s _ { t + 1 } , a ^ { * } )$ . As E $\left\lceil F _ { t } ^ { Q } | P _ { t } \right\rceil \leq \gamma | | \Delta _ { t } | |$ is a well-known result, then condition 3 of lemma 1 holds if it can be shown that $c _ { t }$ converges to 0 with probability 1.

Let $y = r _ { t } + \gamma \operatorname* { m i n } ( Q _ { t } ^ { B } ( s _ { t + 1 } , a ^ { * } ) , Q _ { t } ^ { A } ( s _ { t + 1 } , a ^ { * } ) )$ and $\Delta _ { t } ^ { B A } ( s _ { t } , a _ { t } ) = Q _ { t } ^ { B } ( s _ { t } , a _ { t } ) - Q _ { t } ^ { A } ( s _ { t } , a _ { t } )$ , where $c _ { t }$ converges to 0 if $\Delta ^ { B A }$ converges to 0. The update of $\Delta _ { t } ^ { B A }$ at time t is the sum of updates of $Q ^ { A }$ and $Q ^ { B }$ :

$$
\begin{array}{r l} & {\Delta_ {t + 1} ^ {B A} (s _ {t}, a _ {t}) = \Delta_ {t} ^ {B A} (s _ {t}, a _ {t}) + \alpha_ {t} (s _ {t}, a _ {t}) \left(y - Q _ {t} ^ {B} (s _ {t}, a _ {t}) - (y - Q _ {t} ^ {A} (s _ {t}, a _ {t}))\right)} \\ & {\qquad = \Delta_ {t} ^ {B A} (s _ {t}, a _ {t}) + \alpha_ {t} (s _ {t}, a _ {t}) \left(Q _ {t} ^ {A} (s _ {t}, a _ {t}) - Q _ {t} ^ {B} (s _ {t}, a _ {t})\right)} \\ & {\qquad = (1 - \alpha_ {t} (s _ {t}, a _ {t})) \Delta_ {t} ^ {B A} (s _ {t}, a _ {t}).} \end{array}\tag{21}
$$

Clearly $\Delta _ { t } ^ { B A }$ will converge to 0, which then shows we have satisfied condition 3 of lemma 1, implying that $Q ^ { A } ( s _ { t } , a _ { t } )$ converges to $Q _ { t } ^ { * } ( s _ { t } , a _ { t } )$ . Similarly, we get convergence of $Q ^ { B } ( s _ { t } , a _ { t } )$ to the optimal vale function by choosing $\Delta _ { t } =$ $Q _ { t } ^ { B } - Q ^ { * }$ and repeating the same arguments, thus proving theorem 1.

## B. Overestimation Bias in Deterministic Policy Gradients

If the gradients from the deterministic policy gradient update are unnormalized, this overestimation is still guaranteed to occur under a slightly stronger condition on the expectation of the value estimate. Assume the approximate value function is equal to the true value function, in expectation over the steady-state distribution, with respect to policy parameters between the original policy and in the direction of the true policy update:

$$
\begin{array}{l} \mathbb {E} _ {s \sim \pi} \left[ Q _ {\theta} (s, \pi_ {\text { new }} (s)) \right] = \mathbb {E} _ {s \sim \pi} \left[ Q ^ {\pi} (s, \pi_ {\text { new }} (s)) \right] \\ \forall \phi_ {\text { new }} \in [ \phi , \phi + \beta (\phi_ {\text { true }} - \phi) ] \text {   such   that   } \beta > 0. \end{array}\tag{22}
$$

Noting that $\phi _ { \mathrm { t r u e } }$ maximizes the rate of change of the true value $\Delta _ { \mathrm { t r u e } } ^ { \pi } = Q ^ { \pi } ( s , \pi _ { \mathrm { t r u e } } ( s ) ) - Q ^ { \pi } ( s , \pi _ { \phi } ( s ) ) , \Delta _ { \mathrm { t r u e } } ^ { \pi } \geq \Delta _ { \mathrm { a p p r o x } } ^ { \pi }$ . By the given condition 22 the maximal rate of change of the approximate value must be at least as great $\Delta _ { \mathrm { a p p r o x } } ^ { \theta } \geq \Delta _ { \mathrm { t r u e } } ^ { \pi }$ . Given $Q _ { \theta } ( s , \pi _ { \phi } ) = Q ^ { \pi } ( s , \pi _ { \phi } )$ this implies $Q _ { \theta } ( s , \pi _ { \mathrm { a p p r o x } } ( s ) ) \geq Q ^ { \pi } ( s , \pi _ { \mathrm { t r u e } } ( s ) ) \geq Q ^ { \pi } ( s , \pi _ { \mathrm { a p p r o x } } ( s ) )$ , showing an overestimation of 41

Table 3. A complete comparison of hyper-parameter choices between our DDPG and the OpenAI baselines implementation (Dhariwa et al., 2017).

<table><tr><td>Hyper-parameter</td><td>Ours</td><td>DDPG</td></tr><tr><td>Critic Learning Rate</td><td> $10^{-3}$ </td><td> $10^{-3}$ </td></tr><tr><td>Critic Regularization</td><td>None</td><td> $10^{-2} \cdot ||\theta||^{2}$ </td></tr><tr><td>Actor Learning Rate</td><td> $10^{-3}$ </td><td> $10^{-4}$ </td></tr><tr><td>Actor Regularization</td><td>None</td><td>None</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Adam</td></tr><tr><td>Target Update Rate ( $\tau$ )</td><td> $5 \cdot 10^{-3}$ </td><td> $10^{-3}$ </td></tr><tr><td>Batch Size</td><td>100</td><td>64</td></tr><tr><td>Iterations per time step</td><td>1</td><td>1</td></tr><tr><td>Discount Factor</td><td>0.99</td><td>0.99</td></tr><tr><td>Reward Scaling</td><td>1.0</td><td>1.0</td></tr><tr><td>Normalized Observations</td><td>False</td><td>True</td></tr><tr><td>Gradient Clipping</td><td>False</td><td>False</td></tr><tr><td>Exploration Policy</td><td> $\mathcal{N}(0,0.1)$ </td><td>OU, $\theta = 0.15$ , $\mu = 0$ , $\sigma = 0.2$ </td></tr></table>

## C. DDPG Network and Hyper-parameter Comparison

## DDPG Critic Architecture

```txt
(state dim, 400)
ReLU
(action dim + 400, 300)
ReLU
(300, 1)
```

DDPG Actor Architecture

```txt
(state dim, 400)
ReLU
(400, 300)
ReLU
(300, 1)
tanh
```

Our Critic Architecture

```txt
(state dim + action dim, 400)
ReLU
(action dim + 400, 300)
RelU
(300, 1)
```

Our Actor Architecture

```txt
(state dim, 400)
ReLU
(400, 300)
ReLU
(300, 1)
tanh
```

## D. Additional Implementation Details

For clarity in presentation, certain implementation details were omitted, which we describe here. For the most complete possible description of the algorithm, code can be found on our GitHub (https://github.com/sfujim/TD3).

Our implementation of both DDPG and TD3 follows a standard practice in deep Q-learning, in which the update differs for terminal transitions. For transitions where the episode terminates by reaching some failure state, and not due to the episode running until the max horizon, the value of $Q ( s , \cdot )$ is set to 0 in the target y:

$$
y = \left\{ \begin{array}{l l} r & \text { if   terminal   s^{\prime } and  t <   max   horizon} \\ r + \gamma Q _ {\theta^ {\prime}} (s ^ {\prime}, \pi_ {\phi^ {\prime}} (s ^ {\prime})) & \text { else } \end{array} \right.
$$

For target policy smoothing (Section 5.3), the added noise is clipped to the range of possible actions, to avoid error introduced by using values of impossible actions:

$$
\begin{array}{l} y = r + \gamma Q _ {\theta^ {\prime}} (s ^ {\prime}, \text {clip} (\pi_ {\phi^ {\prime}} (s ^ {\prime}) + \epsilon , \text {min action}, \text {max action})), \\ \epsilon \sim \text {clip} (\mathcal {N} (0, \sigma), - c, c). \end{array}
$$

## E. Soft Actor-Critic Implementation Details

For our implementation of Soft Actor-Critic (Haarnoja et al., 2018) we use the code provided by the author (https: //github.com/haarnoja/sac), using the hyper-parameters described by the paper. We use a Gaussian mixture policy with 4 Gaussian distributions, except for the Reacher-v1 task, where we use a single Gaussian distribution due to numerical instability issues in the provided implementation. We use the environment-dependent reward scaling as described by the authors, multiplying the rewards by 3 for Walker2d-v1 and Ant-v1, and 1 for all remaining environments.

For fair comparison with our method, we train for only 1 iteration per time step, rather than the 4 iterations used by the results reported by the authors. This along with fewer total time steps should explain for the discrepancy in results on some of the environments. Additionally, we note this comparison is against a prior version of Soft Actor-Critic, while the most recent variant includes our Clipped Double Q-learning in the value update and produces competitive results to TD3 on most tasks.

## F. Additional Learning Curves

![](42f6fa92235568712a2323aec6729c5bd62327da124d90efcd4277caa590a7f2.jpg)  
(a) HalfCheetah-v1

![](068303cb5b66a12891e9ec6e720fa1e30ba37d505b6cfec0b986c07847195ee9.jpg)  
(b) Hopper-v1

![](dcae74134c65d518889fa59c290420f716fbd8ee6269e44186d96e16088d6b92.jpg)  
(c) Walker2d-v1

![](3c4043e3411784ed8f8597313a913494d212d4655a41b52248dfebd232a5e72e.jpg)  
(d) Ant-v1

Figure 6. Ablation over the varying modifications to our DDPG (AHE), comparing the subtraction of delayed policy updates (TD3 - DP), target policy smoothing (TD3 - TPS) and Clipped Double Q-learning (TD3 - CDQ).  
![](0c976291eef3d1b5612f320a38cf35dac3f2f8f830cd0b8e2f8041ccdf9ce9c7.jpg)  
(a) HalfCheetah-v1

![](7e35cdd8668bddcc8a1d9248558045adf450f512644ed9a8df1ac7a7255cdf4a.jpg)  
(b) Hopper-v1

![](40a698ef2127b6ac2c787df8f25f0a4e231bcdcff9fbbfd5fff3844f2aea8e6b.jpg)  
(c) Walker2d-v1

![](b74d7369b6367d5758d7147bf4488cf22fe9c2f853ffa73c0e4386f63748502b.jpg)  
(d) Ant-v1

Figure 7. Ablation over the varying modifications to our DDPG (AHE), comparing the addition of delayed policy updates (AHE + DP), target policy smoothing (AHE + TPS) and Clipped Double Q-learning (AHE + CDQ).  
![](39215b0b6341c9fbf11ae64c8ee691317af694130adb749e76d81919bdd226fd.jpg)  
(a) HalfCheetah-v1

![](68a9e52176ed1946ce4c15773424d364064607985697be1be732e4e313e50662.jpg)  
(b) Hopper-v1

![](4751f1958c710a8bae509ae8070c59c7fa683042b30485ea04fdd4c4a85c7382.jpg)  
(c) Walker2d-v1

![](4f057ff5220a9062be0de406128f253c99efaccdd6f0a4a0175aaf28d7bb5f9e.jpg)  
(d) Ant-v1  
Figure 8. Comparison of TD3 and the Double Q-learning (DQ-AC) and Double DQN (DDQN-AC) actor-critic variants, which also leverage delayed policy updates and target policy smoothing.