# Estimating the Probability of Heads for a Biased Coin

## How the game works

We are given a **biased** coin with some probability of flipping heads. There are two players who will employ different strategies to get as close as possible to the true probability. There is a payoff structure in place that will score the players based on their guesses.

## Objective

To test and compare the strategies and determine the best among them after a set number of rounds.

## Functions and variables

- Probability of heads $p$.
- Number of rounds $T$ (default to 1,000 rounds).
- Number of simulations $S$ (default to 20 simulations).
- The function `payoff_function` representing the payoff.
- The functions `strategy_(name of strategy)` that represent the strategies being used. A player can use only _one_ strategy at a time.
- The function `simulate_game` to run the game for the given number of rounds and simulations.
-

## The plan

1. **Choosing a _proper_ scoring rule**
   <details>
   <summary>More on proper scoring rules</summary>
    A proper scoring rule is a way to score probabilistic predictions such that the expected score is maximized (or the expected error is minimized) precisely when the predicted probabilities match the true underlying probabilities. Put differently, a scoring rule is called proper if telling the truth (i.e., reporting the true probability distribution) is your best strategy in expectation.

    When predicting the probability of heads for a biased coin toss, you want a scoring rule that incentivizes you to give honest forecasts. A rule that isn’t proper can create incentives for you to systematically over- or under-state your true belief to gain more points.

    Some examples include - Brier score and log score.

    Given that $\hat{p} =$ predicted probability, $X =$ actual outcome -
    - $\text{Brier}(\hat{p}, X) = (\hat{p}-X)^2$
    - $\text{LogScore}(\hat{p}, X) = \begin{cases} ln(\hat{p}), & \text{if } X = 1 \\ ln(1-\hat{p}), & \text{if } X = 0\end{cases}$
   </details>

2. **Define the repeated setup**
     - Each round, each player picks a guess $p_i$.
     - Then each player observes their payoff.

3. **Implement strategies**

    Some examples include -

    - **Bayesian**: Maintain $\text{Beta}(\alpha,\beta$)$ for $p$. If partial or full feedback is given, update it. Guess the posterior mean each round (for Brier/log, that’s typically optimal).
    - **Frequentist**: Track the fraction of “heads” deduced from partial feedback or from actual coin outcomes (if revealed). Guess that fraction.
    - **Always $p$ (simple baselines)**.
    - **Noisy or Adaptive**: E.g., add random shifts, “If I see poor payoff, guess a different number next time.”
    - **Game-Theoretic**: Introduce a dimension where each player’s payoff also depends on the other’s guess, or do a “who is closer?” contest. Then try “fictitious play” or “best response to the other’s guess distribution.”

4. **Run simulations**
5. **Visualize**
