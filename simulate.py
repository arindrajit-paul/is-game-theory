import numpy as np


def run_single_simulation(
    strategy_player1,
    strategy_player2,
    scoring_func,
    p_true=0.63,
    n_rounds=1000,
    random_seed=None,
):
    """
    Runs one simulation of n_rounds for two players, partial-feedback style.

    Each round:
      1. Generate the coin outcome (heads=1 w.p. p_true, else tails=0).
      2. Each player calls their strategy with (round_idx, last_payoff, state)
         to get a new guess in [0,1].
      3. Compute payoff using scoring_func(guess, outcome).
      4. Store payoff in an array; pass it back to the player's strategy next round
         so they can infer the outcome if possible (partial feedback).

    Parameters
    ----------
    strategy_player1, strategy_player2 : callables
        Each strategy function must have the signature:
          def strategy_name(round_idx, last_payoff, state) -> (guess, new_state)
        last_payoff is None or float; new_state is the updated state.
    scoring_func : callable
        A function(guess, outcome) -> float. E.g., brier_score, log_score, etc.
    p_true : float
        The true probability of heads for the biased coin.
    n_rounds : int
        Number of rounds in each simulation.
    random_seed : int or None
        If set, ensures reproducibility for that simulation.

    Returns
    -------
    payoffs_p1, payoffs_p2 : np.array
        Round-by-round payoffs for each player (length n_rounds).
        Summing or averaging these yields the player's total or mean score.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    payoffs_p1 = np.zeros(n_rounds)
    payoffs_p2 = np.zeros(n_rounds)

    # Each player's state; if a strategy needs persistent data (e.g. alpha,beta)
    state_p1 = None
    state_p2 = None

    # We'll store "last payoff" to pass into each strategy
    # For round 0, there's no previous payoff, so it's None
    last_payoff_p1 = None
    last_payoff_p2 = None

    for t in range(n_rounds):
        # 1) Each player picks a guess in [0,1], based on partial feedback
        guess_p1, state_p1 = strategy_player1(t, last_payoff_p1, state_p1)
        guess_p2, state_p2 = strategy_player2(t, last_payoff_p2, state_p2)

        # 2) Flip the coin with probability p_true = heads
        outcome = 1 if np.random.rand() < p_true else 0

        # 3) Compute payoffs using the chosen scoring function
        payoff_p1 = scoring_func(guess_p1, outcome)
        payoff_p2 = scoring_func(guess_p2, outcome)

        payoffs_p1[t] = payoff_p1
        payoffs_p2[t] = payoff_p2

        # 4) Update last_payoff so next round, the strategy can use it
        last_payoff_p1 = payoff_p1
        last_payoff_p2 = payoff_p2

    return payoffs_p1, payoffs_p2


def run_multiple_simulations(
    strategy_player1,
    strategy_player2,
    scoring_func,
    p_true=0.63,
    n_rounds=1000,
    n_sims=20,
):
    """
    Repeats run_single_simulation n_sims times (with different seeds),
    then returns the average final score for each player.

    Parameters
    ----------
    strategy_player1, strategy_player2 : callables
        As before.
    scoring_func : callable
        e.g., brier_score, log_score, etc.
    p_true : float
        Probability of heads for the coin.
    n_rounds : int
        Rounds per simulation.
    n_sims : int
        How many independent simulations to run.

    Returns
    -------
    avg_score_p1, avg_score_p2 : float
        The mean total payoff of each player, across n_sims simulations.
    """
    final_scores_p1 = []
    final_scores_p2 = []

    for seed in range(n_sims):
        pay_p1, pay_p2 = run_single_simulation(
            strategy_player1,
            strategy_player2,
            scoring_func,
            p_true,
            n_rounds,
            random_seed=seed,
        )
        # sum of payoffs over the n_rounds => final score for that sim
        final_scores_p1.append(np.sum(pay_p1))
        final_scores_p2.append(np.sum(pay_p2))

    avg_score_p1 = np.mean(final_scores_p1)
    avg_score_p2 = np.mean(final_scores_p2)
    return avg_score_p1, avg_score_p2


if __name__ == "__main__":

    from payoffs import *
    from strategies import *

    # Run one simulation
    p1_payoffs, p2_payoffs = run_single_simulation(
        strategy_player1=strategy_always_heads_pf,
        strategy_player2=strategy_bayesian_partial,
        scoring_func=log_score,
        p_true=0.63,
        n_rounds=100,
        random_seed=42,
    )

    print("Single sim final scores:")
    print("P1:", np.sum(p1_payoffs))
    print("P2:", np.sum(p2_payoffs))

    # Run multiple sims, average results
    avg_p1, avg_p2 = run_multiple_simulations(
        strategy_player1=strategy_always_heads_pf,
        strategy_player2=strategy_bayesian_partial,
        scoring_func=log_score,
        p_true=0.63,
        n_rounds=100,
        n_sims=20,
    )
    print(
        f"\nOver 20 sims, average final scores:\nPlayer1: {avg_p1:.3f}, Player2: {avg_p2:.3f}"
    )
