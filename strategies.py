import numpy as np


def strategy_always_heads_pf(round_idx, last_payoff, state):
    """
    Always guess heads (1.0).
    'pf' stands for partial feedback.
    We ignore 'last_payoff' because we never change our behavior.

    Returns
    -------
    guess = 1.0
    new_state = state (unchanged)
    """
    guess = 1.0
    return guess, state  # no state updates needed


def strategy_always_tails_pf(round_idx, last_payoff, state):
    """
    Always guess tails (0.0).
    """
    guess = 0.0
    return guess, state


def strategy_random_pf(round_idx, last_payoff, state):
    """
    Pick a random guess in [0,1] each round. Ignores partial feedback.
    """
    guess = np.random.rand()
    return guess, state


def strategy_freq_partial(round_idx, last_payoff, state):
    """
    A 'frequentist' approach under partial feedback:
    - We keep a heads_count and tails_count in state.
    - Each round, we guess heads_count / (heads_count+tails_count)
      if we have at least 1 known outcome.
      Otherwise, default to 0.5.
    - To decode the last round's outcome, we compare last_payoff to
      hypothetical payoffs for heads vs. tails.
      (Assuming Brier or some rule that can be distinguished.)

    State structure:
      {
        'heads_count': int,
        'tails_count': int,
        'last_guess': float
      }
    """
    # If no state yet, initialize
    if state is None:
        state = {"heads_count": 0, "tails_count": 0, "last_guess": 0.5}  # default guess

    # Try to decode last outcome from last_payoff
    # Only if round_idx > 0
    if round_idx > 0 and last_payoff is not None:
        # Hypothetical payoffs under Brier, for example
        # payoff = -(guess - outcome)^2
        guess_old = state["last_guess"]
        pay_if_0 = -((guess_old - 0.0) ** 2)
        pay_if_1 = -((guess_old - 1.0) ** 2)

        if abs(last_payoff - pay_if_0) < abs(last_payoff - pay_if_1):
            # outcome was likely 0
            state["tails_count"] += 1
        elif abs(last_payoff - pay_if_1) < abs(last_payoff - pay_if_0):
            # outcome was likely 1
            state["heads_count"] += 1
        else:
            # tie => ambiguous => no update
            pass

    # Now decide current guess
    total = state["heads_count"] + state["tails_count"]
    if total > 0:
        guess = state["heads_count"] / total
    else:
        guess = 0.5  # no info yet

    # Store new guess in state
    state["last_guess"] = guess

    return guess, state


def strategy_bayesian_partial(round_idx, last_payoff, state):
    """
    Bayesian updating with partial feedback:
    - Keep Beta(alpha, beta).
    - Each round, guess = alpha/(alpha+beta).
    - If last_payoff suggests outcome=0 vs outcome=1, update alpha/beta accordingly.

    State structure:
      {
        'alpha': float,
        'beta': float,
        'last_guess': float
      }
    """
    if state is None:
        state = {"alpha": 1.0, "beta": 1.0, "last_guess": 0.5}

    alpha = state["alpha"]
    beta = state["beta"]
    last_guess = state["last_guess"]

    # decode last outcome if possible
    if round_idx > 0 and last_payoff is not None:
        pay_if_0 = -((last_guess - 0.0) ** 2)
        pay_if_1 = -((last_guess - 1.0) ** 2)

        if abs(last_payoff - pay_if_0) < abs(last_payoff - pay_if_1):
            beta += 1  # outcome=0
        elif abs(last_payoff - pay_if_1) < abs(last_payoff - pay_if_0):
            alpha += 1  # outcome=1
        else:
            # ambiguous => no update
            pass

    # Posterior mean
    guess = alpha / (alpha + beta)

    # Update state
    state["alpha"] = alpha
    state["beta"] = beta
    state["last_guess"] = guess

    return guess, state


def strategy_moving_average_partial(round_idx, last_payoff, state, alpha=0.1):
    """
    Moving Average (or Exponential Smoothing) under partial feedback.
    - If we decode last outcome=1 or 0, we do:
         estimate <- alpha*(outcome) + (1-alpha)*estimate
    - Then guess = estimate
    - If ambiguous, no update.

    State structure:
      {
        'estimate': float,   # current running estimate of p
        'last_guess': float
      }
    alpha : float
        smoothing parameter in [0,1]
    """
    if state is None:
        state = {"estimate": 0.5, "last_guess": 0.5}  # start with 0.5

    est = state["estimate"]
    last_guess = state["last_guess"]

    # decode outcome from last payoff if round_idx>0
    if round_idx > 0 and last_payoff is not None:
        pay_if_0 = -((last_guess - 0.0) ** 2)
        pay_if_1 = -((last_guess - 1.0) ** 2)

        if abs(last_payoff - pay_if_0) < abs(last_payoff - pay_if_1):
            # outcome=0
            est = alpha * 0.0 + (1.0 - alpha) * est
        elif abs(last_payoff - pay_if_1) < abs(last_payoff - pay_if_0):
            # outcome=1
            est = alpha * 1.0 + (1.0 - alpha) * est
        else:
            # tie => no update
            pass

    guess = est
    # store
    state["estimate"] = est
    state["last_guess"] = guess

    return guess, state
