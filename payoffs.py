import math


def brier_score(guess: float, outcome: int) -> float:
    """
    Brier Score (strictly proper for binary events).

    Returns the single-round payoff as:
        - (guess - outcome)^2
    i.e., we include a negative sign so that *higher* is better
    (since (guess - outcome)^2 is normally a 'cost' or 'loss').

    Parameters:
    -----------
    guess   : float, in [0,1]
        The player's predicted probability of outcome=1 (heads).
    outcome : int,   0 or 1
        The actual realized event.

    Returns:
    --------
    payoff  : float
        Negative quadratic error: -(guess - outcome)^2
        Range: [0, -1], with 0 being a perfect guess and -1 the worst.
    """
    return -((guess - outcome) ** 2)


def log_score(guess: float, outcome: int) -> float:
    """
    Logarithmic Score (strictly proper).

    If outcome=1 (heads), payoff = ln(guess).
    If outcome=0 (tails), payoff = ln(1 - guess).

    Because ln(0) -> -∞, you may wish to clip guesses
    away from 0 and 1 to avoid math errors in practice.

    Parameters:
    -----------
    guess   : float, in [0,1]
    outcome : int,   0 or 1

    Returns:
    --------
    payoff  : float
        Typically in [-∞, 0]. 0 if guess perfectly matches outcome=1 or 0.
    """
    # small epsilon to avoid log(0) numeric error
    eps = 1e-12
    clipped_guess = max(eps, min(1 - eps, guess))

    if outcome == 1:
        return math.log(clipped_guess)
    else:
        return math.log(1 - clipped_guess)


def spherical_score(guess: float, outcome: int) -> float:
    """
    Spherical Score (strictly proper).

    For a binary event:
        payoff = guess / sqrt(guess^2 + (1-guess)^2)  if outcome=1
                (1-guess) / sqrt(guess^2 + (1-guess)^2)  if outcome=0

    Range is [0, 1] for a single round if you interpret
    payoff as above. But typically for a 'loss' style,
    you might invert or subtract from 1.
    Here we keep it as a direct "score" where higher is better.

    Parameters:
    -----------
    guess   : float, in [0,1]
    outcome : int,   0 or 1

    Returns:
    --------
    payoff  : float, in [0, 1]
        1.0 is best (correct guess=1 if outcome=1 or guess=0 if outcome=0)
        0.0 is worst (if guess=0 when outcome=1, or guess=1 when outcome=0)
    """
    # denominator = sqrt(guess^2 + (1 - guess)^2)
    denom = math.sqrt(guess**2 + (1.0 - guess) ** 2)
    if denom < 1e-15:
        # If guess ~0 or ~1:
        # If outcome matches that guess, score near 1, else 0
        # e.g. guess=1, outcome=1 => payoff=1
        # guess=1, outcome=0 => payoff=0
        if guess < 1e-8 and outcome == 0:
            return 1.0
        elif guess > 1 - 1e-8 and outcome == 1:
            return 1.0
        else:
            return 0.0

    if outcome == 1:
        return guess / denom
    else:
        return (1.0 - guess) / denom
