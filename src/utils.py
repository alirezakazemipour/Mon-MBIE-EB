from numba import jit
import numpy as np
import random


def random_argmax(x, rng=np.random):
    """
    Simple random tiebreak for np.argmax() for when there are multiple max values.
    """

    best = np.argwhere(x == x.max())
    i = rng.choice(range(best.shape[0]))
    return tuple(best[i])


def random_argmin(x, rng=np.random):
    """
    Simple random tiebreak for np.argmax() for when there are multiple max values.
    """

    best = np.argwhere(x == x.min())
    i = rng.choice(range(best.shape[0]))
    return tuple(best[i])


# https://en.wikipedia.org/wiki/Pairing_function
def cantor_pairing(x: int, y: int) -> int:
    """
    Cantor pairing function to uniquely encode two
    natural numbers into a single natural number.
    Used for seeding.

    Args:
        x (int): first number,
        y (int): second number,

    Returns:
        A unique integer computed from x and y.
    """

    return int(0.5 * (x + y) * (x + y + 1) + y)


def set_rng_seed(seed: int = None) -> None:
    """
    Set random number generator seed across modules
    with random/stochastic computations.

    Args:
        seed (int)
    """

    np.random.seed(seed)
    random.seed(seed)


def dict_to_id(d: dict) -> str:
    """
    Parse a dictionary and generate a unique id.
    The id will have the initials of every key followed by its value.
    Entries are separated by underscore.

    Example:
        d = {first_key: 0, some_key: True} -> fk0_skTrue
    """

    def make_prefix(key: str) -> str:
        return "".join(w[0] for w in key.split("_"))

    return "_".join([f"{make_prefix(k)}{v}" for k, v in d.items()])


@jit
def kl_divergence(p, q):
    # https://github.com/guptav96/bandit-algorithms/blob/main/algo/klucb.py
    DIV_MAX = 10000

    if q == 0 and p == 0:
        return 0
    elif q == 0 and not p == 0:
        return DIV_MAX
    elif q == 1 and p == 1:
        return 0
    elif q == 1 and not p == 1:
        return DIV_MAX
    elif p == 0:
        return np.log(1 / (1 - q))
    elif p == 1:
        return np.log(1 / q)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


@jit
def kl_confidence(t, emp_mean, num_pulls, precision=1e-5, max_iter=50):
    # https://github.com/guptav96/bandit-algorithms/blob/main/algo/klucb.py
    n = 0
    lower_bound = emp_mean
    upper_bound = 1
    while n < max_iter and upper_bound - lower_bound > precision:
        q = (lower_bound + upper_bound) / 2
        if kl_divergence(emp_mean, q) > (np.log(1 + t * np.log(t) ** 2) / num_pulls):
            upper_bound = q
        else:
            lower_bound = q
        n += 1
    return (lower_bound + upper_bound) / 2.
