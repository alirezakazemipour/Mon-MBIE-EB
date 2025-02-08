from numba import jit
import numpy as np
import random
import time
from tqdm import tqdm


def random_argmax(x, rng=np.random):
    """
    Simple random tiebreak for np.argmax() for when there are multiple max values.
    """

    best = np.argwhere(x == x.max())
    i = rng.choice(range(best.shape[0]))
    return tuple(best[i])


def random_argmin(x, rng=np.random):
    """
    Simple random tiebreak for np.argmin() for when there are multiple max values.
    """

    best = np.argwhere(x == x.min())
    i = rng.choice(range(best.shape[0]))
    return tuple(best[i])


# https://en.wikipedia.org/wiki/Pairing_function
@jit
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


def report_river_swim(env):
    ret = []
    for i in tqdm(range(10000)):
        np.random.seed(i)
        ret_e = 0
        obs, _ = env.reset(seed=i)
        t = 0
        while True:
            # while obs["mon"] == 1:
            #     a = {"env": 0, "mon": 0}
            #     obs, r, term, trunc, _ = env.step(a)
            #     ret_e += (0.99 ** t) * (r["env"] + r["mon"])
            #     t += 1

            a = {"env": 1, "mon": 3}
            obs, r, term, trunc, _ = env.step(a)
            ret_e += (0.99 ** t) * (r["env"] + r["mon"])
            if term or trunc:
                ret.append(ret_e)
                break
            t += 1

    print(np.max(ret))
    print(np.mean(ret) - 1.96 * np.std(ret) / np.sqrt(10000))
    print(np.std(ret))
    exit()


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
def kl_confidence(t, emp_mean, num_pulls, beta, precision=1e-5, max_iter=50):
    # https://github.com/guptav96/bandit-algorithms/blob/main/algo/klucb.py
    n = 0
    lower_bound = emp_mean
    upper_bound = 1
    while n < max_iter and upper_bound - lower_bound > precision:
        q = (lower_bound + upper_bound) / 2
        if kl_divergence(emp_mean, q) > (beta * np.log(1 + t * np.log(t) ** 2) / num_pulls):
            upper_bound = q
        else:
            lower_bound = q
        n += 1
    return (lower_bound + upper_bound) / 2.


@jit
def jittable_joint_max(a):
    res = np.zeros(a.shape[:2])
    for x in range(a.shape[0]):
        for y in range(a.shape[1]):
            res[x, y] = np.max(a[x, y])
    return res


if __name__ == "__main__":
    x = np.random.normal(1, size=(36, 2, 5, 2))
    r = np.max(x, axis=(-1, -2))
    t = jittable_joint_max(x)

    start = time.perf_counter()
    r = np.max(x, axis=(-1, -2))
    end = time.perf_counter()
    print("Elapsed (after compilation) = {}s".format((end - start)))

    start = time.perf_counter()
    t = jittable_joint_max(x)
    end = time.perf_counter()
    print("Elapsed (after compilation) = {}s".format((end - start)))
