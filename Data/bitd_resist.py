from sympy import Symbol, S, summation
from sympy.stats import P, Die, density

def calculate_EY(n):
    # Create a single die
    X = Die('X', 6)
    # Probability that one die is less than or equal to i
    prob_less_than_or_equal = lambda i: P(X <= i)
    # Calculate probability distribution of Y
    dist_Y = {}
    for i in range(1, 7):
        # Probability that the maximum of n dice is exactly i
        prob_max_i = prob_less_than_or_equal(i)**n - prob_less_than_or_equal(i-1)**n
        # Corresponding value of Y is 6 - i
        dist_Y[6 - i] = prob_max_i

    # Compute and return E(Y)
    return sum(value * prob for value, prob in dist_Y.items())

def calculate_EZ(n):
    # Probability of not rolling a 6 with one die
    p_not_six = S(5)/6
    # Probability of rolling at least two 6's
    # 1 - Probability of rolling zero 6's - Probability of rolling exactly one 6
    prob_at_least_two_six = 1 - p_not_six**n - n * (S(1)/6) * (p_not_six**(n-1))
    # E(Z) is the probability of Z=1
    return prob_at_least_two_six

def find_smallest_n():
    n = 1
    while True:
        EY = calculate_EY(n)
        EZ = calculate_EZ(n)
        if EY < EZ:
            return n
        n += 1

# Find and print the smallest n
smallest_n = find_smallest_n()
print(f"The smallest n for which E(Y) is smaller than E(Z) is: {smallest_n}")
