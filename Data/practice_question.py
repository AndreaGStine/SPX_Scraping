import numpy

def solution(f, a, b, y, p):
    # want to find: min({x | f(x) = y})

    # check if function is increasing or decreasing:
    if f(b) > f(a):
        if f(b) < y or f(a) > y:
            return None

        # if abs(f(b) - f(a)) < p:
        #     return (b - a) / 2

        left = a
        right = b
        while (f(right) - f(left)) > p:
            if f((right - left)/2) > y:
                right = (right - left)/2
            elif f((right - left)/2) < y:
                left = (right - left)/2
            else:
                mid = (right - left) / 2
                if f(mid - p) < f(mid):
                    return mid
                else:
                    right = mid

                # return solution(f, a, mid, y, p)

        return (right - left)/2


        # increasing
    elif f(b) < f(a):
        if f(a) < y or f(b) > y:
            return None
        # decreasing
    else:
        return a if f(a) == y else None
