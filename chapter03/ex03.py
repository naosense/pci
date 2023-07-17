# -*- coding:utf-8 _*-
from math import sqrt


def distance(v1: list[float], v2: list[float]) -> float:
    d = sqrt(sum([pow(v1[i] - v2[i], 2) for i in range(len(v1))]))
    return 1.0 - 1.0 / (1.0 + d)
