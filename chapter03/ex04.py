# -*- coding:utf-8 _*-


def manhattan(v1: list[float], v2: list[float]) -> float:
    d = sum([abs(v1[i] - v2[i]) for i in range(len(v1))])
    return 1.0 - 1.0 / (1.0 + d)
