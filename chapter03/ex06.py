# -*- coding:utf-8 _*-
from ex05 import *


def select_k(rows: list[list[float]]) -> list[float]:
    n = len(rows)
    res = []
    for i in range(n):
        _, _, error = kcluster(rows, k=i + 1)
        res.append(error)
    return res
