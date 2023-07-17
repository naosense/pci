# -*- coding:utf-8 _*-
import random
from typing import Callable

from clusters import pearson


def kcluster(
    rows: list[list[float]],
    distance: Callable[[list[float], list[float]], float] = pearson,
    k: int = 4,
) -> tuple[list[list[float]], list[list[float]], float]:
    """返回的是各个群组包含的row的序号"""
    ranges = [
        (min([row[i] for row in rows]), max([row[i] for row in rows]))
        for i in range(len(rows[0]))
    ]

    clusters = [
        [
            random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
            for i in range(len(rows[0]))
        ]
        for _ in range(k)
    ]

    last_matches = None
    total_error = 0.0
    for t in range(100):
        print(f"Iteration {t}")
        best_matches = [[] for i in range(k)]

        for j in range(len(rows)):
            row = rows[j]
            best_match = 0
            # 找到row最匹配（最近的）中心点
            for i in range(k):
                d = distance(clusters[i], row)
                if d < distance(clusters[best_match], row):
                    best_match = i
            best_matches[best_match].append(j)
            total_error += distance(clusters[best_match], row)

        # 收敛返回
        if best_matches == last_matches:
            break
        last_matches = best_matches

        # 重新计算集群中心点，新的中心点属性为群组内各点属性的平均值
        for i in range(k):
            avgs = [0.0] * len(rows[0])
            if len(best_matches[i]) > 0:
                for row_id in best_matches[i]:
                    for m in range(len(rows[row_id])):
                        avgs[m] += rows[row_id][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(best_matches[i])
                clusters[i] = avgs

    # 感觉返回last_matches比较合理，因为作用域够大
    return best_matches, clusters, total_error
