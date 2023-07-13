# -*- coding:utf-8 _*-
from recommendations import *


# ex3 预先计算不同person之间的相似性
def similarity_person(
    prefs: dict[str, dict[str, float]], similarity: callable = sim_pearson
) -> dict[tuple[str, str], float]:
    sim_dict = {}
    for person1 in prefs.keys():
        for person2 in prefs.keys():
            if person1 == person2:
                continue

            sim = similarity(prefs, person1, person2)
            sim_dict[(person1, person2)] = sim
            sim_dict[(person2, person1)] = sim

    return sim_dict


def get_recommendations(
    prefs: dict[str, dict[str, float]],
    person: str,
    similarities: dict[tuple[str, str], float],
) -> list[tuple[float, str]]:
    totals = {}
    sim_sums = {}
    for other in prefs:
        # 不和自己比较
        if other == person:
            continue
        sim = similarities[(person, other)]
        # 忽略评价值小与0的情况，什么时候出现这种情况？
        if sim <= 0:
            continue
        for item in prefs[other]:
            # 只对自己还未看过的影片评价
            if item not in prefs[person] or prefs[person][item] == 0:
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                sim_sums.setdefault(item, 0)
                sim_sums[item] += sim

    # 建立一个归一化的列表
    rankings = [(total / sim_sums[item], item) for item, total in totals.items()]

    rankings.sort(reverse=True)
    return rankings
