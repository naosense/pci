# -*- coding:utf-8 _*-
from math import sqrt

critics = {
    "Lisa Rose": {
        "Lady in the Water": 2.5,
        "Snakes on a Plane": 3.5,
        "Just My Luck": 3.0,
        "Superman Returns": 3.5,
        "You, Me and Dupree": 2.5,
        "The Night Listener": 3.0,
    },
    "Gene Seymour": {
        "Lady in the Water": 3.0,
        "Snakes on a Plane": 3.5,
        "Just My Luck": 1.5,
        "Superman Returns": 5.0,
        "The Night Listener": 3.0,
        "You, Me and Dupree": 3.5,
    },
    "Michael Phillips": {
        "Lady in the Water": 2.5,
        "Snakes on a Plane": 3.0,
        "Superman Returns": 3.5,
        "The Night Listener": 4.0,
    },
    "Claudia Puig": {
        "Snakes on a Plane": 3.5,
        "Just My Luck": 3.0,
        "The Night Listener": 4.5,
        "Superman Returns": 4.0,
        "You, Me and Dupree": 2.5,
    },
    "Mick LaSalle": {
        "Lady in the Water": 3.0,
        "Snakes on a Plane": 4.0,
        "Just My Luck": 2.0,
        "Superman Returns": 3.0,
        "The Night Listener": 3.0,
        "You, Me and Dupree": 2.0,
    },
    "Jack Matthews": {
        "Lady in the Water": 3.0,
        "Snakes on a Plane": 4.0,
        "The Night Listener": 3.0,
        "Superman Returns": 5.0,
        "You, Me and Dupree": 3.5,
    },
    "Toby": {
        "Snakes on a Plane": 4.5,
        "You, Me and Dupree": 1.0,
        "Superman Returns": 4.0,
    },
}


# 返回person1和person2基于距离的相似度评价
def sim_distance(
    prefs: dict[str, dict[str, float]], person1: str, person2: str
) -> float:
    # 得到共同评价过得列表
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1

    # 如果两者没有共同之处，返回0
    if not si:
        return 0

    # 计算平方和
    sum_of_squares = sum(
        [
            pow(prefs[person1][item] - prefs[person2][item], 2)
            for item in prefs[person1]
            if item in prefs[person2]
        ]
    )

    return 1 / (1 + sqrt(sum_of_squares))


def sim_pearson(
    prefs: dict[str, dict[str, float]], person1: str, person2: str
) -> float:
    # 得到双方都评价过得物品列表
    si = {}

    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1

    if not si:
        return 1

    sum1 = sum([prefs[person1][it] for it in si])
    sum2 = sum([prefs[person2][it] for it in si])

    sum1sq = sum([pow(prefs[person1][it], 2) for it in si])
    sum2sq = sum([pow(prefs[person2][it], 2) for it in si])

    psum = sum([prefs[person1][it] * prefs[person2][it] for it in si])

    n = len(si)
    num = psum - (sum1 * sum2 / n)
    den = sqrt((sum1sq - pow(sum1, 2) / n) * (sum2sq - pow(sum2, 2) / n))
    if den == 0:
        return 0

    r = num / den
    return r


# 从反映偏好的字典中返回最佳匹配者
def top_matches(
    prefs: dict[str, dict[str, float]],
    person: str,
    n: int = 5,
    similarity: callable = sim_pearson,
) -> list[tuple[float, str]]:
    scores = [
        (similarity(prefs, person, other), other) for other in prefs if other != person
    ]

    scores.sort(reverse=True)
    return scores[0:n]


def get_recommendations(
    prefs: dict[str, dict[str, float]], person: str, similarity: callable = sim_pearson
) -> list[tuple[float, str]]:
    totals = {}
    sim_sums = {}
    for other in prefs:
        # 不和自己比较
        if other == person:
            continue
        sim = similarity(prefs, person, other)
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


def transform_prefs(prefs: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            result[item][person] = prefs[person][item]

    return result


def calculate_similar_items(
    prefs: dict[
        str,
        dict[str, float],
    ],
    n: int = 10,
) -> dict[str, list[tuple[float, str]]]:
    result = {}

    item_prefs = transform_prefs(prefs)
    c = 0
    for item in item_prefs:
        c += 1
        if c % 100 == 0:
            print(f"{c} / {len(item_prefs)}")
        scores = top_matches(item_prefs, item, n=n, similarity=sim_distance)
        result[item] = scores

    return result


def get_recommended_items(
    prefs: dict[str, dict[str, float]],
    item_match: dict[str, list[tuple[float, str]]],
    user: str,
) -> list[tuple[float, str]]:
    user_ratings = prefs[user]
    scores = {}
    total_sim = {}

    for item, rating in user_ratings.items():
        for similarity, item2 in item_match[item]:
            if item2 in user_ratings:
                continue
            scores.setdefault(item2, 0)
            scores[item2] += similarity * rating

            total_sim.setdefault(item2, 0)
            total_sim[item2] += similarity

    rankings = [(score / total_sim[item], item) for item, score in scores.items()]
    rankings.sort(reverse=True)
    return rankings


def load_movie_lens(path: str = "movielens") -> dict[str, float]:
    movies = {}
    for line in open(f"{path}/u.item", encoding="ISO-8859-1"):
        id, title = line.split("|")[0:2]
        movies[id] = title

    prefs = {}
    for line in open(f"{path}/u.data", encoding="ISO-8859-1"):
        user, movieid, rating, ts = line.split("\t")
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)
    return prefs
