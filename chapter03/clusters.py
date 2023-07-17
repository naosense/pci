# -*- coding:utf-8 _*-
import random
from math import sqrt
from typing import Callable

from PIL import Image, ImageDraw
from typing_extensions import Self


def read_file(filename: str) -> tuple[list[str], list[str], list[list[float]]]:
    # py2: lines = [line for line in file(filename)]
    lines = [line for line in open(filename)]

    # 第一行是列标题
    colnames = lines[0].strip().split("\t")[1:]
    rownames = []
    data = []
    for line in lines[1:]:
        p = line.strip().split("\t")
        # 每行的第一列是列名
        rownames.append(p[0])
        # 剩余的是该行对应的数据
        data.append([float(x) for x in p[1:]])
    return rownames, colnames, data


def pearson(v1: list[float], v2: list[float]) -> float:
    sum1 = sum(v1)
    sum2 = sum(v2)

    sum1sq = sum([pow(v, 2) for v in v1])
    sum2sq = sum([pow(v, 2) for v in v2])

    psum = sum([v1[i] * v2[i] for i in range(len(v1))])

    num = psum - (sum1 * sum2 / len(v1))
    den = sqrt((sum1sq - pow(sum1, 2) / len(v1)) * (sum2sq - pow(sum2, 2) / len(v1)))
    if den == 0:
        return 0
    return 1.0 - num / den


class BiCluster:
    def __init__(
        self,
        vec: list[float],
        left: Self | None = None,
        right: Self | None = None,
        distance: float = 0.0,
        id: int | None = None,
    ):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance


def hcluster(
    rows: list[list[float]],
    distance: Callable[[list[float], list[float]], float] = pearson,
) -> BiCluster:
    distances = {}
    current_clust_id = -1

    clust = [BiCluster(rows[i], id=i) for i in range(len(rows))]

    while len(clust) > 1:
        lowest_pair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(
                        clust[i].vec, clust[j].vec
                    )

                d = distances[(clust[i].id, clust[j].id)]

                if d < closest:
                    closest = d
                    lowest_pair = (i, j)

        merge_vec = [
            (clust[lowest_pair[0]].vec[i] + clust[lowest_pair[1]].vec[i]) / 2.0
            for i in range(len(clust[0].vec))
        ]

        new_cluster = BiCluster(
            merge_vec,
            left=clust[lowest_pair[0]],
            right=clust[lowest_pair[1]],
            distance=closest,
            id=current_clust_id,
        )

        current_clust_id -= 1
        del clust[lowest_pair[1]]
        del clust[lowest_pair[0]]
        clust.append(new_cluster)

    return clust[0]


def print_clust(clust: BiCluster, labels: list[str] | None = None, n: int = 0):
    for i in range(n):
        print(" ", end="")

    if clust.id < 0:
        print("-")
    else:
        if labels is None:
            print(clust.id)
        else:
            print(labels[clust.id])

    if clust.left is not None:
        print_clust(clust.left, labels, n + 1)
    if clust.right is not None:
        print_clust(clust.right, labels, n + 1)


def get_height(clust: BiCluster) -> int:
    if clust.left is None and clust.right is None:
        return 1

    return get_height(clust.left) + get_height(clust.right)


def get_depth(clust: BiCluster) -> float:
    if clust.left is None and clust.right is None:
        return 0

    return max(get_depth(clust.left), get_depth(clust.right)) + clust.distance


def draw_dendrogram(clust: BiCluster, labels: list[str], jpeg: str = "clusters.jpg"):
    h = get_height(clust) * 20
    w = 1200
    depth = get_depth(clust)

    scaling = float(w - 150) / depth

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.line((0, h / 2, 10, h / 2), fill=(255, 0, 0))

    draw_node(draw, clust, 10, h / 2, scaling, labels)
    img.save(jpeg, "JPEG")


def draw_node(
    draw: ImageDraw,
    clust: BiCluster,
    x: float,
    y: float,
    scaling: float,
    labels: list[str],
):
    if clust.id < 0:
        h1 = get_height(clust.left) * 20
        h2 = get_height(clust.right) * 20
        top = y - (h1 + h2) / 2
        bottom = y + (h1 + h2) / 2

        ll = clust.distance * scaling

        draw.line((x, top + h1 / 2, x, bottom - h2 / 2), fill=(255, 0, 0))
        draw.line((x, top + h1 / 2, x + ll, top + h1 / 2), fill=(255, 0, 0))
        draw.line((x, bottom - h2 / 2, x + ll, bottom - h2 / 2), fill=(255, 0, 0))

        draw_node(draw, clust.left, x + ll, top + h1 / 2, scaling, labels)
        draw_node(draw, clust.right, x + ll, bottom - h2 / 2, scaling, labels)
    else:
        draw.text((x + 5, y - 7), labels[clust.id], (0, 0, 0))


def rotate_matrix(data: list[list[float]]) -> list[list[float]]:
    new_data = []
    for i in range(len(data[0])):
        new_row = [data[j][i] for j in range(len(data))]
        new_data.append(new_row)
    return new_data


def kcluster(
    rows: list[list[float]],
    distance: Callable[[list[float], list[float]], float] = pearson,
    k: int = 4,
) -> list[list[int]]:
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
    return best_matches


def tanimoto(v1: list[float], v2: list[float]) -> float:
    c1, c2, shr = 0, 0, 0

    for i in range(len(v1)):
        if v1[i] != 0:
            c1 += 1
        if v2[i] != 0:
            c2 += 1
        if v1[i] != 0 and v2[i] != 0:
            shr += 1

    return 1.0 - float(shr) / (c1 + c2 - shr)


def scale_down(
    data: list[list[float]],
    distance: Callable[[list[float], list[float]], float] = pearson,
    rate: float = 0.01,
) -> list[list[float]]:
    n = len(data)

    # n x n的相似性关系表
    real_dist = [[distance(data[i], data[j]) for j in range(n)] for i in range(0, n)]

    # n x 2
    loc = [[random.random(), random.random()] for _ in range(n)]
    # n x n，data[i]与data[j]的距离
    fake_dist = [[0.0 for _ in range(n)] for _ in range(n)]

    last_error = None
    for _ in range(0, 1000):
        for i in range(n):
            for j in range(n):
                fake_dist[i][j] = sqrt(
                    sum([pow(loc[i][x] - loc[j][x], 2) for x in range(len(loc[i]))])
                )

        # n x 2
        grad = [[0.0, 0.0] for _ in range(n)]

        total_error = 0
        # 这里i，j的顺序不太好理解，既然fake_dist[i][j] == fake_dist[j][i]，那么调换下顺序也无妨
        # real_dist同理
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                error_term = (fake_dist[i][j] - real_dist[i][j]) / real_dist[i][j]

                # (x1 - x2) / d * diff
                grad[i][0] += (loc[i][0] - loc[j][0]) / fake_dist[i][j] * error_term
                # (y1 - y2) / d * diff
                grad[i][1] += (loc[i][1] - loc[j][1]) / fake_dist[i][j] * error_term

                total_error += abs(error_term)

        print(total_error)

        if last_error and last_error < total_error:
            break
        last_error = total_error

        for i in range(n):
            loc[i][0] -= rate * grad[i][0]
            loc[i][1] -= rate * grad[i][1]

    return loc


def draw2d(data: list[list[float]], labels: list[str], jpeg: str = "mds2d.jpg"):
    img = Image.new("RGB", (2000, 2000), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(len(data)):
        x = (data[i][0] + 0.5) * 1000
        y = (data[i][1] + 0.5) * 1000
        draw.text((x, y), labels[i], (0, 0, 0))
    img.save(jpeg, "JPEG")
