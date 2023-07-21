# -*- coding:utf-8 _*-
import sqlite3
from enum import Enum

import nn

net = nn.SearchNet("nn.db")


class ScoreType(Enum):
    FREQUENCY = 1
    LOCATION = 2
    DISTANCE = 3
    INBOUND_COUNT = 4
    PAGERANK = 5
    LINK_TEXT = 6
    NN = 7
    MIX = 0


class Searcher:
    def __init__(self, dbname: str):
        self.con = sqlite3.connect(dbname)

    def __del__(self):
        self.con.close()

    def get_match_rows(self, q: str) -> tuple[list[tuple[int, int, int]], list[int]]:
        field_list = "w0.urlid"
        table_list = ""
        clause_list = ""
        word_ids = []

        words = q.split(" ")
        table_number = 0

        for word in words:
            word_row = self.con.execute(
                f"select rowid from wordlist where word='{word}'"
            ).fetchone()
            if word_row:
                word_id = word_row[0]
                word_ids.append(word_id)
                if table_number > 0:
                    table_list += ","
                    clause_list += " and "
                    clause_list += (
                        f"w{table_number - 1}.urlid=w{table_number}.urlid and "
                    )
                field_list += f",w{table_number}.location"
                table_list += f"wordlocation w{table_number}"
                clause_list += f"w{table_number}.wordid={word_id}"
                table_number += 1

        full_query = f"select {field_list} from {table_list} where {clause_list}"
        print(full_query)
        cur = self.con.execute(full_query)
        rows = [row for row in cur]

        return rows, word_ids

    def get_scored_list(
        self,
        rows: list[tuple[int, int, int]],
        word_ids: list[int],
        score_type: ScoreType = ScoreType.FREQUENCY,
    ) -> dict[int, float]:
        total_scores = dict([(row[0], 0) for row in rows])

        match score_type:
            case ScoreType.FREQUENCY:
                weights = [(1.0, self.frequency_score(rows))]
            case ScoreType.LOCATION:
                weights = [(1.0, self.location_score(rows))]
            case ScoreType.DISTANCE:
                weights = [(1.0, self.distance_score(rows))]
            case ScoreType.INBOUND_COUNT:
                weights = [(1.0, self.inbound_link_score(rows))]
            case ScoreType.PAGERANK:
                weights = [(1.0, self.pagerank_score(rows))]
            case ScoreType.LINK_TEXT:
                weights = [(1.0, self.link_text_score(rows, word_ids))]
            case ScoreType.NN:
                weights = [(1.0, self.nn_score(rows, word_ids))]
            case ScoreType.MIX:
                weights = [
                    (1.0, self.frequency_score(rows)),
                    (1.5, self.location_score(rows)),
                ]

        for weight, scores in weights:
            for url in total_scores:
                total_scores[url] += weight * scores[url]

        return total_scores

    def get_url_name(self, id: int) -> str:
        return self.con.execute(f"select url from urllist where rowid={id}").fetchone()[
            0
        ]

    def query(
        self, q: str, score_type: ScoreType = ScoreType.FREQUENCY
    ) -> tuple[list[int], list[int]]:
        rows, word_ids = self.get_match_rows(q)
        scores = self.get_scored_list(rows, word_ids, score_type)
        ranked_scores = sorted(
            [(score, url) for url, score in scores.items()], reverse=True
        )
        for score, urlid in ranked_scores[0:10]:
            print(f"{score}\t{self.get_url_name(urlid)}")
        return word_ids, [r[1] for r in ranked_scores[0:10]]

    def normalize_scores(
        self, scores: dict[int, float], small_is_better=False
    ) -> dict[int, float]:
        vsmall = 0.00001

        if small_is_better:
            min_score = min(scores.values())
            return dict(
                [(u, float(min_score) / max(vsmall, l)) for u, l in scores.items()]
            )
        else:
            max_score = max(scores.values())
            if max_score == 0:
                max_score = vsmall
            return dict([(u, float(c) / max_score) for u, c in scores.items()])

    def frequency_score(self, rows: list[tuple[int, int, int]]) -> dict[int, float]:
        counts = dict([(row[0], 0) for row in rows])
        for row in rows:
            counts[row[0]] += 1
        return self.normalize_scores(counts)

    def location_score(self, rows: list[tuple[int, int, int]]) -> dict[int, float]:
        locations = dict([(row[0], 1000000) for row in rows])
        for row in rows:
            loc = sum(row[1:])
            if loc < locations[row[0]]:
                locations[row[0]] = loc

        return self.normalize_scores(locations, small_is_better=True)

    def distance_score(self, rows: list[tuple[int, int, int]]) -> dict[int, float]:
        if len(rows[0]) <= 2:
            return dict([(row[0], 1.0) for row in rows])

        mind_distance = dict([(row[0], 1000000) for row in rows])

        for row in rows:
            dist = sum([abs(row[i] - row[i - 1]) for i in range(2, len(row))])
            if dist < mind_distance[row[0]]:
                mind_distance[row[0]] = dist
        return self.normalize_scores(mind_distance, small_is_better=True)

    def inbound_link_score(self, rows: list[tuple[int, int, int]]) -> dict[int, float]:
        unique_urls = set([row[0] for row in rows])
        inbound_count = dict(
            [
                (
                    u,
                    self.con.execute(
                        f"select count(*) from link where toid={u}"
                    ).fetchone()[0],
                )
                for u in unique_urls
            ]
        )
        return self.normalize_scores(inbound_count)

    def calculate_pagerank(self, iterations: int = 20):
        self.con.execute("drop table if exists pagerank")
        self.con.execute("create table pagerank(urlid primary key,score)")

        self.con.execute("insert into pagerank select rowid, 1.0 from urllist")
        self.con.commit()

        # 为什么可以迭代求得pagerank值
        for i in range(iterations):
            print(f"Iteration {i}")
            for (urlid,) in self.con.execute("select rowid from urllist"):
                pr = 0.15

                for (linker,) in self.con.execute(
                    f"select distinct fromid from link where toid={urlid}"
                ):
                    linking_gpr = self.con.execute(
                        f"select score from pagerank where urlid={linker}"
                    ).fetchone()[0]
                    linking_count = self.con.execute(
                        f"select count(*) from link where fromid={linker}"
                    ).fetchone()[0]
                    pr += 0.85 * (linking_gpr / linking_count)
                self.con.execute(f"update pagerank set score={pr} where urlid={urlid}")
                self.con.commit()

    def pagerank_score(self, rows: list[tuple[int, int, int]]) -> dict[int, float]:
        pageranks = dict(
            [
                (
                    row[0],
                    self.con.execute(
                        f"select score from pagerank where urlid={row[0]}"
                    ).fetchone()[0],
                )
                for row in rows
            ]
        )
        max_rank = max(pageranks.values())
        return dict([(u, float(l) / max_rank) for u, l in pageranks.items()])

    def link_text_score(
        self, rows: list[tuple[int, int, int]], word_ids: list[int]
    ) -> dict[int, float]:
        link_scores = dict([(row[0], 0) for row in rows])
        for word_id in word_ids:
            cur = self.con.execute(
                f"select link.fromid, link.toid from linkwords,link "
                f"where wordid={word_id} and linkwords.linkid=link.rowid"
            )
            for fromid, toid in cur:
                if toid in link_scores:
                    pr = self.con.execute(
                        f"select score from pagerank where urlid={fromid}"
                    ).fetchone()[0]
                    link_scores[toid] += pr
        max_score = max(link_scores.values())
        return dict([(u, float(l) / max_score) for u, l in link_scores.items()])

    def nn_score(self, rows: list[tuple[int, int, int]], word_ids: list[int]):
        url_ids = [urlid for urlid in set([row[0] for row in rows])]
        nn_res = net.get_result(word_ids, url_ids)
        scores = dict([(url_ids[i], nn_res[i]) for i in range(len(url_ids))])
        return self.normalize_scores(scores)
