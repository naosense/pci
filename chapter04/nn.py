# -*- coding:utf-8 _*-
import sqlite3
from math import tanh


def dtanh(y: float) -> float:
    return 1.0 - y * y


class SearchNet:
    def __init__(self, dbname: str) -> None:
        self.con = sqlite3.connect(dbname)

    def __del__(self) -> None:
        self.con.close()

    def make_tables(self) -> None:
        self.con.execute("create table hiddennode(create_key)")
        self.con.execute("create table wordhidden(fromid,toid,strength)")
        self.con.execute("create table hiddenurl(fromid,toid,strength)")
        self.con.commit()

    def get_strength(self, fromid: int, toid: int, layer: int) -> float:
        if layer == 0:
            table = "wordhidden"
        else:
            table = "hiddenurl"
        res = self.con.execute(
            f"select strength from {table} where fromid={fromid} and toid={toid}"
        ).fetchone()
        if not res:
            if layer == 0:
                return -0.2
            if layer == 1:
                return 0
        return res[0]

    def set_strength(self, fromid: int, toid: int, layer: int, strength: float):
        if layer == 0:
            table = "wordhidden"
        else:
            table = "hiddenurl"
        res = self.con.execute(
            f"select rowid from {table} where fromid={fromid} and toid={toid}"
        ).fetchone()
        if not res:
            self.con.execute(
                f"insert into {table} (fromid,toid,strength) values ({fromid},{toid},{strength})"
            )
        else:
            rowid = res[0]
            self.con.execute(
                f"update {table} set strength={strength} where rowid={rowid}"
            )

    def generate_hidden_node(self, word_ids: list[int], urls: list[int]):
        if len(word_ids) > 3:
            return None
        create_key = "_".join(sorted([str(wi) for wi in word_ids]))
        res = self.con.execute(
            f"select rowid from hiddennode where create_key='{create_key}'"
        ).fetchone()

        if not res:
            cur = self.con.execute(
                f"insert into hiddennode (create_key) values ('{create_key}')"
            )
            hidden_id = cur.lastrowid
            for word_id in word_ids:
                self.set_strength(word_id, hidden_id, 0, 1.0 / len(word_ids))
            for urlid in urls:
                self.set_strength(hidden_id, urlid, 1, 0.1)
            self.con.commit()

    def get_all_hidden_ids(self, word_ids: list[int], url_ids: list[int]) -> list[int]:
        l1 = {}
        for word_id in word_ids:
            cur = self.con.execute(
                f"select toid from wordhidden where fromid={word_id}"
            )
            for row in cur:
                l1[row[0]] = 1
        for url_id in url_ids:
            cur = self.con.execute(f"select fromid from hiddenurl where toid={url_id}")
            for row in cur:
                l1[row[0]] = 1
        return list(l1.keys())

    def setup_network(self, word_ids: list[int], url_ids: list[int]) -> None:
        self.word_ids = word_ids
        self.hidden_ids = self.get_all_hidden_ids(word_ids, url_ids)
        self.url_ids = url_ids

        self.ai = [1.0] * len(self.word_ids)
        self.ah = [1.0] * len(self.hidden_ids)
        self.ao = [1.0] * len(self.url_ids)

        self.wi = [
            [self.get_strength(word_id, hidden_id, 0) for hidden_id in self.hidden_ids]
            for word_id in self.word_ids
        ]
        self.wo = [
            [self.get_strength(hidden_id, url_id, 1) for url_id in self.url_ids]
            for hidden_id in self.hidden_ids
        ]

    def feed_forward(self) -> list[float]:
        for i in range(len(self.word_ids)):
            self.ai[i] = 1.0

        for j in range(len(self.hidden_ids)):
            sum = 0.0
            for i in range(len(self.word_ids)):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = tanh(sum)

        for k in range(len(self.url_ids)):
            sum = 0.0
            for j in range(len(self.hidden_ids)):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = tanh(sum)

        return self.ao[:]

    def get_result(self, word_ids: list[int], url_ids: list[int]) -> list[float]:
        self.setup_network(word_ids, url_ids)
        return self.feed_forward()

    def back_propagate(self, targets: list[float], N: int = 0.5) -> None:
        output_deltas = [0.0] * len(self.url_ids)
        for k in range(len(self.url_ids)):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dtanh(self.ao[k]) * error

        hidden_deltas = [0.0] * len(self.hidden_ids)
        for j in range(len(self.hidden_ids)):
            error = 0.0
            for k in range(len(self.url_ids)):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dtanh(self.ah[j]) * error

        for j in range(len(self.hidden_ids)):
            for k in range(len(self.url_ids)):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += N * change

        for i in range(len(self.word_ids)):
            for j in range(len(self.hidden_ids)):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] += N * change

    def train_query(
        self, word_ids: list[int], url_ids: list[int], selected_url: int
    ) -> None:
        self.generate_hidden_node(word_ids, url_ids)

        self.setup_network(word_ids, url_ids)
        self.feed_forward()
        targets = [0.0] * len(url_ids)
        targets[url_ids.index(selected_url)] = 1.0
        self.back_propagate(targets)
        self.update_database()

    def update_database(self) -> None:
        for i in range(len(self.word_ids)):
            for j in range(len(self.hidden_ids)):
                self.set_strength(
                    self.word_ids[i], self.hidden_ids[j], 0, self.wi[i][j]
                )
        for j in range(len(self.hidden_ids)):
            for k in range(len(self.url_ids)):
                self.set_strength(self.hidden_ids[j], self.url_ids[k], 1, self.wo[j][k])
        self.con.commit()


if __name__ == "__main__":
    import os

    if not os.path.exists("nn.db"):
        net = SearchNet("nn.db")
        net.make_tables()
        wWorld, wRiver, wBank = 101, 102, 103
        uWorldBank, uRiver, uEarth = 201, 202, 203
        net.generate_hidden_node([wWorld, wBank], [uWorldBank, uRiver, uEarth])
        print(net.get_result([wWorld, wBank], [uWorldBank, uRiver, uEarth]))
        net.train_query([wWorld, wBank], [uWorldBank, uRiver, uEarth], uWorldBank)
        print(net.get_result([wWorld, wBank], [uWorldBank, uRiver, uEarth]))
        all_urls = [uWorldBank, uRiver, uEarth]
        for i in range(30):
            net.train_query([wWorld, wBank], all_urls, uWorldBank)
            net.train_query([wRiver, wBank], all_urls, uRiver)
            net.train_query([wWorld], all_urls, uEarth)
        print(net.get_result([wWorld, wBank], all_urls))
        print(net.get_result([wRiver, wBank], all_urls))
        print(net.get_result([wBank], all_urls))

        from search_engine import Searcher, ScoreType

        e = Searcher("searchindex.db")
        word_ids, url_ids = e.query("functional programming", score_type=ScoreType.NN)
        net.generate_hidden_node(word_ids, url_ids)
        for _ in range(30):
            for url_id in url_ids:
                net.train_query(word_ids, url_ids, url_id)
    else:
        print("nn.db is already exists.")
