# backend/app/neo4j_layer.py
import os
from neo4j import GraphDatabase


class Neo4jLayer:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        pwd = os.getenv("NEO4J_PASS", "test")

        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        if self.driver:
            self.driver.close()

    def seed_criterion(self, criterion_id, keywords):
        with self.driver.session() as session:
            session.run(
                "MERGE (c:Criterion {id:$cid})",
                cid=criterion_id
            )
            for kw in keywords:
                session.run(
                    """
                    MERGE (k:Keyword {text:$kw})
                    MERGE (c:Criterion {id:$cid})
                    MERGE (c)-[:HAS_KEYWORD]->(k)
                    """,
                    kw=kw, cid=criterion_id
                )

    def seed_from_rubric(self, rubric: dict):
        categories = rubric.get("categories", [])
        for cat in categories:
            for crit in cat.get("criteria", []):
                kws = []

                if crit.get("keywords"):
                    kws.extend(crit["keywords"])

                if crit.get("keyword_groups"):
                    for group in crit["keyword_groups"]:
                        kws.extend(group)

                self.seed_criterion(crit["id"], kws)

    def criterion_relevance(self, criterion_id, transcript):
        lc = transcript.lower()

        with self.driver.session() as session:
            res = session.run(
                "MATCH (c:Criterion {id:$cid})-[:HAS_KEYWORD]->(k) RETURN collect(k.text) AS kws",
                cid=criterion_id
            ).single()

            if not res:
                return 0.0

            kws = res.get("kws") or []
            found = sum(1 for k in kws if k.lower() in lc)

            return min(1.0, found / max(len(kws), 1))
