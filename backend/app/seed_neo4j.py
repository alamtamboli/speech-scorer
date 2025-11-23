# backend/app/seed_neo4j.py
import json
import os
from neo4j_layer import Neo4jLayer

def main():
    base = os.path.dirname(__file__)
    rubric_path = os.path.join(base, "rubric.json")

    with open(rubric_path, "r", encoding="utf-8") as f:
        rubric = json.load(f)

    neo = Neo4jLayer()
    neo.seed_from_rubric(rubric)
    neo.close()

    print("Neo4j seeded successfully.")

if __name__ == "__main__":
    main()
