"""One-off migration: rename TTS provider IndicMio -> SpringLab in AgentConfig.

Context: the Indic-Mio TTS integration originally stored the *model* brand
("IndicMio") in the provider slot (tts_model.name). The provider is the org
SPRINGLab; the model stays "indic-mio". This script fixes existing agent docs
created before the rename.

Changes only agent_config.tts_model.name: "IndicMio" -> "SpringLab".
Leaves agent_config.tts_model.model ("indic-mio") untouched.

Usage:
    python -m scripts.migrate_indicmio_to_springlab            # dry-run (default)
    python -m scripts.migrate_indicmio_to_springlab --apply    # perform update

Run from the voicera_backend directory with the same env (MONGODB_*) as the app.
"""
import argparse
import sys

from app.config import settings
from pymongo import MongoClient

OLD_NAME = "IndicMio"
NEW_NAME = "SpringLab"
COLLECTION = "AgentConfig"
FIELD = "agent_config.tts_model.name"


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate IndicMio -> SpringLab TTS provider name")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the update. Without this flag the script only reports what would change (dry-run).",
    )
    args = parser.parse_args()

    client = MongoClient(settings.mongodb_uri)
    db = client[settings.MONGODB_DATABASE]
    coll = db[COLLECTION]

    query = {FIELD: OLD_NAME}
    match_count = coll.count_documents(query)

    print(f"DB={settings.MONGODB_DATABASE} collection={COLLECTION}")
    print(f"Agents with {FIELD} == '{OLD_NAME}': {match_count}")

    if match_count == 0:
        print("Nothing to migrate.")
        return 0

    if not args.apply:
        print("Dry-run. Re-run with --apply to update these documents.")
        return 0

    result = coll.update_many(query, {"$set": {FIELD: NEW_NAME}})
    print(f"Matched={result.matched_count} Modified={result.modified_count} ('{OLD_NAME}' -> '{NEW_NAME}')")

    remaining = coll.count_documents(query)
    if remaining:
        print(f"WARNING: {remaining} docs still match '{OLD_NAME}' after update.")
        return 1

    print("Migration complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
