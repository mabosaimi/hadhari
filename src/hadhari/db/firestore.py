import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore

_DEFAULT_CRED_PATH = Path(__file__).resolve().parents[3] / "firebaseServiceAccount.json"


@lru_cache(maxsize=1)
def get_db() -> Any:
    cred_path = Path(os.environ.get("FIREBASE_CREDENTIALS_PATH", str(_DEFAULT_CRED_PATH)))
    try:
        firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred)

    return firestore.client()
