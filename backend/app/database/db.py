import json
from pathlib import Path
from backend.app.database.schemas import UserProfile

# JSON-based lightweight DB (perfect for portfolio project)
DB_PATH = Path("user_profiles.json")


def load_db():
    if not DB_PATH.exists():
        return {}
    with open(DB_PATH, "r") as f:
        return json.load(f)


def save_db(data):
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_user(user_id: str) -> UserProfile:
    """
    Fetch user profile.
    If user does not exist, create a new one.
    """
    data = load_db()

    if user_id not in data:
        profile = UserProfile(user_id=user_id)
        data[user_id] = profile.dict()
        save_db(data)
        return profile

    return UserProfile(**data[user_id])


def save_user(profile: UserProfile):
    """
    Persist updated user profile.
    """
    data = load_db()
    data[profile.user_id] = profile.dict()
    save_db(data)
