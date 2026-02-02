from backend.app.database.db import get_user
from backend.app.adaptive_engine.adaptive_engine import AdaptiveEngine


class AdaptiveService:
    def __init__(self):
        self.engine = AdaptiveEngine()

    def get_recommendation(self, user_id: str):
        profile = get_user(user_id)

        recommendation = self.engine.recommend_practice(
            digit_accuracy=profile.digit_accuracy,
            confusion_matrix=profile.confusion_matrix
        )

        return {
            "user_id": user_id,
            "overall_accuracy": profile.accuracy(),
            "adaptive_plan": recommendation
        }
