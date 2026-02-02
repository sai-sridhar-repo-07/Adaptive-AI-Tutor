from backend.app.database.db import get_user

class AnalyticsService:
    def get_analytics(self, user_id: str):
        profile = get_user(user_id)

        return {
            "user_id": user_id,
            "overall_accuracy": profile.accuracy(),
            "total_attempts": profile.total_attempts,
            "digit_accuracy": profile.digit_accuracy,
            "confusion_matrix": profile.confusion_matrix
        }
