from backend.app.database.db import get_user

class ConfusionService:
    def get_matrix(self, user_id: str):
        profile = get_user(user_id)

        # Initialize 10x10 matrix
        matrix = [[0 for _ in range(10)] for _ in range(10)]

        for key, count in profile.confusion_matrix.items():
            true_d, pred_d = map(int, key.split("->"))
            matrix[true_d][pred_d] = count

        return {
            "labels": list(range(10)),
            "matrix": matrix
        }
