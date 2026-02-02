from backend.app.database.db import get_user

class AdaptivePracticeService:
    def recommend(self, user_id: str):
        profile = get_user(user_id)

        digit_accuracy = profile.digit_accuracy
        confusion = profile.confusion_matrix

        # --- 1. Most confused digit ---
        confused_scores = {}

        for key, count in confusion.items():
            true_d, pred_d = map(int, key.split("->"))
            confused_scores[true_d] = confused_scores.get(true_d, 0) + count

        most_confused_digit = (
            max(confused_scores, key=confused_scores.get)
            if confused_scores else None
        )

        # --- 2. Lowest accuracy digit ---
        weak_digits = sorted(
            digit_accuracy.items(),
            key=lambda x: x[1]
        )

        lowest_accuracy_digit = (
            weak_digits[0][0] if weak_digits else None
        )

        # --- 3. Decide recommendation ---
        recommended_digit = (
            most_confused_digit
            if most_confused_digit is not None
            else lowest_accuracy_digit
        )

        return {
            "recommended_digit": recommended_digit,
            "weak_digits": [
                d for d, acc in digit_accuracy.items() if acc < 0.7
            ],
            "confusion_summary": confused_scores,
            "message": (
                f"Practice digit {recommended_digit}"
                if recommended_digit is not None
                else "You're doing great! No practice needed."
            )
        }
