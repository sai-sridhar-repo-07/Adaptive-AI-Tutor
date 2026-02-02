from backend.app.database.db import get_user, save_user


class UserProfiler:
    def update_profile(
        self,
        user_id: str,
        true_digit: int,
        predicted_digit: int
    ):
        profile = get_user(user_id)

        profile.total_attempts += 1

        if true_digit == predicted_digit:
            profile.correct_attempts += 1

        # Update confusion matrix
        if true_digit != predicted_digit:
            key = f"{true_digit}->{predicted_digit}"
            profile.confusion_matrix[key] = (
                profile.confusion_matrix.get(key, 0) + 1
            )


        # Update per-digit accuracy
        digit_attempts = {
            k.split("->")[0]: v
            for k, v in profile.confusion_matrix.items()
        }

        correct = sum(
            v for k, v in profile.confusion_matrix.items()
            if k.split("->")[0] == k.split("->")[1]
        )

        for d in range(10):
            total = sum(
                v for k, v in profile.confusion_matrix.items()
                if int(k.split("->")[0]) == d
            )
            if total > 0:
                profile.digit_accuracy[d] = round(
                    sum(
                        v for k, v in profile.confusion_matrix.items()
                        if k == f"{d}->{d}"
                    ) / total,
                    3
                )

        save_user(profile)
        return profile
