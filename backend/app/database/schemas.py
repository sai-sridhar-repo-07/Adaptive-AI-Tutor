from typing import Dict
from pydantic import BaseModel


class UserProfile(BaseModel):
    user_id: str

    total_attempts: int = 0
    correct_attempts: int = 0

    # digit -> accuracy
    digit_accuracy: Dict[int, float] = {}

    # (true, predicted) -> count
    confusion_matrix: Dict[str, int] = {}

    def accuracy(self):
        if self.total_attempts == 0:
            return 0.0
        return self.correct_attempts / self.total_attempts
