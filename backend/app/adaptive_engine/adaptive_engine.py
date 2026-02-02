from typing import Dict, List, Tuple


class AdaptiveEngine:
    """
    Converts user learning data into practice recommendations
    """

    def __init__(self, mastery_threshold: float = 0.7):
        self.mastery_threshold = mastery_threshold

    def get_weak_digits(self, digit_accuracy: Dict[int, float]) -> List[int]:
        """
        Digits below mastery threshold
        """
        return [
            digit for digit, acc in digit_accuracy.items()
            if acc < self.mastery_threshold
        ]

    def get_confusion_pairs(
        self,
        confusion_matrix: Dict[str, int],
        min_count: int = 2
    ) -> List[Tuple[int, int]]:
        """
        Extract frequent confusion pairs like (3 -> 8)
        """
        pairs = []
        for key, count in confusion_matrix.items():
            true_d, pred_d = map(int, key.split("->"))
            if true_d != pred_d and count >= min_count:
                pairs.append((true_d, pred_d))
        return pairs

    def recommend_practice(
        self,
        digit_accuracy: Dict[int, float],
        confusion_matrix: Dict[str, int]
    ) -> Dict:
        """
        Final adaptive recommendation
        """
        weak_digits = self.get_weak_digits(digit_accuracy)
