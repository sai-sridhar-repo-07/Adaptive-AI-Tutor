from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import io

from backend.app.services.inference import InferenceService
from backend.app.services.user_profile import UserProfiler

# -------------------------------------------------------------------
# Router initialization
# -------------------------------------------------------------------

router = APIRouter()

# -------------------------------------------------------------------
# Services (loaded once)
# -------------------------------------------------------------------

inference_service = InferenceService(
    model_path="digit_cnn.pth"
)

profiler = UserProfiler()

# -------------------------------------------------------------------
# Health Check
# -------------------------------------------------------------------

@router.get("/health")
def health_check():
    return {
        "status": "OK",
        "service": "Adaptive AI Tutor API"
    }

# -------------------------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------------------------

@router.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    """
    Upload a handwritten digit image and get:
    - predicted digit
    - confidence
    - Grad-CAM heatmap
    """

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    result = inference_service.predict(image)
    return result

# -------------------------------------------------------------------
# Feedback / Human-in-the-loop Learning
# -------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    user_id: str
    true_digit: int
    predicted_digit: int


@router.post("/feedback")
def submit_feedback(data: FeedbackRequest):
    """
    User corrects the model.
    This updates the personalized learning profile.
    """

    profile = profiler.update_profile(
        user_id=data.user_id,
        true_digit=data.true_digit,
        predicted_digit=data.predicted_digit
    )

    return {
        "message": "User profile updated successfully",
        "overall_accuracy": profile.accuracy(),
        "digit_accuracy": profile.digit_accuracy,
        "confusion_matrix": profile.confusion_matrix
    }
from backend.app.services.adaptive_service import AdaptiveService

adaptive_service = AdaptiveService()


@router.get("/adaptive/{user_id}")
def get_adaptive_plan(user_id: str):
    """
    Returns personalized learning recommendations
    """
    return adaptive_service.get_recommendation(user_id)


from backend.app.services.analytics_service import AnalyticsService

analytics_service = AnalyticsService()

@router.get("/analytics/{user_id}")
def get_analytics(user_id: str):
    return analytics_service.get_analytics(user_id)


from backend.app.services.adaptive_practice_service import AdaptivePracticeService

adaptive_practice_service = AdaptivePracticeService()

@router.get("/adaptive-practice/{user_id}")
def adaptive_practice(user_id: str):
    return adaptive_practice_service.recommend(user_id)


from backend.app.services.confusion_service import ConfusionService

confusion_service = ConfusionService()

@router.get("/confusion-matrix/{user_id}")
def confusion_matrix(user_id: str):
    return confusion_service.get_matrix(user_id)


import base64
from backend.app.services.digit_generator import DigitGenerator

digit_generator = DigitGenerator()

@router.get("/generate-practice/{digit}")
def generate_practice(digit: int):
    images = digit_generator.generate(digit)

    encoded = []
    for img in images:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded.append(
            base64.b64encode(buffer.getvalue()).decode()
        )

    return {
        "digit": digit,
        "samples": encoded
    }
