from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.rate_limit import rate_limit
from app.schemas.models import FeedbackRequest, FeedbackResponse
from app.services.feedback_store import add_feedback

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse, dependencies=[Depends(rate_limit)])
def feedback(req: FeedbackRequest, settings: Settings = Depends(get_settings)) -> FeedbackResponse:
    # Write-only on purpose: there is no endpoint to READ feedback, so it cannot leak.
    add_feedback(settings.db_path, req.request_id, req.rating, req.reason)
    return FeedbackResponse(status="saved")
