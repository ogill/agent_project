# schemas.py

from pydantic import BaseModel, Field


class GetTimeArgs(BaseModel):
    """Arguments schema for get_time tool."""
    city: str


class GetWeatherArgs(BaseModel):
    """Arguments schema for get_weather tool."""
    city: str


class FetchUrlArgs(BaseModel):
    """Arguments schema for fetch_url tool."""
    url: str


class SummarizeTextArgs(BaseModel):
    """Arguments schema for summarize_text tool."""
    text: str
    bullets: int = Field(default=3, ge=1, le=10)


class AlwaysFailArgs(BaseModel):
    """
    Arguments schema for always_fail tool.

    Explicit schema is REQUIRED so:
    - planner can emit valid JSON
    - agent can validate args
    - replanning works deterministically
    """
    reason: str = "forced failure for replanning test"