# schemas.py

from __future__ import annotations
from pydantic import BaseModel, Field


class AlwaysFailArgs(BaseModel):
    reason: str = Field(default="forced failure for replanning test")


class GetTimeArgs(BaseModel):
    city: str


class GetWeatherArgs(BaseModel):
    city: str


class FetchUrlArgs(BaseModel):
    url: str


class SummarizeTextArgs(BaseModel):
    text: str
    bullets: int = Field(default=3, ge=1, le=10)