# schemas.py

from pydantic import BaseModel


class GetTimeArgs(BaseModel):
    """
    Arguments schema for get_time tool.
    """
    city: str


class GetWeatherArgs(BaseModel):
    """
    Arguments schema for get_weather tool.
    """
    city: str


class FetchUrlArgs(BaseModel):
    """
    Arguments schema for fetch_url tool.
    """
    url: str


class SummarizeTextArgs(BaseModel):
    """
    Arguments schema for summarize_text tool.
    """
    text: str
    bullets: int = 3