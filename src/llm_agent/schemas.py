from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class FoodCategory(StrEnum):
    PROTEIN = "protein"
    GRAINS = "grains"
    VEGETABLES = "vegetables"
    FRUITS = "fruits"
    DAIRY = "dairy"
    FAST_FOOD = "fast_food"
    SNACKS = "snacks"
    SWEETS = "sweets"
    BEVERAGES = "beverages"
    OTHER = "other"


class EntryInput(BaseModel):
    date: date
    description: str = Field(min_length=1)
    calories: int = Field(ge=0, strict=True)
    category: FoodCategory
    tags: list[str]
