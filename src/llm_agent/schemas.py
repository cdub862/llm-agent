from datetime import date as Date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


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
    model_config = ConfigDict(extra="forbid")

    date: Date
    description: str = Field(min_length=1)
    calories: int = Field(ge=0, strict=True)
    category: FoodCategory
    tags: list[str]


class EntryUpdate(BaseModel):
    date: Date | None = None
    description: str | None = Field(default=None, min_length=1)
    calories: int | None = Field(default=None, ge=0, strict=True)
    category: FoodCategory | None = None
    tags: list[str] | None = None


class ListEntriesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: Date | None = Field(
        default=None,
        description="Inclusive start date in YYYY-MM-DD format.",
    )
    end_date: Date | None = Field(
        default=None,
        description="Inclusive end date in YYYY-MM-DD format.",
    )
