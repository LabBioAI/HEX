# schema.py
from pydantic import BaseModel, Field
from typing import List

class EntityRow(BaseModel):
    entity_name: str = Field(description="The unique name of the health facility or community organization.")
    location: str = Field(description="City, township, region, or full address inside Ontario boundaries.")
    description: str = Field(description="A clear summary of specific public healthcare services provided.")
    contact_info: str = Field(description="Available phone numbers, branch emails, or website links.")

class EntityTable(BaseModel):
    rows: List[EntityRow] = Field(description="A complete aggregated collection of structured web table records.")
