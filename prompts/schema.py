# schema.py
from pydantic import BaseModel, Field
from typing import List

class ExtractedEntity(BaseModel):
    service_name: str = Field(description="The formal name of the company, initiative, project, or service provider.")
    location: str = Field(description="The city, region, town, or specific physical/geographical location.")
    contact: str = Field(description="Phone numbers, email addresses, contact forms, or specific contact names found.")
    description: str = Field(description="A concise summary detailing the service, objective, or initiative.")
    source_url: str = Field(description="The source URL this entity information was extracted from.")

class EntityCollection(BaseModel):
    entities: List[ExtractedEntity] = Field(description="List of all unique entities extracted from the web document.")
