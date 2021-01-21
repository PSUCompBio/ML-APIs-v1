from typing import List, Optional
from pydantic import BaseModel,Field


class MPSData(BaseModel):
    simulation_path: str = Field(..., title="simulation path in S3",min_length=5)
