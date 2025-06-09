from pydantic import BaseModel
from typing import List
class url(BaseModel):
    url:List[str]