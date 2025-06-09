# app/schemas/connection.py
from pydantic import BaseModel

class Logout(BaseModel):
    refresh_token: str
