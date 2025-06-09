from pydantic import BaseModel, Field

class ChangePassword(BaseModel):
    username: str
    old_password: str
    new_password: str
