from fastapi import Request, HTTPException, Depends
from typing import List
import jwt

def roles_required(allowed_roles: List[str]):
    async def role_checker(request: Request):
        authorization_header = request.headers.get('Authorization')

        if authorization_header is None or not authorization_header.startswith("Bearer "):
            raise HTTPException(status_code=400, detail="Invalid or missing Authorization header")

        token = authorization_header.split(" ")[1]
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            # print(payload)
        except:
            raise HTTPException(status_code=401, detail="Invalid token")

        user_roles = payload.get('resource_access', {}).get('darch', {}).get('roles', [])
        print(user_roles)
        if not any(role in allowed_roles for role in user_roles):
            raise HTTPException(status_code=403, detail="Access forbidden")

    return Depends(role_checker)
