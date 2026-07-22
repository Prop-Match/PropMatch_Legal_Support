import secrets
from typing import Annotated, Any

import jwt
from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


def require_user(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
    x_internal_service_key: Annotated[str | None, Header()] = None,
    x_propmatch_user_id: Annotated[str | None, Header()] = None,
    x_propmatch_user_role: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    if not settings.auth_required:
        return {"sub": "development", "role": "tenant"}

    if settings.internal_service_api_key:
        if not x_internal_service_key or not secrets.compare_digest(
            x_internal_service_key, settings.internal_service_api_key
        ):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="غير مصرح")
        if not x_propmatch_user_id or not x_propmatch_user_role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="بيانات المستخدم الداخلية مطلوبة",
            )
        return {"sub": x_propmatch_user_id, "role": x_propmatch_user_role}

    if not settings.jwt_secret:
        raise HTTPException(status_code=500, detail="JWT_SECRET is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="غير مصرح")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="غير مصرح") from exc
    if not payload.get("sub"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="غير مصرح")
    return payload


CurrentUser = Annotated[dict[str, Any], Depends(require_user)]
