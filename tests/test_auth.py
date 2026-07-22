import pytest
from fastapi import HTTPException

from app.auth import require_user
from app.config import Settings


def test_internal_service_key_accepts_nestjs_user_context():
    settings = Settings(
        auth_required=True,
        internal_service_api_key="shared-internal-key",
        jwt_secret="",
    )

    user = require_user(
        settings=settings,
        authorization=None,
        x_internal_service_key="shared-internal-key",
        x_propmatch_user_id="user-1",
        x_propmatch_user_role="TENANT",
    )

    assert user == {"sub": "user-1", "role": "TENANT"}


def test_internal_service_key_rejects_direct_or_mismatched_requests():
    settings = Settings(
        auth_required=True,
        internal_service_api_key="shared-internal-key",
    )

    with pytest.raises(HTTPException) as error:
        require_user(
            settings=settings,
            authorization="Bearer otherwise-valid-user-token",
            x_internal_service_key=None,
            x_propmatch_user_id=None,
            x_propmatch_user_role=None,
        )

    assert error.value.status_code == 401


def test_internal_service_request_requires_user_context_headers():
    settings = Settings(
        auth_required=True,
        internal_service_api_key="shared-internal-key",
    )

    with pytest.raises(HTTPException) as error:
        require_user(
            settings=settings,
            authorization=None,
            x_internal_service_key="shared-internal-key",
            x_propmatch_user_id=None,
            x_propmatch_user_role=None,
        )

    assert error.value.status_code == 400

