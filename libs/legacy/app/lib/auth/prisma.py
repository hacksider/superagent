import json
from datetime import datetime, timedelta, timezone
from typing import Dict

import bcrypt
import jwt
import requests as req
from azure.identity import DefaultAzureCredential
from decouple import config
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.transport import requests
from google.oauth2 import id_token

from app.lib.prisma import prisma

jwtSecret = config("JWT_SECRET")


def signJWT(user_id: str) -> Dict[str, str]:
    EXPIRES = datetime.now(tz=timezone.utc) + timedelta(days=365)

    payload = {
        "exp": EXPIRES,
        "userId": user_id,
    }
    return jwt.encode(payload, jwtSecret, algorithm="HS256")


def decodeJWT(token: str) -> dict:
    try:
        decoded = jwt.decode(token, jwtSecret, algorithms=["HS256"])
        return decoded if decoded["exp"] else None

    except jwt.ExpiredSignatureError:
        print("Token expired. Get new one")
        return None

    except Exception:
        return None


def encryptPassword(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def validatePassword(password: str, encrypted: str) -> str:
    return bcrypt.checkpw(password.encode("utf-8"), encrypted.encode("utf-8"))


class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super(
            JWTBearer, self
        ).__call__(request)

        if not credentials:
            raise HTTPException(status_code=403, detail="Invalid authorization code.")
        if credentials.scheme != "Bearer":
            raise HTTPException(
                status_code=403, detail="Invalid token or expired token."
            )

        if credentials.credentials.startswith("oauth_"):
            accessToken = credentials.credentials.split("oauth_")[-1]
            oauth_data = prisma.user.find_first(where={"accessToken": accessToken})
            self.validateOAuthData(oauth_data)
            return dict({"userId": oauth_data.id})
        else:
            if not self.verify_jwt(credentials.credentials):
                if tokens_data := prisma.apitoken.find_first(
                    where={"token": credentials.credentials}
                ):
                    return json.loads(tokens_data.json())

                else:
                    raise HTTPException(
                        status_code=403, detail="Invalid token or expired token."
                    )

        return decodeJWT(credentials.credentials)

    def verify_jwt(self, jwtToken: str) -> bool:
        try:
            payload = decodeJWT(jwtToken)

        except Exception:
            payload = None

        return bool(payload)

    def validateOAuthData(self, oauth_data) -> bool:
        if oauth_data.provider == "google":
            self.verify_google_token(oauth_data.accessToken)
        elif oauth_data.provider == "github":
            self.verify_github_token(oauth_data.accessToken)
        elif oauth_data.provider == "azure-ad":
            self.verify_azure_token(oauth_data.accessToken)
        return True

    def verify_github_token(self, accessToken: str) -> bool:
        uri = "https://api.github.com/user"
        headers = {"Authorization": f"token {accessToken}"}
        res = req.get(uri, headers=headers)
        return res.status_code == 200

    def verify_google_token(self, accessToken: str) -> bool:
        try:
            id_info = id_token.verify_oauth2_token(
                accessToken, requests.Request(), config("GOOGLE_CLIENT_ID")
            )
            return id_info["aud"] == config("GOOGLE_CLIENT_ID")
        except ValueError:
            return False

    def verify_azure_token(self, accessToken: str) -> bool:
        try:
            credentials = DefaultAzureCredential(
                exclude_managed_identity_credential=False
            )
            token = credentials.get_token("https://management.azure.com/.default")
            return token.token == accessToken
        except Exception:
            return False
