import uuid


def generate_api_token() -> str:
    return uuid.uuid4().hex
