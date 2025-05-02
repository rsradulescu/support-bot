from pydantic import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    jira_user: str
    jira_token: str
    jira_server: str
    slack_token: str

    class Config:
        env_file = "../.env_vars"

settings = Settings()
