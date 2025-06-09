# token_generator.py

from google_auth_oauthlib.flow import InstalledAppFlow
import json

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send"
]

flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
creds = flow.run_local_server(port=0)

# Save token.json with required fields
with open("token.json", "w") as token_file:
    token_file.write(creds.to_json())

print(" token.json created successfully!")
