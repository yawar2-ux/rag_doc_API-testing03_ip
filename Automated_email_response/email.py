import os
import json
import base64
import datetime
from googletrans import Translator
import json
import googletrans
from langdetect import detect, LangDetectException
import smtplib
import os
import difflib  
from pydantic import BaseModel
import google.auth.transport.requests
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from typing import Optional
from email.mime.text import MIMEText
from googleapiclient.errors import HttpError  #
from fastapi.requests import Request
from email.mime.multipart import MIMEMultipart
from fastapi import FastAPI, HTTPException, Depends, Query, Body, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pathlib import Path
import psycopg2.extras
import requests
import traceback
import random
from contextlib import asynccontextmanager
import secrets
import psycopg2
import hashlib
import jwt
import re
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from fastapi import FastAPI, HTTPException, Query, Body, Depends, Header, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from typing import List, Dict, Any, Optional
from langdetect import detect
from langdetect import detect, LangDetectException
from googletrans import Translator
import re
import threading
import time
from fastapi import Path, HTTPException
from fastapi.security import APIKeyHeader
from fastapi import APIRouter, FastAPI
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv


router = APIRouter()



# Define the API key header security scheme
api_key_header = APIKeyHeader(name="Authorization", auto_error=True)


# Keycloak Configuration
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET")



# JWT configuration
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "RS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))


GROQ_API_KEY = os.getenv("GROQ_API_KEY1")


# Database Configuration
DB_PARAMS = {
    "dbname": os.getenv("dbname"),
    "user": os.getenv("user"),
    "password": os.getenv("password"),
    "host": os.getenv("host"),
    "port": os.getenv("port")
}
user_interaction_cache = {}


RESPONSE_TEMPLATES = {}

# Load response templates with proper error handling
try:
    template_path = os.path.join(os.path.dirname(__file__), "response_templates.json")
    with open(template_path, "r", encoding='utf-8') as f:
        RESPONSE_TEMPLATES = json.load(f)
        print(f"Templates loaded successfully from: {template_path}")
        print(f"Loaded template categories: {', '.join(RESPONSE_TEMPLATES.keys())}")
except FileNotFoundError:
    print(f"Error: Template file not found at: {template_path}")
    RESPONSE_TEMPLATES = {}
except json.JSONDecodeError as e:
    print(f"Error parsing template JSON: {str(e)}")
    RESPONSE_TEMPLATES = {}
except Exception as e:
    print(f"Unexpected error loading templates: {str(e)}")
    RESPONSE_TEMPLATES = {}

# Helper function to check if templates are loaded
def are_templates_loaded() -> bool:
    """Check if templates were successfully loaded"""
    return bool(RESPONSE_TEMPLATES)

# === Models ===
class EmailResponse(BaseModel):
    subject: str
    body: str
    recipient: str
    cc: Optional[List[str]] = []
    bcc: Optional[List[str]] = []
class TemplateRequest(BaseModel):
    template_path: str
    replacements: Optional[Dict[str, str]] = {}
    cc: Optional[List[str]] = []
    bcc: Optional[List[str]] = []


class FeedbackModel(BaseModel):
    rating: int
    comment: Optional[str] = None
    improvement_suggestion: Optional[str] = None

class AdminLogin(BaseModel):
    username: str
    password: str
from pydantic import BaseModel

class Login(BaseModel):
    username: str
    password: str
class TicketDetail(BaseModel):
    ticket_id: str
    subject: str
    status: str
    priority: str
    customer_name: str
    body: str
    assigned_to: Optional[str] = None
    
class TicketUpdate(BaseModel):
    ticket_status: Optional[str] = None
    assigned_to: Optional[str] = None
    priority: Optional[str] = None
    #updated_at: Optional[datetime] = None

#== Your setup logic ===
def create_tables():
    print(" Tables created!")

# === Lifespan: replaces deprecated startup/shutdown ===


# === Create app first ===
def validate_email_environment():
    """Validate required email-related environment variables"""
    required_vars = [
        "GMAIL_TOKEN",
        "GMAIL_REFRESH_TOKEN",
        "GMAIL_TOKEN_URL",
        "GMAIL_CLIENT_ID",
        "GMAIL_SECRET_KEY",
        "GMAIL_SENDER_EMAIL"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required email environment variables: {', '.join(missing_vars)}"
        )
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Validate environment variables
        validate_email_environment()
        
        # Create database tables
        create_tables()
        
        print("Environment variables loaded and validated")
        yield
    except Exception as e:
        print(f"Startup error: {str(e)}")
        raise
# app = FastAPI(lifespan=lifespan)
 
# === Authentication with Gmail API ===
def get_gmail_credentials():
    """Create credentials object from environment variables"""
    try:
        # Get all required credentials from environment variables
        token = os.getenv("GMAIL_TOKEN")
        refresh_token = os.getenv("GMAIL_REFRESH_TOKEN")
        client_id = os.getenv("GMAIL_CLIENT_ID")
        client_secret = os.getenv("GMAIL_SECRET_KEY")  # Note: Using GMAIL_SECRET_KEY from .env
        token_uri = os.getenv("GMAIL_TOKEN_URL")

        # Validate all required credentials are present
        if not all([token, refresh_token, client_id, client_secret, token_uri]):
            missing = []
            if not token: missing.append("GMAIL_TOKEN")
            if not refresh_token: missing.append("GMAIL_REFRESH_TOKEN")
            if not client_id: missing.append("GMAIL_CLIENT_ID")
            if not client_secret: missing.append("GMAIL_SECRET_KEY")
            if not token_uri: missing.append("GMAIL_TOKEN_URL")
            raise ValueError(f"Missing required Gmail credentials: {', '.join(missing)}")

        # Create credentials object
        creds = Credentials(
            token=token,
            refresh_token=refresh_token,
            token_uri=token_uri,
            client_id=client_id,
            client_secret=client_secret,
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/gmail.send",
            ]
        )

        # Check if token needs refreshing
        if creds.expired and creds.refresh_token:
            request = google.auth.transport.requests.Request()
            creds.refresh(request)
            
            # Update the token in environment
            os.environ["GMAIL_TOKEN"] = creds.token
            print("Token refreshed successfully")

        return creds
    except Exception as e:
        print(f"Error creating Gmail credentials: {str(e)}")
        raise
# === Extract Email  body from Multipart emails  and generate Simple single-part emails ===
def extract_email_body(email_data):
    body = ""
    
    if "parts" in email_data.get("payload", {}):
        parts = email_data["payload"]["parts"]
        for part in parts:
            if part.get("mimeType") == "text/plain":
                body_data = part["body"].get("data", "")
                if body_data:
                    body += base64.urlsafe_b64decode(body_data).decode("utf-8")
    elif "body" in email_data.get("payload", {}):
        body_data = email_data["payload"]["body"].get("data", "")
        if body_data:
            body = base64.urlsafe_b64decode(body_data).decode("utf-8")
    
    return body

# === Detect Language ===
def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return "en"  # Default to English if detection fails

#--authenticate_____
def authenticate_gmail():
    """Authenticate with Gmail API using credentials from environment variables"""
    try:
        # Get credentials using the existing get_gmail_credentials function
        creds = get_gmail_credentials()
        
        # Check if token needs refreshing
        if creds.expired and creds.refresh_token:
            request = google.auth.transport.requests.Request()
            creds.refresh(request)
            
            # Update the token in environment variable
            os.environ["GMAIL_TOKEN"] = creds.token
            print("Token refreshed successfully")

        # Build and return the Gmail service
        service = build("gmail", "v1", credentials=creds)
        return service

    except Exception as e:
        print(f"Gmail authentication error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Gmail authentication failed: {str(e)}"
        )

# === Fetch Emails from Gmail ===
@router.get("/api/fetch-emails")
def fetch_emails(
    is_unread: bool = Query(False, description="Fetch only unread emails"),
    max_results: int = Query(20, description="Maximum number of emails to fetch"),
    include_spam: bool = Query(False, description="Include emails marked as spam"),
    sender: str = Query(None, description="Filter by sender email"),
    subject_contains: str = Query(None, description="Filter by subject text"),
    date_after: str = Query(None, description="Filter by date (YYYY/MM/DD)"),
    date_before: str = Query(None, description="Filter by date (YYYY/MM/DD)"),
    label: str = Query(None, description="Filter by Gmail label")
):
    """Fetch emails from Gmail with various filtering options and save to database"""
    service = authenticate_gmail()
    conn = None
    cur = None

    try:
        # Connect to database
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        # Build query string based on parameters
        query_parts = []

        if is_unread:
            query_parts.append("is:unread")

        if include_spam:
            query_parts.append("in:spam")  # Explicitly include spam emails
        else:
            query_parts.append("NOT in:spam")  # Exclude spam emails

        if sender:
            query_parts.append(f"from:{sender}")

        if subject_contains:
            query_parts.append(f"subject:{subject_contains}")

        if date_after:
            query_parts.append(f"after:{date_after}")

        if date_before:
            query_parts.append(f"before:{date_before}")

        if label:
            query_parts.append(f"label:{label}")

        query = " ".join(query_parts)

        # Fetch emails
        results = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
        messages = results.get("messages", [])

        if not messages:
            return {"status": "success", "message": "No emails found", "emails": []}

        email_list = []
        saved_count = 0

        for msg in messages:
            msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
            headers = msg_data["payload"]["headers"]

            # Extract email metadata
            sender = next((h["value"] for h in headers if h["name"].lower() == "from"), "Unknown")
            subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "No Subject")
            recipient = next((h["value"] for h in headers if h["name"].lower() == "to"), "Unknown")
            date = next((h["value"] for h in headers if h["name"].lower() == "date"), "Unknown")
            
            # Extract CC and BCC headers
            cc = next((h["value"] for h in headers if h["name"].lower() == "cc"), None)
            # Note: BCC is typically not visible in received emails, 
            # but we'll include the field for consistency and for cases where it might be available
            bcc = next((h["value"] for h in headers if h["name"].lower() == "bcc"), None)

            # Convert the date to a datetime object
            try:
                email_received_at = datetime.datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z")
            except ValueError:
                email_received_at = None  # Handle invalid date formats

            # Extract thread ID
            thread_id = msg_data.get("threadId", "")

            # Extract body
            body = extract_email_body(msg_data)

            email_info = {
                "id": msg["id"],
                "thread_id": thread_id,
                "sender": sender,
                "recipient": recipient,
                "cc": cc,
                "bcc": bcc,
                "subject": subject,
                "date": date,
                "email_received_at": email_received_at,
                "body": body,
                "labels": msg_data.get("labelIds", []),
                "unread": "UNREAD" in msg_data.get("labelIds", [])
            }
            email_list.append(email_info)

            # Save email to database for analysis
            try:
                # Check if email already exists
                cur.execute("SELECT COUNT(*) FROM email_analysis WHERE email_id = %s", (msg["id"],))
                if cur.fetchone()[0] > 0:
                    # Update existing record
                    cur.execute("""
                        UPDATE email_analysis 
                        SET subject = %s, recipient = %s, cc = %s, bcc = %s, body = %s, sender = %s, updated_at = NOW()
                        WHERE email_id = %s
                    """, (subject, recipient, cc, bcc, body, sender, msg["id"]))
                else:
                    # Insert new record
                    cur.execute("""
                        INSERT INTO email_analysis (email_id, subject, recipient, cc, bcc, body, sender, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    """, (msg["id"], subject, recipient, cc, bcc, body, sender))

                saved_count += 1
            except Exception as db_error:
                print(f"Error saving email {msg['id']} to database: {str(db_error)}")
                # Continue with other emails even if one fails

        # Commit all database changes at once
        conn.commit()

        if email_list:
            print(f"Saved {saved_count} emails to database")
        else:
            print("No emails found")

        return {
            "status": "success",
            "message": f"Found {len(email_list)} emails, saved {saved_count} to database",
            "emails": email_list
        }
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error fetching emails: {str(e)}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()



# === Analyze Email ===
@router.post("/api/analyze-email/{email_id}")
def analyze_email(email_id: str):
    """Analyze an email and return its sentiment, category, priority, summary, escalation details, and detected language (including Hinglish and Punjabi)."""
    conn = None
    cur = None
    try:
     
        # Connect to the database
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        # Fetch email details
        cur.execute("SELECT subject, body, sender, recipient FROM email_analysis WHERE email_id = %s", (email_id,))
        email = cur.fetchone()
        if not email:
            raise HTTPException(status_code=404, detail=f"Email with ID {email_id} not found")

        subject, body, sender, recipient = email

        # Enhanced language detection with improved Hindi and Punjabi detection
        
        detected_language = "en"  # Default to English
        is_hinglish = False
        
        try:
            if body and len(body.strip()) > 5:
                # Check for Punjabi-specific characters and patterns
                # Punjabi Unicode range: \u0A00-\u0A7F
                has_punjabi_script = bool(re.search(r'[\u0A00-\u0A7F]', body))
                
                # Check for Devanagari script (Hindi)
                has_devanagari = bool(re.search(r'[\u0900-\u097F]', body))
                
                # Punjabi common words that can help distinguish from Hindi
                punjabi_indicators = [
                    "ਕਿਉਂ", "ਮੈਂ", "ਤੁਸੀਂ", "ਕੀ", "ਨਹੀਂ", "ਹਾਂ", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", 
                    "ਪੰਜਾਬੀ", "ਕਿਤੇ", "ਜੀ", "ਹੈ", "ਦਾ", "ਦੀ", "ਵਿੱਚ", "ਤੇ", "ਨਾਲ"
                ]
                
                # Count Punjabi indicator words
                punjabi_word_count = sum(1 for word in punjabi_indicators if word in body)
                
                # First priority: Check for Punjabi script or words
                if has_punjabi_script or punjabi_word_count >= 2:
                    detected_language = "pa"  # Punjabi ISO code
                # Second: Check for Devanagari script for Hindi
                elif has_devanagari:
                    detected_language = "hi"  # Hindi
                else:
                    # Use langdetect for initial language detection
                    try:
                        detected_lang = detect(body)
                        detected_language = detected_lang
                        
                        # If detection gives 'hi', 'pa', or 'und', verify with Google Translate
                        if detected_lang in ['hi', 'pa', 'und']:
                            try:
                                translator = Translator()
                                detection = translator.detect(body)
                                if detection and detection.lang:
                                    # Update detected language
                                    if detection.lang == 'pa':
                                        detected_language = 'pa'  # Confirmed Punjabi
                                    elif detection.lang == 'hi':
                                        detected_language = 'hi'  # Confirmed Hindi
                                    else:
                                        detected_language = detection.lang
                            except Exception as e:
                                print(f"Google Translate detection failed: {e}")
                    except LangDetectException as e:
                        print(f"LangDetect failed, trying Google Translate: {e}")
                        try:
                            translator = Translator()
                            detection = translator.detect(body)
                            if detection and detection.lang:
                                detected_language = detection.lang
                        except Exception as e:
                            print(f"Google Translate detection also failed: {e}")
                
                # Check for Hinglish if not already detected as Hindi or Punjabi
                if detected_language not in ['hi', 'pa']:
                    # Check for Hinglish: mixture of English characters and Hindi-derived words
                    has_english_chars = any(char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' for char in body)
                    
                    # Common Hinglish word indicators
                    hinglish_word_indicators = [
                        "kyunki", "toh", "lekin", "abhi", "thoda", "zyada", "kaafi", 
                        "ho gaya", "kar raha", "milta", "ummid", "jald", "intezaar",
                        "matlab", "pata hai", "samajh", "dekho", "suno", "aap", "mai",
                        "hum", "accha", "kya", "ko", "hai", "nahi", "karo", "gaya"
                    ]
                    
                    # Count Hinglish words in the text
                    hinglish_word_count = sum(1 for word in hinglish_word_indicators if word in body.lower())
                    
                    # Define criteria for Hinglish: has English chars, detected as English/undetermined, 
                    # and has at least 2 Hinglish indicators
                    if has_english_chars and detected_language in ['en', 'id', 'und'] and hinglish_word_count >= 2:
                        is_hinglish = True
                        detected_language = "hinglish"
            else:
                print("Email body too short for reliable language detection")
        except Exception as e:
            print(f"Error during language detection: {e}")

        # Store the original body
        original_body = body

        # Translate the subject to English if needed (excluding Hinglish)
        subject_in_english = subject
        if subject and detected_language not in ["en", "hinglish"]:
            try:
                translator = Translator()
                subject_translation = translator.translate(subject, dest='en')
                subject_in_english = subject_translation.text
            except Exception as e:
                print(f"Error translating subject to English: {str(e)}")
                # Fallback to original subject if translation fails

        # Prepare the prompt for analysis
        prompt = f"""
        Analyze the following email:

        Subject: {subject_in_english}

        Email Content:
        {body}

        1. **Categorization** Classify the email into one of the following categories:
            - Support: Technical issues, product functionality problems, or assistance requests.
            - Complaint: Expressions of dissatisfaction with product, service, or experience.
            - Query: General questions about products, services, or policies.
            - Feedback: Suggestions, ideas, or general feedback.
            - Sales: Inquiries about purchasing, pricing, or product availability.
            - Billing: Questions or issues related to payments, invoices, or subscriptions.
            - Feature Request: Request for new features or functionality.
            - Bug Report: Reporting of specific software or service issues.
            - Other: Emails that don't fit into the categories above.

        2. **Sentiment Analysis**: Classify the sentiment as one of the following:
           - Empathetic: Expressing understanding, compassion, or emotional support — often in response to frustration or personal concerns.
           - Corrective: Providing clarification or gently correcting a misunderstanding, usually in a respectful and informative tone.
           - Neutral: Factual or informational communication without strong emotion or judgment.
           - Mixed: Containing both empathetic and corrective (or other) elements — combining care with clarification
        3. **Tone Analysis**: Identify the dominant tone:
            - Professional: Formal business communication.
            - Casual: Informal and conversational.
            - Urgent: Expressing need for immediate action.
            - Frustrated: Showing irritation or annoyance.
            - Confused: Indicating lack of clarity or understanding.
            - Appreciative: Showing gratitude or thanks.

        4. **Priority**:
            - Low: Standard request, can be handled with regular SLA
            - Medium: Important but not time-sensitive
            - High: Requires prompt attention
            - Critical: Requires immediate attention

        5. **Escalation Check**: Determine if this email needs human escalation based on:
            - Urgent matters requiring immediate attention
            - Legal threats or implications
            - Complex technical issues beyond automated resolution
            - High-value customer accounts or VIPs
            - Multiple repeated complaints
            - Explicit request for human assistance
            - Sensitive personal or financial information requiring human review
            - Highly emotional content indicating significant customer distress

        6. **Summary**: Provide a detailed summary of the email content in 4-5 sentences, explaining the key points, concerns, and any specific requests or issues raised by the sender.

        Return a structured JSON response following this exact format:

        ```json
        {{
            "sentiment": "string",
            "category": "string",
            "tone": "string",
            "summary": "string",
            "priority": "string",
            "is_escalated": boolean,
            "escalation_reason": "string"
        }}
        ```

        Ensure the response is properly formatted as valid JSON.
        """

        # Send the prompt to Groq's API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Groq API error: {response.text}")

        # Parse the response from Groq
        ai_response = response.json().get("choices", [{}])[0].get("message", {}).get("content", {})
        if not ai_response:
            raise HTTPException(status_code=500, detail="Failed to parse analysis result from Groq API")

        # Parse JSON string to dict
        analysis_dict = json.loads(ai_response)

        # Add language detection information to the analysis_dict
        analysis_dict["detected_language"] = detected_language
        analysis_dict["is_hinglish"] = is_hinglish
        analysis_dict["original_body"] = original_body
        analysis_dict["sender"] = sender
        analysis_dict["recipient"] = recipient

        # Save individual analysis fields to database along with the full analysis result
        cur.execute("""
            UPDATE email_analysis
            SET analysis_result = %s,
                detected_language = %s,
                is_hinglish = %s,
                sentiment = %s,
                category = %s,
                priority = %s,
                summary = %s,
                is_escalated = %s,
                escalation_reason = %s
            WHERE email_id = %s
        """, (
            json.dumps(analysis_dict),
            detected_language,
            is_hinglish,
            analysis_dict.get("sentiment"),
            analysis_dict.get("category"),
            analysis_dict.get("priority"),
            analysis_dict.get("summary"),
            analysis_dict.get("is_escalated", False),
            analysis_dict.get("escalation_reason", ""),
            email_id
        ))
        conn.commit()

        return {
            "status": "success",
            "email_id": email_id,
            "analysis": analysis_dict
        }
    except Exception as e:
        print(f"Error in analyze_email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
# === Generate and Save Response ===
@router.post("/api/generate-response/{email_id}")
def generate_response(email_id: str):
    """
    Generate a response for an email by checking templates first, then falling back to AI.
    """
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Get email details from database
        query = """
        SELECT email_id, sender, recipient, subject, body, analysis_result, category
        FROM email_analysis
        WHERE email_id = %s
        """
        cur.execute(query, (email_id,))
        result = cur.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Email with ID {email_id} not found")
        
        email_id, sender, recipient, subject, body, analysis_result, category = result
        
        # Parse analysis result
        try:
            analysis = json.loads(analysis_result) if isinstance(analysis_result, str) else (analysis_result or {})
        except (json.JSONDecodeError, TypeError):
            analysis = {}
            
        # Find matching templates
        matched_templates = []
        if RESPONSE_TEMPLATES:
            email_category = category or "general"
            for cat, subcats in RESPONSE_TEMPLATES.items():
                if not isinstance(subcats, dict):
                    continue
                    
                for subcat, tones in subcats.items():
                    if not isinstance(tones, dict):
                        continue
                        
                    for tone, template in tones.items():
                        if cat.lower() == email_category.lower():
                            matched_templates.append({
                                "path": f"{cat}.{subcat}.{tone}",
                                "subject": template.get("subject", f"RE: {subject}"),
                                "preview": template.get("body", "")[:100] + "..."
                            })
        
        # Store matched templates
        templates_data = {
            "matched_templates": matched_templates,
            "best_match": matched_templates[0]["path"] if matched_templates else None,
            "match_count": len(matched_templates)
        }
        
        cur.execute("""
            UPDATE email_analysis
            SET matched_templates = %s
            WHERE email_id = %s
        """, (json.dumps(templates_data), email_id))
        conn.commit()
        
        response_data = {
            "email_id": email_id,
            "email_info": {
                "sender": sender,
                "recipient": recipient,
                "subject": subject
            },
            "template_options": matched_templates,
            "next_steps": [
                {
                    "action": "use_template",
                    "endpoint": f"/api/manual-template-response/{email_id}",
                    "method": "POST"
                },
                {
                    "action": "use_ai",
                    "endpoint": f"/api/ai-response/{email_id}",
                    "method": "POST"
                }
            ]
        }
        
        # If no templates matched, generate AI response
        if not matched_templates:
            try:
                ai_response = generate_ai_response(
                    email_id, sender, recipient, subject, 
                    body, analysis, 
                    analysis.get("Recipient Information", "Customer"),
                    analysis.get("detected_language", "en")
                )
                response_data["ai_response"] = ai_response
            except Exception as e:
                print(f"Error generating AI response: {str(e)}")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
# New endpoint for AI-generated responses
@router.post("/api/ai-response/{email_id}")
def ai_response(
    email_id: str,
    send_email: bool = False
):
    """Generate an AI response for the specified email in the detected language."""
    conn = None
    cur = None
    email_sent_at = None
    email_sent = False
    message_id = None
    try:
        # Connect to the database
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        # Fetch email data from the database
        query = """
        SELECT sender, recipient, subject, body, analysis_result, email_received_at, detected_language, is_hinglish
        FROM email_analysis
        WHERE email_id = %s
        """
        cur.execute(query, (email_id,))
        result = cur.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail=f"Email with ID {email_id} not found")
        sender, recipient, subject, body_text, analysis_result, email_received_at, detected_language, is_hinglish = result

        # Parse the analysis result
        analysis = {}
        if analysis_result:
            try:
                analysis = json.loads(analysis_result) if isinstance(analysis_result, str) else analysis_result
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Could not parse analysis_result for email_id {email_id}")
                analysis = {}

        # Get recipient name
        # Default to "Customer" if not found or invalid
        recipient_name = analysis.get("Recipient Information", "Customer")
        if not recipient_name or recipient_name in ["Unknown", "Not found"]:
            recipient_name = "Customer"

        # Get language and Hinglish flag
        # Set default values if not found
        language = detected_language or "en"
        is_hinglish = is_hinglish or False

        # Generate AI response in the detected language
        response_data = generate_ai_response(email_id, sender, recipient, subject, body_text, analysis, recipient_name, language, is_hinglish)

        # Check if response_data indicates an error
        if not response_data.get("success"):
            raise HTTPException(status_code=500, detail=response_data.get("error", "Failed to generate AI response"))

        response_subject = response_data["response"]["subject"]
        response_body = response_data["response"]["body"]
        response_language = response_data["response"]["language"]

        # Optionally send the email
        if send_email:
            try:
                send_result = send_email_via_gmail_oauth2(
                    recipient_email=recipient,
                    subject=response_subject,
                    body=response_body,
                    cc=[],
                    bcc=[]
                )
                email_sent = send_result.get("success", False)
                message_id = send_result.get("message_id") 
                if email_sent:
                    email_sent_at = datetime.datetime.now()
                    update_query = """
                        UPDATE email_analysis
                        SET email_sent = %s,
                            email_sent_at = %s,
                            message_id = %s
                        WHERE email_id = %s
                        """
                    cur.execute(update_query, (email_sent, email_sent_at,message_id, email_id))
                    conn.commit()
                    
            except Exception as e:
                print(f"Error sending email: {str(e)}")
                email_sent = False
                email_sent_at = None
                message_id = None

        return {
            "email_id": email_id,
            "response": {
                "subject": response_subject,
                "body": response_body,
                "recipient": recipient,
                "language": response_language
            },
            "metadata": {
                "email_sent": email_sent,
                "response_time": None,
                "email_sent_at": email_sent_at.isoformat() if email_sent_at else None,
                "message_id": message_id,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /api/ai-response: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating AI response: {str(e)}")

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
from deep_translator import GoogleTranslator

def translate_text_sync(text: str, target_lang: str) -> str:
    """Synchronous translation helper function"""
    try:
        if not text or target_lang == 'en':
            return text

        # Create a new translator instance for each translation
        translator = GoogleTranslator(source='en', target=target_lang)
        
        # Perform translation
        translation = translator.translate(text)
        return translation
    except Exception as e:
        print(f"Translation error: {str(e)} - falling back to English")
        return text
@router.post("/api/manual-template-response/{email_id}")
def generate_response_from_template(
    email_id: str,
    body: dict = Body(...),
    send_email: bool = Query(False, description="Whether to send the email immediately")
):
    """Generate a response using a specific template and translate it to Hinglish if detected."""
    conn = None
    cur = None

    try:
        template_path = body.get("template_path")
        if not template_path or not isinstance(template_path, str):
            raise HTTPException(status_code=400, detail="Valid template_path is required")

        path_parts = template_path.split(".")
        if len(path_parts) != 3:
            raise HTTPException(status_code=400, detail="Template path must be in format: category.subcategory.tone")

        category, subcategory, tone = path_parts

        if (category not in RESPONSE_TEMPLATES or
                subcategory not in RESPONSE_TEMPLATES[category] or
                tone not in RESPONSE_TEMPLATES[category][subcategory]):
            raise HTTPException(status_code=404, detail=f"Template not found: {template_path}")

        template_data = RESPONSE_TEMPLATES[category][subcategory][tone]
        custom_replacements = body.get("replacements", {})
        if not isinstance(custom_replacements, dict):
            custom_replacements = {}

        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        # Fetch email details from the database
        cur.execute("""
            SELECT sender, recipient, subject, body, analysis_result, detected_language, is_hinglish
            FROM email_analysis WHERE email_id = %s
        """, (email_id,))
        result = cur.fetchone()

        if not result:
            raise HTTPException(status_code=404, detail=f"Email with ID {email_id} not found")

        sender, recipient, subject, email_body, analysis_result, detected_language, is_hinglish = result

        # Set a default language if none was detected
        if not detected_language:
            detected_language = 'en'

        try:
            analysis = json.loads(analysis_result) if isinstance(analysis_result, str) else analysis_result
        except Exception as e:
            print(f"Error parsing analysis_result: {str(e)}")
            analysis = {}

        # Extract recipient name from analysis
        recipient_name = analysis.get("Recipient Information", "Customer")
        if recipient_name in [None, "", "Unknown", "Not found"]:
            recipient_name = "Customer"

        # Generate the response in English first - only replace customer_name and custom replacements
        response_body_en = template_data["body"].replace("{{customer_name}}", recipient_name)
        for placeholder, value in custom_replacements.items():
            response_body_en = response_body_en.replace(f"{{{{{placeholder}}}}}", value)

        response_subject_en = template_data.get("subject", f"RE: {subject}").replace("{{original_subject}}", subject)

        # Initialize response with English version
        response_body = response_body_en
        response_subject = response_subject_en
        response_language = detected_language

        if is_hinglish:
            # Updated Hinglish conversion approach using a refined prompt for better results
            groq_prompt = f"""
            Convert the following English text to natural Hinglish (mix of Hindi and English). 
            Keep technical terms, brand names, and common English phrases in English.
            Use Hinglish as it's naturally spoken - not formal Hindi with English words.
            
            DO NOT translate to pure Hindi - this must be Hinglish with a natural mix of both languages.
            
            SUBJECT: {response_subject_en}
            
            BODY: {response_body_en}
            
            Format your response as:
            SUBJECT: [Hinglish subject]
            
            BODY: [Hinglish body]
            """
            
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": groq_prompt}],
                "temperature": 0.4,  # Lower temperature for more consistent output
                "max_tokens": 1000
            }
            
            try:
                # Send request to Groq API
                groq_response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                                             headers=headers, 
                                             json=payload, 
                                             timeout=30)
                groq_response.raise_for_status()
                hinglish_response = groq_response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if hinglish_response:
                    # Parse the response to extract subject and body separately
                    lines = hinglish_response.strip().split('\n')
                    
                    # Extract subject - look for "SUBJECT:" prefix
                    subject_line = next((line for line in lines if line.strip().startswith("SUBJECT:")), None)
                    if subject_line:
                        response_subject = subject_line.replace("SUBJECT:", "", 1).strip()
                    else:
                        # Fallback if format wasn't followed
                        response_subject = f"RE: {subject}"
                    
                    # Extract body - everything after "BODY:" 
                    body_start_index = -1
                    for i, line in enumerate(lines):
                        if line.strip().startswith("BODY:"):
                            body_start_index = i
                            break
                    
                    if body_start_index != -1:
                        # Join all lines after body_start_index, removing the "BODY:" prefix
                        body_lines = lines[body_start_index:]
                        if body_lines:
                            body_lines[0] = body_lines[0].replace("BODY:", "", 1).strip()
                        response_body = "\n".join(line for line in body_lines if line.strip())
                    else:
                        # If we can't find the BODY marker, just use everything after the first two lines
                        # (assuming the first two lines are subject related)
                        response_body = "\n".join(lines[2:]) if len(lines) > 2 else hinglish_response
                    
                    response_language = "hinglish"
                    print("Successfully generated Hinglish response using LLM")
                else:
                    print("Failed to generate Hinglish response using LLM, falling back to English.")
            except Exception as e:
                print(f"Error generating Hinglish response using LLM: {e}, falling back to English.")
                
        elif detected_language != 'en':
            # For non-English, non-Hinglish languages, use translation service
            try:
                # Use synchronous translation
                response_body = translate_text_sync(response_body_en, detected_language)
                response_subject = translate_text_sync(response_subject_en, detected_language)
                print(f"Translated response to {detected_language}")
            except Exception as e:
                print(f"Translation error: {str(e)} - falling back to English response")
                response_body = response_body_en
                response_subject = response_subject_en
                response_language = 'en'

        response_data = {
            "subject": response_subject,
            "body": response_body,
            "subject_english": response_subject_en,
            "body_english": response_body_en,
            "recipient": sender,
            "cc": body.get("cc", []),
            "bcc": body.get("bcc", []),
            "source": "manual_template",
            "template_used": template_path,
            "response_language": response_language
        }

        current_time = datetime.datetime.now()

        cur.execute("""
            UPDATE email_analysis
            SET response_subject = %s,
                response_body = %s,
                response_subject_english = %s,
                response_body_english = %s,
                response_generated_at = %s,
                response_source = %s,
                template_used = %s,
                response_language = %s,
                message_id = %s,
                email_sent = %s,
                email_sent_at = %s
            WHERE email_id = %s
        """, (
            response_data["subject"],
            response_data["body"],
            response_data["subject_english"],
            response_data["body_english"],
            current_time,
            response_data["source"],
            response_data["template_used"],
            response_data["response_language"],
            "not_sent",  # Initialize message_id
            False,      # Initialize email_sent
            None,       # Initialize email_sent_at
            email_id
        ))
        conn.commit()

        email_sent = False
        email_sent_at = None
        message_id = "not_sent"

        if send_email:
            try:
                send_result = send_email_via_gmail_oauth2(
                    recipient_email=response_data["recipient"],
                    subject=response_data["subject"],
                    body=response_data["body"],
                    cc=response_data.get("cc"),
                    bcc=response_data.get("bcc")
                )
                email_sent = send_result.get("success", False)
                message_id = send_result.get("message_id", f"failed_to_get_id:{datetime.datetime.now().timestamp()}")

                if email_sent:
                    email_sent_at = datetime.datetime.now()
                    cur.execute("""
                        UPDATE email_analysis
                        SET email_sent = %s,
                            email_sent_at = %s,
                            message_id = %s
                        WHERE email_id = %s
                    """, (email_sent, email_sent_at, message_id, email_id))
                    conn.commit()
            except Exception as e:
                print(f"Error sending email: {str(e)}")
                message_id = f"failed_to_send:{str(e)[:30]}"
        else:
            message_id = "not_requested"

        return {
            "email_id": email_id,
            "response": {
                "subject": response_data["subject"],
                "body": response_data["body"],
                "subject_english": response_data["subject_english"],
                "body_english": response_data["body_english"],
                "recipient": response_data["recipient"],
                "cc": response_data["cc"],
                "bcc": response_data["bcc"],
                "language": response_data["response_language"]
            },
            "metadata": {
                "source": response_data["source"],
                "template_used": response_data["template_used"],
                "response_generated_at": current_time.isoformat(),
                "email_sent": email_sent,
                "email_sent_at": email_sent_at.isoformat() if email_sent_at else None,
                "message_id": message_id
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error generating template response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
def fallback_response(email_id: str, subject: str, recipient_name: str, sender_email: str, error_type: str, details: Optional[str] = None) -> Dict:
    """Fallback response in case of errors."""
    return {
        "email_id": email_id,
        "error": f"AI response generation failed ({error_type}): {details if details else ''}",
        "success": False,
        "response": {
            "subject": f"RE: {subject}",
            "body": f"Dear {recipient_name},\n\nWe encountered an issue while trying to process your request. Please bear with us while we investigate.\n\nSincerely,\nOur Support Team",
            "language": "en"  # Add a default language here
        },
        "metadata": {
            "source": "ai_fallback",
            "response_generated_at": datetime.datetime.now().isoformat(),
            "email_sent": False,
            "email_sent_at": None,
            "message_id": None
        }
    }
def generate_ai_response(email_id: str, sender: str, recipient: str, subject: str, body: str, analysis: Dict, recipient_name: str, language: str, is_hinglish: bool):
    """Generate an AI response to an email, considering the detected language and Hinglish."""
    conn = None
    cur = None
    response_subject = f"RE: {subject}"
    response_body = ""
    response_language = language  # Default to detected language

    try:
        # Validate email_id
        valid_email_id = get_valid_email_id(email_id)
        if not valid_email_id:
            return {"error": f"Could not find a valid email ID for: {email_id}", "success": False}
        email_id = valid_email_id

        # Get email sentiment, category, priority
        sentiment = analysis.get("sentiment", "neutral")
        category = analysis.get("category", "")
        priority = analysis.get("priorityLevel", "medium")

        # Get user history for context
        sender_email = None
        history_context = ""
        try:
            if "<" in sender and ">" in sender:
                sender_email = sender.split("<")[1].split(">")[0]
            else:
                sender_email = sender
            user_history = get_user_history(sender_email)
            if user_history:
                pass
        except Exception as e:
            print(f"Error getting user history: {e}")

        # Prepare the prompt for the LLM based on the detected language and Hinglish
        prompt_instruction = ""
        if is_hinglish:
            prompt_instruction = "Generate a response in Hinglish (Hindi-English mix)."
            response_language = "hinglish"
        elif language.lower() != 'en':
            language_name = {
                "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German", "zh": "Chinese",
                "ja": "Japanese", "ar": "Arabic", "ru": "Russian"
            }.get(language.lower(), language)
            prompt_instruction = f"Generate a response in {language_name}."
        else:
            prompt_instruction = "Generate a response in English."
            response_language = "en"

        prompt = f"""{prompt_instruction}

        Respond to the following email:
        Subject: {subject}
        Body: {body}

        Consider the sentiment: {sentiment}, category: {category}, priority: {priority}.
        Recipient name: {recipient_name}.
        {history_context}

        Format your response as a JSON object with "subject" and "body" keys.
        """

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",  # Or your preferred model
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        response_data_ai = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

        try:
            response_json = json.loads(response_data_ai)
            response_subject = response_json.get("subject", f"RE: {subject}")
            response_body = response_json.get("body", "")

            # Save to database with the determined response language
            conn = psycopg2.connect(**DB_PARAMS)
            cur = conn.cursor()
            current_time = datetime.datetime.now()
            update_query = """
                UPDATE email_analysis
                SET response_subject = %s,
                    response_body = %s,
                    response_generated_at = %s,
                    response_source = %s,
                    message_id = %s,
                    email_sent = FALSE,
                    email_sent_at = NULL,
                    response_language = %s
                WHERE email_id = %s
                """
            cur.execute(update_query, (response_subject, response_body, current_time, "ai_generated", "not_sent", response_language, email_id))
            conn.commit()

            return {
                "email_id": email_id,
                "response": {"subject": response_subject, "body": response_body, "language": response_language},
                "metadata": {"source": "ai_generated", "response_generated_at": current_time.isoformat()},
                "success": True
            }
        except json.JSONDecodeError:
            return fallback_response(email_id, subject, recipient_name, sender_email, "ai_json_error")

    except requests.exceptions.RequestException as e:
        print(f"Error during Groq API request: {e}")
        return fallback_response(email_id, subject, recipient_name, sender_email, "ai_api_error", str(e))
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return fallback_response(email_id, subject, recipient_name, sender_email, "general_error", str(e))
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

# Helper function to get a valid email ID from the database (for other endpoints to use)
def get_valid_email_id(requested_id):
    """Find a valid email ID in the database based on requested ID"""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Try exact match first
        cur.execute("SELECT email_id FROM email_analysis WHERE email_id = %s", (requested_id,))
        result = cur.fetchone()
        
        if result:
            return result[0]
        
        # Try partial match
        cur.execute("SELECT email_id FROM email_analysis WHERE email_id LIKE %s OR %s LIKE CONCAT('%%', email_id, '%%') LIMIT 1", 
                   (f"%{requested_id}%", requested_id))
        result = cur.fetchone()
        
        if result:
            return result[0]
        
        # Try most recent as last resort
        cur.execute("SELECT email_id FROM email_analysis ORDER BY created_at DESC LIMIT 1")
        result = cur.fetchone()
        
        if result:
            return result[0]
            
        return None
    except Exception as e:
        print(f"Error finding valid email ID: {str(e)}")
        return None
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        

# Add this to other endpoint functions that need to handle email_id
# For example:


def send_email_via_gmail_oauth2(recipient_email, subject, body, cc=None, bcc=None):
    """
    Send an email via Gmail API using OAuth2 credentials from environment variables

    Args:
        recipient_email (str): Email address of the recipient
        subject (str): Email subject
        body (str): Email body content
        cc (list, optional): List of CC email addresses
        bcc (list, optional): List of BCC email addresses

    Returns:
        dict: Status of the email sending operation
    """
    try:
        creds = get_gmail_credentials()
        
        # Check if credentials need refreshing
        if creds.expired:
            request = google.auth.transport.requests.Request()
            creds.refresh(request)

        # Build Gmail API service
        service = build('gmail', 'v1', credentials=creds)

        # Create email message
        message = MIMEMultipart()
        message['To'] = recipient_email
        message['Subject'] = subject

        # Set sender email from credentials or configuration
        sender_email = os.getenv("GMAIL_SENDER_EMAIL", "noreply@yourdomain.com")
        message['From'] = sender_email

        # Add CC recipients if provided
        if cc:
            if isinstance(cc, list):
                message['Cc'] = ", ".join(cc)
            else:
                message['Cc'] = cc

        # Add message body - support both plain text and HTML
        if '<html>' in body.lower():
            message.attach(MIMEText(body, 'html'))
        else:
            message.attach(MIMEText(body, 'plain'))

        # Encode the message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        # Create the email request body
        email_request = {
            'raw': encoded_message
        }

        print(f"Sending email to: {recipient_email}")
        # Send the message
        sent_message = service.users().messages().send(userId='me', body=email_request).execute()
        print("Email API call executed successfully")

        # Verify we have a message ID
        if 'id' not in sent_message:
            print("Warning: No message ID returned from Gmail API")
            message_id = f"generated_{int(time.time())}"
        else:
            message_id = sent_message['id']
            print(f"Message ID received: {message_id}")

        return {
            "success": True,
            "message": "Email sent successfully",
            "message_id": message_id
        }

    except HttpError as error:
        error_msg = f"Gmail API error: {str(error)}"
        print(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "message_id": f"api_error_{int(time.time())}"
        }
    except Exception as e:
        error_msg = str(e)
        print(f"Exception during email sending: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": error_msg,
            "message_id": f"error_{int(time.time())}"
        }


# Add a new endpoint to check if an email_id exists in the database
@router.get("/api/check-email/{email_id}")
def check_email_exists(email_id: str):
    """Check if an email with the given ID exists in the database"""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Clean up the email_id
        email_id = email_id.strip()
        
        # First check with exact match
        cur.execute("SELECT email_id FROM email_analysis WHERE email_id = %s", (email_id,))
        exact_match = cur.fetchone()
        
        if exact_match:
            return {
                "exists": True,
                "email_id": exact_match[0],
                "match_type": "exact"
            }
        
        # Try case insensitive match
        cur.execute("SELECT email_id FROM email_analysis WHERE LOWER(email_id) = LOWER(%s)", (email_id,))
        case_insensitive_match = cur.fetchone()
        
        if case_insensitive_match:
            return {
                "exists": True,
                "email_id": case_insensitive_match[0],  # Return the actual ID in the database
                "match_type": "case_insensitive"
            }
        
        return {
            "exists": False,
            "email_id": email_id
        }
    except Exception as e:
        print(f"Error checking email: {str(e)}")
        return {
            "exists": False,
            "error": str(e)
        }
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
# === Get User History ===
def get_user_history(email):
  
    """Get user interaction history"""
    # Check cache first
    if email in user_interaction_cache:
        return user_interaction_cache[email]
    
    # If not in cache, check database
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        query = """
        SELECT interaction_count, categories, sentiment_history, last_interaction
        FROM user_interactions
        WHERE email = %s
        """
        
        cur.execute(query, (email,))
        result = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if result:
            interaction_count, categories, sentiment_history, last_interaction = result
            
            data = {
                "interaction_count": interaction_count,
                "categories": categories,
                "sentiment_history": sentiment_history,
                "last_interaction": last_interaction.isoformat() if isinstance(last_interaction, datetime.datetime) else last_interaction
            }
            
            return json.dumps(data)
        
        return None
    except Exception:
        return None


def offset_to_page_token(offset):
    """
    Convert a numeric offset to a Gmail API page token.
    
    For Gmail API, we need to use their pagination system which uses page tokens
    rather than simple numeric offsets. This function handles that conversion.
    
    Args:
        offset (int): The numeric offset for pagination
        
    Returns:
        str or None: The page token for Gmail API, or None for the first page
    """
   
    if offset == 0:
        return None
@router.get("/api/escalated-emails")
async def get_escalated_emails(
    limit: int = Query(50, description="Maximum number of results"),
    offset: int = Query(0, description="Result offset for pagination")
):
    """
    Get all escalated emails with keywords like 'urgent' directly from Gmail
    and fetch emails from the email_analysis table with priority 'High'.
    """
    try:
        # Initialize result list
        escalated_emails = []

        # === Fetch emails from Gmail ===
        try:
            print("Fetching emails from Gmail...")
            creds = get_gmail_credentials()

            
            gmail_service = build('gmail', 'v1', credentials=creds)

            # Build Gmail search query
            query = "(subject:urgent OR subject:critical OR subject:emergency OR subject:important OR subject:asap)"

            # Execute Gmail search
            response = gmail_service.users().messages().list(
                userId='me',
                q=query,
                maxResults=limit,
                pageToken=offset_to_page_token(offset)
            ).execute()

            messages = response.get('messages', [])
            print(f"Fetched {len(messages)} messages from Gmail")
            for message in messages:
                msg_id = message['id']
                msg_data = gmail_service.users().messages().get(
                    userId='me',
                    id=msg_id,
                    format='full'
                ).execute()

                # Extract and process email data
                email_details = parse_gmail_message(msg_data)
                email_details['source'] = 'Gmail'
                escalated_emails.append(email_details)

        except Exception as gmail_error:
            print(f"Error fetching emails from Gmail: {str(gmail_error)}")

        # === Fetch emails from the database ===
        try:
            print("Fetching emails from the database...")
            limit = int(limit)  # Ensure limit is an integer
            offset = int(offset)  # Ensure offset is an integer

            conn = psycopg2.connect(**DB_PARAMS)
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            # Query to fetch emails with priority 'High' from the email_analysis table
            db_query = """
                SELECT email_id, sender, recipient, subject, body, priority, is_escalated, escalation_reason, created_at
                FROM email_analysis
                WHERE priority = 'High'
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            cur.execute(db_query, (limit, offset))
            db_emails = cur.fetchall()
            print(f"Fetched {len(db_emails)} emails from the database")

            for db_email in db_emails:
                email_details = {
                    "id": db_email["email_id"],
                    "sender": db_email["sender"],
                    "recipient": db_email["recipient"],
                    "subject": db_email["subject"],
                    "body": db_email["body"],
                    "priority": db_email["priority"],
                    "is_escalated": db_email["is_escalated"],
                    "escalation_reason": db_email["escalation_reason"],
                    "created_at": db_email["created_at"].isoformat() if db_email["created_at"] else None,
                    "source": "Database"
                }
                escalated_emails.append(email_details)

            cur.close()
            conn.close()

        except Exception as db_error:
            print(f"Error fetching emails from the database: {str(db_error)}")

        # Combine and return results
        return {
            "status": "success",
            "total": len(escalated_emails),
            "offset": offset,
            "limit": limit,
            "escalated_emails": escalated_emails
        }

    except Exception as e:
        print(f"Error in /api/escalated-emails: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to fetch escalated emails: {str(e)}"
        }

def parse_gmail_message(msg_data):
    """
    Parse a Gmail message into a structured format.
    
    Args:
        msg_data (dict): The Gmail message data from the API
        
    Returns:
        dict: Structured email details
    """
    # Initialize an empty dictionary for email details
    email_details = {
        'id': msg_data['id'],
        'thread_id': msg_data['threadId'],
        'subject': '',
        'body': '',
        'sender': '',
        'recipient': '',
        'date': '',
        'attachments': []
    }
    
    # Get headers
    headers = msg_data['payload']['headers']
    for header in headers:
        name = header['name'].lower()
        value = header['value']
        
        if name == 'subject':
            email_details['subject'] = value
        elif name == 'from':
            email_details['sender'] = value
        elif name == 'to':
            email_details['recipient'] = value
        elif name == 'date':
            email_details['date'] = value
    
    # Extract body
    if 'parts' in msg_data['payload']:
        # Handle multipart messages
        for part in msg_data['payload']['parts']:
            # Check if this part is the email body (text or HTML)
            if part.get('mimeType') == 'text/plain':
                body_data = part.get('body', {}).get('data', '')
                if body_data:
                    email_details['body'] = decode_base64_text(body_data)
                break
            
            # If no plain text, try HTML
            elif part.get('mimeType') == 'text/html' and not email_details['body']:
                body_data = part.get('body', {}).get('data', '')
                if body_data:
                    email_details['body'] = decode_base64_text(body_data)
            
            # Check for attachments
            elif 'filename' in part and part.get('filename'):
                attachment = {
                    'filename': part.get('filename'),
                    'mimeType': part.get('mimeType'),
                    'size': part.get('body', {}).get('size', 0),
                    'attachment_id': part.get('body', {}).get('attachmentId', '')
                }
                email_details['attachments'].append(attachment)
    else:
        # Handle single part message
        body_data = msg_data['payload'].get('body', {}).get('data', '')
        if body_data:
            email_details['body'] = decode_base64_text(body_data)
    
    return email_details
#Decodes Base64-encoded text, typically used for decoding email content retrieved from the Gmail AP
def decode_base64_text(encoded_text):
    """
    Decode base64 encoded text from Gmail API.
    
    Args:
        encoded_text (str): Base64 encoded text
        
    Returns:
        str: Decoded text
    """
    # Gmail API uses URL-safe base64 encoding with padding removed
    # Add padding back if needed
    padding_needed = len(encoded_text) % 4
    if padding_needed:
        encoded_text += '=' * (4 - padding_needed)
    
    # Decode from base64
    try:
        decoded_bytes = base64.urlsafe_b64decode(encoded_text)
        return decoded_bytes.decode('utf-8')
    except Exception as e:
        return f"[Error decoding message: {str(e)}]"


# === Feedback Mechanism ===
@router.post("/api/feedback/{message_id}")
def add_feedback(message_id: str, feedback: FeedbackModel):
    """Add feedback for an automated message response."""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Check if message exists
        cur.execute("SELECT email_id FROM email_analysis WHERE message_id = %s", (message_id,))
        email_record = cur.fetchone()
        if not email_record:
            raise HTTPException(status_code=404, detail=f"No email found for message_id: {message_id}")
        
        email_id = email_record[0]
        
        # Get current timestamp
        current_time = datetime.datetime.now()
        
        # Insert or update feedback with created_at timestamp
        cur.execute("""
            INSERT INTO response_feedback (message_id, email_id, rating, comment, improvement_suggestion, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (message_id) DO UPDATE
            SET rating = EXCLUDED.rating,
                comment = EXCLUDED.comment,
                improvement_suggestion = EXCLUDED.improvement_suggestion,
                updated_at = NOW()
        """, (message_id, email_id, feedback.rating, feedback.comment, feedback.improvement_suggestion, current_time, current_time))
        conn.commit()
        
        # Update feedback rating in email_analysis
        cur.execute("""
            UPDATE email_analysis
            SET feedback_rating = %s
            WHERE message_id = %s
        """, (feedback.rating, message_id))
        conn.commit()
        
        return {
            "status": "success",
            "message": "Feedback recorded successfully",
            "message_id": message_id,
            "created_at": current_time.isoformat()
        }
    except Exception as e:
        print(f"Error in add_feedback: {str(e)}")
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
    
@router.get("/api/email-detail/{email_id}")
def get_email_detail(
    email_id: str = Path(..., description="Email ID to analyze")
):
    """Generate detailed analytics for a single email including status, sentiment, response details, and feedback"""
    conn = None
    cur = None
    
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Fetch email details from the database
        query = """
        SELECT email_id, sender, subject, body, email_received_at, email_sent_at, 
               response_subject, response_body, response_generated_at, sentiment, priority, 
               category, is_escalated, escalation_reason, feedback_rating, message_id
        FROM email_analysis
        WHERE email_id = %s
        """
        cur.execute(query, (email_id,))
        result = cur.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Email with ID {email_id} not found")
        
        # Get message_id for feedback lookup - use the one from the record
        message_id = result["message_id"]
        
        # FIX: If message_id is None, we can't properly link to feedback
        if not message_id:
            print(f"Warning: Email {email_id} has no message_id, feedback cannot be retrieved")
            feedback_result = None
        else:
            # Lookup feedback using message_id
            feedback_query = """
            SELECT rating, comment, improvement_suggestion, created_at, updated_at
            FROM response_feedback
            WHERE message_id = %s
            ORDER BY updated_at DESC NULLS LAST, created_at DESC
            LIMIT 1
            """
            cur.execute(feedback_query, (message_id,))
            feedback_result = cur.fetchone()
            
            # FIX: If no feedback found but the email has a feedback_rating, 
            # check if feedback exists with a different message_id format
            if not feedback_result and result["feedback_rating"] is not None:
                print(f"Trying alternative lookup for feedback with email_id {email_id}")
                alt_feedback_query = """
                SELECT rf.rating, rf.comment, rf.improvement_suggestion, rf.created_at, rf.updated_at
                FROM response_feedback rf
                JOIN email_analysis ea ON (ea.email_id = %s)
                WHERE rf.rating = ea.feedback_rating
                ORDER BY rf.updated_at DESC NULLS LAST, rf.created_at DESC
                LIMIT 1
                """
                cur.execute(alt_feedback_query, (email_id,))
                feedback_result = cur.fetchone()
        
        # Calculate response time
        response_time = None
        if result["email_received_at"] and result["response_generated_at"]:
            response_time = (result["response_generated_at"] - result["email_received_at"]).total_seconds()
        
        # Calculate total duration
        total_duration = None
        if result["email_received_at"] and result["email_sent_at"]:
          total_duration_seconds = (result["email_sent_at"] - result["email_received_at"]).total_seconds()
          total_duration = {
        "seconds": total_duration_seconds,
        "formatted": format_time_duration(total_duration_seconds)
    }
        
        # Format the response
        email_details = {
            "email_id": result["email_id"],
            "sender": result["sender"],
            "subject": result["subject"],
            "body": result["body"],
            "email_received_at": result["email_received_at"].isoformat() if result["email_received_at"] else None,
            "email_sent_at": result["email_sent_at"].isoformat() if result["email_sent_at"] else None,
            "category": result["category"],
            "priority": result["priority"],
            "is_escalated": result["is_escalated"],
            "escalation_reason": result["escalation_reason"],
            "message_id": message_id
        }
        
        response_details = {
            "response_subject": result["response_subject"],
            "response_body": result["response_body"],
            "response_generated_at": result["response_generated_at"].isoformat() if result["response_generated_at"] else None,
            "response_time_seconds": response_time,
            "response_time_formatted": format_time_duration(response_time) if response_time else None
        }
        
        sentiment_info = {
            "initial_sentiment": result["sentiment"],
            "feedback_rating": result["feedback_rating"]
        }
        
        # Feedback information
        feedback_info = {"has_feedback": False}
        if feedback_result:
            last_updated = feedback_result["updated_at"] if feedback_result["updated_at"] else feedback_result["created_at"]
            feedback_info = {
                "has_feedback": True,
                "rating": feedback_result["rating"],
                "comment": feedback_result["comment"],
                "improvement_suggestion": feedback_result["improvement_suggestion"],
                "submitted_at": feedback_result["created_at"].isoformat() if feedback_result["created_at"] else None,
                "last_updated": last_updated.isoformat() if last_updated else None
            }
        elif result["feedback_rating"] is not None:
            # FIX: If we have a feedback_rating in email_analysis but no feedback record,
            # at least show the rating
            feedback_info = {
                "has_feedback": True,
                "rating": result["feedback_rating"],
                "comment": None,
                "improvement_suggestion": None,
                "submitted_at": None,
                "last_updated": None,
                "note": "Rating available but detailed feedback not found"
            }
        
        # Audit trail
        audit_trail = []
        if result["email_received_at"]:
            audit_trail.append({
                "action": "Email received",
                "timestamp": result["email_received_at"].isoformat(),
                "details": "Email was received in the system"
            })
        if result["response_generated_at"]:
            audit_trail.append({
                "action": "Response generated",
                "timestamp": result["response_generated_at"].isoformat(),
                "details": "Response was generated"
            })
        if result["email_sent_at"]:
            audit_trail.append({
                "action": "Response sent",
                "timestamp": result["email_sent_at"].isoformat(),
                "details": "Response was sent to the recipient"
            })
        
        # If feedback exists, add it to audit trail
        if feedback_info["has_feedback"]:
            if feedback_result and feedback_result["created_at"]:
                audit_trail.append({
                    "action": "Feedback submitted",
                    "timestamp": feedback_result["created_at"].isoformat(),
                    "details": f"Feedback rating: {feedback_info['rating']}/5"
                })
                
                # If feedback was updated, add that too
                if feedback_result["updated_at"] and feedback_result["updated_at"] != feedback_result["created_at"]:
                    audit_trail.append({
                        "action": "Feedback updated",
                        "timestamp": feedback_result["updated_at"].isoformat(),
                        "details": f"Feedback rating updated: {feedback_info['rating']}/5"
                    })
            elif result["feedback_rating"] is not None:
                # Add a generic entry for feedback if we only have the rating
               
                audit_trail.append({
                    "action": "Feedback submitted",
                    "timestamp": result["email_sent_at"].isoformat() if result["email_sent_at"] else 
                                 result["response_generated_at"].isoformat() if result["response_generated_at"] else 
                                 datetime.datetime.now().isoformat(),
                    "details": f"Feedback rating: {result['feedback_rating']}/5"
                })
        
        # Sort audit trail by timestamp
        audit_trail.sort(key=lambda x: x["timestamp"])
        
        return {
            "status": "success",
            "email_details": email_details,
            "response_details": response_details,
            "sentiment_info": sentiment_info,
            "feedback_info": feedback_info,
            "audit_trail": audit_trail,
            "total_duration": total_duration
        }
    
    except Exception as e:
        print(f"Error fetching email details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
def format_time_duration(seconds):
    """Format seconds into a human-readable duration"""
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{int(minutes)} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{int(hours)} hours"
    else:
        days = seconds / 86400
        return f"{int(days)} days"

# 

# Security scheme for JWT



security = HTTPBearer()

# Updated LoginModel to use username instead of email
class LoginModel(BaseModel):
    username: str  # Changed from email to username
    password: str

# Authentication functions
def verify_token(token: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify the token provided in the Authorization header.
    Returns the username from the token if valid.
    """
    try:
        # Verify token with Keycloak
        introspect_url = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token/introspect"
        
        payload = {
            "token": token.credentials,
            "client_id": KEYCLOAK_CLIENT_ID,
            "client_secret": KEYCLOAK_CLIENT_SECRET
        }
        
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(introspect_url, data=payload, headers=headers, verify=False)
        
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Failed to validate token")
        
        token_info = response.json()
        
        if not token_info.get("active", False):
            raise HTTPException(status_code=401, detail="Token is inactive or expired")
        
        # Decode token to get user information
        decoded_token = jwt.decode(
            token.credentials,
            options={"verify_signature": False}
        )
        
        # Return the username from the token
        return decoded_token.get('preferred_username', 'unknown')
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")

# Updated authentication functions for Keycloak roles
def verify_admin_token(token: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify the token and check if the user has the Global role in GenAI client for admin access.
    Returns the username if the user is an admin.
    """
    try:
        # Keycloak token validation
        introspect_url = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token/introspect"
        
        payload = {
            "token": token.credentials,
            "client_id": KEYCLOAK_CLIENT_ID,
            "client_secret": KEYCLOAK_CLIENT_SECRET
        }
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        response = requests.post(introspect_url, data=payload, headers=headers, verify=False)
        
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Failed to validate token")
        
        token_info = response.json()
        
        # Check if token is active
        if not token_info.get("active", False):
            raise HTTPException(status_code=401, detail="Token is inactive or expired")
        
        # Decode token to get roles (without verification as we just verified with introspection)
        decoded_token = jwt.decode(
            token.credentials,
            options={"verify_signature": False}
        )
        
        # Check client-level role first (preferred approach)
        resource_access = decoded_token.get('resource_access', {})
        client_roles = resource_access.get(KEYCLOAK_CLIENT_ID, {}).get('roles', [])
        
        # Check for the "Global" role in client roles
        if "Global" in client_roles:
            return decoded_token.get('preferred_username')
            
        # Fallback: check realm roles if client roles check fails
        realm_roles = decoded_token.get('realm_access', {}).get('roles', [])
        if "Global" in realm_roles:
            return decoded_token.get('preferred_username')
            
        raise HTTPException(status_code=403, detail="Admin access required")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")

def verify_ticket_access(ticket_id: str, token: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify if the user has access to the specified ticket.
    Users with 'Global' role in GenAI client can access all tickets.
    Regular users can only access tickets they created or are assigned to.
    """
    try:
        # Verify token with Keycloak
        introspect_url = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token/introspect"
        
        payload = {
            "token": token.credentials,
            "client_id": KEYCLOAK_CLIENT_ID,
            "client_secret": KEYCLOAK_CLIENT_SECRET
        }
        
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(introspect_url, data=payload, headers=headers, verify=False)
        
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Failed to validate token")
        
        token_info = response.json()
        
        if not token_info.get("active", False):
            raise HTTPException(status_code=401, detail="Token is inactive or expired")
        
        # Decode token to get user information
        decoded_token = jwt.decode(
            token.credentials,
            options={"verify_signature": False}
        )
        
        user_email = decoded_token.get('email')
        
        # Check for Global role in GenAI client roles (primary approach)
        resource_access = decoded_token.get('resource_access', {})
        client_roles = resource_access.get(KEYCLOAK_CLIENT_ID, {}).get('roles', [])
        
        # Check realm roles as fallback
        realm_roles = decoded_token.get('realm_access', {}).get('roles', [])
        
        # Check if user has Global role for admin access
        if "Global" in client_roles or "Global" in realm_roles:
            return decoded_token.get('preferred_username')  # Admin users can view all tickets
        
        # If not admin, check if user has access to this specific ticket
        assigned_to, sender = get_ticket_assignment_and_sender(ticket_id)
        
        if user_email == assigned_to or user_email == sender:
            return decoded_token.get('preferred_username')
        
        raise HTTPException(status_code=403, detail="Not authorized to view this ticket")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")

# Example helper function to check user roles
def get_user_roles(token: HTTPAuthorizationCredentials = Depends(security)):
    """
    Get user roles from the token for role-based access control.
    Returns a dictionary with username and roles.
    """
    try:
        # Verify token with Keycloak
        introspect_url = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token/introspect"
        
        payload = {
            "token": token.credentials,
            "client_id": KEYCLOAK_CLIENT_ID,
            "client_secret": KEYCLOAK_CLIENT_SECRET
        }
        
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(introspect_url, data=payload, headers=headers, verify=False)
        
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Failed to validate token")
        
        token_info = response.json()
        
        if not token_info.get("active", False):
            raise HTTPException(status_code=401, detail="Token is inactive or expired")
        
        # Decode token to get user information
        decoded_token = jwt.decode(
            token.credentials,
            options={"verify_signature": False}
        )
        
        username = decoded_token.get('preferred_username')
        
        # Get client roles
        resource_access = decoded_token.get('resource_access', {})
        client_roles = resource_access.get(KEYCLOAK_CLIENT_ID, {}).get('roles', [])
        
        # Get realm roles
        realm_roles = decoded_token.get('realm_access', {}).get('roles', [])
        
        # Combine all roles
        all_roles = set(client_roles + realm_roles)
        
        return {
            "username": username,
            "roles": list(all_roles),
            "is_admin": "Global" in all_roles
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")
# Helper function
def get_ticket_assignment_and_sender(ticket_id: str):
    """
    Retrieves who is assigned to a ticket and who sent it, based on the ticket ID.
    
    Args:
        ticket_id (str): The ID of the ticket to check
        
    Returns:
        tuple: (assigned_to, sender) - email addresses of the agent assigned to the ticket
               and the original sender of the ticket
    """
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        query = """
            SELECT assigned_to, sender
            FROM email_analysis
            WHERE email_id = %s;
        """
        cur.execute(query, (ticket_id,))
        result = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if result:
            return result[0], result[1]  # assigned_to, sender
        return None, None
    except Exception as e:
        print(f"Error fetching ticket assignment: {e}")
        return None, None



# API Endpoints with Keycloak Authentication

@router.get("/api/tickets/{ticket_id}", response_model=TicketDetail)
def get_ticket_detail(
    ticket_id: str = Path(..., title="The ID of the ticket to retrieve"),
    token: HTTPAuthorizationCredentials = Depends(security)
):
    """Retrieve details for a specific ticket (accessible to all authenticated users)."""
    username = verify_token(token)
    
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Updated query to select correct fields in correct order
        query = """
            SELECT 
                email_id,
                subject,
                COALESCE(ticket_status, 'new') as status,
                COALESCE(priority, 'medium') as priority,
                sender as customer_name,
                body,
                COALESCE(assigned_to, NULL) as assigned_to
            FROM email_analysis
            WHERE email_id = %s;
        """
        cur.execute(query, (ticket_id,))
        result = cur.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Ticket with ID {ticket_id} not found")
        
        # Create TicketDetail object with proper field mapping
        ticket_detail = TicketDetail(
            ticket_id=result[0],
            subject=result[1],
            status=result[2],
            priority=result[3],
            customer_name=result[4],
            body=result[5],
            assigned_to=result[6] if result[6] else None  # Convert None to null for optional field
        )
        
        return ticket_detail
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving ticket {ticket_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve ticket: {str(e)}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
@router.put("/api/admin/update-ticket/{email_id}")
def update_ticket(
    email_id: str,
    update_data: TicketUpdate,
    admin_username: str = Depends(verify_admin_token)
):
    """Update ticket status, assignment, or priority - admin only and send assignment email."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        # Check if ticket exists and get current ticket info
        check_query = "SELECT email_id, subject, body FROM email_analysis WHERE email_id = %s"
        cur.execute(check_query, (email_id,))
        ticket_info = cur.fetchone()

        if not ticket_info:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail=f"Email ticket {email_id} not found")
        
        ticket_id, ticket_subject, ticket_body = ticket_info

        # Build update query dynamically based on provided fields
        update_query = "UPDATE email_analysis SET "
        update_parts = []
        params = []
        assigned_to_email = None  # To store the assigned email if it's being updated
        status_change = None  # To track status changes

        if update_data.ticket_status is not None:
            update_parts.append("ticket_status = %s")
            params.append(update_data.ticket_status)
            status_change = update_data.ticket_status

        if update_data.assigned_to is not None:
            update_parts.append("assigned_to = %s")
            params.append(update_data.assigned_to)
            assigned_to_email = update_data.assigned_to

        if update_data.priority is not None:
            update_parts.append("priority = %s")
            params.append(update_data.priority)

        # Add updated_at timestamp
        update_parts.append("updated_at = %s")
        params.append(datetime.datetime.now())

        # If no fields to update, return early
        if not update_parts:
            cur.close()
            conn.close()
            return {"status": "no_change", "message": "No fields to update"}

        # Complete query
        update_query += ", ".join(update_parts)
        update_query += " WHERE email_id = %s"
        params.append(email_id)

        # Execute update
        cur.execute(update_query, params)
        conn.commit()

        cur.close()
        conn.close()

         # Send assignment notification email if the ticket was assigned
        if assigned_to_email:
            subject = f"Ticket #{email_id} Assigned to You"
            body = f"""
            Hi,

            Ticket #{email_id} has been assigned to you for further action.

            You can view the ticket details in the system.

            Best regards,
            The Support Team
            """
            send_email_via_gmail_oauth2(assigned_to_email, subject, body)

        return {
            "status": "success",
            "message": f"Ticket {email_id} updated successfully",
            "updates": {k: v for k, v in update_data.dict().items() if v is not None}
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Log the error
        print(f"Error updating ticket: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Ticket update failed: {str(e)}"
        )
@router.get("/api/admin/dashboard")
def admin_dashboard(
    data_type: str = Query("tickets", description="Type of data to fetch (tickets, feedback, stats)"),
    status: str = Query(None, description="Filter by status"),
    priority: str = Query(None, description="Filter by priority"),
    category: str = Query(None, description="Filter by category"),
    #sort_by: str = Query(None, description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order (asc or desc)"),
    limit: int = Query(50, description="Maximum number of results"),
    offset: int = Query(0, description="Result offset for pagination"),
    admin_username: str = Depends(verify_admin_token)
):
    """
    Consolidated admin dashboard endpoint that handles different types of data based on the data_type parameter.
    Supports tickets, feedback, and statistical data.
    """
    try:
        # Dispatch to appropriate handler based on data_type
        if data_type == "tickets":
            return get_tickets_data(status, priority, category, limit, offset)
        elif data_type == "feedback":
            return get_feedback_data(sort_order, limit, offset)
        elif data_type == "stats":
            return get_dashboard_stats()
        else:
            raise HTTPException(status_code=400, detail=f"Invalid data_type: {data_type}")
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Log the error
        print(f"Error in admin dashboard: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Dashboard operation failed: {str(e)}"
        )

# Helper functions for the admin dashboard
def get_tickets_data(status, priority, category, limit, offset):
    """Helper function to fetch ticket data with filters"""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    query = """
    SELECT 
        email_id, sender, subject,
        category, sentiment, priority,
        is_escalated, escalation_reason, 
        ticket_status, assigned_to, created_at,
        updated_at, response_generated_at
    FROM 
        email_analysis
    WHERE 1=1
    """
    
    params = []
    
    if status:
        query += " AND ticket_status = %s"
        params.append(status)
    
    if priority:
        query += " AND priority = %s"
        params.append(priority)
    
    if category:
        query += " AND category = %s"
        params.append(category)
    
    query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
    params.append(limit)
    params.append(offset)
    
    cur.execute(query, params)
    results = cur.fetchall()
    
    columns = [desc[0] for desc in cur.description]
    tickets = []
    
    for row in results:
        ticket_dict = dict(zip(columns, row))
        
        # Convert datetime objects to strings
        for key, value in ticket_dict.items():
            if isinstance(value, datetime.datetime):
                ticket_dict[key] = value.isoformat()
        
        tickets.append(ticket_dict)
    
    # Get total count
    count_query = query.split("ORDER BY")[0]
    count_query = f"SELECT COUNT(*) FROM ({count_query}) as count_query"
    
    cur.execute(count_query, params[:-2])
    total_count = cur.fetchone()[0]
    
    cur.close()
    conn.close()
    
    return {
        "status": "success",
        "data_type": "tickets",
        "total": total_count,
        "offset": offset,
        "limit": limit,
        "tickets": tickets
    }

def get_feedback_data(sort_order, limit, offset):
    """Helper function to fetch feedback data"""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Validate sorting
    valid_sort_orders = ["asc", "desc"]
    if sort_order.lower() not in valid_sort_orders:
        sort_order = "desc"

    # Query response_feedback table - Add a column to sort by
    query = f"""
        SELECT email_id, rating, comment, improvement_suggestion, created_at, updated_at
        FROM response_feedback
        ORDER BY created_at {sort_order}  
        LIMIT %s OFFSET %s
    """
    cur.execute(query, (limit, offset))
    rows = cur.fetchall()

    # Rest of your function remains the same
    cur.execute("SELECT COUNT(*) FROM response_feedback")
    total_count = cur.fetchone()[0]

    cur.execute("SELECT AVG(rating) FROM response_feedback")
    avg_rating = cur.fetchone()[0]

    feedback_list = []
    for row in rows:
        feedback_list.append({
            "email_id": row["email_id"],
            "rating": row["rating"],
            "comment": row["comment"],
            "improvement_suggestion": row["improvement_suggestion"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None
        })

    cur.close()
    conn.close()

    return {
        "status": "success",
        "data_type": "feedback",
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "average_rating": round(avg_rating, 1) if avg_rating else 0,
        "feedback": feedback_list
    }
def get_dashboard_stats():
    """Helper function to fetch simplified ticket statistics with satisfaction score"""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    # Time periods
    today = datetime.datetime.now().date()
    yesterday = today - datetime.timedelta(days=1)
    last_week = today - datetime.timedelta(days=7)
    
    # Get today's tickets
    cur.execute("SELECT COUNT(*) FROM email_analysis WHERE DATE(created_at) = %s", (today,))
    today_count = cur.fetchone()[0]
    
    # Get yesterday's tickets
    cur.execute("SELECT COUNT(*) FROM email_analysis WHERE DATE(created_at) = %s", (yesterday,))
    yesterday_count = cur.fetchone()[0]
    
    # Calculate percent change
    percent_change = 0
    if yesterday_count > 0:
        percent_change = ((today_count - yesterday_count) / yesterday_count) * 100
    
    # Get ticket status counts
    cur.execute("""
        SELECT 
            ticket_status,
            COUNT(*) as count
        FROM email_analysis
        GROUP BY ticket_status
    """)
    status_results = cur.fetchall()
    
    total_tickets = sum(count for _, count in status_results)
    status_counts = {}
    for status, count in status_results:
        status_counts[status] = count
    
    # Get unresolved High priority tickets
    cur.execute("""
        SELECT COUNT(*) FROM email_analysis
        WHERE priority IN ('High', 'critical')
        AND ticket_status NOT IN ('resolved', 'closed')
    """)
    High_priority_count = cur.fetchone()[0]
    
    # Get aging ticket stats - tickets older than 24 hours that aren't resolved
    cur.execute("""
        SELECT COUNT(*) FROM email_analysis
        WHERE created_at < %s
        AND ticket_status NOT IN ('resolved', 'closed')
    """, (today - datetime.timedelta(hours=24),))
    aging_tickets = cur.fetchone()[0]
    
    # Get satisfaction score from feedback ratings
    cur.execute("""
        SELECT 
            AVG(rating) as avg_rating,
            COUNT(*) as total_ratings
        FROM response_feedback
        WHERE created_at >= %s
    """, (last_week,))
    feedback_result = cur.fetchone()
    
    avg_rating = 0
    total_ratings = 0
    if feedback_result[0] is not None:
        avg_rating = round(float(feedback_result[0]), 1)
        total_ratings = feedback_result[1]
    
    # Calculate satisfaction rate (percentage of ratings that are 4 or Higher)
    cur.execute("""
        SELECT 
            SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as satisfied,
            COUNT(*) as total
        FROM response_feedback
        WHERE created_at >= %s
    """, (last_week,))
    satisfaction_result = cur.fetchone()
    
    satisfaction_rate = 0
    if satisfaction_result[1] > 0:  # If there's any feedback
        satisfaction_rate = round((satisfaction_result[0] / satisfaction_result[1]) * 100, 1)
    
    cur.close()
    conn.close()
    
    return {
        "status": "success",
        "today_count": today_count,
        "percent_change_from_yesterday": round(percent_change, 1),
        "total_tickets": total_tickets,
        "ticket_status": status_counts,
        "High_priority_unresolved": High_priority_count,
        "aging_tickets": aging_tickets,
        "satisfaction": {
            "avg_rating": avg_rating,
            "total_ratings": total_ratings,
            "satisfaction_rate": satisfaction_rate  # Percentage of ratings 4 or Higher
        }
    }

# Add a login endpoint to get a token
@router.post("/auth/login")
async def login(username: str, password: str):
    """
    Login endpoint to authenticate users and get a Keycloak token.
    Returns only the access token without the refresh token.
    """
    try:
        # Get token from Keycloak
        token_url = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"
        payload = {
            "grant_type": "password",
            "client_id": KEYCLOAK_CLIENT_ID,
            "client_secret": KEYCLOAK_CLIENT_SECRET,
            "username": username,
            "password": password
        }
        
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(token_url, data=payload, headers=headers, verify=False)
        
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Get token data
        token_data = response.json()
        
        return {
            "access_token": token_data["access_token"],
            "token_type": "bearer",
            "expires_in": token_data["expires_in"]
            # refresh_token removed as requested
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")


def create_tables():
    """Create necessary database tables if they don't exist"""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    # Create email_analysis table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS email_analysis (
        email_id VARCHAR(255) PRIMARY KEY,
        thread_id VARCHAR(255),
        sender VARCHAR(255),
        recipient VARCHAR(255),
        subject TEXT,
        body TEXT,
        category VARCHAR(100),
        subcategory VARCHAR(100),
        sentiment VARCHAR(50),
        priority VARCHAR(50),
        escalation_flag BOOLEAN DEFAULT FALSE,
        escalation_reason TEXT,
        recipient_name VARCHAR(255),
        language VARCHAR(10),
        ticket_status VARCHAR(50) DEFAULT 'new',
        assigned_to VARCHAR(100),
        flag_for_review BOOLEAN DEFAULT FALSE,
        response_time FLOAT,
        analysis_result JSONB,
        created_at TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Create email_responses table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS email_responses (
        id SERIAL PRIMARY KEY,
        email_id VARCHAR(255) REFERENCES email_analysis(email_id),
        response_subject TEXT,
        response_body TEXT,
        sent_at TIMESTAMP,
        UNIQUE(email_id)
    )
    """)
    
    # Create response_feedback table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS response_feedback (
        id SERIAL PRIMARY KEY,
        email_id VARCHAR(255) REFERENCES email_analysis(email_id),
        rating INTEGER,
        comment TEXT,
        improvement_suggestion TEXT,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        UNIQUE(email_id)
    )
    """)
    
    # Create user_interactions table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_interactions (
        email VARCHAR(255) PRIMARY KEY,
        interaction_count INTEGER DEFAULT 0,
        categories JSONB DEFAULT '{}'::jsonb,
        sentiment_history JSONB DEFAULT '[]'::jsonb,
        last_interaction TIMESTAMP
    )
    """)

    conn.commit()
    cur.close()
    conn.close()

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)