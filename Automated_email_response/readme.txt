Automated Email Response Management System
==========================================

Project Description
-------------------
This project is an Automated Email Response Management (ERM) System built with FastAPI, PostgreSQL, and integration with Gmail and Groq LLMs. The system fetches emails, categorizes them using AI, generates intelligent responses based on templates or LLMs, and provides analytics and feedback mechanisms for improving customer service operations.

Key Features
------------
1. **Email Fetching**
   - Fetches emails from Gmail using filters (subject, unread, spam, date, etc.)
   - Supports IMAP or Gmail API-based access

2. **Email Analysis**
   - Sentiment Analysis (Positive, Neutral, Negative)
   - Email Categorization (Complaint, Request, Inquiry, Feedback, Escalation)
   - Priority Detection (High, Medium, Low)
   - Summary Generation (TL;DR)
   - Language detection

3. **Automated Response Generation**
   - Uses structured templates categorized by tone and context
   - Falls back to Groq LLM for custom responses when no template is found

4. **Admin & User Dashboards**
   - Admin access to all user activity, system performance, and analytics
   - User-level email processing, insights, and templates details

5. **Analytics & Reports : Email details**
   - Tracks response time, escalation rate, and sentiment trends, priority ,category ,response etc.
   

6. **Feedback Loop**
   - Users can rate responses for continuous improvement

7. **Multilingual Support**
   - Translates responses to the detected email language in which email is received 

8. **Authentication & Roles**
   - Secure login system with role-based access (admin/user) using keycloack 
   - JWT token-based authentication

Technology Stack
----------------
- **Backend**: FastAPI
- **Database**: PostgreSQL
- **AI/LLM**: Groq LLM 
- **Email Integration**: Gmail API / 
- **Language Translation**: Google Translate API / LangChain tools
- **Authentication**: Using keycloack
- **Language**: Python

----------PROJECT SETUP FILES ----------------
1)**** email.py****
 PURPOSE:The main entry point of the project which handles all the endpoints operations

 -GMAILAPI:Uses gmail credentials(token.json or gmail.json{})
 -Database: Connects to a PostgreSQL database using DB_PARAMS
 -Response Templates: Loads response_templates.json for predefined email responses.

 OUTPUT:
-API Responses: Returns JSON responses for API endpoints.
-Email Sending: Sends emails via Gmail API if requested.
-Database Updates and stored data 
-perform all the operations of endpoints 


2)***Tokenjson.py
-Handles OAuth2 flow and saves credentials in JSON format.
-It generates : token.json file from which main information is  extracted and  is  created in proper format in the form of Gmail.json{}.

3) ***gmail_service.py****
_This file does not create the OAuth2 flow but uses the token.json file (created by tokenjson.py) to authenticate and create a Gmail API service object.

4) Credentials.json{}: It is generated while creating GmailAPI using googlecloud


All these json credentials are in .env format 


------Project setup--------
1) Create project in google cloud in terms of  gmail API which result in creation of credential.json file 

2) Create token.json file in order to create the token.json file with all the credentials related to  Gmail API-related data.and from this main information is extracted and a new gmail.json file is created from it which is used in main.py file for all the operation related to gmailAPI such as Stores Gmail API credentials  for reuse

3)Create gmail_service.py file which is used to authenticate with the Gmail API and create a Gmail API service object that allows your application to interact with Gmail. This service object is essential for performing operations like sending emails, reading emails, and managing Gmail messages.

4)Create a main.py file which includes above credentials to operate and perform all the endpoints operations:
-GROQ_API KEY   : groqAPI key
-DB_PARAMS      : database configuration
-CREDENTIAL_FILE: gmail.json{}
-JWT SECRET KEY : keycloack
-Response_templates{} : to generate manual template responses 


