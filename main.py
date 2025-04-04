import os
import json
import uuid
import tempfile
import shutil
import asyncio
import logging
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from transformers import pipeline

import boto3
from botocore.exceptions import NoCredentialsError
from mangum import Mangum

# Import your sentiment analysis functions from your module (sentiment.py)
from Sentiment import (
    flatten_reviews,
    preprocess_text,
    is_extremely_short,
    compute_cosine_similarity,
    check_review_flags,
    analyze_sentiment_batch,
    detect_review_language,
    detect_aspects_dominant,
    detect_customer_journey_stage,
    detect_complaint_or_compliment_multilingual,
    convert_star_rating_to_numeric,
    compute_token_frequencies_by_location,
    enhance_sentiment_count_json,
    convert_numpy_types,
    extract_date_from_datetime,
    clean_locality
)

# -----------------------
# AWS SNS Setup for Error Alerts
# -----------------------
sns_client = boto3.client('sns', region_name='ap-south-1')  # Updated region
SNS_TOPIC_ARN = 'arn:aws:sns:ap-south-1:024943590030:API-Deployment:f5a97a05-5009-4b45-8d78-4851448bddc4'  # Updated SNS ARN

def send_sns_alert(message: str, error_type: str = "Unknown"):
    try:
        timestamp = datetime.utcnow().isoformat()
        full_message = f"Timestamp: {timestamp}\nError Type: {error_type}\nError Description: {message}"
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=full_message,
            Subject="Critical Error Alert: Review Sentiment API"
        )
        logger.info("SNS Alert Sent Successfully")
    except NoCredentialsError:
        logger.error("SNS Alert: No AWS credentials found.")
    except Exception as e:
        logger.error(f"Error sending SNS alert: {str(e)}")

# -----------------------
# LoggerWriter for Capturing stdout and stderr
# -----------------------
class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = ""
    
    def write(self, message):
        if message and not message.isspace():
            self.buffer += message
            if self.buffer.endswith('\n'):
                self.logger.log(self.level, self.buffer.rstrip())
                self.buffer = ""
    
    def flush(self):
        if self.buffer:
            self.logger.log(self.level, self.buffer)
            self.buffer = ""

# -----------------------
# Configuration & Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ReviewAPI")

# Redirect stdout and stderr to logger
sys.stdout = LoggerWriter(logger, logging.INFO)
sys.stderr = LoggerWriter(logger, logging.ERROR)

# -----------------------
# Global In-Memory Job Storage
# -----------------------
jobs: Dict[str, Dict[str, Any]] = {}

# -----------------------
# Security: HTTP Basic Auth
# -----------------------
security = HTTPBasic()
VALID_USERNAME = "admin"
VALID_PASSWORD = "secret"

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != VALID_USERNAME or credentials.password != VALID_PASSWORD:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    return credentials.username

# -----------------------
# Rate Limiting (Simple in-memory per-IP limiter)
# -----------------------
RATE_LIMIT = 5  # maximum requests per minute per IP
rate_limit_storage: Dict[str, List[datetime]] = {}

async def rate_limiter(request: Request):
    client_ip = request.client.host
    now = datetime.utcnow()
    window_start = now - timedelta(minutes=1)
    request_times = rate_limit_storage.get(client_ip, [])
    request_times = [t for t in request_times if t > window_start]
    if len(request_times) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    request_times.append(now)
    rate_limit_storage[client_ip] = request_times

# -----------------------
# Pydantic Model for Input Validation (Extend as needed)
# -----------------------
class ReviewUpload(BaseModel):
    email_id: str
    accounts: List[Dict[str, Any]]

# -----------------------
# Global Models (Loaded once)
# -----------------------
sentiment_model = None
aspect_model = None

def load_models() -> bool:
    global sentiment_model, aspect_model
    try:
        logger.info("Loading sentiment analysis model...")
        sentiment_model = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
        logger.info("Sentiment model loaded successfully")
        logger.info("Loading aspect classification model...")
        aspect_model = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", multi_label=True)
        logger.info("Aspect model loaded successfully")
        return True
    except Exception as e:
        error_message = f"Error loading models: {str(e)}"
        logger.error(error_message)
        send_sns_alert(error_message, error_type="Model Loading Failure")
        traceback.print_exc()
        return False

# -----------------------
# FastAPI App Setup
# -----------------------
app = FastAPI(
    title="Review Sentiment Analysis API",
    description="API for analyzing customer reviews with sentiment analysis and aspect detection."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mangum integration for serverless deployment
handler = Mangum(app)

# -----------------------
# Background Task: Processing Reviews
# -----------------------
async def process_reviews_task(job_id: str, input_file: str):
    try:
        temp_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_")
        logger.info(f"[Job {job_id}] Created temporary directory: {temp_dir}")
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 0.1

        # Validate file size (limit: 50MB)
        if os.path.getsize(input_file) > 50 * 1024 * 1024:
            error_message = f"[Job {job_id}] Uploaded file size exceeds the allowed limit."
            logger.error(error_message)
            send_sns_alert(error_message, error_type="File Size Limit Exceeded")
            raise HTTPException(status_code=413, detail="File size exceeds the limit")

        # Step 1: JSON Decoding and Validation
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"[Job {job_id}] JSON file validated successfully.")
        except json.JSONDecodeError as e:
            error_message = f"[Job {job_id}] JSON decoding failed: {str(e)}"
            logger.error(error_message)
            send_sns_alert(error_message, error_type="JSON Decoding Failure")
            raise HTTPException(status_code=400, detail="Invalid JSON file")

        # Calculate counts from the raw JSON before processing
        num_emails = len(data)
        num_accounts = sum(len(record.get("accounts", [])) for record in data)
        num_locations = sum(len(account.get("profiles", [])) for record in data for account in record.get("accounts", []))
        num_reviews = sum(len(profile.get("reviews", [])) for record in data for account in record.get("accounts", []) for profile in account.get("profiles", []))

        # Step 13 (Aggregated Results) will include these counts in the summary
        # Save counts in the jobs dictionary to later include in the output
        jobs[job_id]["raw_counts"] = {
            "num_emails": num_emails,
            "num_accounts": num_accounts,
            "num_locations": num_locations,
            "num_reviews": num_reviews
        }

        # Step 2: Flatten Reviews and Create DataFrame
        try:
            flattened_reviews = flatten_reviews(data)
            import pandas as pd
            df = pd.DataFrame(flattened_reviews)
            jobs[job_id]["progress"] = 0.2
        except Exception as e:
            error_message = f"[Job {job_id}] Failed to flatten reviews and create DataFrame: {str(e)}"
            logger.error(error_message)
            send_sns_alert(error_message, error_type="DataFrame Creation Failure")
            raise

        # Step 3: Convert Star Ratings
        try:
            df['review_star_rating_number'] = df['review_star_rating'].apply(convert_star_rating_to_numeric)
        except Exception as e:
            error_message = f"[Job {job_id}] Star rating conversion failed: {str(e)}"
            logger.error(error_message)
            send_sns_alert(error_message, error_type="Star Rating Conversion Failure")
            raise

        # Step 4: Sentiment Analysis
        try:
            reviews = df['review_comment'].tolist()
            logger.info(f"[Job {job_id}] Analyzing sentiment for {len(reviews)} reviews with batch size 500")
            sentiments = analyze_sentiment_batch(reviews, sentiment_model, batch_size=500)
            df['sentiment'] = sentiments
            jobs[job_id]["progress"] = 0.4
        except Exception as e:
            error_message = f"[Job {job_id}] Sentiment analysis failed: {str(e)}"
            logger.error(error_message)
            send_sns_alert(error_message, error_type="Sentiment Analysis Failure")
            raise HTTPException(status_code=500, detail="Sentiment analysis failed")

        # Step 5: Language Detection
        try:
            logger.info(f"[Job {job_id}] Detecting review languages...")
            df['language'] = [detect_review_language(str(review)) for review in df['review_comment']]
        except Exception as e:
            error_message = f"[Job {job_id}] Language detection failed: {str(e)}"
            logger.error(error_message)
            send_sns_alert(error_message, error_type="Language Detection Failure")
            raise HTTPException(status_code=500, detail="Language detection failed")

        # Step 6: Token Preprocessing
        try:
            logger.info(f"[Job {job_id}] Preprocessing tokens...")
            df['tokens'] = df.apply(
                lambda row: preprocess_text(
                    str(row['review_comment']),
                    language=row['language'] if row['language'] != 'unknown' else 'english'
                ),
                axis=1
            )
            jobs[job_id]["progress"] = 0.6
        except Exception as e:
            error_message = f"[Job {job_id}] Token preprocessing failed: {str(e)}"
            logger.error(error_message)
            send_sns_alert(error_message, error_type="Token Preprocessing Failure")
            raise HTTPException(status_code=500, detail="Token preprocessing failed")

        # Step 7: Aspect Detection
        try:
            logger.info(f"[Job {job_id}] Detecting aspects...")
            aspects = [detect_aspects_dominant(str(review), aspect_model) for review in df['review_comment']]
            df['aspects'] = [', '.join(aspect_list) if aspect_list else '' for aspect_list in aspects]
        except Exception as e:
            error_message = f"[Job {job_id}] Aspect detection failed: {str(e)}"
            logger.error(error_message)
            send_sns_alert(error_message, error_type="Aspect Detection Failure")
            raise HTTPException(status_code=500, detail="Aspect detection failed")

        # Step 8: Customer Journey Stage Detection
        try:
            logger.info(f"[Job {job_id}] Detecting customer journey stages...")
            customer_journey_stages = [
                detect_customer_journey_stage(str(review), sentiment)
                for review, sentiment in zip(df['review_comment'], df['sentiment'])
            ]
            df['customer_journey_stage'] = customer_journey_stages
        except Exception as e:
            error_message = f"[Job {job_id}] Customer journey stage detection failed: {str(e)}"
            logger.error(error_message)
            send_sns_alert(error_message, error_type="Customer Journey Detection Failure")
            raise

        # Step 9: Complaint/Compliment Detection
        try:
            logger.info(f"[Job {job_id}] Detecting complaint/compliment...")
            complaint_or_compliment = [
                detect_complaint_or_compliment_multilingual(str(review), sentiment, language)
                for review, sentiment, language in zip(df['review_comment'], df['sentiment'], df['language'])
            ]
            df['complaint_or_compliment'] = complaint_or_compliment
        except Exception as e:
            error_message = f"[Job {job_id}] Complaint/Compliment detection failed: {str(e)}"
            logger.error(error_message)
            send_sns_alert(error_message, error_type="Complaint/Compliment Detection Failure")
            raise HTTPException(status_code=500, detail="Complaint/Compliment detection failed")

        # Step 10: Flagging Logic
        try:
            logger.info(f"[Job {job_id}] Applying flagging logic...")
            df['flag'] = check_review_flags(df, review_column='review_comment', user_id_column='reviewer_name', rating_column='review_star_rating')
            jobs[job_id]["progress"] = 0.8
        except Exception as e:
            error_message = f"[Job {job_id}] Flagging logic failed: {str(e)}"
            logger.error(error_message)
            send_sns_alert(error_message, error_type="Flagging Logic Failure")
            raise

        # Step 11: Save Processed Results
        try:
            output_json_path = os.path.join(temp_dir, f"results_{job_id}.json")
            logger.info(f"[Job {job_id}] Saving processed results to {output_json_path}")
            output_data = df.to_dict(orient='records')
            output_data = [convert_numpy_types(item) for item in output_data]
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            error_message = f"[Job {job_id}] Error saving processed results: {str(e)}"
            logger.error(error_message)
            send_sns_alert(error_message, error_type="File Saving Failure")
            raise HTTPException(status_code=500, detail="Error saving processed results")

        # Step 12: Compute Token Frequencies and Sentiment Count
        try:
            token_frequencies = compute_token_frequencies_by_location(df)
            token_freq_output = os.path.join(temp_dir, f"word_count_{job_id}.json")
            with open(token_freq_output, 'w', encoding='utf-8') as f:
                json.dump(convert_numpy_types(token_frequencies), f, indent=4, ensure_ascii=False)

            sentiment_count_output = os.path.join(temp_dir, f"sentiment_count_{job_id}.json")
            enhance_sentiment_count_json(df, sentiment_count_output)
        except Exception as e:
            error_message = f"[Job {job_id}] Error computing token frequencies or sentiment count: {str(e)}"
            logger.error(error_message)
            send_sns_alert(error_message, error_type="Aggregation Failure")
            raise

        
        # Step 13: Prepare Aggregated Results (with raw JSON counts in summary)
        try:
            raw_counts = jobs[job_id].get("raw_counts", {})
            results = {
                "processed_reviews": output_data,
                "word_count": json.load(open(token_freq_output, 'r', encoding='utf-8')),
                "sentiment_count": json.load(open(sentiment_count_output, 'r', encoding='utf-8')),
                "summary": {
                    "num_emails": raw_counts.get("num_emails", 0),
                    "num_accounts": raw_counts.get("num_accounts", 0),
                    "num_locations": raw_counts.get("num_locations", 0),
                    "num_reviews": raw_counts.get("num_reviews", 0),
                    "num_reviews_with_sentiment": int(df['sentiment'].notnull().sum()),
                    "total_records": len(df),
                    "sentiment_distribution": {
                        "positive": int(df[df['sentiment'] == 'positive'].shape[0]),
                        "negative": int(df[df['sentiment'] == 'negative'].shape[0]),
                        "neutral": int(df[df['sentiment'] == 'neutral'].shape[0])
                    },
                    "language_distribution": df['language'].value_counts().to_dict(),
                    "customer_journey_distribution": df['customer_journey_stage'].value_counts().to_dict(),
                    "complaint_compliment_distribution": df['complaint_or_compliment'].value_counts().to_dict()
                }
            }
        except Exception as e:
            error_message = f"[Job {job_id}] Aggregating results failed: {str(e)}"
            logger.error(error_message)
            send_sns_alert(error_message, error_type="Results Aggregation Failure")
            raise


        # Finalize Job Status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        jobs[job_id]["results"] = results

        logger.info(f"[Job {job_id}] Completed successfully.")

        # Cleanup temporary files
        shutil.rmtree(temp_dir)
        if os.path.exists(input_file):
            os.remove(input_file)

    except Exception as e:
        logger.error(f"[Job {job_id}] Error during processing: {str(e)}")
        send_sns_alert(str(e), error_type="Processing Failure")
        traceback.print_exc()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        try:
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(input_file):
                os.remove(input_file)
        except Exception as cleanup_error:
            logger.error(f"[Job {job_id}] Error cleaning up temporary files: {str(cleanup_error)}")

# -----------------------
# API Endpoints
# -----------------------
@app.on_event("startup")
async def startup_event():
    if not load_models():
        logger.error("Failed to load models during startup")
    else:
        logger.info("Models loaded successfully. API startup complete.")

@app.post("/upload", dependencies=[Depends(rate_limiter), Depends(verify_credentials)])
async def upload_reviews(background_tasks: BackgroundTasks, file: UploadFile = File(...), request: Request = None):
    if file.content_type != "application/json":
        logger.error(f"Invalid file type: {file.content_type}")
        send_sns_alert(f"Invalid file type: {file.content_type}", error_type="Upload Endpoint Failure")
        raise HTTPException(status_code=400, detail="Invalid file type. Only JSON is allowed.")
    try:
        job_id = str(uuid.uuid4())
        logger.info(f"[Job {job_id}] Initiating upload process.")
        temp_file_path = tempfile.mktemp(prefix=f"input_{job_id}_", suffix=".json")
        with open(temp_file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"[Job {job_id}] File saved temporarily at {temp_file_path}.")

        # Validate JSON file
        try:
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            logger.info(f"[Job {job_id}] JSON file validated successfully.")
        except json.JSONDecodeError as e:
            logger.error(f"[Job {job_id}] JSON validation failed: {str(e)}")
            os.remove(temp_file_path)
            send_sns_alert(f"JSON validation failed: {str(e)}", error_type="JSON Validation Failure")
            raise HTTPException(status_code=400, detail="Invalid JSON file")

        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "error": None,
            "results": None
        }

        background_tasks.add_task(process_reviews_task, job_id, temp_file_path)
        logger.info(f"[Job {job_id}] Queued for processing.")
        return JSONResponse(content={"job_id": job_id, "status": "queued"})
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        send_sns_alert(f"Upload endpoint error: {str(e)}", error_type="Upload Endpoint Failure")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

@app.get("/status/{job_id}", dependencies=[Depends(rate_limiter), Depends(verify_credentials)])
async def get_job_status(job_id: str):
    try:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
        job = jobs[job_id]
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "created_at": job["created_at"],
            "completed_at": job["completed_at"],
            "error": job["error"]
        }
    except Exception as e:
        logger.error(f"Error in status endpoint for job {job_id}: {str(e)}")
        send_sns_alert(f"Status endpoint error for job {job_id}: {str(e)}", error_type="Status Endpoint Failure")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching job status: {str(e)}")

@app.get("/results/{job_id}", dependencies=[Depends(rate_limiter), Depends(verify_credentials)])
async def get_job_results(job_id: str):
    try:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
        job = jobs[job_id]
        if job["status"] == "completed":
            return JSONResponse(content=job["results"])
        elif job["status"] == "failed":
            raise HTTPException(status_code=500, detail=f"Job failed: {job['error']}")
        else:
            raise HTTPException(status_code=202, detail=f"Job still processing. Progress: {job['progress']}")
    except Exception as e:
        logger.error(f"Error in results endpoint for job {job_id}: {str(e)}")
        send_sns_alert(f"Results endpoint error for job {job_id}: {str(e)}", error_type="Results Endpoint Failure")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching job results: {str(e)}")

@app.delete("/jobs/{job_id}", dependencies=[Depends(rate_limiter), Depends(verify_credentials)])
async def delete_job(job_id: str):
    try:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
        del jobs[job_id]
        logger.info(f"Job {job_id} deleted successfully")
        return {"message": f"Job {job_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error in delete endpoint for job {job_id}: {str(e)}")
        send_sns_alert(f"Delete endpoint error for job {job_id}: {str(e)}", error_type="Jobs Endpoint Failure")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting job: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


