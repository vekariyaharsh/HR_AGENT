
from flask import Flask, request, jsonify, Blueprint
from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier
import os
import threading
import time
import logging
from dotenv import load_dotenv
import hragent

# Load environment variables
load_dotenv()


slack_bot = Blueprint('slack_bot', __name__)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Slack client
SLACK_BOT_TOKEN=os.getenv("SLACK_SIGNING_SECRET")
slack_client =  WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
signature_verifier = SignatureVerifier(os.environ.get("SLACK_SIGNING_SECRET"))

# Your existing LLM function - modify this to use your current logic
def generate_llm_response(message, user_id=None, channel_id=None):
    """
    Replace this function with your existing LLM logic from your Flask app
    """
    try:
    
        question = message
        response= hragent.start_chatbot(question)
        answer=response["answer"]
        return answer
        
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        return "Sorry, I'm having trouble processing your request right now."

def send_slack_message(channel, text, thread_ts=None):
    """Send message to Slack channel"""
    try:
        response = slack_client.chat_postMessage(
            channel=channel,
            text=text,
            thread_ts=thread_ts
        )
        return response
    except Exception as e:
        logger.error(f"Error sending Slack message: {str(e)}")
        return None

def process_slack_message_async(channel, message, user_id, thread_ts=None):
    """Process message asynchronously to avoid Slack timeout"""
    def process():
        #try:
            # Show typing indicator
        slack_client.chat_postMessage(
            channel=channel,
            text="🤔 Thinking...",
            thread_ts=thread_ts
        )
        
        # Generate LLM response using your existing logic
        llm_response = generate_llm_response(message, user_id, channel)
        print("llm_response",llm_response)
        
        # Send the actual response
        send_slack_message(channel, llm_response, thread_ts)
            
        '''except Exception as e:
            logger.error(f"Error in async processing: {str(e)}")
            send_slack_message(
                channel, 
                "Sorry, something went wrong. Please try again.", 
                thread_ts
            )'''
    
    # Run in separate thread to avoid Slack timeout
    thread = threading.Thread(target=process)
    thread.start()

@slack_bot.route("/slack/events", methods=["POST"])
def slack_events():
    """Handle Slack Events API"""
    
    # Raw body for signature verification
    data = request.json
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data.get("challenge")})
    
    print(data)
    # 2. Validate signature to ensure the request is from Slack
    if not signature_verifier.is_valid_request(request.get_data(), request.headers):
        print("forbidden")
        return "Invalid signature", 403

    # 3. Handle real events (e.g., message events)
    # Example: user sent a message
    event = data.get("event", {})
    if event.get("type") == "message" and "subtype" not in event:
        user = event.get("user")
        text = event.get("text")
        channel = event.get("channel")
        print(f"Message from {user} in {channel}: {text}")
        # Handle or respond to message as needed
    
    # Handle URL verification challenge
    if "challenge" in data:
        return {"challenge": data["challenge"]}
    
    # Handle events
    if "event" in data:
        event = data["event"]
        
        # Ignore bot messages and retries
        if event.get("bot_id") or "client_msg_id" not in event:
            return "OK", 200
        
        # Handle app mentions (@botname)
        if event["type"] == "app_mention":
            channel = event["channel"]
            user = event["user"]
            text = event["text"]
            ts = event["ts"]
            
            # Remove bot mention from message
            import re
            message = re.sub(r'<@[^>]+>', '', text).strip()
            
            if not message:
                send_slack_message(channel, "Hi! How can I help you today?", ts)
            else:
                # Process message asynchronously
                process_slack_message_async(channel, message, user, ts)
        
        # Handle direct messages
        elif event["type"] == "message" and event.get("channel_type") == "im":
            channel = event["channel"]
            user = event["user"]
            message = event["text"]
            
            # Process message asynchronously
            process_slack_message_async(channel, message, user)
    
    return "OK", 200


    

@slack_bot.route("/slack/commands", methods=["POST"])
def slack_commands():
    """Handle Slack slash commands"""
    
    # Verify request signature
    if not signature_verifier.is_valid_request(request.get_data(), request.headers):
        return "Invalid signature", 403
    
    command_data = request.form
    text = command_data.get("text", "")
    user_id = command_data.get("user_id")
    channel_id = command_data.get("channel_id")
    response_url = command_data.get("response_url")
    
    if not text:
        return jsonify({
            "response_type": "ephemeral",
            "text": "Please provide a message for me to process."
        })
    
    # Send immediate acknowledgment
    def delayed_response():
        try:
            time.sleep(1)  # Small delay to ensure immediate response is sent
            
            # Generate LLM response
            llm_response = generate_llm_response(text, user_id, channel_id)
            
            # Send delayed response via response_url
            import requests
            requests.post(response_url, json={
                "response_type": "in_channel",
                "text": llm_response
            })
        except Exception as e:
            logger.error(f"Error in delayed response: {str(e)}")
    
    # Start delayed response in background
    threading.Thread(target=delayed_response).start()
    
    return jsonify({
        "response_type": "in_channel",
        "text": "🤔 Processing your request..."
    })

@slack_bot.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "OK",
        "timestamp": time.time()
    })




# Check required environment variables
required_vars = [ "SLACK_SIGNING_SECRET"]
for var in required_vars:
        if not os.environ.get(var):
            logger.error(f"Missing required environment variable: {var}")
            exit(1)
    
logger.info("Starting Flask Slack bot...")