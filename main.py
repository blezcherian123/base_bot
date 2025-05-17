import random
import pickle
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.responses import FileResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model components with error handling
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model.pkl: {str(e)}")
    raise HTTPException(status_code=500, detail="Failed to load model")

try:
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    logger.info("Label encoder loaded successfully")
except Exception as e:
    logger.error(f"Failed to load label_encoder.pkl: {str(e)}")
    raise HTTPException(status_code=500, detail="Failed to load label encoder")

try:
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("TFIDF vectorizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load tfidf_vectorizer.pkl: {str(e)}")
    raise HTTPException(status_code=500, detail="Failed to load vectorizer")

try:
    with open("responses.pkl", "rb") as f:
        responses = pickle.load(f)
    logger.info("Responses loaded successfully")
except Exception as e:
    logger.error(f"Failed to load responses.pkl: {str(e)}")
    raise HTTPException(status_code=500, detail="Failed to load responses")

try:
    with open("intents.json", "r", encoding="utf-8") as f:
        intents_data = json.load(f)
    logger.info("Intents loaded successfully")
except Exception as e:
    logger.error(f"Failed to load intents.json: {str(e)}")
    raise HTTPException(status_code=500, detail="Failed to load intents")

class ChatRequest(BaseModel):
    message: str | None = Field(None, alias="text")  # Accept 'text' or 'message', allow None

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"

    def get_message(self) -> str:
        # Return message or default if None/empty
        return self.message.strip() if self.message and isinstance(self.message, str) else "hello"

def get_response(user_input: str) -> dict:
    try:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)
        tag = label_encoder.inverse_transform(prediction)[0]
        
        quick_replies = []
        for intent in intents_data["intents"]:
            if tag in (intent["tag"] if isinstance(intent["tag"], list) else [intent["tag"]]):
                quick_replies = intent.get("quick_replies", [])
                break
        
        return {
            "response": random.choice(responses[tag]),
            "quick_replies": quick_replies
        }
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing input")

@app.post("/chat")
async def chat(request: ChatRequest):
    input_message = request.get_message()
    logger.info(f"Received message: {input_message}")
    if input_message.lower() == "quit":
        return {"response": "Bye! 👋", "quick_replies": []}
    result = get_response(input_message)
    return result

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "healthy"}    