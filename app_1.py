from fastapi import FastAPI
from pydantic import BaseModel
import torch
from torch.nn.functional import softmax
from transformers import BertForSequenceClassification, AutoTokenizer
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time


# print("Loading model and tokenizer...")
# # start_time = time.time()

model_checkpoint = r".\results_1\checkpoint-300-4_12"


model = BertForSequenceClassification.from_pretrained(
    model_checkpoint,
    local_files_only=True,
    torch_dtype=torch.float16,  # Use half precision for faster inference
)

# Set model to evaluation mode and move to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set to evaluation mode permanently

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Create thread pool for CPU-bound tokenization
executor = ThreadPoolExecutor(max_workers=4)

# print(f"Model loaded in {time.time() - start_time:.2f} seconds")
# print(f"Using device: {device}")

# Pre-compile the model with a dummy input for faster first inference
with torch.no_grad():
    dummy_input = tokenizer(
        "dummy text", return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    dummy_input = {k: v.to(device) for k, v in dummy_input.items()}
    _ = model(**dummy_input)
# print("Model warmed up and ready")


def classify_text(text: str) -> float:
    """Optimized classification function"""

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():  # Run inference with no gradient calculation
        outputs = model(**inputs)

    # Calculate spam probability
    probs = softmax(outputs.logits, dim=1)
    spam_score = float(probs[0][1])

    return spam_score


# FastAPI app
app = FastAPI(title="Spam Detection API")


class InputData(BaseModel):
    Massage_Id: str
    Tanent_id: str
    mailbox_id: str


class OutputData(BaseModel):
    signal: str
    value: float
    # processing_time_ms: float


@app.post("/process", response_model=OutputData)
async def process_text(data: InputData):
    """Process text and return spam score"""
    # start = time.time()

    # Run classification in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    spam_score = await loop.run_in_executor(executor, classify_text, data.Massage_Id)

    # processing_time = (time.time() - start) * 1000  # Convert to milliseconds

    return {
        "signal": "Content_Spam_Score",
        "value": spam_score,
        # "processing_time_ms": processing_time,
    }


@app.get("/")
async def root():
    return {
        "message": "FastAPI Spam Detection is running!",
        "device": str(device),
        "model_loaded": True,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(device),
    }


# Cleanup on shutdown
@app.on_event("shutdown")
def shutdown_event():
    executor.shutdown(wait=True)
