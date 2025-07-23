from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

# Import the agent factory for testing
from agents.agent_factory import create_agent, AgentType


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class AgentTaskRequest(BaseModel):
    agent_type: str = Field(..., description="Type of agent (researcher, analyst, writer)")
    task_type: str = Field(..., description="Type of task to perform")
    description: str = Field(..., description="Task description or prompt")
    complexity: str = Field(default="medium", description="Task complexity (low, medium, high)")
    urgency: str = Field(default="normal", description="Task urgency (low, normal, high)")
    additional_data: Optional[Dict[str, Any]] = Field(default=None, description="Additional task data")

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

@api_router.post("/agents/test", response_model=Dict[str, Any])
async def test_agent(request: AgentTaskRequest):
    """Test endpoint for LLM-enabled agents."""
    try:
        # Create agent
        agent = await create_agent(
            agent_type=request.agent_type,
            auto_start=True
        )
        
        # Prepare task data
        task_data = {
            "id": f"test_{uuid.uuid4().hex[:8]}",
            "type": request.task_type,
            "description": request.description,
            "complexity": request.complexity,
            "urgency": request.urgency,
            **(request.additional_data or {})
        }
        
        # Execute task
        result = await agent.execute_task(task_data)
        
        # Clean up agent
        await agent.stop()
        
        return {
            "success": True,
            "agent_type": request.agent_type,
            "task_type": request.task_type,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Agent test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent test failed: {str(e)}")

@api_router.get("/agents/health")
async def check_agent_health():
    """Check health of agent system."""
    try:
        # Test creating a simple researcher agent
        agent = await create_agent(
            agent_type="researcher",
            auto_start=True
        )
        
        # Test LLM health
        llm_health = await agent.get_llm_health_status()
        
        # Clean up
        await agent.stop()
        
        return {
            "status": "healthy",
            "llm_integration": llm_health,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
