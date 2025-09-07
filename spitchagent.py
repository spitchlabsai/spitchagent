import os
import json
import sys
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import JobContext, WorkerOptions, cli, Agent, AutoSubscribe, AgentSession, RoomInputOptions
from livekit.plugins import (
    spitch,
    assemblyai,
    google,
    noise_cancellation,
    silero,
)

from vector_search import RAGService
from sentence_transformers import SentenceTransformer

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

class SalesAgent(Agent):
    def __init__(self, user_context: Dict[str, Any], vector_service: RAGService):
        self.user_context = user_context
        self.user_id = user_context["userId"]  # Fail if missing
        self.user_name = user_context.get("name", "there")
        self.agent_name = user_context.get("agentName")
        self.company_name = user_context.get("companyName")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.vector_service = RAGService(SUPABASE_URL, SUPABASE_KEY, self.embedder)
        # self.vector_service = vector_service
        
        # Load all business context once at initialization
        self.business_context = self._load_business_context()
        print("loaded business content")

        self.cold_call_script = self._load_cold_call_script()
        print("loaded cold call script")

        # Simple base instructions with loaded context
#         base_instructions = f"""You are {self.agent_name}, a professional outbound sales representative for {self.company_name}.

# BUSINESS KNOWLEDGE:
# {self.business_context}

# Use this knowledge to engage prospects, introduce products/services, and convert qualified leads."""
        base_instructions = f"""You are {self.agent_name}, a professional outbound sales representative for {self.company_name}.

BUSINESS KNOWLEDGE:
{self.business_context}

COLD CALL SCRIPT:
{self.cold_call_script}

Use this script as your guide to structure calls with prospects. 
- Start with the opening pitch. 
- Ask discovery questions conversationally (don’t read them robotically).
- Always capture Budget, Authority, Need, Timeline (BANT).
- Summarize and hand off to a closer if qualified.
- Keep tone friendly and consultative.
"""
        
        super().__init__(instructions=base_instructions)

    def _load_business_context(self) -> str:
        """Load all business context once during initialization"""
        # Get comprehensive business context using broad search terms
        context = self.vector_service.get_context(
            "Company overview", 
            self.user_id,
            max_chunks=15  # Get more chunks for complete business picture
        )
        
        if not context:
            raise ValueError(f"No business context found for user {self.user_id}")
            
        return context
    
    def _load_cold_call_script(self) -> str:
        """Static Orthodox Gadgets cold call script"""
        return json.dumps({
            "script_title": "Orthodox Gadgets Cold Call Script (Entry-Level Techies)",
            "sections": [
                {
                    "title": "Opening",
                    "content": "Good day, this is [Your Name] with Orthodox Gadgets. We work with people just getting started in tech—helping them find reliable laptops without breaking the bank. Did I catch you at an okay time?"
                },
                {
                    "title": "Build Rapport",
                    "content": "Awesome. Just so I don’t waste your time—are you currently studying, starting a new tech role, or just getting into tech as a hobby?"
                },
                {
                    "title": "Discovery Questions (BANT)",
                    "Need": [
                        "What are you mainly planning to use a laptop for—coding, design, studying, or more general use?",
                        "What’s been your biggest frustration with the device you’re using now, if any?"
                    ],
                    "Budget": [
                        "Do you already have a budget range in mind for a new laptop, or are you open to options?"
                    ],
                    "Timeline": [
                        "Are you looking to get a new laptop right away, or sometime in the next few months?"
                    ],
                    "Authority": [
                        "Will this be a personal purchase, or will someone else (like family or employer) be helping with the decision?"
                    ]
                },
                {
                    "title": "Qualification Recap",
                    "content": "Got it—so you’re looking for [summary]. That’s definitely something we can help with."
                },
                {
                    "title": "Handoff to Closer",
                    "content": "The best next step would be a quick call with one of our laptop specialists. They can recommend the right model for your needs and budget, and even walk you through some exclusive offers. Does [suggest time] work for you?"
                },
                {
                    "title": "Close the Call",
                    "content": "Perfect, I’ll send you a quick confirmation email/text for [date/time]. Thanks for your time today, [Name]. Excited to help you get set up with the right laptop as you kick off your tech journey!"
                }
            ],
            "agent_notes": [
                "Keep it warm and encouraging—many are students or first-jobbers.",
                "Focus on listening to their needs (coding vs. design vs. general use).",
                "Don’t sell specs—just qualify and hand off.",
                "Capture BANT answers in CRM for the closer."
            ]
        }, indent=2)

    def get_context_instructions(self, query: str) -> str:
        """Get instructions for responding to user query"""
        # Get specific context relevant to their query
        specific_context = self.vector_service.get_context(query, self.user_id, max_chunks=5)
        
        if specific_context:
            return f"""The prospect just said: "{query}"

MOST RELEVANT INFORMATION FOR THIS RESPONSE:
{specific_context}

GENERAL BUSINESS CONTEXT (if needed):
{self.business_context}

Respond naturally to what they said using the most relevant information first. Work toward scheduling or closing."""
        else:
            # Fall back to general business context
            return f"""The prospect just said: "{query}"

BUSINESS CONTEXT:
{self.business_context}

Respond naturally to what they said using your business knowledge. Work toward scheduling or closing."""


class SalesAgentService:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.vector_service = RAGService(supabase_url, supabase_key)

    async def start_session(self, ctx: JobContext, user_context: Dict[str, Any]):
        """Start agent session - strict mode, no fallbacks"""
        
        # Create sales agent (this will load all business context)
        sales_agent = SalesAgent(user_context, self.vector_service)
        # Create session
        session = AgentSession(
            turn_detection="stt",
            stt=assemblyai.STT(),
            vad=silero.VAD.load(),
            llm=google.LLM(model="gemini-2.0-flash-exp"),
            tts=spitch.TTS(language="en", voice="lina"),
        )

        # Handle user speech with context lookup
        @session.on("user_speech_commit")
        def on_speech(event):
            async def process_speech():
                if event.alternatives:
                    message = event.alternatives[0].text
                    instructions = sales_agent.get_context_instructions(message)
                    await session.generate_reply(instructions=instructions)
            
            asyncio.create_task(process_speech())

        # Start session
        await session.start(
            room=ctx.room,
            agent=sales_agent,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )

        # Send opening pitch (agent already has all context loaded)
        await self._send_opening(sales_agent, session)
        
        return session

#     async def _send_opening(self, sales_agent: SalesAgent, session: AgentSession):
#         """Send opening pitch - context already loaded in agent"""
#         opening_instructions = f"""Start the sales conversation immediately with this greeting or a variation of it:

# "Hi! This is {sales_agent.agent_name} calling from {sales_agent.company_name}. I hope I'm catching you at a good time? . Do you have a couple minutes to hear about it?"

# You already have all your business knowledge loaded. Begin the call with the greeting above."""

#         await session.generate_reply(instructions=opening_instructions)

    async def _send_opening(self, sales_agent: SalesAgent, session: AgentSession):
        """Send opening pitch using cold call script"""
    
        # Extract script opening (we stored it as JSON string in SalesAgent)
        cold_call = json.loads(sales_agent.cold_call_script)
        opening_section = next(
            (s for s in cold_call["sections"] if s["title"] == "Opening"), 
            None
        )

        if opening_section:
            opening_line = opening_section["content"]
        else:
            # fallback if script not found
            opening_line = f"Hi! This is {sales_agent.agent_name} from {sales_agent.company_name}. I hope I'm catching you at a good time?"

        opening_instructions = f"""Start the sales conversation immediately with this greeting or a natural variation:

    {opening_line}

    Remember:
    - Keep tone friendly and consultative.
    - Use the script and business context to guide the conversation.
    - Transition into discovery questions naturally after the opening.
    """

        await session.generate_reply(instructions=opening_instructions)



# Global service
agent_service = None


async def entrypoint(ctx: JobContext):
    """Main entrypoint - strict mode"""
    global agent_service
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Initialize service
    if agent_service is None:
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        agent_service = SalesAgentService(SUPABASE_URL, SUPABASE_KEY)
    
    # Parse user context
    user_context = {}
    if ctx.job.metadata:
        user_context.update(json.loads(ctx.job.metadata))
        print(user_context)
    if ctx.room.metadata:
        print(user_context)
        user_context.update(json.loads(ctx.room.metadata))
    
    if not user_context.get("userId"):
        raise ValueError("userId required in metadata")
    
    # Start session
    await agent_service.start_session(ctx, user_context)
    
    # Wait for participant
    await ctx.wait_for_participant()

if __name__ == "__main__":
    # Check if this is the "download-files" build step
    if len(sys.argv) > 1 and sys.argv[1] == "download-files":
        print("Running in download-files mode, skipping SUPABASE env validation...")
        # call any model/file prefetch logic here if you need
        sys.exit(0)

    # Normal agent startup → enforce required environment vars
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing environment variables: {missing}")

    # Launch the agent worker
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))