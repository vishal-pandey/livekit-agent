from dotenv import load_dotenv
from typing import Optional
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path

from livekit import agents, rtc, api
from livekit.agents import AgentSession, Agent, RoomInputOptions, UserStateChangedEvent
from livekit.plugins import (
    openai,
    sarvam,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import aiohttp
import asyncio
import logging
import google.generativeai as genai  # Import the Google Generative AI SDK
import wave  # For WAV file creation
from motor.motor_asyncio import AsyncIOMotorClient  # MongoDB async client
from datetime import datetime, timezone

logger = logging.getLogger("hindi-voice-agent")

load_dotenv()


# MongoDB Configuration
MONGO_CONNECTION_STRING = "mongodb+srv://lumiq:{db_password}@cluster0.x40mx40.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "telemer"
COLLECTIONS = {
    "call_sessions": "call_sessions",
    "medical_summaries": "medical_summaries", 
    "transcripts": "transcripts"
}


class MongoDBManager:
    """Manages MongoDB operations for storing session data"""
    
    def __init__(self):
        self.client = None
        self.database = None
        
    async def connect(self):
        """Connect to MongoDB"""
        try:
            # Get MongoDB password from environment
            mongo_password = os.getenv("MONGO_DB_PASSWORD")
            if not mongo_password:
                logger.error("MONGO_DB_PASSWORD environment variable not set")
                return False
                
            connection_string = MONGO_CONNECTION_STRING.replace("{db_password}", mongo_password)
            self.client = AsyncIOMotorClient(connection_string)
            self.database = self.client[DATABASE_NAME]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    async def update_call_session(self, room_name: str, recording_url: Optional[str] = None, 
                                status: str = "completed", duration: Optional[str] = None,
                                transcript_confidence: Optional[int] = None):
        """Update call session record with recording path and metadata"""
        try:
            if self.database is None:
                logger.error("Database not connected")
                return False
                
            # Extract proposal number from room name
            # Handle different room name formats
            proposal_no = room_name
            if "voice_assistant_room_" in room_name:
                proposal_no = room_name.replace("voice_assistant_room_", "")
            elif "room_" in room_name:
                proposal_no = room_name.replace("room_", "")
            
            logger.info(f"Mapping room '{room_name}' to proposal '{proposal_no}'")
            
            update_data = {
                "status": status,
                "updatedAt": datetime.now(timezone.utc)
            }
            
            if recording_url:
                update_data["recordingPath"] = recording_url
            if duration:
                update_data["duration"] = duration
            if transcript_confidence:
                update_data["transcriptConfidence"] = transcript_confidence
                
            collection = self.database[COLLECTIONS["call_sessions"]]
            result = await collection.update_one(
                {"proposalNo": proposal_no},
                {"$set": update_data}
            )
            
            if result.matched_count > 0:
                logger.info(f"Updated call session for proposal: {proposal_no}")
                return True
            else:
                logger.warning(f"No call session found for proposal: {proposal_no}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update call session: {e}")
            return False
    
    async def store_transcript(self, room_name: str, transcript_data: dict):
        """Store transcript data in transcripts collection"""
        try:
            if self.database is None:
                logger.error("Database not connected")
                return False
                
            # Extract proposal number from room name  
            # Handle different room name formats
            proposal_no = room_name
            if "voice_assistant_room_" in room_name:
                proposal_no = room_name.replace("voice_assistant_room_", "")
            elif "room_" in room_name:
                proposal_no = room_name.replace("room_", "")
            
            logger.info(f"Storing transcript for proposal '{proposal_no}'")
            
            transcript_record = {
                "proposalNo": proposal_no,
                "roomName": proposal_no,  # Use clean proposal number for both fields
                "transcriptData": transcript_data,
                "createdAt": datetime.now(timezone.utc),
                "updatedAt": datetime.now(timezone.utc)
            }
            
            collection = self.database[COLLECTIONS["transcripts"]]
            # Upsert: update if exists, insert if not
            result = await collection.replace_one(
                {"proposalNo": proposal_no},
                transcript_record,
                upsert=True
            )
            
            logger.info(f"Stored transcript for proposal: {proposal_no}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store transcript: {e}")
            return False
    
    async def store_medical_summary(self, room_name: str, medical_report: dict, 
                                  enhanced_summary: Optional[str] = None):
        """Store medical summary and report in medical_summaries collection"""
        try:
            if self.database is None:
                logger.error("Database not connected")
                return False
                
            # Extract proposal number from room name
            # Handle different room name formats  
            proposal_no = room_name
            if "voice_assistant_room_" in room_name:
                proposal_no = room_name.replace("voice_assistant_room_", "")
            elif "room_" in room_name:
                proposal_no = room_name.replace("room_", "")
            
            logger.info(f"Storing medical summary for proposal '{proposal_no}'")
            
            summary_record = {
                "proposalNo": proposal_no,
                "roomName": proposal_no,  # Use clean proposal number for consistency
                "medicalReport": medical_report,
                "enhancedSummary": enhanced_summary,
                "createdAt": datetime.now(timezone.utc),
                "updatedAt": datetime.now(timezone.utc)
            }
            
            collection = self.database[COLLECTIONS["medical_summaries"]]
            # Upsert: update if exists, insert if not
            result = await collection.replace_one(
                {"proposalNo": proposal_no},
                summary_record,
                upsert=True
            )
            
            logger.info(f"Stored medical summary for proposal: {proposal_no}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store medical summary: {e}")
            return False


# Global MongoDB manager instance
mongo_manager = MongoDBManager()


class VolumeBasedVAD:
    """Custom VAD that filters based on volume/energy to focus on primary speaker"""
    
    def __init__(self, volume_threshold: float = 0.02, energy_threshold: float = 0.05):
        self.volume_threshold = volume_threshold
        self.energy_threshold = energy_threshold
        self.recent_volumes = []
        self.max_history = 50  # Keep last 50 frames for baseline
    
    def should_process_audio(self, audio_frame: rtc.AudioFrame) -> bool:
        """Determine if audio frame has sufficient volume/energy to be from primary speaker"""
        # Convert audio data to numpy array
        audio_data = np.frombuffer(audio_frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Calculate RMS (Root Mean Square) for volume
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Calculate energy
        energy = np.sum(audio_data**2)
        
        # Keep track of recent volumes for adaptive thresholding
        self.recent_volumes.append(rms)
        if len(self.recent_volumes) > self.max_history:
            self.recent_volumes.pop(0)
        
        # Adaptive threshold based on recent audio
        if len(self.recent_volumes) > 10:
            avg_volume = np.mean(self.recent_volumes)
            dynamic_threshold = max(self.volume_threshold, avg_volume * 0.6)
        else:
            dynamic_threshold = self.volume_threshold
        
        # Check if this frame meets our criteria for primary speaker
        is_primary_speaker = (rms > dynamic_threshold and energy > self.energy_threshold)
        
        if is_primary_speaker:
            logger.debug(f"Primary speaker detected - RMS: {rms:.4f}, Energy: {energy:.6f}")
        
        return is_primary_speaker


# Custom Agent with volume-based filtering using LiveKit's built-in session history
class VolumeFilteredAssistant(Agent):
    def __init__(self, room_name: str = "unknown") -> None:
        super().__init__(instructions="""
        You are a tele medical examination assistant that responds only to the primary speaker.
        your role is to conduct the telemedical examination by asking questions and ask the reflexive questions based on the user's responses.
        
        You need to be smart to get all the answers correctly and efficiently from the user.
        You are a female voice assistant with a friendly and professional tone.
        
        You speak in hindi but you also do understand english.
        You generally speak in hinglish and use daily use words and phrases in english only, also whenever you need to talk about the medical terms you need to pronounce the meidical terms and medecine names in english only.
                         
        Below I am providing you the questions that you need to ask the user in order to conduct the telemedical examination.
        The questions are in json format and it is a reflexive questionnaire, which means you need to ask the questions based on the user's responses.
                         

        There might be multiple person in the room while this call is being conducted, so you need to ingore the content which are not relevant to the primary speaker.
        And anytime if you have some confustion around what user has said, you can ask the user to answer the question again or provide more details.
                         

        {
  "questions": [
    {
      "Q.No.": "Q1",
      "Question Description": "Has any member ever applied for a policy with Care Heath Insurance in the past?",
      "Input Type": "Yes/No Checkbox"
    },
    {
      "Q.No.": "Q2",
      "Question Description": "Has any of the person(s) to be insured ever filed a claim with their current / previous or any other insurer?",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q2a",
          "Question Description": "Please Provide Detail",
          "Input Type": "Free Text"
        }
      ]
    },
    {
      "Q.No.": "Q 3",
      "Question Description": "Is any member suffering from Diabetes/Sugar problem or has been tested to have high blood sugar?",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q3a",
          "Question Description": "When was any member first detected with Diabetes/Sugar?",
          "Input Type": "Year/Month, No. of Years/Months"
        },
        {
          "Q.No.": "Q3b",
          "Question Description": "Has any member ever been prescribed or taken Insulin?",
          "Input Type": "Yes/No Checkbox"
        },
        {
          "Q.No.": "Q3c",
          "Question Description": "Has any member suffered from any complications of Diabetes like reduced vision, kidney complications, non healing ulcer?",
          "Input Type": "Yes/No Checkbox"
        }
      ]
    },
    {
      "Q.No.": "Q4",
      "Question Description": "Has any member been diagnosed with high cholesterol or Lipid disorder?",
      "Input Type": "Yes/No Checkbox"
    },
    {
      "Q.No.": "Q5",
      "Question Description": "Has any member been detected to have high BP or blood pressure or Hypertension?",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q5a",
          "Question Description": "When was any member first detected for having high BP/Blood Pressure/ Hypertension?",
          "Input Type": "Year/Month, No. of Years/Months"
        }
      ]
    },
    {
      "Q.No.": "Q6",
      "Question Description": "Does any member have a Cardiac or Heart problem or experienced chest pain in the past?",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q6a",
          "Question Description": "Please choose the problem.",
          "Input Type": "Dropdown + Free Text (For 'Any Other..)",
          "Dropdown List": [
            "Angioplasty/Stenting/ Bypass surgery (No. of Year / Months) - Note : Need to add + Sign with Years and month option for Multi event",
            "Pace maker implantation (No. of Year / Months)",
            "Heart Valve disorder (No. of Year / Months ) - ( Operated / Unoperated )",
            "Genetic heart disorders such as Hole in heart (No. of Year / Months ) - ( Operated / Corrected /Unoperated )",
            "Increased or slow heart rate For e.g Palpitations, Tachycardia or Bradycardia (No. of Year / Months )",
            "Any other type of disorder"
          ]
        }
      ]
    },
    {
      "Q.No.": "Q7",
      "Question Description": "Has any member experienced any symptoms of joint pain in Knee, Shoulder, Hip etc.?",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q7a",
          "Question Description": "Please provide detail along with medication prescribed/ taken, if any.",
          "Input Type": "Free Text"
        }
      ]
    },
    {
      "Q.No.": "Q8",
      "Question Description": "Has any member suffered any vision related problem like blurry or hazy vision.",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q8a",
          "Question Description": "Please choose the Problem.",
          "Input Type": "Dropdown + Free Text (Fot 'Any Other..')",
          "Dropdown List": [
            "Cataract (No. of Year / Months) - ( Operated / Unoperated )",
            "Retinal disorder (No. of Year / Months) - ( Operated / Unoperated )",
            "Glaucoma (No. of Year / Months) - ( Operated / Unoperated )",
            "Any other type of disorder"
          ]
        }
      ]
    },
    {
      "Q.No.": "Q9",
      "Question Description": "Has any member been diagnosed for gall bladder, kidney or urinary stones?",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q9a",
          "Question Description": "Please choose the problem.",
          "Input Type": "Dropdown",
          "Dropdown List": [
            "Gall Bladder Stone (No. of Year / Months) - ( Operated / Unoperated )",
            "Kidney or Urinary Stone (No. of Year / Months) - ( Operated / Conservatively resolved / Unoperated )"
          ]
        }
      ]
    },
    {
      "Q.No.": "Q10",
      "Question Description": "Has any member been diagnosed for prostrate related problem, any complaints of increased urinary frequency, urgency or retention?",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q10 a",
          "Question Description": "When was member first Diagnosed with Prostate or urinary disorder ?",
          "Input Type": "Year/Month, No. of Years/Months"
        },
        {
          "Q.No.": "Q10 b",
          "Question Description": "Please specify",
          "Input Type": "Free Text"
        }
      ]
    },
    {
      "Q.No.": "Q11",
      "Question Description": "Has any member ever been diagnosed with any gynaecological problems like abnormal bleeding, cyst or fibroid in ovaries etc.?",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q 11 a",
          "Question Description": "When was member first Diagnosed with gynaecological problems ?",
          "Input Type": "Year/Month, No. of Years/Months"
        },
        {
          "Q.No.": "Q 11 b",
          "Question Description": "Please specify",
          "Input Type": "Free Text"
        }
      ]
    },
    {
      "Q.No.": "Q12",
      "Question Description": "Has any member ever been diagnosed with any form of Thyroid disorder",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q12a",
          "Question Description": "Please confirm type of Thyroid Disorder and/or name of medicine prescribed?",
          "Input Type": "Free Text"
        },
        {
          "Q.No.": "Q12b",
          "Question Description": "How long has any member of the policy been suffering from thyroid disorder?",
          "Input Type": "Year/Month, No. of ears/Months"
        }
      ]
    },
    {
      "Q.No.": "Q13",
      "Question Description": "Has any member ever been admitted to a hospital or undergone or advised for a surgery",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q13a",
          "Question Description": "Please specify the reason for hospitalization or surgery.",
          "Input Type": "Free Text with Limit up to 400 Words"
        }
      ]
    },
    {
      "Q.No.": "Q14",
      "Question Description": "Has any member ever done medical test like Ultrasound/ CT scan/ MRI, 2D echo or any major investigation with positive finding? Please share report",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q14 a",
          "Question Description": "Please specify",
          "Input Type": "Free Text"
        }
      ]
    },
    {
      "Q.No.": "Q15",
      "Question Description": "Has any member ever experienced symptoms such as pain in abdomen or any other part of body, breathlessness?",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q15a",
          "Question Description": "Please specify.",
          "Input Type": "Free Text"
        }
      ]
    },
    {
      "Q.No.": "Q16",
      "Question Description": "Do you want to disclose any other condition/illness/procedure for any member, other than the ones already answered above?",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q16a",
          "Question Description": "Please specify.",
          "Input Type": "Free Text with Limit up to 400 Words"
        }
      ]
    },
    {
      "Q.No.": "Q17",
      "Question Description": "Does any member smoke?",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q17a",
          "Question Description": "How many cigarettes/ bidi does MemberName smoke?",
          "Input Type": "Number Dropdown"
        },
        {
          "Q.No.": "Q17b",
          "Question Description": "Since when has any member been smoking?",
          "Input Type": "Year/Month, No. of Years/Months"
        }
      ]
    },
    {
      "Q.No.": "Q18",
      "Question Description": "Does any member consume alcohol?",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q18a",
          "Question Description": "How often does any member drink?",
          "Input Type": "Dropdown: Occasionally, Daily, Weekly"
        },
        {
          "Q.No.": "Q18b",
          "Question Description": "Please specify quantity? (Ex. Unit would be 30 ml of liquor per day/week) (Alcohol consumption (e.g., 100 ml) throws an error due to the digit limit.)",
          "Input Type": "Number Dropdown"
        }
      ]
    },
    {
      "Q.No.": "Q19",
      "Question Description": "Does any member have a habit of chewing tobacco/pan masala/gutka?",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q19a",
          "Question Description": "How often does any member consume chewing tobacco/pan masala/gutka?",
          "Input Type": "Dropdown: Occasionally, Daily, Weekly"
        },
        {
          "Q.No.": "Q19b",
          "Question Description": "How much does any member consume? Enter number (Ex. Sachets/grams per day (The current input limit of 2 digits is inadequate.),",
          "Input Type": "Number Dropdown"
        }
      ]
    },
    {
      "Q.No.": "Q20",
      "Question Description": "Does any member have any other prohibitive habits?",
      "Input Type": "Yes/No Checkbox",
      "sub_options": [
        {
          "Q.No.": "Q20a",
          "Question Description": "Please Specify.",
          "Input Type": "Free Text"
        }
      ]
    },
    {
      "Q.No.": "Q21",
      "Question Description": "What is the height of member/s in feet and inches? (Ex: 5,3 for 5 feet and 3 inches)",
      "Input Type": "Number Dropdown: Feet and Inches"
    },
    {
      "Q.No.": "Q22",
      "Question Description": "What is the weight of member/s in Kgs? (Ex: 82 for 82 KGs)",
      "Input Type": "Number Dropdown: Kgs"
    }
  ]
}
""")
        self.volume_filter = VolumeBasedVAD(volume_threshold=0.025, energy_threshold=0.08)
        self.ignored_frames = 0
        self.processed_frames = 0
        
        # Initialize local audio recorder as fallback
        self.local_recorder = LocalAudioRecorder(room_name)
    
    def should_process_audio(self, audio_frame: rtc.AudioFrame) -> bool:
        """Filter audio based on volume to focus on primary speaker"""
        should_process = self.volume_filter.should_process_audio(audio_frame)
        
        # Always record audio frames (even if not processing for STT)
        self.local_recorder.add_audio_frame(audio_frame)
        
        if should_process:
            self.processed_frames += 1
        else:
            self.ignored_frames += 1
        
        # Log statistics periodically
        if (self.processed_frames + self.ignored_frames) % 100 == 0:
            total = self.processed_frames + self.ignored_frames
            logger.info(f"Audio filtering stats - Processed: {self.processed_frames}/{total} ({self.processed_frames/total*100:.1f}%)")
        
        return should_process


class LocalAudioRecorder:
    """Manual audio recording for local LiveKit servers without Egress"""
    
    def __init__(self, room_name: str):
        self.room_name = room_name
        self.audio_frames = []
        self.is_recording = False
        self.sample_rate = 16000  # Standard sample rate for voice
        self.channels = 1  # Mono recording
        
        # Create recordings directory
        self.recordings_dir = Path("./recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        
        logger.info(f"LocalAudioRecorder initialized for room: {room_name}")
    
    def start_recording(self):
        """Start manual audio recording"""
        self.is_recording = True
        self.audio_frames = []
        logger.info("Started local audio recording")
    
    def add_audio_frame(self, audio_frame: rtc.AudioFrame):
        """Add an audio frame to the recording"""
        if self.is_recording:
            # Convert audio frame to bytes
            audio_data = bytes(audio_frame.data)
            self.audio_frames.append(audio_data)
    
    def stop_recording(self):
        """Stop recording and save to WAV file"""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        
        if not self.audio_frames:
            logger.warning("No audio frames recorded")
            return None
        
        try:
            # Create WAV file
            wav_filename = self.recordings_dir / f"session_{self.room_name}.wav"
            
            with wave.open(str(wav_filename), 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(self.sample_rate)
                
                # Write all audio frames
                for frame_data in self.audio_frames:
                    wav_file.writeframes(frame_data)
            
            logger.info(f"Local audio recording saved to: {wav_filename}")
            return str(wav_filename)
            
        except Exception as e:
            logger.error(f"Failed to save local audio recording: {e}")
            return None


def calculate_session_duration(transcript_data: dict) -> str:
    """Calculate session duration from transcript data"""
    try:
        items = transcript_data.get('items', [])
        if len(items) < 2:
            return "0:00"
        
        # Estimate duration based on number of exchanges (rough estimate)
        # Each exchange averages about 10-15 seconds
        estimated_seconds = len(items) * 12  # 12 seconds per exchange average
        
        minutes = estimated_seconds // 60
        seconds = estimated_seconds % 60
        
        return f"{minutes}:{seconds:02d}"
        
    except Exception as e:
        logger.warning(f"Could not calculate session duration: {e}")
        return "0:00"


def calculate_transcript_confidence(transcript_data: dict) -> int:
    """Calculate average transcript confidence score"""
    try:
        items = transcript_data.get('items', [])
        if not items:
            return 85  # Default confidence
        
        # For LiveKit session history, we'll estimate confidence based on content quality
        # Longer responses generally indicate higher confidence in STT
        total_confidence = 0
        count = 0
        
        for item in items:
            content = item.get('content', [])
            if isinstance(content, list):
                text_length = sum(len(str(c)) for c in content)
            else:
                text_length = len(str(content))
            
            # Estimate confidence based on text length and complexity
            if text_length > 50:
                confidence = 90
            elif text_length > 20:
                confidence = 85
            else:
                confidence = 80
                
            total_confidence += confidence
            count += 1
        
        return total_confidence // count if count > 0 else 85
        
    except Exception as e:
        logger.warning(f"Could not calculate transcript confidence: {e}")
        return 85


async def generate_structured_report(transcript_data):
    """
    Generates a structured medical report from the conversation transcript using a direct Gemini API call.
    This function is adapted from old_agent.py to work with LiveKit session history.
    """
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    except KeyError:
        logger.error("Error: GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")
        return {"error": "Google API Key not configured."}

    # Use the model tag 'gemini-2.5-flash' as in the original implementation
    model = genai.GenerativeModel('gemini-2.5-flash') 

    full_transcript_content = ""
    for item in transcript_data['items']:
        role = item['role'].capitalize()
        content = " ".join(item['content']) if isinstance(item['content'], list) else str(item['content'])
        full_transcript_content += f"{role}: {content}\n"

    # Medical report generation prompt adapted for the telemedical examination questionnaire
    prompt = f"""
    You are an expert medical summarizer. Your task is to extract specific medical information from a conversation transcript and format it as a structured JSON object. Focus only on providing the requested JSON. Do not include any conversational text outside the JSON.

    The following is a transcript of a Tele Medical Examination Report (TMER) conversation for insurance policy evaluation.
    Please analyze the transcript and extract the medical information into a structured JSON object.
    For each medical condition listed below, determine if the response was 'Yes' or 'No' and provide the detailed response from the insured.
    If the condition was mentioned or inferred, even if not explicitly stated as 'Yes', mark it as 'Yes' and provide the details.
    If details are inferred, clearly mark them as "Inferred Details".

    Medical Conditions to Extract (Based on Insurance Questionnaire):
    1. Previous Insurance Claims
    2. Diabetes/Sugar Problem
    3. High Cholesterol/Lipid Disorder
    4. Hypertension/High Blood Pressure
    5. Cardiac/Heart Problems
    6. Joint Pain
    7. Vision Problems
    8. Gall Bladder/Kidney/Urinary Stones
    9. Prostate Problems
    10. Gynaecological Problems
    11. Thyroid Disorder
    12. Hospitalization/Surgery
    13. Medical Tests (CT/MRI/Echo etc.)
    14. Other Symptoms (Pain/Breathlessness)
    15. Smoking Habits
    16. Alcohol Consumption
    17. Tobacco/Pan Masala Habits
    18. Height and Weight

    Output Format (JSON):
    ```json
    {{
        "Previous Insurance Claims": {{
            "answered_yes_no": "Yes/No",
            "details": "Extracted Details or Inferred Details"
        }},
        "Diabetes/Sugar Problem": {{
            "answered_yes_no": "Yes/No",
            "details": "Details including detection date, insulin usage, complications"
        }},
        "High Cholesterol/Lipid Disorder": {{
            "answered_yes_no": "Yes/No",
            "details": "Details"
        }},
        "Hypertension/High Blood Pressure": {{
            "answered_yes_no": "Yes/No",
            "details": "Details including detection date"
        }},
        "Cardiac/Heart Problems": {{
            "answered_yes_no": "Yes/No",
            "details": "Details including type of problem, procedures, timeline"
        }},
        "Joint Pain": {{
            "answered_yes_no": "Yes/No",
            "details": "Details including location, medications"
        }},
        "Vision Problems": {{
            "answered_yes_no": "Yes/No",
            "details": "Details including type, operated/unoperated"
        }},
        "Gall Bladder/Kidney/Urinary Stones": {{
            "answered_yes_no": "Yes/No",
            "details": "Details including type, treatment status"
        }},
        "Prostate Problems": {{
            "answered_yes_no": "Yes/No",
            "details": "Details including diagnosis date, symptoms"
        }},
        "Gynaecological Problems": {{
            "answered_yes_no": "Yes/No",
            "details": "Details including diagnosis date, type"
        }},
        "Thyroid Disorder": {{
            "answered_yes_no": "Yes/No",
            "details": "Details including type, medication, duration"
        }},
        "Hospitalization/Surgery": {{
            "answered_yes_no": "Yes/No",
            "details": "Details including reason, date, location"
        }},
        "Medical Tests": {{
            "answered_yes_no": "Yes/No",
            "details": "Details including type of test, findings"
        }},
        "Other Symptoms": {{
            "answered_yes_no": "Yes/No",
            "details": "Details including pain location, breathlessness"
        }},
        "Smoking Habits": {{
            "answered_yes_no": "Yes/No",
            "details": "Details including quantity, duration"
        }},
        "Alcohol Consumption": {{
            "answered_yes_no": "Yes/No",
            "details": "Details including frequency, quantity"
        }},
        "Tobacco/Pan Masala Habits": {{
            "answered_yes_no": "Yes/No",
            "details": "Details including frequency, quantity"
        }},
        "Height and Weight": {{
            "height": "Height in feet and inches",
            "weight": "Weight in KGs"
        }}
    }}
    ```

    Conversation Transcript:
    ---
    {full_transcript_content}
    ---
    """

    response_text = ""
    try:
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
            )
        )
        
        response_text = response.text

        start_idx = response_text.find("```json")
        end_idx = response_text.rfind("```")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx + len("```json"):end_idx].strip()
        else:
            json_str = response_text.strip()

        report = json.loads(json_str)
        logger.info("Structured medical report generated successfully using Gemini API")
        return report

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from Gemini response: {e}")
        logger.error(f"Gemini raw response: \n{response_text}")
        return {"error": "Failed to parse report from Gemini.", "raw_llm_response": response_text}
    except Exception as e:
        logger.error(f"An unexpected error occurred during Gemini API call: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


async def start_room_recording(room_name: str):
    """
    Start recording using Track Recording instead of Room Composite for better Agent compatibility.
    Track recording is more reliable with LiveKit Agents as it doesn't wait for video signals.
    """
    livekit_api = None
    try:
        # Initialize LiveKit API client
        livekit_api = api.LiveKitAPI(
            url=os.getenv("LIVEKIT_URL"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET")
        )
        
        # Generate filename without timestamp
        filename_prefix = f"medical-recordings/sessions/{room_name}"
        
        # Get AWS S3 configuration
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "ap-south-1")
        s3_bucket = os.getenv("AWS_S3_BUCKET")
        
        if not all([aws_access_key, aws_secret_key, s3_bucket]):
            logger.warning("AWS S3 credentials not configured - falling back to local recording")
            await livekit_api.aclose()
            return None
        
        logger.info(f"S3 Configuration - Bucket: {s3_bucket}, Region: {aws_region}, Key: {aws_access_key[:8] if aws_access_key else 'None'}***")
        
        # Use Track Composite Recording instead of Room Composite for better Agent compatibility
        # This records all audio tracks mixed together without waiting for video
        request = api.TrackCompositeEgressRequest(
            room_name=room_name,
            file=api.EncodedFileOutput(
                # S3 output configuration
                s3=api.S3Upload(
                    access_key=aws_access_key,
                    secret=aws_secret_key,
                    region=aws_region,
                    bucket=s3_bucket,
                ),
                filepath=f"{filename_prefix}.mp3",
            )
        )
        
        # Start recording
        logger.info(f"Starting track composite recording for {room_name} with S3 storage...")
        egress_info = await livekit_api.egress.start_track_composite_egress(request)
        logger.info(f"Started track recording with Egress ID: {egress_info.egress_id}")
        
        # Generate the public S3 URL
        public_s3_url = f"https://{s3_bucket}.s3.{aws_region}.amazonaws.com/{filename_prefix}.mp3"
        logger.info(f"Recording will be saved to S3: {public_s3_url}")
        
        return egress_info.egress_id, public_s3_url
        
    except Exception as e:
        logger.warning(f"Track Egress recording failed: {e}")
        
        # Fallback to Room Composite if Track Composite fails
        try:
            logger.info("Falling back to Room Composite recording...")
            
            # Re-initialize API client if needed
            if not livekit_api:
                livekit_api = api.LiveKitAPI(
                    url=os.getenv("LIVEKIT_URL"),
                    api_key=os.getenv("LIVEKIT_API_KEY"),
                    api_secret=os.getenv("LIVEKIT_API_SECRET")
                )
            
            # Re-initialize variables for fallback (without timestamp)
            filename_prefix = f"medical-recordings/sessions/{room_name}"
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_REGION", "ap-south-1")
            s3_bucket = os.getenv("AWS_S3_BUCKET")
            

            request = api.RoomCompositeEgressRequest(
                room_name=room_name,
                audio_only=True,
                file_outputs=[api.EncodedFileOutput(
                    file_type=api.EncodedFileType.OGG,
                    filepath=f"{filename_prefix}.ogg",
                    s3=api.S3Upload(
                        access_key=aws_access_key,
                        secret=aws_secret_key,
                        region=aws_region,
                        bucket=s3_bucket,
                    ),
                )],
            )

            lkapi = api.LiveKitAPI()
            egress_info = await lkapi.egress.start_room_composite_egress(request)

            await lkapi.aclose()


            # # Room composite recording with modified settings for Agent compatibility
            # request = api.RoomCompositeEgressRequest(
            #     room_name=room_name,
            #     layout="speaker",  # Layout type - focus on active speaker
            #     audio_only=True,   # Audio-only recording for medical sessions
            #     file=api.EncodedFileOutput(
            #         # S3 output configuration
            #         s3=api.S3Upload(
            #             access_key=aws_access_key,
            #             secret=aws_secret_key,
            #             region=aws_region,
            #             bucket=s3_bucket,
            #         ),
            #         filepath=f"{filename_prefix}.mp3",
            #     )
            # )
            
            # egress_info = await livekit_api.egress.start_room_composite_egress(request)
            logger.info(f"Started room composite recording with Egress ID: {egress_info.egress_id}")
            
            # Generate the public S3 URL for Room Composite (OGG format)
            public_s3_url = f"https://{s3_bucket}.s3.{aws_region}.amazonaws.com/{filename_prefix}.ogg"
            logger.info(f"Recording will be saved to S3: {public_s3_url}")
            
            return egress_info.egress_id, public_s3_url
            
        except Exception as fallback_error:
            logger.warning(f"Both Track and Room Composite recording failed: {fallback_error}")
            logger.info("Continuing with local audio recording fallback")
            return None
    finally:
        if livekit_api:
            await livekit_api.aclose()


async def stop_room_recording(egress_id: str):
    """Stops the room recording using LiveKit's Egress API."""
    if not egress_id:
        return None
    try:
        livekit_api = api.LiveKitAPI(
            url=os.getenv("LIVEKIT_URL"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET")
        )
        
        # Create a StopEgressRequest object
        request = api.StopEgressRequest(egress_id=egress_id)
        
        # Stop the recording
        egress_info = await livekit_api.egress.stop_egress(request)
        
        logger.info(f"Stopped room recording. Final status: {egress_info.status}")
        
        if egress_info.status == api.EgressStatus.EGRESS_COMPLETE:
            logger.info("Recording completed successfully and uploaded to S3")
            if hasattr(egress_info, 'file') and egress_info.file:
                logger.info(f"S3 file location confirmed: {egress_info.file.location}")
        elif egress_info.status == api.EgressStatus.EGRESS_FAILED:
            logger.error("Recording failed - check Egress logs for details")
            if hasattr(egress_info, 'error') and egress_info.error:
                logger.error(f"Egress error: {egress_info.error}")
        elif egress_info.status == api.EgressStatus.EGRESS_ABORTED:
            logger.warning("Recording was aborted - possibly due to S3 configuration issues")
            if hasattr(egress_info, 'error') and egress_info.error:
                logger.error(f"Abort reason: {egress_info.error}")
        
        await livekit_api.aclose()
        return egress_info
        
    except Exception as e:
        logger.warning(f"Could not stop egress recording: {e}")
        return None

async def write_transcript_and_generate_summary(session: AgentSession, room_name: str, recording_url: Optional[str] = None):
    """
    Use LiveKit's built-in session history to generate transcript and summary.
    This leverages LiveKit's native recording capabilities, adds Gemini-based medical report generation,
    and stores everything in MongoDB.
    """
    # Connect to MongoDB
    await mongo_manager.connect()
    
    # Create recordings directory
    output_dir = Path("./recordings")
    output_dir.mkdir(exist_ok=True)
    
    # Use LiveKit's built-in session history
    transcript_data = session.history.to_dict()
    
    # Save the raw transcript using LiveKit's session history (without timestamp)
    transcript_filename = output_dir / f"transcript_{room_name}.json"
    with open(transcript_filename, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"LiveKit session transcript saved to {transcript_filename}")
    
    # Store transcript in MongoDB
    await mongo_manager.store_transcript(room_name, transcript_data)
    
    # Generate structured medical report using Gemini API (from old_agent.py)
    logger.info("Generating structured medical report with Gemini API...")
    structured_report = await generate_structured_report(transcript_data)
    
    # Save the structured medical report (without timestamp)
    report_filename = output_dir / f"medical_report_{room_name}.json"
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(structured_report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Structured medical report saved to {report_filename}")
    
    # Create enhanced summary from LiveKit session history (existing functionality)
    enhanced_summary_text = await create_enhanced_summary(transcript_data, room_name, output_dir)
    
    # Store medical summary in MongoDB
    await mongo_manager.store_medical_summary(room_name, structured_report, enhanced_summary_text)
    
    # Calculate session duration and confidence (basic estimates)
    session_duration = calculate_session_duration(transcript_data)
    transcript_confidence = calculate_transcript_confidence(transcript_data)
    
    # Update call session with recording URL and metadata
    await mongo_manager.update_call_session(
        room_name=room_name,
        recording_url=recording_url,
        status="completed",
        duration=session_duration,
        transcript_confidence=transcript_confidence
    )
    
    # Close MongoDB connection
    await mongo_manager.close()
    
    logger.info("All data successfully stored in MongoDB")


async def create_enhanced_summary(transcript_data: dict, room_name: str, output_dir: Path) -> str:
    """Create an enhanced summary from LiveKit session history and return the text"""
    
    # Parse LiveKit session history format
    conversation_items = transcript_data.get('items', [])
    
    # Analyze conversation
    total_items = len(conversation_items)
    user_messages = []
    assistant_messages = []
    
    # Extract messages from LiveKit session history format
    for item in conversation_items:
        role = item.get('role', '')
        content = item.get('content', [])
        
        # Handle content which can be a list of strings or a single string
        if isinstance(content, list):
            message_text = ' '.join(str(c) for c in content)
        else:
            message_text = str(content)
        
        if role == 'user':
            user_messages.append(message_text)
        elif role == 'assistant':
            assistant_messages.append(message_text)
    
    # Extract medical keywords and insights
    medical_keywords = [
        'diabetes', 'blood pressure', 'hypertension', 'cholesterol', 'heart', 'cardiac',
        'thyroid', 'kidney', 'liver', 'surgery', 'hospital', 'medicine', 'treatment',
        'pain', 'symptoms', 'diagnosis', 'medical', 'doctor', 'health', 'insurance',
        'policy', 'claim', 'diabetic', 'insulin', 'bp', 'chest pain'
    ]
    
    medical_mentions = []
    for message in user_messages:
        message_lower = message.lower()
        for keyword in medical_keywords:
            if keyword in message_lower:
                medical_mentions.append({
                    'keyword': keyword,
                    'context': message[:100] + '...' if len(message) > 100 else message
                })
    
    # Create enhanced summary file
    summary_filename = output_dir / f"enhanced_summary_{room_name}.txt"
    
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TELEMEDICAL EXAMINATION SESSION SUMMARY\n")
        f.write("(Generated using LiveKit Session History)\n")
        f.write("=" * 80 + "\n\n")
        
        # Session Information
        f.write("SESSION INFORMATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Room: {room_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Conversation Statistics
        f.write("CONVERSATION STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Conversation Items: {total_items}\n")
        f.write(f"User Messages: {len(user_messages)}\n")
        f.write(f"Assistant Messages: {len(assistant_messages)}\n")
        f.write(f"Medical Keywords Detected: {len(medical_mentions)}\n\n")
        
        # Medical Insights
        if medical_mentions:
            f.write("MEDICAL KEYWORDS DETECTED:\n")
            f.write("-" * 30 + "\n")
            unique_keywords = list(set(mention['keyword'] for mention in medical_mentions))
            for keyword in unique_keywords:
                f.write(f"• {keyword.title()}\n")
            f.write("\n")
            
            f.write("MEDICAL CONTEXTS:\n")
            f.write("-" * 30 + "\n")
            for i, mention in enumerate(medical_mentions[:10], 1):  # Show first 10
                f.write(f"{i}. {mention['keyword'].title()}: {mention['context']}\n")
            f.write("\n")
        
        # Full Conversation from LiveKit Session History
        f.write("FULL CONVERSATION TRANSCRIPT:\n")
        f.write("-" * 50 + "\n\n")
        
        for i, item in enumerate(conversation_items, 1):
            role = item.get('role', 'unknown').title()
            content = item.get('content', [])
            
            # Handle content format
            if isinstance(content, list):
                message_text = ' '.join(str(c) for c in content)
            else:
                message_text = str(content)
            
            f.write(f"[{i}] {role}: {message_text}\n\n")
        
        # File Information
        f.write("\n" + "=" * 80 + "\n")
        f.write("FILES GENERATED:\n")
        f.write("-" * 20 + "\n")
        f.write(f"• Raw Transcript: transcript_{room_name}.json\n")
        f.write(f"• Structured Medical Report: medical_report_{room_name}.json\n")
        f.write(f"• Enhanced Summary: enhanced_summary_{room_name}.txt\n")
        f.write(f"• Audio Recording: session_{room_name}.wav (Local Recording)\n")
        f.write(f"• Source: LiveKit Session History + Gemini API Analysis + Local Audio Recording\n")
    
    logger.info(f"Enhanced summary saved to {summary_filename}")
    
    # Return the summary text for MongoDB storage
    with open(summary_filename, 'r', encoding='utf-8') as f:
        return f.read()


async def entrypoint(ctx: agents.JobContext):
    """Main entrypoint for the agent."""
    http_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False))

    session = AgentSession(
        stt=sarvam.STT(language="hi-IN", model="saarika:v2.5"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=sarvam.TTS(target_language_code="hi-IN", speaker="anushka", http_session=http_session),
        vad=ctx.proc.userdata.get("vad") or silero.VAD.load(activation_threshold=0.85, min_silence_duration=0.8),
        turn_detection=MultilingualModel(),
    )

    recording_egress_id: Optional[str] = None
    recording_s3_url: Optional[str] = None
    agent = VolumeFilteredAssistant(ctx.room.name)
    recording_started = False

    async def cleanup_and_record():
        nonlocal recording_egress_id, recording_s3_url
        logger.info("Session ending, generating transcript and summary...")
        
        if recording_egress_id:
            logger.info("Stopping LiveKit Egress room recording...")
            egress_result = await stop_room_recording(recording_egress_id)
            # We already have the S3 URL from when we started recording
        
        agent.local_recorder.stop_recording()
        
        await http_session.close()
        await write_transcript_and_generate_summary(session, ctx.room.name, recording_s3_url)

    ctx.add_shutdown_callback(cleanup_and_record)

    # FIX: Removed noise_cancellation for self-hosting
    await session.start(room=ctx.room, agent=agent)
    
    await ctx.connect()

    await session.say("नमस्ते! मैं आपका तेलीमेडिकल एग्जामिनेशन असिस्टेंट हूँ। मैं आपसे कुछ मेडिकल सवाल पूछूंगी। क्या आप तैयार हैं?")

    # Wait for participant and let the session run naturally
    logger.info("Agent is running and waiting for participants...")
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")
    
    # Start local recording immediately as fallback
    agent.local_recorder.start_recording()
    logger.info("Started local audio recording fallback")
    
    # Wait for first user response before starting Egress recording
    logger.info("Waiting for user to start speaking before starting S3 recording...")
    
    async def start_recording_after_user_speech():
        nonlocal recording_egress_id, recording_started, recording_s3_url
        
        # Wait for the first user transcript to ensure audio is flowing
        user_spoke = False
        max_wait_time = 60  # Maximum 60 seconds to wait for user speech
        wait_start = asyncio.get_event_loop().time()
        
        while not user_spoke and (asyncio.get_event_loop().time() - wait_start) < max_wait_time:
            await asyncio.sleep(0.5)
            # Check if we have any user messages in session history
            if hasattr(session, 'history') and session.history:
                history_dict = session.history.to_dict()
                user_messages = [item for item in history_dict.get('items', []) if item.get('role') == 'user']
                if len(user_messages) > 0:
                    user_spoke = True
                    logger.info("User speech detected, starting S3 recording now...")
                    break
        
        if user_spoke and not recording_started:
            recording_started = True
            # Additional delay to ensure audio is stable
            await asyncio.sleep(1.0)
            
            logger.info("Attempting to start LiveKit Egress room recording...")
            recording_result = await start_room_recording(ctx.room.name)
            if recording_result:
                if isinstance(recording_result, tuple) and len(recording_result) == 2:
                    recording_egress_id, recording_s3_url = recording_result
                    logger.info(f"Recording started with Egress ID: {recording_egress_id}")
                    logger.info(f"S3 URL captured: {recording_s3_url}")
                else:
                    # Handle case where only egress_id is returned (shouldn't happen now)
                    recording_egress_id = recording_result
                    logger.warning("Recording started but no S3 URL returned")
            else:
                logger.info("Egress not available - continuing with local audio recording fallback ONLY.")
        elif not user_spoke:
            logger.warning("No user speech detected within 60 seconds - continuing with local recording only")
    
    # Start the recording task in background
    recording_task = asyncio.create_task(start_recording_after_user_speech())
    
    # Keep the agent alive until the session ends
    await asyncio.sleep(0.1)  # Small delay to ensure everything is set up
    
    # The session will continue until participants leave and cleanup_and_record is called


def prewarm(proc: agents.JobProcess):
    """Prewarm VAD model to improve startup times"""
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.6,  # Higher threshold for primary speaker (matching NOISY_CONFIG)
        min_silence_duration=0.8,
        min_speech_duration=0.15,  # Require longer speech to activate
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm
        )
    )
