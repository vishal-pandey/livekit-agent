from dotenv import load_dotenv
from typing import Optional
import numpy as np
import json
import os
import re
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
import time

logger = logging.getLogger("hindi-voice-agent")

load_dotenv()


# MongoDB Configuration
MONGO_CONNECTION_STRING = "mongodb+srv://lumiq:{db_password}@cluster0.x40mx40.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "telemer"
POLICY_DATABASE_NAME = "policy_db"
COLLECTIONS = {
    "call_sessions": "call_sessions",
    "medical_summaries": "medical_summaries", 
    "transcripts": "transcripts"
}
POLICY_COLLECTIONS = {
    "proposals": "proposals"
}


class MongoDBManager:
    """
    Manages MongoDB operations for storing session data.
    
    All database operations use proposalNo as the primary key for upserting:
    - Extracts proposalNo from room_name (handles formats like 'voice_assistant_room_123', 'room_456', or plain '789')
    - Uses upsert operations to create new records or update existing ones based on proposalNo
    - Maintains consistency across call_sessions, transcripts, and medical_summaries collections
    - Stores original room_name in roomName field while using clean proposalNo for indexing
    - Captures complete call session data including customer details, call metadata, and recording paths
    """
    
    def __init__(self):
        self.client = None
        self.database = None
        self.policy_database = None
        
    def _extract_proposal_no(self, room_name: str) -> str:
        """
        Extract proposal number from room name.
        Handles different room name formats and returns clean proposal number.
        
        Examples:
        - 'voice_assistant_room_123' -> '123'
        - 'room_456' -> '456' 
        - '789' -> '789'
        """
        proposal_no = room_name
        if "voice_assistant_room_" in room_name:
            proposal_no = room_name.replace("voice_assistant_room_", "")
        elif "room_" in room_name:
            proposal_no = room_name.replace("room_", "")
        return proposal_no
        
    def _extract_customer_info(self, family_data: Optional[dict]) -> tuple[Optional[str], Optional[str]]:
        """
        Extract customer name and phone from family data.
        Returns tuple of (customer_name, customer_phone)
        """
        if not family_data:
            return None, None
            
        customer_name = None
        customer_phone = None
        
        # Try to get customer info from main fields (different possible field names)
        customer_name = (family_data.get('customerName') or 
                        family_data.get('proposerName') or 
                        family_data.get('policyHolderName'))
        
        customer_phone = (family_data.get('customerPhone') or 
                         family_data.get('proposerPhone') or 
                         family_data.get('proposerMobile') or
                         family_data.get('policyHolderPhone'))
        
        # If not found in main fields, try to get from first member details
        if not customer_name and family_data.get('memberDetails'):
            member_details = family_data.get('memberDetails', [])
            if member_details:
                first_member = member_details[0]
                basic_info = first_member.get('basicInfo', {})
                customer_name = basic_info.get('name')
                # Phone might be in contact info
                contact_info = first_member.get('contactInfo', {})
                if not customer_phone:
                    customer_phone = contact_info.get('phone') or contact_info.get('mobile')
        
        return customer_name, customer_phone

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
            self.policy_database = self.client[POLICY_DATABASE_NAME]
            
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
                                transcript_confidence: Optional[int] = None,
                                customer_name: Optional[str] = None,
                                customer_phone: Optional[str] = None,
                                language: str = "hindi"):
        """Update or create call session record with recording path and metadata using proposalNo as key"""
        try:
            if self.database is None:
                logger.error("Database not connected")
                return False
                
            # Extract proposal number from room name using helper method
            proposal_no = self._extract_proposal_no(room_name)
            
            logger.info(f"Upserting call session for proposal '{proposal_no}' (room: '{room_name}')")
            
            # Prepare the record data for upsert with all required fields
            call_session_record = {
                "proposalNo": proposal_no,
                "roomName": room_name,  # Store original room name, not the cleaned proposal number
                "status": status,
                "language": language,
                "callDate": datetime.now(timezone.utc),  # Set call date to current time
                "updatedAt": datetime.now(timezone.utc)
            }
            
            # Add optional fields if provided
            if recording_url:
                call_session_record["recordingPath"] = recording_url
            if duration:
                call_session_record["duration"] = duration
            if transcript_confidence:
                call_session_record["transcriptConfidence"] = transcript_confidence
            if customer_name:
                call_session_record["customerName"] = customer_name
            if customer_phone:
                call_session_record["customerPhone"] = customer_phone
            
            # Set createdAt only when creating new records
            update_operation = {
                "$set": call_session_record,
                "$setOnInsert": {
                    "createdAt": datetime.now(timezone.utc)
                }
            }
                
            collection = self.database[COLLECTIONS["call_sessions"]]
            # Upsert: update if exists, insert if not, based on proposalNo
            result = await collection.update_one(
                {"proposalNo": proposal_no},
                update_operation,
                upsert=True
            )
            
            if result.upserted_id:
                logger.info(f"Created new call session for proposal: {proposal_no}")
            elif result.matched_count > 0:
                logger.info(f"Updated existing call session for proposal: {proposal_no}")
            else:
                logger.warning(f"No changes made to call session for proposal: {proposal_no}")
            
            return True
                
        except Exception as e:
            logger.error(f"Failed to update call session: {e}")
            return False
    
    async def store_transcript(self, room_name: str, transcript_data: dict):
        """Store transcript data in transcripts collection using proposalNo as key"""
        try:
            if self.database is None:
                logger.error("Database not connected")
                return False
                
            # Extract proposal number from room name using helper method
            proposal_no = self._extract_proposal_no(room_name)
            
            logger.info(f"Storing transcript for proposal '{proposal_no}' (room: '{room_name}')")
            
            transcript_record = {
                "proposalNo": proposal_no,
                "roomName": room_name,  # Store original room name, not the cleaned proposal number
                "transcriptData": transcript_data,
                "createdAt": datetime.now(timezone.utc),
                "updatedAt": datetime.now(timezone.utc)
            }
            
            collection = self.database[COLLECTIONS["transcripts"]]
            # Upsert: update if exists, insert if not, based on proposalNo
            result = await collection.replace_one(
                {"proposalNo": proposal_no},
                transcript_record,
                upsert=True
            )
            
            if result.upserted_id:
                logger.info(f"Created new transcript for proposal: {proposal_no}")
            elif result.matched_count > 0:
                logger.info(f"Updated existing transcript for proposal: {proposal_no}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store transcript: {e}")
            return False
    
    async def store_medical_summary(self, room_name: str, medical_report: dict, 
                                  enhanced_summary: Optional[str] = None):
        """Store medical summary and report in medical_summaries collection using proposalNo as key"""
        try:
            if self.database is None:
                logger.error("Database not connected")
                return False
                
            # Extract proposal number from room name using helper method
            proposal_no = self._extract_proposal_no(room_name)
            
            logger.info(f"Storing medical summary for proposal '{proposal_no}' (room: '{room_name}')")
            
            summary_record = {
                "proposalNo": proposal_no,
                "roomName": room_name,  # Store original room name, not the cleaned proposal number
                "medicalReport": medical_report,
                "enhancedSummary": enhanced_summary,
                "createdAt": datetime.now(timezone.utc),
                "updatedAt": datetime.now(timezone.utc)
            }
            
            collection = self.database[COLLECTIONS["medical_summaries"]]
            # Upsert: update if exists, insert if not, based on proposalNo
            result = await collection.replace_one(
                {"proposalNo": proposal_no},
                summary_record,
                upsert=True
            )
            
            if result.upserted_id:
                logger.info(f"Created new medical summary for proposal: {proposal_no}")
            elif result.matched_count > 0:
                logger.info(f"Updated existing medical summary for proposal: {proposal_no}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store medical summary: {e}")
            return False

    async def get_family_details(self, room_name: str) -> Optional[dict]:
        """Get family details from the policy database using proposalNo as key"""
        try:
            if self.policy_database is None:
                logger.error("Policy database not connected")
                return None
                
            # Extract proposal number from room name using helper method
            proposal_no = self._extract_proposal_no(room_name)
            
            logger.info(f"Fetching family details for proposal '{proposal_no}' (room: '{room_name}')")
            
            collection = self.policy_database[POLICY_COLLECTIONS["proposals"]]
            proposal = await collection.find_one({"proposalNo": proposal_no})
            
            if proposal:
                logger.info(f"Found family details for proposal: {proposal_no}")
                return proposal
            else:
                logger.warning(f"No proposal found for proposal: {proposal_no}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get family details: {e}")
            return None

    def format_family_details_for_prompt(self, member_details: list) -> str:
        """Format family details into a readable format for the AI prompt"""
        if not member_details:
            return "No family details available."
        
        formatted_text = "Family Members for Telemedical Examination:\n\n"
        
        for i, member in enumerate(member_details, 1):
            basic_info = member.get("basicInfo", {})
            medical_questions = member.get("medicalQuestions", [])
            eligible = member.get("eligibleForTelemedical", False)
            reason = member.get("telemedicalReason", "")
            
            formatted_text += f"Member {i}:\n"
            formatted_text += f"- Name: {basic_info.get('name', 'N/A')}\n"
            formatted_text += f"- Age: {basic_info.get('age', 'N/A')}\n"
            formatted_text += f"- Gender: {basic_info.get('gender', 'N/A')}\n"
            formatted_text += f"- Relationship: {basic_info.get('relationship', 'N/A')}\n"
            formatted_text += f"- Occupation: {basic_info.get('occupation', 'N/A')}\n"
            formatted_text += f"- Sum Insured: ₹{basic_info.get('sumInsured', 'N/A')}\n"
            formatted_text += f"- Eligible for Telemedical: {'Yes' if eligible else 'No'}\n"
            
            if reason:
                formatted_text += f"- Telemedical Reason: {reason}\n"
            
            if medical_questions:
                formatted_text += f"- Pre-filled Medical Information:\n"
                for question in medical_questions:
                    if question.get("requiresTelemedical", False):
                        formatted_text += f"  * {question.get('question', 'N/A')}: {question.get('answer', 'N/A')}"
                        if question.get('details'):
                            formatted_text += f" - {question.get('details')}"
                        formatted_text += "\n"
            
            formatted_text += "\n"
        
        formatted_text += "Focus the telemedical examination on members who are 'Eligible for Telemedical' and pay special attention to their pre-filled medical information.\n"
        
        return formatted_text


# Global MongoDB manager instance
mongo_manager = MongoDBManager()


class SilenceDetector:
    """
    Detects prolonged user silence and triggers prompts with dynamic timeout based on agent response length.
    
    Features:
    - Dynamic timeout calculation based on agent message word count
    - Automatic timeout adjustment for longer/shorter agent responses  
    - Configurable speaking rate and buffer time
    - Min/max timeout constraints for safety
    - Statistics tracking for monitoring timeout behavior
    
    The dynamic timeout works by:
    1. Counting words in agent messages
    2. Calculating estimated speaking time (words / speaking_rate)
    3. Adding buffer time for user processing (speaking_time * buffer_multiplier)  
    4. Applying min/max constraints
    5. Using this timeout until next activity is detected
    """
    
    def __init__(self, base_silence_timeout: float = 20.0, prompt_interval: float = 20.0):
        self.base_silence_timeout = base_silence_timeout  # Base timeout for silence detection
        self.current_silence_timeout = base_silence_timeout  # Current dynamic timeout
        self.prompt_interval = prompt_interval  # Time between subsequent prompts
        self.last_user_activity = time.time()
        self.last_prompt_time = 0
        self.is_monitoring = False
        self.prompt_count = 0
        self.max_prompts = 3  # Maximum number of silence prompts before giving up
        
        # Dynamic timeout configuration
        self.min_timeout = 10.0  # Minimum timeout regardless of word count
        self.max_timeout = 60.0  # Maximum timeout regardless of word count
        self.words_per_second = 2.5  # Average speaking rate in Hindi/Hinglish (words per second)
        self.buffer_multiplier = 1.5  # Extra buffer time (50% more than speaking time)
        
    def calculate_dynamic_timeout_from_message(self, message_content: str) -> float:
        """
        Calculate dynamic silence timeout based on agent message word count.
        
        Args:
            message_content: The agent's message content
            
        Returns:
            Calculated timeout in seconds
        """
        if not message_content:
            return self.base_silence_timeout
        
        # Count words in the message
        word_count = len(message_content.split())
        
        # Calculate estimated speaking time
        speaking_time = word_count / self.words_per_second
        
        # Add buffer time for user to process and respond
        timeout_with_buffer = speaking_time * self.buffer_multiplier
        
        # Apply min/max constraints
        dynamic_timeout = max(self.min_timeout, min(self.max_timeout, timeout_with_buffer))
        
        logger.debug(f"📊 Dynamic timeout calculation - Words: {word_count}, "
                    f"Speaking time: {speaking_time:.1f}s, "
                    f"Timeout with buffer: {timeout_with_buffer:.1f}s, "
                    f"Final timeout: {dynamic_timeout:.1f}s")
        
        return dynamic_timeout
    
    def set_dynamic_timeout(self, timeout: float):
        """Set the current dynamic timeout"""
        self.current_silence_timeout = max(self.min_timeout, min(self.max_timeout, timeout))
        logger.debug(f"🔄 Silence timeout updated to {self.current_silence_timeout:.1f}s")
    
    def reset_to_base_timeout(self):
        """Reset timeout to base value"""
        self.current_silence_timeout = self.base_silence_timeout
        logger.debug(f"🔄 Silence timeout reset to base value: {self.base_silence_timeout:.1f}s")
        
    def reset_user_activity(self):
        """Reset the timer when user speaks"""
        self.last_user_activity = time.time()
        self.prompt_count = 0
        
    def reset_activity_for_any_message(self):
        """Reset the timer for any activity (user or agent messages)"""
        current_time = time.time()
        time_since_last_activity = current_time - self.last_user_activity
        self.last_user_activity = current_time
        self.prompt_count = 0
        
        # Reset to base timeout when activity is detected (user responds)
        if self.current_silence_timeout != self.base_silence_timeout:
            logger.debug(f"🔄 Activity detected - reset silence timer and timeout "
                        f"(was silent for {time_since_last_activity:.1f}s, "
                        f"timeout was {self.current_silence_timeout:.1f}s, now {self.base_silence_timeout:.1f}s)")
            self.current_silence_timeout = self.base_silence_timeout
        else:
            logger.debug(f"🔄 Activity detected - reset silence timer (was silent for {time_since_last_activity:.1f}s)")
        
    def get_timeout_stats(self) -> dict:
        """Get current timeout statistics"""
        return {
            'base_timeout': self.base_silence_timeout,
            'current_timeout': self.current_silence_timeout,
            'min_timeout': self.min_timeout,
            'max_timeout': self.max_timeout,
            'words_per_second': self.words_per_second,
            'buffer_multiplier': self.buffer_multiplier
        }
        
    def start_monitoring(self):
        """Start monitoring for silence"""
        self.is_monitoring = True
        self.last_user_activity = time.time()
        self.last_prompt_time = 0
        self.prompt_count = 0
        
    def stop_monitoring(self):
        """Stop monitoring for silence"""
        self.is_monitoring = False
        
    def should_prompt_user(self) -> bool:
        """Check if we should prompt the user due to silence"""
        if not self.is_monitoring:
            return False
            
        current_time = time.time()
        time_since_activity = current_time - self.last_user_activity
        time_since_last_prompt = current_time - self.last_prompt_time
        
        # First prompt after initial silence timeout (using dynamic timeout)
        if (self.prompt_count == 0 and 
            time_since_activity >= self.current_silence_timeout):
            return True
            
        # Subsequent prompts after prompt interval
        if (self.prompt_count > 0 and 
            self.prompt_count < self.max_prompts and
            time_since_last_prompt >= self.prompt_interval):
            return True
            
        return False
        
    def mark_prompt_sent(self):
        """Mark that a prompt was sent"""
        self.last_prompt_time = time.time()
        self.prompt_count += 1
        
    def should_end_session(self) -> bool:
        """Check if we should end the session due to too many silence prompts"""
        return self.prompt_count >= self.max_prompts
        
    def get_prompt_message(self) -> str:
        """Get appropriate prompt message based on prompt count"""
        if self.prompt_count == 0:
            return "क्या आप वहाँ हैं? क्या आप मेडिकल एग्जामिनेशन जारी रखना चाहते हैं?"
        elif self.prompt_count == 1:
            return "मुझे लगता है आप मुझे सुन नहीं पा रहे। कृपया बताएं कि क्या आप तैयार हैं?"
        else:
            return "अगर आप वहाँ नहीं हैं तो मैं यह सेशन समाप्त कर दूंगी। कृपया जवाब दें।"


class EnhancedNoiseFilter:
    """
    Enhanced noise filtering system for telemedical examinations in noisy environments.
    Uses multiple techniques to identify and focus on the primary speaker while 
    filtering out background conversations and noise.
    """
    
    def __init__(self, 
                 volume_threshold: float = 0.015,  # Lower threshold for better sensitivity
                 energy_threshold: float = 0.05,   # Lower energy threshold
                 consistency_threshold: int = 2,   # Reduced consistency requirement
                 speaker_change_sensitivity: float = 0.3):
        # Basic thresholds - made more sensitive for better user voice pickup
        self.volume_threshold = volume_threshold
        self.energy_threshold = energy_threshold
        
        # Advanced filtering parameters - reduced for better responsiveness
        self.consistency_threshold = consistency_threshold  # Frames needed to confirm speaker change
        self.speaker_change_sensitivity = speaker_change_sensitivity
        
        # Audio analysis history
        self.recent_volumes = []
        self.recent_energies = []
        self.recent_frequencies = []
        self.max_history = 100  # Increased for better baseline
        
        # Primary speaker characteristics
        self.primary_speaker_volume_range = None
        self.primary_speaker_energy_range = None
        self.primary_speaker_established = False
        self.frames_since_primary_established = 0
        
        # Background noise estimation
        self.background_noise_level = 0.01
        self.noise_samples = []
        self.max_noise_samples = 30
        
        # Speaker consistency tracking
        self.consistent_speaker_frames = 0
        self.speaker_change_candidates = 0
        
        # Statistics
        self.total_frames = 0
        self.primary_speaker_frames = 0
        self.background_noise_frames = 0
        self.rejected_frames = 0
        
        # Emergency fallback mode
        self.use_basic_filtering = False  # Can be enabled if enhanced filtering is too strict
        
        logger.info("Enhanced noise filter initialized for telemedical examination")
    
    def _calculate_spectral_features(self, audio_data: np.ndarray) -> dict:
        """Calculate spectral features to help distinguish speakers"""
        try:
            # Calculate FFT for frequency analysis
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(magnitude)
            
            # Calculate spectral centroid (brightness measure)
            freqs = np.fft.rfftfreq(len(audio_data), 1/16000)  # Assuming 16kHz sample rate
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
            
            # Calculate spectral rolloff (90% of energy)
            cumsum = np.cumsum(magnitude)
            rolloff_idx = np.where(cumsum >= 0.9 * cumsum[-1])[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            
            return {
                'dominant_frequency': freqs[dominant_freq_idx] if dominant_freq_idx < len(freqs) else 0,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'high_frequency_energy': np.sum(magnitude[len(magnitude)//2:]) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
            }
        except Exception as e:
            logger.debug(f"Error calculating spectral features: {e}")
            return {'dominant_frequency': 0, 'spectral_centroid': 0, 'spectral_rolloff': 0, 'high_frequency_energy': 0}
    
    def _update_background_noise_estimation(self, rms: float, energy: float):
        """Update background noise level estimation"""
        # If this appears to be background noise (low energy, consistent level)
        if rms < self.volume_threshold * 1.5 and energy < self.energy_threshold * 1.5:
            self.noise_samples.append(rms)
            if len(self.noise_samples) > self.max_noise_samples:
                self.noise_samples.pop(0)
            
            if len(self.noise_samples) >= 10:
                self.background_noise_level = np.mean(self.noise_samples)
    
    def _establish_primary_speaker_profile(self, rms: float, energy: float, spectral_features: dict):
        """Establish or update the primary speaker's audio profile"""
        if not self.primary_speaker_established:
            # More lenient requirements to establish primary speaker
            if (rms > self.volume_threshold * 1.5 and  # Reduced from 2x to 1.5x
                energy > self.energy_threshold * 1.5 and  # Reduced from 2x to 1.5x
                rms > self.background_noise_level * 2):   # Reduced from 3x to 2x
                
                self.consistent_speaker_frames += 1
                
                if self.consistent_speaker_frames >= self.consistency_threshold:
                    # Establish primary speaker profile with wider tolerance ranges
                    recent_rms = self.recent_volumes[-self.consistency_threshold:]
                    recent_energy = self.recent_energies[-self.consistency_threshold:]
                    
                    self.primary_speaker_volume_range = (
                        np.mean(recent_rms) * 0.3,  # Lower bound - more inclusive
                        np.mean(recent_rms) * 4.0   # Upper bound - more inclusive
                    )
                    
                    self.primary_speaker_energy_range = (
                        np.mean(recent_energy) * 0.3,  # Lower bound - more inclusive
                        np.mean(recent_energy) * 4.0   # Upper bound - more inclusive
                    )
                    
                    self.primary_speaker_established = True
                    self.frames_since_primary_established = 0
                    
                    logger.info(f"🎯 Primary speaker established for telemedical examination!")
                    logger.info(f"   📊 Volume range: {self.primary_speaker_volume_range[0]:.4f} - {self.primary_speaker_volume_range[1]:.4f}")
                    logger.info(f"   ⚡ Energy range: {self.primary_speaker_energy_range[0]:.4f} - {self.primary_speaker_energy_range[1]:.4f}")
                    logger.info(f"   🔇 Background noise level: {self.background_noise_level:.4f}")
                    logger.info(f"   🎙️ Enhanced noise filtering now active - focusing on primary speaker only")
            else:
                self.consistent_speaker_frames = 0
    
    def _is_primary_speaker(self, rms: float, energy: float, spectral_features: dict) -> bool:
        """Determine if the current audio frame is from the primary speaker"""
        if not self.primary_speaker_established or self.primary_speaker_volume_range is None or self.primary_speaker_energy_range is None:
            return False
        
        # Check if audio characteristics match primary speaker profile
        volume_match = (self.primary_speaker_volume_range[0] <= rms <= self.primary_speaker_volume_range[1])
        energy_match = (self.primary_speaker_energy_range[0] <= energy <= self.primary_speaker_energy_range[1])
        
        # Additional checks for background noise rejection - made more lenient
        above_noise_floor = rms > self.background_noise_level * 1.8  # Reduced from 2.5x to 1.8x
        sufficient_energy = energy > self.energy_threshold * 0.8     # Reduced threshold
        
        # Spectral consistency check - made more forgiving
        reasonable_frequency = 30 <= spectral_features.get('dominant_frequency', 0) <= 8000  # Wider frequency range
        
        # More lenient decision - allow if most criteria are met
        criteria_met = sum([volume_match, energy_match, above_noise_floor, sufficient_energy, reasonable_frequency])
        is_primary = bool(criteria_met >= 3)  # Need at least 3 out of 5 criteria
        
        if is_primary:
            self.frames_since_primary_established += 1
            # Gradually adapt the profile to account for natural variation
            if self.frames_since_primary_established % 30 == 0:  # More frequent adaptation
                self._adapt_speaker_profile(rms, energy)
        
        return is_primary
    
    def _adapt_speaker_profile(self, rms: float, energy: float):
        """Gradually adapt the primary speaker profile to account for natural variation"""
        if (self.primary_speaker_established and 
            self.primary_speaker_volume_range is not None and 
            self.primary_speaker_energy_range is not None):
            
            # Slight adjustment toward current values (10% adaptation rate)
            current_vol_center = (self.primary_speaker_volume_range[0] + self.primary_speaker_volume_range[1]) / 2
            current_energy_center = (self.primary_speaker_energy_range[0] + self.primary_speaker_energy_range[1]) / 2
            
            new_vol_center = current_vol_center * 0.9 + rms * 0.1
            new_energy_center = current_energy_center * 0.9 + energy * 0.1
            
            vol_range_width = self.primary_speaker_volume_range[1] - self.primary_speaker_volume_range[0]
            energy_range_width = self.primary_speaker_energy_range[1] - self.primary_speaker_energy_range[0]
            
            self.primary_speaker_volume_range = (
                new_vol_center - vol_range_width/2,
                new_vol_center + vol_range_width/2
            )
            
            self.primary_speaker_energy_range = (
                new_energy_center - energy_range_width/2,
                new_energy_center + energy_range_width/2
            )
    
    def should_process_audio(self, audio_frame: rtc.AudioFrame) -> bool:
        """
        Determine if audio frame should be processed for STT based on enhanced noise filtering.
        Returns True if the frame is likely from the primary speaker.
        """
        self.total_frames += 1
        
        # Emergency fallback to basic filtering if enabled
        if self.use_basic_filtering:
            try:
                audio_data = np.frombuffer(audio_frame.data, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(audio_data**2)))
                energy = float(np.sum(audio_data**2))
                return rms > 0.01 and energy > 0.03
            except:
                return True  # Process if we can't analyze
        
        try:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(audio_frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Calculate basic audio features
            rms = float(np.sqrt(np.mean(audio_data**2)))
            energy = float(np.sum(audio_data**2))
            
            # Calculate spectral features for better speaker discrimination
            spectral_features = self._calculate_spectral_features(audio_data)
            
            # Update rolling history
            self.recent_volumes.append(rms)
            self.recent_energies.append(energy)
            self.recent_frequencies.append(spectral_features.get('dominant_frequency', 0))
            
            if len(self.recent_volumes) > self.max_history:
                self.recent_volumes.pop(0)
                self.recent_energies.pop(0)
                self.recent_frequencies.pop(0)
            
            # Update background noise estimation
            self._update_background_noise_estimation(rms, energy)
            
            # Establish or update primary speaker profile
            self._establish_primary_speaker_profile(rms, energy, spectral_features)
            
            # Determine if this frame should be processed
            if self.primary_speaker_established:
                should_process = self._is_primary_speaker(rms, energy, spectral_features)
            else:
                # Before primary speaker is established, use more permissive basic thresholds
                # This ensures we can pick up initial user speech to establish the profile
                should_process = (rms > self.volume_threshold * 0.8 and  # Even more lenient
                                energy > self.energy_threshold * 0.7 and  # More lenient
                                rms > self.background_noise_level * 1.2)   # Much more lenient
            
            # Update statistics
            if should_process:
                self.primary_speaker_frames += 1
            elif rms < self.background_noise_level * 1.5:
                self.background_noise_frames += 1
            else:
                self.rejected_frames += 1
            
            # Log statistics periodically
            if self.total_frames % 200 == 0:
                self._log_filtering_stats()
            
            if should_process:
                logger.debug(f"✅ Primary speaker audio - RMS: {rms:.4f}, Energy: {energy:.6f}, "
                           f"Freq: {spectral_features.get('dominant_frequency', 0):.1f}Hz, "
                           f"Established: {self.primary_speaker_established}")
            else:
                if self.total_frames % 100 == 0:  # Log rejections less frequently
                    logger.debug(f"❌ Audio rejected - RMS: {rms:.4f} (thresh: {self.volume_threshold:.4f}), "
                               f"Energy: {energy:.6f} (thresh: {self.energy_threshold:.6f}), "
                               f"Noise: {self.background_noise_level:.4f}, "
                               f"Established: {self.primary_speaker_established}")
            
            return bool(should_process)
            
        except Exception as e:
            logger.error(f"Error in enhanced noise filtering: {e}")
            # Fallback to basic processing
            return True
    
    def _log_filtering_stats(self):
        """Log detailed filtering statistics"""
        if self.total_frames > 0:
            primary_pct = (self.primary_speaker_frames / self.total_frames) * 100
            noise_pct = (self.background_noise_frames / self.total_frames) * 100
            rejected_pct = (self.rejected_frames / self.total_frames) * 100
            
            logger.info(f"Enhanced Noise Filter Stats - Total: {self.total_frames}, "
                       f"Primary: {primary_pct:.1f}%, Background: {noise_pct:.1f}%, "
                       f"Rejected: {rejected_pct:.1f}%, "
                       f"Noise Level: {self.background_noise_level:.4f}, "
                       f"Speaker Established: {self.primary_speaker_established}")
    
    def reset_primary_speaker(self):
        """Reset the primary speaker profile (useful for new sessions)"""
        self.primary_speaker_established = False
        self.primary_speaker_volume_range = None
        self.primary_speaker_energy_range = None
        self.consistent_speaker_frames = 0
        self.frames_since_primary_established = 0
        logger.info("Primary speaker profile reset")
    
    def enable_basic_filtering(self):
        """Enable basic filtering mode (fallback when enhanced filtering is too strict)"""
        self.use_basic_filtering = True
        logger.warning("🔄 Switched to basic filtering mode - enhanced filtering was too strict")
    
    def disable_basic_filtering(self):
        """Disable basic filtering mode (return to enhanced filtering)"""
        self.use_basic_filtering = False
        logger.info("🎯 Switched back to enhanced filtering mode")
    
    def get_filtering_summary(self) -> dict:
        """Get a summary of filtering performance"""
        return {
            'total_frames': self.total_frames,
            'primary_speaker_frames': self.primary_speaker_frames,
            'background_noise_frames': self.background_noise_frames,
            'rejected_frames': self.rejected_frames,
            'primary_speaker_established': self.primary_speaker_established,
            'background_noise_level': self.background_noise_level,
            'primary_speaker_percentage': (self.primary_speaker_frames / self.total_frames * 100) if self.total_frames > 0 else 0
        }


# Custom Agent with enhanced noise filtering for telemedical examinations in noisy environments
class VolumeFilteredAssistant(Agent):
    """
    Enhanced Voice Assistant with sophisticated noise filtering for telemedical examinations.
    
    Features:
    - Primary speaker identification and tracking
    - Background noise estimation and filtering  
    - Spectral analysis for speaker discrimination
    - Adaptive thresholds based on session characteristics
    - Comprehensive noise filtering statistics
    
    This implementation is specifically designed for telemedical examinations where:
    1. Multiple people may be present in the room
    2. Background conversations need to be filtered out
    3. Only the primary speaker (patient/family member) should be processed
    4. Robust operation in noisy household environments is required
    """
    def __init__(self, room_name: str = "unknown", family_details: str = "") -> None:
        
        # Family details section
        family_section = family_details or "No family details available."

        logger.info("VISHAL_PANDEY")
        logger.info(family_section)
        
        # Base instructions with the JSON questions (no placeholders needed in JSON)
        final_instructions = f"""
        

System Prompt:

You are 'Erica Tele MER Agent,' a friendly, sophisticated, and multilingual AI voice assistant. You are female in gender, so it is importatnt that you make all the conversation in the female voice. You act as a virtual Medical examination Tellercaller representative from Care Health Insurance. Your job is to conduct a Tele Medical Examination Report (TMER) by asking health-related questions in a natural, human-like conversation based on the family details and question list provided below.

Your default language is Hindi. You always need to write in the language in which communication is happening.

---

1. Core Mission and Context

Your single goal is to accurately and efficiently conduct a medical history examination. The data you collect is of the utmost importance for the health underwriting process, so precision, clarity, and detail are critical. You must be an active listener, understanding user responses to clarify medical conditions with intelligent, reflexive questions.

---

2. Language, Tone, and Persona

Language Support & Dynamic Switching:
FIRST QUESTION - Language Preference: Always start the conversation with a clear, purposeful greeting that establishes your role and asks for language preference.
Example Script: "नमस्ते, मेरा नाम एरिक है और मैं Care Health Insurance से आपका Tele-Medical Examination करने के लिए कॉल कर रहा हूँ। To begin, आप किस भाषा में बात करना पसंद करेंगे - Hindi या English?"

Language Rules:
- Immediately switch to the user's preferred language.
- Use natural code-mixing (e.g., Hinglish, Tanglish) as a native speaker would.
- Keep ALL medical terms, procedures, and tests in English (e.g., Diabetes, Angioplasty, MRI), but explain them in the user's chosen language.
- If a user finds a medical term complex, explain it in simple, layman language.
- If the user switches language mid-conversation, adapt immediately and seamlessly.

Tone & Style:
- Voice & Tone: Use a warm, empathetic, and conversational tone.
- Cultural Sensitivity: Use appropriate cultural greetings and a soft approach.
- Empathy: Use empathetic phrases when needed. Example: "Don't worry, मैं समझ सकती हूँ कि आप कैसा महसूस कर रहे हैं।"

---

3. Family Members for Examination

Below are the details of the family for this policy. Your task is to conduct the examination for all the members where examination_required is true.

{family_section}

---

4. Session Flow & Master Plan

Part A: Pre-Examination Setup

1. Language Selection: Start with the language preference question as defined above.

2. Introduction: After language selection, continue with a warm greeting.

3. Set the Agenda: Using the provided family details, identify the members who need the examination. Explain the new, efficient process to the user.
Script: "तो जैसा कि मैं देख सकती हूँ, हमें [Person 1 Name] और [Person 2 Name] का मेडिकल एग्जामिनेशन करना है। मैं एक-एक करके हेल्थ कंडीशंस के बारे में पूछूँगी, और आप बस यह बताइएगा कि यह कंडीशन दोनों में से किसी को है या नहीं। इससे प्रोसेस जल्दी और आसान हो जाएगा।"

4. Consent (CRITICAL): Before asking any medical questions, you must get the user's consent.
Script: "Before we proceed, please know that this session will be recorded for medical and quality purposes. क्या आप आगे बढ़ने के लिए अपनी सहमति देते हैं? I need your consent to proceed."
- If User says "Yes" (Haan): Continue to the examination.
- If User says "No" (Nahi): End the call politely. Say: "No problem at all. मैं समझती हूँ। We'll end the session here. आपके टाइम के लिए थैंक यू!"

Part B: The Medical Examination (Question-by-Question)

This is the main part of your task. For each question in the Master List, you will ask it for the group and then clarify.

Guiding Principles for This Section:
- Principle 1: Ask Generically. For each bullet point, formulate a single question for the whole group. Use their names for clarity.
Example: For the "Diabetes" bullet point, ask: "तो शुरू करते हैं। क्या [Person 1 Name] या [Person 2 Name] में से किसी को Diabetes या शुगर की प्रॉब्लम है?"

- Principle 2: Clarify and Probe Dynamically. This is the most important step.
- If the answer is NO: Simply move to the next bullet point.
- If the answer is YES: Your immediate goal is to find out who has the condition. Generate a natural, dynamic question to clarify. Do not use the same sentence every time.
Example of a good follow-up: "ओके, आप प्लीज़ कन्फर्म कर सकते हैं यह प्रॉब्लम किसको है? [Person 1 Name] को, [Person 2 Name] को, या दोनों को?" Another example: "ठीक है, यह कंडीशन किन के लिए है?"

- Principle 3: Ask Listed Sub-Questions. Once the specific person(s) is identified, ask the follow-up questions listed under the bullet point for that person only. Also ask the followup questions according the question asked from your knowledge also.
Example: "ठीक है। तो [Person's Name] के लिए, यह उन्हें पहली बार कब डिटेक्ट हुआ था? ..." and so on from your knowledge

- Principle 4: Handle the "Both" Scenario. If the user says both members have the condition, you must get the details for each one sequentially.
Example: "ओके, दोनों के लिए नोट कर लेती हूँ। पहले [Person 1 Name] के लिए बताएं... (ask all sub-questions). ... बहुत शुक्रिया। अब [Person 2 Name] के लिए बताएं... (ask all sub-questions)."

- Principle 5: Avoid Endless Loops. Your role is to clarify the listed questions, not to conduct a full diagnosis. If a user's answer to a sub-question is vague, you may ask one or two clarifying follow-ups. After that, you must move on to the next main bullet point in the list. Do not create new lines of medical inquiry. You need to make sure that you do not get stuck in endless loop of question.

---
Master Question List (Health Conditions)

- Previous Insurance Policy with Care Health Insurance, or any previous insurance claims.
IF YES: (Clarify for whom). "प्लीज़ उसकी डिटेल बता दीजिए।"

- Diabetes, sugar problem, or high blood sugar.
IF YES: (Clarify for whom, then ask) "पहली बार कब डिटेक्ट हुआ था?" ... "क्या कभी Insulin लेने के लिए कहा गया है?" ... "क्या Diabetes से जुड़ी कोई कॉम्प्लीकेशन्स हुई हैं, जैसे विज़न या किडनी प्रॉब्लम?"

- High cholesterol or Lipid disorder.

- High BP, यानी Hypertension.
IF YES: (Clarify for whom, then ask) "यह उन्हें पहली बार कब डिटेक्ट हुआ था?"

- Cardiac (heart) problem, or experienced chest pain.
IF YES: (Clarify for whom, then ask) "प्लीज़ बताइए इनमें से कौन सी प्रॉब्लम है: Angioplasty / Bypass, Pacemaker, Heart Valve disorder, या दिल की dhadkan से जुड़ी कोई प्रॉब्लम?"

- Joint pain in Knee, Shoulder, Hip etc.
IF YES: (Clarify for whom, then ask) "प्लीज़ इसकी डिटेल बताइए और यह भी बताइए कि इसके लिए कौन सी दवा ली या प्रिसक्राइब की गई थी।"

- Vision related problem like blurry or hazy vision.
IF YES: (Clarify for whom, then ask) "प्लीज़ बताइए इनमें से कौन सी प्रॉब्लम है: Cataract, Retinal disorder, या Glaucoma?"

- Gall bladder, kidney or urinary stones (pathri).
IF YES: (Clarify for whom, then ask) "यह Gall Bladder Stone है या Kidney/Urinary Stone?"

- Prostate related problem, or any urinary complaints. (This question is for male members only, so ask it specifically if there are males in the group e.g., "अब एक सवाल [Male Person's Name] के लिए, क्या उन्हें...").
IF YES: "यह कब डायग्नोस हुआ था और इसके बारे में थोड़ा और बताइए प्लीज़।"

- Gynaecological problems like abnormal bleeding, cyst or fibroid. (This question is for female members only, ask it specifically e.g., "अब एक सवाल [Female Person's Name] के लिए, क्या उन्हें...")
IF YES: "यह कब डायग्नोस हुआ था और इसके बारे में थोड़ा और बताइए प्लीज़।"

- Any form of Thyroid disorder.
IF YES: (Clarify for whom, then ask) "यह किस टाइप का Thyroid डिसऑर्डर है और इसके लिए कौन सी मेडिसिन लेते हैं? और यह कब से है?"

- Admitted to a hospital, or undergone or advised for a surgery.
IF YES: (Clarify for whom, then ask) "प्लीज़ हॉस्पिटल में एडमिट होने या सर्जरी का रीज़न बताइए।"

- Any medical test like Ultrasound, CT scan, MRI, 2D Echo with a positive finding.
IF YES: (Clarify for whom, then ask) "प्लीज़ उसके बारे में बताइए।"

- Symptoms such as pain in abdomen, breathlessness, or pain in any other part of the body.
IF YES: (Clarify for whom, then ask) "प्लीज़ इसके बारे में डिटेल में बताइए।"

- Habit of smoking, consuming alcohol, or chewing tobacco/pan masala/gutka.
IF YES: (Clarify for whom and for which habit, then ask) "कितनी क्वांटिटी और कब से?"

---

Part C: Final Personal Details

After completing the group questions, switch to a person-by-person mode for the final details.
Script: "ग्रेट, हमने हेल्थ कंडीशंस पूरी कर ली हैं। अब बस आखिर में, मैं दोनों की हाइट और वेट अलग-अलग पूछना चाहूँगी।"

1. Height & Weight:
- "पहले [Person 1 Name] की हाइट, फ़ीट और इंच में, बता सकते हैं?"
- "और उनका वेट, Kgs में?"
- "शुक्रिया। अब [Person 2 Name] की हाइट, फ़ीट और इंच में, क्या है?"
- "और उनका वेट, Kgs में?"

2. Final Disclosure:
- "आख़िरी सवाल। अब तक हमने जो भी डिस्कस किया, उसके अलावा क्या कोई और कंडीशन, बीमारी, या प्रोसीजर है जो आप दोनों में से किसी के बारे में बताना चाहेंगे?"
IF YES: (Clarify for whom and ask for details).

---

Part D: Concluding the Call

Once everything is complete, end the call politely.
Script: "बहुत बहुत शुक्रिया। हमने [Person 1 Name] और [Person 2 Name] दोनों के लिए सभी ज़रूरी मेडिकल जानकारी कलेक्ट कर ली है। आपके अनमोल समय और सहयोग के लिए धन्यवाद। आपका दिन शुभ हो!"
"""
        
        super().__init__(instructions=final_instructions)
        self.enhanced_noise_filter = EnhancedNoiseFilter(
            volume_threshold=0.015,      # Lower for better pickup
            energy_threshold=0.05,       # Lower for better pickup  
            consistency_threshold=2,     # Require only 2 consistent frames
            speaker_change_sensitivity=0.3
        )
        self.ignored_frames = 0
        self.processed_frames = 0
        
        # Initialize local audio recorder as fallback
        self.local_recorder = LocalAudioRecorder(room_name)
        
        # Initialize silence detector - DISABLED
        # self.silence_detector = SilenceDetector(
        #     base_silence_timeout=20.0,  # Base timeout for silence detection
        #     prompt_interval=20.0   # Time between subsequent prompts
        # )
        
        # Session reference to send prompts
        self._session: Optional[AgentSession] = None
        self._last_user_message_count = 0
        self._last_agent_message_count = 0  # Track agent messages as activity too
        
        # Speech state tracking to prevent interruptions
        self._is_speaking = False
        self._speech_start_time = 0.0
        self._last_speech_text = ""
    
    def should_process_audio(self, audio_frame: rtc.AudioFrame) -> bool:
        """Filter audio based on enhanced noise filtering to focus on primary speaker"""
        should_process = self.enhanced_noise_filter.should_process_audio(audio_frame)
        
        # Fallback: If enhanced filter hasn't established primary speaker after many frames, 
        # use basic filtering to ensure we don't miss user input
        total_frames = self.enhanced_noise_filter.total_frames
        if (not self.enhanced_noise_filter.primary_speaker_established and 
            total_frames > 500 and  # After 500 frames (~25 seconds)
            total_frames % 100 == 0):  # Check periodically
            
            # Calculate basic audio characteristics for fallback
            audio_data = np.frombuffer(audio_frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio_data**2)))
            energy = float(np.sum(audio_data**2))
            
            # Very basic threshold check as fallback
            if rms > 0.01 and energy > 0.03:
                should_process = True
                logger.warning(f"🔄 Using fallback audio processing - no primary speaker established after {total_frames} frames")
        
        # Always record audio frames (even if not processing for STT)
        self.local_recorder.add_audio_frame(audio_frame)
        
        if should_process:
            self.processed_frames += 1
            # Reset silence detector when user speaks (using universal activity reset) - DISABLED
            # self.silence_detector.reset_activity_for_any_message()
        else:
            self.ignored_frames += 1
        
        # Log statistics periodically with enhanced filtering info and timeout stats
        if (self.processed_frames + self.ignored_frames) % 200 == 0:
            total = self.processed_frames + self.ignored_frames
            filtering_summary = self.enhanced_noise_filter.get_filtering_summary()
            # timeout_stats = self.get_silence_timeout_stats()  # DISABLED
            logger.info(f"Enhanced Audio Filtering Stats - Processed: {self.processed_frames}/{total} ({self.processed_frames/total*100:.1f}%), "
                       f"Primary Speaker Established: {filtering_summary['primary_speaker_established']}, "
                       f"Background Noise Level: {filtering_summary['background_noise_level']:.4f}")
                       # f"Silence Timeout: {timeout_stats['current_silence_timeout']:.1f}s/{timeout_stats['base_timeout']:.1f}s")  # DISABLED
        
        return should_process
    
    def set_session(self, session):
        """Set the session reference for sending prompts"""
        self._session = session
        
    def start_silence_monitoring(self):
        """Start monitoring for user silence - DISABLED"""
        # self.silence_detector.start_monitoring()
        logger.info("Silence monitoring is disabled")
        
    def stop_silence_monitoring(self):
        """Stop monitoring for user silence - DISABLED"""
        # self.silence_detector.stop_monitoring()
        logger.info("Silence monitoring is disabled")
    
    def reset_noise_filter(self):
        """Reset the enhanced noise filter for a new session"""
        self.enhanced_noise_filter.reset_primary_speaker()
        logger.info("Enhanced noise filter reset for new session")
    
    def enable_basic_audio_filtering(self):
        """Enable basic filtering if enhanced filtering is too strict"""
        self.enhanced_noise_filter.enable_basic_filtering()
        logger.warning("Switched to basic audio filtering mode")
    
    def disable_basic_audio_filtering(self):
        """Disable basic filtering and return to enhanced filtering"""
        self.enhanced_noise_filter.disable_basic_filtering()
        logger.info("Returned to enhanced audio filtering mode")
    
    def get_noise_filtering_stats(self) -> dict:
        """Get comprehensive noise filtering statistics"""
        basic_stats = {
            'total_audio_frames': self.processed_frames + self.ignored_frames,
            'processed_frames': self.processed_frames,
            'ignored_frames': self.ignored_frames,
            'processing_rate': (self.processed_frames / (self.processed_frames + self.ignored_frames) * 100) if (self.processed_frames + self.ignored_frames) > 0 else 0
        }
        enhanced_stats = self.enhanced_noise_filter.get_filtering_summary()
        return {**basic_stats, **enhanced_stats}
        
    def get_silence_timeout_stats(self) -> dict:
        """Get silence timeout statistics for monitoring - DISABLED"""
        # Return empty stats since silence detection is disabled
        return {
            'current_silence_timeout': 0,
            'base_timeout': 0,
            'is_monitoring': False,
            'prompt_count': 0,
            'disabled': True
        }
        
    def set_dynamic_silence_timeout_for_message(self, message: str):
        """Set dynamic silence timeout based on a message that the agent is about to send - DISABLED"""
        # dynamic_timeout = self.silence_detector.calculate_dynamic_timeout_from_message(message)
        # self.silence_detector.set_dynamic_timeout(dynamic_timeout)
        logger.debug(f"🤖 Silence detection disabled - ignoring timeout setting for message")
        
    async def reliable_say(self, message: str, wait_for_completion: bool = True) -> bool:
        """
        Reliably deliver speech with proper completion waiting and error handling.
        Prevents speech interruption by tracking speech state.
        
        Args:
            message: The text message to speak
            wait_for_completion: Whether to wait for speech completion before returning
            
        Returns:
            True if speech was delivered successfully, False otherwise
        """
        if not self._session or not message.strip():
            logger.warning("Cannot speak: No session or empty message")
            return False
            
        try:
            # Mark that we're starting to speak
            self._is_speaking = True
            self._speech_start_time = time.time()
            self._last_speech_text = message
            
            # Log the full message that should be spoken
            logger.info(f"🎤 Speaking complete message: '{message}'")
            
            # Ensure message ends with proper punctuation for better TTS delivery
            if not message.strip().endswith(('.', '!', '?', '।')):
                message = message.strip() + '।'  # Add Hindi punctuation
            
            # Split very long messages into smaller chunks if needed (to prevent TTS timeouts)
            max_chunk_length = 200  # characters
            if len(message) > max_chunk_length:
                logger.info(f"Message is long ({len(message)} chars), splitting into chunks")
                sentences = message.split('।')
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence + '।') <= max_chunk_length:
                        current_chunk += sentence + '।'
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + '।'
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Speak each chunk sequentially
                for i, chunk in enumerate(chunks):
                    logger.info(f"🎤 Speaking chunk {i+1}/{len(chunks)}: '{chunk}'")
                    await self._session.say(chunk)
                    
                    # Add small delay between chunks to ensure proper delivery
                    if i < len(chunks) - 1:  # Don't wait after the last chunk
                        await asyncio.sleep(0.5)  # Increased delay for reliability
                        
            else:
                # Speak the complete message at once
                await self._session.say(message)
            
            # Wait for speech completion if requested
            if wait_for_completion:
                # Estimate speech duration and wait accordingly
                word_count = len(message.split())
                estimated_duration = (word_count / 2.0) + 1.5  # Conservative: 2 words/sec + 1.5s buffer
                logger.debug(f"Waiting {estimated_duration:.1f}s for speech completion ({word_count} words)")
                await asyncio.sleep(estimated_duration)
            
            # Mark that we've finished speaking
            self._is_speaking = False
            speech_duration = time.time() - self._speech_start_time
            
            logger.info(f"✅ Successfully delivered complete message (took {speech_duration:.1f}s)")
            return True
            
        except Exception as e:
            self._is_speaking = False  # Ensure we reset state even on error
            logger.error(f"❌ Failed to deliver speech: {e}")
            return False
    
    def should_interrupt_speech(self) -> bool:
        """
        Determine if agent speech should be interrupted.
        This prevents premature interruption while allowing necessary stops.
        """
        if not self._is_speaking:
            return True  # Not speaking, so can be interrupted
            
        # Don't interrupt if we just started speaking (give minimum time)
        min_speech_time = 1.0  # seconds
        speech_duration = time.time() - self._speech_start_time
        
        if speech_duration < min_speech_time:
            logger.debug(f"Preventing early interruption (only {speech_duration:.1f}s of {min_speech_time}s min)")
            return False
            
        return True  # Allow interruption after minimum speech time

    async def safe_say(self, message: str) -> bool:
        """
        Safely say a message, checking for speech state to prevent interruptions.
        This is the main method that should be used for all agent speech.
        """
        if self._is_speaking:
            logger.warning(f"Already speaking, queuing message: '{message[:50]}...'")
            # Wait for current speech to finish before starting new speech
            max_wait = 10.0  # Maximum wait time in seconds
            wait_start = time.time()
            
            while self._is_speaking and (time.time() - wait_start) < max_wait:
                await asyncio.sleep(0.1)
                
            if self._is_speaking:
                logger.error(f"Timeout waiting for speech completion, forcing new speech")
                self._is_speaking = False
        
        return await self.reliable_say(message, wait_for_completion=True)
    
    def is_currently_speaking(self) -> bool:
        """Check if the agent is currently speaking"""
        return self._is_speaking
        
    def get_speech_state(self) -> dict:
        """Get current speech state information"""
        return {
            'is_speaking': self._is_speaking,
            'speech_start_time': self._speech_start_time,
            'last_speech_text': self._last_speech_text,
            'speech_duration': time.time() - self._speech_start_time if self._is_speaking else 0
        }
        
    async def check_and_handle_silence(self) -> bool:
        """Check for silence and send prompt if needed. Returns True if session should end. - DISABLED"""
        # Silence detection is disabled, always return False (don't end session)
        return False


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


async def generate_structured_report(transcript_data, family_details: str = ""):
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

    This summary we need to gemerate for each member in the family, the final report should have the json for each member in the family for which we need to conduct the tele medical examination.

    Below are the basic details about the family members:

    {family_details}

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
        "<Family Member 1 Name>":
        {{
            "Previous Insurance Claims": {{
                "answered_yes_no": "Yes or No",
                "details": "Extracted Details or Inferred Details"
            }},
            "Diabetes/Sugar Problem": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including detection date, insulin usage, complications"
            }},
            "High Cholesterol/Lipid Disorder": {{
                "answered_yes_no": "Yes or No",
                "details": "Details"
            }},
            "Hypertension/High Blood Pressure": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including detection date"
            }},
            "Cardiac/Heart Problems": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including type of problem, procedures, timeline"
            }},
            "Joint Pain": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including location, medications"
            }},
            "Vision Problems": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including type, operated/unoperated"
            }},
            "Gall Bladder/Kidney/Urinary Stones": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including type, treatment status"
            }},
            "Prostate Problems": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including diagnosis date, symptoms"
            }},
            "Gynaecological Problems": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including diagnosis date, type"
            }},
            "Thyroid Disorder": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including type, medication, duration"
            }},
            "Hospitalization/Surgery": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including reason, date, location"
            }},
            "Medical Tests": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including type of test, findings"
            }},
            "Other Symptoms": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including pain location, breathlessness"
            }},
            "Smoking Habits": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including quantity, duration"
            }},
            "Alcohol Consumption": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including frequency, quantity"
            }},
            "Tobacco/Pan Masala Habits": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including frequency, quantity"
            }},
            "Height and Weight": {{
                "height": "Height in feet and inches",
                "weight": "Weight in KGs"
            }}
        }},

        "<Family Member 2 Name>": {{
            "Previous Insurance Claims": {{
                "answered_yes_no": "Yes or No",
                "details": "Extracted Details or Inferred Details"
            }},
            "Diabetes/Sugar Problem": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including detection date, insulin usage, complications"
            }},
            "High Cholesterol/Lipid Disorder": {{
                "answered_yes_no": "Yes or No",
                "details": "Details"
            }},
            "Hypertension/High Blood Pressure": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including detection date"
            }},
            "Cardiac/Heart Problems": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including type of problem, procedures, timeline"
            }},
            "Joint Pain": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including location, medications"
            }},
            "Vision Problems": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including type, operated/unoperated"
            }},
            "Gall Bladder/Kidney/Urinary Stones": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including type, treatment status"
            }},
            "Prostate Problems": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including diagnosis date, symptoms"
            }},
            "Gynaecological Problems": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including diagnosis date, type"
            }},
            "Thyroid Disorder": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including type, medication, duration"
            }},
            "Hospitalization/Surgery": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including reason, date, location"
            }},
            "Medical Tests": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including type of test, findings"
            }},
            "Other Symptoms": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including pain location, breathlessness"
            }},
            "Smoking Habits": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including quantity, duration"
            }},
            "Alcohol Consumption": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including frequency, quantity"
            }},
            "Tobacco/Pan Masala Habits": {{
                "answered_yes_no": "Yes or No",
                "details": "Details including frequency, quantity"
            }},
            "Height and Weight": {{
                "height": "Height in feet and inches",
                "weight": "Weight in KGs"
            }}
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
    # Fetch family details for the report prompt
    family_data = await mongo_manager.get_family_details(room_name)
    member_details = family_data.get('memberDetails', []) if family_data else []
    family_details_str = mongo_manager.format_family_details_for_prompt(member_details)
    structured_report = await generate_structured_report(transcript_data, family_details=family_details_str)
    
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
    
    # Extract customer information from family data using helper method
    customer_name, customer_phone = mongo_manager._extract_customer_info(family_data)
    
    # Update call session with recording URL and metadata
    await mongo_manager.update_call_session(
        room_name=room_name,
        recording_url=recording_url,
        status="completed",
        duration=session_duration,
        transcript_confidence=transcript_confidence,
        customer_name=customer_name,
        customer_phone=customer_phone,
        language="hindi"  # Default language for telemedical examination
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

    # Store the room name
    room_name = ctx.room.name
    logger.info(f"Starting agent for room: {room_name}")

    # Initialize MongoDB connection and fetch family details
    mongo_manager = MongoDBManager()
    await mongo_manager.connect()
    
    family_details = ""
    
    try:
        # Use room name directly to fetch family details (same as used for storing medical reports)
        logger.info(f"Fetching family details for room: {room_name}")
        
        # Fetch family details using room name directly
        family_data = await mongo_manager.get_family_details(room_name)
        if family_data:
            # Extract member details from the proposal data
            member_details = family_data.get('memberDetails', [])
            family_details = mongo_manager.format_family_details_for_prompt(member_details)
            logger.info("Successfully fetched and formatted family details")
        else:
            logger.warning(f"No family details found for room: {room_name}")
            
    except Exception as e:
        logger.error(f"Error fetching family details: {e}")
        # Continue with empty family details rather than failing
    
    # Close MongoDB connection
    await mongo_manager.close()

    session = AgentSession(
        stt=sarvam.STT(language="hi-IN", model="saarika:v2.5"),
        llm=openai.LLM(model="gpt-4o"),
        tts=sarvam.TTS(
            target_language_code="hi-IN", 
            speaker="anushka", 
            http_session=http_session, 
            speech_sample_rate=16000,
            enable_preprocessing=True  # Disabled to prevent speech interruptions and improve reliability
        ),
        # Optimized VAD configuration for faster response times while preventing speech interruption
        vad=ctx.proc.userdata.get("vad") or silero.VAD.load(
            activation_threshold=0.8,    # Slightly higher to prevent false activation during agent speech
            min_silence_duration=0.5,    # Increased slightly to ensure agent finishes speaking
            min_speech_duration=0.2,     # Require more speech duration to avoid interrupting agent
        ),
        # Use more conservative turn detection to prevent speech interruption
        turn_detection=MultilingualModel(),
    )

    recording_egress_id: Optional[str] = None
    recording_s3_url: Optional[str] = None
    agent = VolumeFilteredAssistant(room_name=room_name, family_details=family_details)
    recording_started = False

    # Set the session reference in the agent for silence prompts
    agent._session = session  # Store session reference directly
    
    # Reset and prepare the enhanced noise filter for the new session
    agent.reset_noise_filter()
    
    # Start silence monitoring task
    silence_monitoring_task = None

    async def silence_monitor():
        """Background task to monitor for silence and send prompts - DISABLED"""
        logger.info("Silence monitoring is disabled - background task will not run")
        return  # Exit immediately since silence detection is disabled

    async def cleanup_and_record():
        nonlocal recording_egress_id, recording_s3_url, silence_monitoring_task
        logger.info("Session ending, generating transcript and summary...")
        
        # Log final noise filtering statistics
        noise_stats = agent.get_noise_filtering_stats()
        logger.info(f"Final Enhanced Noise Filtering Performance:")
        logger.info(f"  - Total Audio Frames: {noise_stats['total_audio_frames']}")
        logger.info(f"  - Primary Speaker Frames: {noise_stats['primary_speaker_frames']} ({noise_stats.get('primary_speaker_percentage', 0):.1f}%)")
        logger.info(f"  - Background Noise Frames: {noise_stats['background_noise_frames']}")
        logger.info(f"  - Rejected Frames: {noise_stats['rejected_frames']}")
        logger.info(f"  - Primary Speaker Established: {noise_stats['primary_speaker_established']}")
        logger.info(f"  - Final Background Noise Level: {noise_stats['background_noise_level']:.4f}")
        
        # Stop silence monitoring - DISABLED
        if silence_monitoring_task:
            silence_monitoring_task.cancel()
            
        # agent.stop_silence_monitoring()  # DISABLED
        
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

    initial_greeting = "नमस्ते! मैं आपका तेलीमेडिकल एग्जामिनेशन असिस्टेंट हूँ। मैं आपसे कुछ मेडिकल सवाल पूछूंगी। क्या आप तैयार हैं?"
    
    # Set dynamic timeout for the initial greeting - DISABLED (but keep for compatibility)
    agent.set_dynamic_silence_timeout_for_message(initial_greeting)
    
    # Use safe speech delivery to prevent interruptions and ensure complete message is spoken
    success = await agent.safe_say(initial_greeting)
    if not success:
        logger.warning("Failed to deliver initial greeting with safe_say, falling back to standard method")
        await session.say(initial_greeting)

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
            await asyncio.sleep(0.2)  # Reduced from 0.5s for faster user detection
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
            # Reduced delay to ensure audio is stable while maintaining responsiveness
            await asyncio.sleep(0.3)  # Reduced from 1.0s for faster response
            
            # Start silence monitoring once user has spoken - DISABLED
            logger.info("Silence monitoring disabled")
            # agent.start_silence_monitoring()
            
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
    
    # Wait a bit for the initial setup (reduced for faster startup)
    await asyncio.sleep(1.0)  # Reduced from 2.0s for faster agent readiness
    
    # Start the silence monitoring task after initial setup - DISABLED
    # silence_monitoring_task = asyncio.create_task(silence_monitor())
    logger.info("Silence monitoring task disabled")
    
    # Keep the agent alive until the session ends
    await asyncio.sleep(0.1)  # Small delay to ensure everything is set up
    
    # The session will continue until participants leave and cleanup_and_record is called


def prewarm(proc: agents.JobProcess):
    """Prewarm VAD model to improve startup times with speech-interruption-prevention configuration"""
    # Optimized VAD configuration that prevents agent speech interruption while maintaining responsiveness
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.8,    # Slightly higher to prevent false activation during agent speech
        min_silence_duration=0.5,    # Increased slightly to ensure agent finishes speaking
        min_speech_duration=0.2,     # Require more speech duration to avoid interrupting agent
        # These settings balance responsiveness with speech completion reliability
    )
    logger.info("VAD prewarmed with speech-interruption-prevention configuration for telemedical examination")


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm
        )
    )
