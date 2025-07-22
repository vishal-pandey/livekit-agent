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
    """Manages MongoDB operations for storing session data"""
    
    def __init__(self):
        self.client = None
        self.database = None
        self.policy_database = None
        
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

    async def get_family_details(self, room_name: str) -> Optional[dict]:
        """Get family details from the policy database for a given room name"""
        try:
            if self.policy_database is None:
                logger.error("Policy database not connected")
                return None
                
            logger.info(f"Fetching family details for room: {room_name}")
            
            collection = self.policy_database[POLICY_COLLECTIONS["proposals"]]
            proposal = await collection.find_one({"proposalNo": room_name})
            
            if proposal:
                logger.info(f"Found family details for room: {room_name}")
                return proposal
            else:
                logger.warning(f"No proposal found for room: {room_name}")
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
    """Detects prolonged user silence and triggers prompts"""
    
    def __init__(self, silence_timeout: float = 30.0, prompt_interval: float = 30.0):
        self.silence_timeout = silence_timeout  # Time in seconds before first prompt
        self.prompt_interval = prompt_interval  # Time between subsequent prompts
        self.last_user_activity = time.time()
        self.last_prompt_time = 0
        self.is_monitoring = False
        self.prompt_count = 0
        self.max_prompts = 3  # Maximum number of silence prompts before giving up
        
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
        logger.debug(f"🔄 Activity detected - reset silence timer (was silent for {time_since_last_activity:.1f}s)")
        
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
        
        # First prompt after initial silence timeout
        if (self.prompt_count == 0 and 
            time_since_activity >= self.silence_timeout):
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
        You are a tele medical examination assistant.
        your role is to conduct the telemedical examination by asking questions and ask the reflexive questions based on the user's responses.
        
        You need to be smart to get all the answers correctly and efficiently from the user about each member of the family for whom eligibleForTelemedical is required 
        You are a female voice assistant with a friendly and professional tone.
        
        You speak in hindi but you also do understand english.
        You generally speak in hinglish and use daily use words and phrases in english only, also whenever you need to talk about the medical terms you need to pronounce the meidical terms and medecine names in english only.

        Below I am providing you the questions that you need to ask the user in order to conduct the telemedical examination.
        The questions are in json format and it is a reflexive questionnaire, which means you need to ask the questions based on the user's responses.
                         
        This tele medical examination is being conducted collect the medical history of multiple people in a family. The questions below are provided to be answered for each perspon in the family for which we need to conduct the tele medical examination.
        Below is the details of the people in the family for which we need to conduct the tele medical examination.
        It also has some preliminary information about the people in the family.
        Please make sure all the relevent answer for the questions listed below are collected for all the users who are relevant for telemer.

        {family_section}
                         

        ** Make sure when you conduct the medical examination you ask questions refering to specific persons for whon telemedical examination need to be conducted, do not talk in generic terms like any of the member or something, we have Names of all the persons for whom we need to conduct the tele medical examination in the family **
        
        ** While Asking questions make sure you address the name of the person for whom the question is related to, you need to make sure you collect answer of all the questions for all the members in the family for which eligibleForTelemedical is required**

        There might be multiple person in the room while this call is being conducted, so you need to ingore the content which are not relevant to the primary speaker.
        And anytime if you have some confustion around what user has said, you can ask the user to answer the question again or provide more details.
                         

        {{
  "questions": [
    {{
      "Question Description": "Has any member ever applied for a policy with Care Heath Insurance in the past?",
      "Input Type": "Yes or No"
    }},
    {{
      "Question Description": "Has any of the person(s) to be insured ever filed a claim with their current / previous or any other insurer?",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "Please Provide Detail",
          "Input Type": "Free Text"
        }}
      ]
    }},
    {{
      "Question Description": "Is any member suffering from Diabetes/Sugar problem or has been tested to have high blood sugar?",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "When was any member first detected with Diabetes/Sugar?",
          "Input Type": "Year/Month, No. of Years/Months"
        }},
        {{
          "Question Description": "Has any member ever been prescribed or taken Insulin?",
          "Input Type": "Yes or No"
        }},
        {{
          "Question Description": "Has any member suffered from any complications of Diabetes like reduced vision, kidney complications, non healing ulcer?",
          "Input Type": "Yes or No"
        }}
      ]
    }},
    {{
      "Question Description": "Has any member been diagnosed with high cholesterol or Lipid disorder?",
      "Input Type": "Yes or No"
    }},
    {{
      "Question Description": "Has any member been detected to have high BP or blood pressure or Hypertension?",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "When was any member first detected for having high BP/Blood Pressure/ Hypertension?",
          "Input Type": "Year/Month, No. of Years/Months"
        }}
      ]
    }},
    {{
      "Question Description": "Does any member have a Cardiac or Heart problem or experienced chest pain in the past?",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
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
        }}
      ]
    }},
    {{
      "Question Description": "Has any member experienced any symptoms of joint pain in Knee, Shoulder, Hip etc.?",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "Please provide detail along with medication prescribed/ taken, if any.",
          "Input Type": "Free Text"
        }}
      ]
    }},
    {{
      "Question Description": "Has any member suffered any vision related problem like blurry or hazy vision.",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "Please choose the Problem.",
          "Input Type": "Dropdown + Free Text (Fot 'Any Other..')",
          "Dropdown List": [
            "Cataract (No. of Year / Months) - ( Operated / Unoperated )",
            "Retinal disorder (No. of Year / Months) - ( Operated / Unoperated )",
            "Glaucoma (No. of Year / Months) - ( Operated / Unoperated )",
            "Any other type of disorder"
          ]
        }}
      ]
    }},
    {{
      "Question Description": "Has any member been diagnosed for gall bladder, kidney or urinary stones?",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "Please choose the problem.",
          "Input Type": "Dropdown",
          "Dropdown List": [
            "Gall Bladder Stone (No. of Year / Months) - ( Operated / Unoperated )",
            "Kidney or Urinary Stone (No. of Year / Months) - ( Operated / Conservatively resolved / Unoperated )"
          ]
        }}
      ]
    }},
    {{
      "Question Description": "Has any member been diagnosed for prostrate related problem, any complaints of increased urinary frequency, urgency or retention?",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "When was member first Diagnosed with Prostate or urinary disorder ?",
          "Input Type": "Year/Month, No. of Years/Months"
        }},
        {{
          "Question Description": "Please specify",
          "Input Type": "Free Text"
        }}
      ]
    }},
    {{
      "Question Description": "Has any member ever been diagnosed with any gynaecological problems like abnormal bleeding, cyst or fibroid in ovaries etc.?",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "When was member first Diagnosed with gynaecological problems ?",
          "Input Type": "Year/Month, No. of Years/Months"
        }},
        {{
          "Question Description": "Please specify",
          "Input Type": "Free Text"
        }}
      ]
    }},
    {{
      "Question Description": "Has any member ever been diagnosed with any form of Thyroid disorder",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "Please confirm type of Thyroid Disorder and/or name of medicine prescribed?",
          "Input Type": "Free Text"
        }},
        {{
          "Question Description": "How long has any member of the policy been suffering from thyroid disorder?",
          "Input Type": "Year/Month, No. of ears/Months"
        }}
      ]
    }},
    {{
      "Question Description": "Has any member ever been admitted to a hospital or undergone or advised for a surgery",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "Please specify the reason for hospitalization or surgery.",
          "Input Type": "Free Text with Limit up to 400 Words"
        }}
      ]
    }},
    {{
      "Question Description": "Has any member ever done medical test like Ultrasound/ CT scan/ MRI, 2D echo or any major investigation with positive finding? Please share report",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "Please specify",
          "Input Type": "Free Text"
        }}
      ]
    }},
    {{
      "Question Description": "Has any member ever experienced symptoms such as pain in abdomen or any other part of body, breathlessness?",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "Please specify.",
          "Input Type": "Free Text"
        }}
      ]
    }},
    {{
      "Question Description": "Do you want to disclose any other condition/illness/procedure for any member, other than the ones already answered above?",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "Please specify.",
          "Input Type": "Free Text with Limit up to 400 Words"
        }}
      ]
    }},
    {{
      "Question Description": "Does any member smoke?",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "How many cigarettes/ bidi does MemberName smoke?",
          "Input Type": "Number Dropdown"
        }},
        {{
          "Question Description": "Since when has any member been smoking?",
          "Input Type": "Year/Month, No. of Years/Months"
        }}
      ]
    }},
    {{
      "Question Description": "Does any member consume alcohol?",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "How often does any member drink?",
          "Input Type": "Dropdown: Occasionally, Daily, Weekly"
        }},
        {{
          "Question Description": "Please specify quantity? (Ex. Unit would be 30 ml of liquor per day/week) (Alcohol consumption (e.g., 100 ml) throws an error due to the digit limit.)",
          "Input Type": "Number Dropdown"
        }}
      ]
    }},
    {{
      "Question Description": "Does any member have a habit of chewing tobacco/pan masala/gutka?",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "How often does any member consume chewing tobacco/pan masala/gutka?",
          "Input Type": "Dropdown: Occasionally, Daily, Weekly"
        }},
        {{
          "Question Description": "How much does any member consume? Enter number (Ex. Sachets/grams per day (The current input limit of 2 digits is inadequate.),",
          "Input Type": "Number Dropdown"
        }}
      ]
    }},
    {{
      "Question Description": "Does any member have any other prohibitive habits?",
      "Input Type": "Yes or No",
      "sub_options": [
        {{
          "Question Description": "Please Specify.",
          "Input Type": "Free Text"
        }}
      ]
    }},
    {{
      "Question Description": "What is the height of member/s in feet and inches? (Ex: 5,3 for 5 feet and 3 inches)",
      "Input Type": "Number Dropdown: Feet and Inches"
    }},
    {{
      "Question Description": "What is the weight of member/s in Kgs? (Ex: 82 for 82 KGs)",
      "Input Type": "Number Dropdown: Kgs"
    }}
  ]
}}
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
        
        # Initialize silence detector
        self.silence_detector = SilenceDetector(
            silence_timeout=20.0,  # 10 seconds of silence before first prompt
            prompt_interval=20.0   # 10 seconds between subsequent prompts
        )
        
        # Session reference to send prompts
        self._session: Optional[AgentSession] = None
        self._last_user_message_count = 0
        self._last_agent_message_count = 0  # Track agent messages as activity too
    
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
            # Reset silence detector when user speaks (using universal activity reset)
            self.silence_detector.reset_activity_for_any_message()
        else:
            self.ignored_frames += 1
        
        # Log statistics periodically with enhanced filtering info
        if (self.processed_frames + self.ignored_frames) % 200 == 0:
            total = self.processed_frames + self.ignored_frames
            filtering_summary = self.enhanced_noise_filter.get_filtering_summary()
            logger.info(f"Enhanced Audio Filtering Stats - Processed: {self.processed_frames}/{total} ({self.processed_frames/total*100:.1f}%), "
                       f"Primary Speaker Established: {filtering_summary['primary_speaker_established']}, "
                       f"Background Noise Level: {filtering_summary['background_noise_level']:.4f}")
        
        return should_process
    
    def set_session(self, session):
        """Set the session reference for sending prompts"""
        self._session = session
        
    def start_silence_monitoring(self):
        """Start monitoring for user silence"""
        self.silence_detector.start_monitoring()
        logger.info("Started silence monitoring")
        
    def stop_silence_monitoring(self):
        """Stop monitoring for user silence"""
        self.silence_detector.stop_monitoring()
        logger.info("Stopped silence monitoring")
    
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
        
    async def check_and_handle_silence(self) -> bool:
        """Check for silence and send prompt if needed. Returns True if session should end."""
        if self.silence_detector.should_prompt_user():
            prompt_message = self.silence_detector.get_prompt_message()
            
            if self._session:
                logger.info(f"Sending silence prompt (count: {self.silence_detector.prompt_count + 1}): {prompt_message}")
                await self._session.say(prompt_message)
                # Reset activity timer since agent just spoke
                self.silence_detector.reset_activity_for_any_message()
                logger.debug("🤖 Silence prompt sent - reset activity timer")
                
            self.silence_detector.mark_prompt_sent()
            
            # Check if we should end the session after too many prompts
            if self.silence_detector.should_end_session():
                logger.warning("Maximum silence prompts reached, ending session")
                if self._session:
                    await self._session.say("मुझे लगता है आप वहाँ नहीं हैं। मैं यह सेशन समाप्त कर रही हूँ। धन्यवाद!")
                    # Reset activity timer for the final message too
                    self.silence_detector.reset_activity_for_any_message()
                    logger.debug("🤖 Final message sent - reset activity timer")
                return True
                
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
        tts=sarvam.TTS(target_language_code="hi-IN", speaker="anushka", http_session=http_session),
        # Enhanced VAD configuration for noisy environments - prioritizes primary speaker but not too aggressive
        vad=ctx.proc.userdata.get("vad") or silero.VAD.load(
            activation_threshold=0.75,   # Reduced from 0.85 for better pickup
            min_silence_duration=0.8,   # Reduced from 1.0 for more responsiveness
            min_speech_duration=0.2,    # Reduced from 0.3 for faster activation
        ),
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
        """Background task to monitor for silence and send prompts"""
        while True:
            try:
                should_end = await agent.check_and_handle_silence()
                if should_end:
                    logger.info("Session ended due to prolonged silence")
                    # End the session gracefully by breaking the loop
                    # The cleanup will be handled by the session end callback
                    break
                    
                # Check for both user and agent messages in session history to reset activity timer
                if hasattr(session, 'history') and session.history:
                    history_dict = session.history.to_dict()
                    all_items = history_dict.get('items', [])
                    
                    # Filter user and agent messages (check for multiple possible agent role names)
                    user_messages = [item for item in all_items if item.get('role') in ['user']]
                    agent_messages = [item for item in all_items if item.get('role') in ['agent', 'assistant', 'system']]
                    
                    # Debug: Log all roles we see (only every 20 checks to avoid spam)
                    if len(all_items) > 0 and (len(all_items) % 20 == 0):
                        unique_roles = set(item.get('role', 'unknown') for item in all_items)
                        logger.debug(f"Session history roles detected: {unique_roles}, Total items: {len(all_items)}")
                    
                    # Initialize counters if they don't exist
                    if not hasattr(agent, '_last_user_message_count'):
                        agent._last_user_message_count = 0
                    if not hasattr(agent, '_last_agent_message_count'):
                        agent._last_agent_message_count = 0
                    
                    # Check for new user messages
                    current_user_message_count = len(user_messages)
                    if current_user_message_count > agent._last_user_message_count:
                        agent.silence_detector.reset_activity_for_any_message()
                        logger.debug(f"✅ User message detected - reset silence timer (count: {current_user_message_count})")
                        agent._last_user_message_count = current_user_message_count
                    
                    # Check for new agent messages (agent speaking = activity)
                    current_agent_message_count = len(agent_messages)
                    if current_agent_message_count > agent._last_agent_message_count:
                        agent.silence_detector.reset_activity_for_any_message()
                        logger.debug(f"🤖 Agent message detected - reset silence timer (count: {current_agent_message_count})")
                        agent._last_agent_message_count = current_agent_message_count
                        
                        # Add a small buffer to ensure agent is done speaking before resuming silence monitoring
                        await asyncio.sleep(2.0)  # Give agent extra time to finish speaking
                
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in silence monitoring: {e}")
                await asyncio.sleep(5)

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
        
        # Stop silence monitoring
        if silence_monitoring_task:
            silence_monitoring_task.cancel()
            
        agent.stop_silence_monitoring()
        
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
            
            # Start silence monitoring once user has spoken
            logger.info("Starting silence monitoring...")
            agent.start_silence_monitoring()
            
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
    
    # Wait a bit for the initial setup
    await asyncio.sleep(2.0)
    
    # Start the silence monitoring task after initial setup
    silence_monitoring_task = asyncio.create_task(silence_monitor())
    
    # Keep the agent alive until the session ends
    await asyncio.sleep(0.1)  # Small delay to ensure everything is set up
    
    # The session will continue until participants leave and cleanup_and_record is called


def prewarm(proc: agents.JobProcess):
    """Prewarm VAD model to improve startup times with balanced noise filtering configuration"""
    # Balanced VAD configuration for noisy environments
    # Not too aggressive to ensure user voice pickup while still filtering background noise
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.7,    # Reduced from 0.8 for better user voice pickup
        min_silence_duration=1.0,   # Reduced from 1.2 for better responsiveness
        min_speech_duration=0.2,    # Reduced from 0.25 for faster activation
        # These settings balance noise filtering with user voice pickup
    )
    logger.info("VAD prewarmed with balanced noise filtering configuration for telemedical examination")


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm
        )
    )
