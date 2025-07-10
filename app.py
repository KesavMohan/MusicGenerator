import os
import uuid
from flask import Flask, render_template, request, jsonify, send_file, url_for, Response
import torch
import torchaudio
import numpy as np
from datetime import datetime
import logging
import time
import tempfile
from werkzeug.utils import secure_filename
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create directories for generated music and uploads
GENERATED_MUSIC_DIR = 'generated_music'
UPLOADS_DIR = 'uploads'
if not os.path.exists(GENERATED_MUSIC_DIR):
    os.makedirs(GENERATED_MUSIC_DIR)
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

# Audio specifications
TARGET_SAMPLE_RATE = 44100  # 44.1 kHz
TARGET_CHANNELS = 2  # Stereo
TARGET_BIT_DEPTH = 16  # 16-bit

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

# Global variable to store the model
music_generator = None

# Progress tracking
generation_progress = {}
progress_lock = threading.Lock()

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def update_progress(generation_id, progress, status="Generating"):
    """Update progress for a specific generation"""
    with progress_lock:
        generation_progress[generation_id] = {
            'progress': progress,
            'status': status,
            'timestamp': time.time()
        }

def get_progress(generation_id):
    """Get progress for a specific generation"""
    with progress_lock:
        return generation_progress.get(generation_id, {'progress': 0, 'status': 'Not found'})

def cleanup_progress(generation_id):
    """Clean up progress data after generation completes"""
    with progress_lock:
        if generation_id in generation_progress:
            del generation_progress[generation_id]

def load_model():
    """Load a music generation model"""
    global music_generator
    try:
        logger.info("Loading music generation model...")
        
        # Try to use MusicGen model which is more compatible
        try:
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
            
            # Use Facebook's MusicGen model as it's more reliable
            model_name = "facebook/musicgen-small"
            processor = AutoProcessor.from_pretrained(model_name)
            model = MusicgenForConditionalGeneration.from_pretrained(model_name)
            
            music_generator = {
                "processor": processor,
                "model": model,
                "type": "musicgen"
            }
            
            logger.info("MusicGen model loaded successfully!")
            return True
            
        except Exception as e:
            logger.warning(f"MusicGen not available: {str(e)}")
            
            # Fallback to a demo version that generates synthetic audio
            logger.info("Using demo audio generation...")
            music_generator = {
                "type": "demo"
            }
            return True
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def process_reference_audio(file_path):
    """Process uploaded reference audio for analysis"""
    try:
        # Load the audio file
        audio_tensor, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if stereo (for analysis)
        if audio_tensor.shape[0] > 1:
            audio_mono = torch.mean(audio_tensor, dim=0, keepdim=True)
        else:
            audio_mono = audio_tensor
        
        # Resample to target sample rate if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=TARGET_SAMPLE_RATE
            )
            audio_mono = resampler(audio_mono)
        
        # Analyze audio characteristics
        duration = audio_mono.shape[1] / TARGET_SAMPLE_RATE
        
        # Calculate basic audio features
        # RMS (loudness)
        rms = torch.sqrt(torch.mean(audio_mono**2))
        
        # Spectral centroid (brightness)
        if hasattr(torchaudio.transforms, 'Spectrogram'):
            spectrogram = torchaudio.transforms.Spectrogram()(audio_mono)
            spectral_centroid = torch.mean(spectrogram, dim=1)
        else:
            spectral_centroid = torch.tensor([0.5])  # Default value
        
        # Tempo estimation (simplified)
        # This is a very basic tempo detection - in a real app you'd use librosa
        tempo_estimate = 120.0  # Default BPM
        
        analysis = {
            'duration': duration,
            'rms': rms.item(),
            'spectral_centroid': spectral_centroid.mean().item(),
            'tempo_estimate': tempo_estimate,
            'sample_rate': TARGET_SAMPLE_RATE
        }
        
        return analysis, audio_mono
        
    except Exception as e:
        logger.error(f"Error processing reference audio: {str(e)}")
        return None, None

def generate_demo_audio(prompt, duration=10, reference_analysis=None, generation_id=None):
    """Generate demo audio with synthesized sounds and progress tracking"""
    try:
        # Generate a simple synthesized audio based on prompt keywords
        t = np.linspace(0, duration, int(TARGET_SAMPLE_RATE * duration))
        
        # Use reference analysis if available
        if reference_analysis:
            # Adjust generation based on reference characteristics
            reference_rms = reference_analysis.get('rms', 0.3)
            reference_centroid = reference_analysis.get('spectral_centroid', 0.5)
            reference_tempo = reference_analysis.get('tempo_estimate', 120)
            
            # Adjust frequency based on spectral centroid
            base_freq = 220 + (reference_centroid * 440)  # 220-660 Hz range
            # Adjust amplitude based on RMS
            amplitude = min(0.8, reference_rms * 2)
            # Adjust rhythm based on tempo
            beat_freq = reference_tempo / 60  # Convert BPM to Hz
        else:
            base_freq = 330
            amplitude = 0.25
            beat_freq = 2
        
        # Simulate progress updates for demo generation
        if generation_id:
            update_progress(generation_id, 10, "Initializing audio generation...")
            time.sleep(0.5)
            update_progress(generation_id, 25, "Analyzing prompt characteristics...")
            time.sleep(0.5)
            update_progress(generation_id, 40, "Generating base frequencies...")
            time.sleep(0.5)
            update_progress(generation_id, 60, "Applying audio effects...")
            time.sleep(0.5)
            update_progress(generation_id, 80, "Finalizing audio mix...")
            time.sleep(0.5)
        
        # Create different tones based on prompt content
        if any(word in prompt.lower() for word in ['upbeat', 'dance', 'energetic', 'fast']):
            # Higher frequency for upbeat music
            freq = base_freq * 1.5
            audio = np.sin(2 * np.pi * freq * t) * amplitude
            # Add some rhythm
            envelope = (np.sin(2 * np.pi * beat_freq * t) + 1) * 0.5
            audio = audio * envelope
            
        elif any(word in prompt.lower() for word in ['ambient', 'calm', 'gentle', 'soft']):
            # Lower frequency for ambient music
            freq = base_freq * 0.8
            audio = np.sin(2 * np.pi * freq * t) * amplitude * 0.7
            # Add some gentle modulation
            mod_freq = 0.5
            modulation = np.sin(2 * np.pi * mod_freq * t) * 0.1 + 1
            audio = audio * modulation
            
        elif any(word in prompt.lower() for word in ['bass', 'heavy', 'dubstep']):
            # Very low frequency for bass-heavy music
            freq = base_freq * 0.5
            audio = np.sin(2 * np.pi * freq * t) * amplitude * 1.2
            # Add some distortion effect
            audio = np.tanh(audio * 3) * 0.3
            
        else:
            # Default electronic sound
            freq = base_freq
            audio = np.sin(2 * np.pi * freq * t) * amplitude
            # Add some harmonics
            audio += np.sin(2 * np.pi * freq * 2 * t) * amplitude * 0.4
            audio += np.sin(2 * np.pi * freq * 0.5 * t) * amplitude * 0.6
        
        # Add some fade in/out to make it sound more natural
        fade_samples = int(TARGET_SAMPLE_RATE * 0.1)  # 0.1 second fade
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
        
        # Convert to stereo by duplicating the mono channel
        audio_stereo = np.stack([audio, audio], axis=0)
        
        # Convert to tensor and ensure proper shape (channels, samples)
        audio_tensor = torch.from_numpy(audio_stereo).float()
        
        if generation_id:
            update_progress(generation_id, 100, "Generation complete!")
            time.sleep(0.5)
        
        return audio_tensor, TARGET_SAMPLE_RATE
        
    except Exception as e:
        logger.error(f"Error generating demo audio: {str(e)}")
        if generation_id:
            update_progress(generation_id, 0, f"Error: {str(e)}")
        return None, None

def process_audio_to_specs(audio_tensor, original_sample_rate):
    """Process audio to meet target specifications"""
    try:
        # Ensure audio is the right shape (channels, samples)
        if audio_tensor.dim() == 1:
            # Mono to stereo
            audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1)
        elif audio_tensor.dim() == 2 and audio_tensor.shape[0] == 1:
            # Mono to stereo
            audio_tensor = audio_tensor.repeat(2, 1)
        elif audio_tensor.dim() == 2 and audio_tensor.shape[0] > 2:
            # Too many channels, take first two
            audio_tensor = audio_tensor[:2, :]
        
        # Resample if needed
        if original_sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sample_rate,
                new_freq=TARGET_SAMPLE_RATE
            )
            audio_tensor = resampler(audio_tensor)
        
        # Ensure stereo (2 channels)
        if audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.repeat(2, 1)
        elif audio_tensor.shape[0] > 2:
            audio_tensor = audio_tensor[:2, :]
        
        return audio_tensor
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for reference audio"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOADS_DIR, f"{uuid.uuid4().hex}_{filename}")
            
            # Save the file
            file.save(file_path)
            
            # Process the reference audio
            analysis, audio_tensor = process_reference_audio(file_path)
            
            if analysis is None:
                return jsonify({'error': 'Failed to process uploaded audio'}), 500
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            return jsonify({
                'success': True,
                'analysis': analysis,
                'message': f'Reference audio analyzed successfully! Duration: {analysis["duration"]:.1f}s'
            })
        else:
            return jsonify({'error': 'Invalid file type. Allowed: wav, mp3, flac, ogg, m4a'}), 400
            
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_music():
    """Generate music based on text prompt"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        duration = data.get('duration', 10)  # Default 10 seconds
        track_count = data.get('track_count', 1)  # Default 1 track
        reference_analysis = data.get('reference_analysis', None)  # Reference audio analysis
        
        if not prompt:
            return jsonify({'error': 'Please provide a text prompt'}), 400
        
        # Validate duration (up to 3 minutes = 180 seconds)
        if duration < 5 or duration > 180:
            return jsonify({'error': 'Duration must be between 5 and 180 seconds (3 minutes)'}), 400
        
        # Generate unique ID for this generation
        generation_id = str(uuid.uuid4())
        update_progress(generation_id, 0, "Starting generation...")
        
        logger.info(f"Generating {track_count} track(s) for prompt: {prompt}, duration: {duration}s")
        if reference_analysis:
            logger.info(f"Using reference audio analysis: {reference_analysis}")
        
        generated_files = []
        
        # Generate multiple tracks
        for track_num in range(track_count):
            try:
                # Update progress for each track
                track_progress = (track_num / track_count) * 100
                update_progress(generation_id, track_progress, f"Generating track {track_num + 1}/{track_count}...")
                
                # Generate unique filename for each track
                filename = f"generated_{uuid.uuid4().hex}.wav"
                filepath = os.path.join(GENERATED_MUSIC_DIR, filename)
                
                # Generate music
                if music_generator:
                    try:
                        if music_generator["type"] == "musicgen":
                            # Use MusicGen model
                            processor = music_generator["processor"]
                            model = music_generator["model"]
                            
                            update_progress(generation_id, track_progress + 10, f"Processing prompt for track {track_num + 1}...")
                            
                            # Enhance prompt with reference analysis if available
                            enhanced_prompt = prompt
                            if reference_analysis:
                                ref_duration = reference_analysis.get('duration', 0)
                                ref_tempo = reference_analysis.get('tempo_estimate', 120)
                                enhanced_prompt = f"{prompt} (inspired by reference: {ref_duration:.1f}s duration, {ref_tempo:.0f} BPM)"
                            
                            # Process the prompt
                            inputs = processor(
                                text=[enhanced_prompt],
                                padding=True,
                                return_tensors="pt",
                            )
                            
                            update_progress(generation_id, track_progress + 30, f"Generating audio for track {track_num + 1}...")
                            
                            # Calculate tokens based on duration (roughly 50 tokens per second)
                            max_new_tokens = int(duration * 50)
                            
                            # Generate audio
                            with torch.no_grad():
                                audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
                            
                            update_progress(generation_id, track_progress + 70, f"Processing audio for track {track_num + 1}...")
                            
                            # Get sample rate from model
                            original_sample_rate = model.config.audio_encoder.sampling_rate
                            
                            # Convert to the right format
                            audio_tensor = audio_values[0].cpu()
                            
                            # Process audio to meet specifications
                            audio_tensor = process_audio_to_specs(audio_tensor, original_sample_rate)
                            
                            if audio_tensor is None:
                                update_progress(generation_id, 0, f"Failed to process audio for track {track_num + 1}")
                                return jsonify({'error': f'Failed to process audio for track {track_num + 1}'}), 500
                            
                        elif music_generator["type"] == "demo":
                            # Use demo generation with reference analysis
                            audio_tensor, original_sample_rate = generate_demo_audio(prompt, duration, reference_analysis, generation_id)
                            
                            if audio_tensor is None:
                                update_progress(generation_id, 0, "Failed to generate demo audio")
                                return jsonify({'error': 'Failed to generate demo audio'}), 500
                            
                            # Process audio to meet specifications
                            audio_tensor = process_audio_to_specs(audio_tensor, original_sample_rate)
                            
                            if audio_tensor is None:
                                update_progress(generation_id, 0, f"Failed to process demo audio for track {track_num + 1}")
                                return jsonify({'error': f'Failed to process demo audio for track {track_num + 1}'}), 500
                        
                        update_progress(generation_id, track_progress + 90, f"Saving track {track_num + 1}...")
                        
                        # Ensure the audio is the right length
                        expected_length = int(TARGET_SAMPLE_RATE * duration)
                        if audio_tensor.shape[-1] < expected_length:
                            # Pad with silence if too short
                            padding_length = expected_length - audio_tensor.shape[-1]
                            padding = torch.zeros(audio_tensor.shape[0], padding_length)
                            audio_tensor = torch.cat([audio_tensor, padding], dim=-1)
                        elif audio_tensor.shape[-1] > expected_length:
                            # Trim if too long
                            audio_tensor = audio_tensor[:, :expected_length]
                        
                        # Save audio file with specific encoding
                        torchaudio.save(
                            filepath, 
                            audio_tensor, 
                            TARGET_SAMPLE_RATE,
                            encoding="PCM_S",
                            bits_per_sample=TARGET_BIT_DEPTH
                        )
                        
                        generated_files.append({
                            'filename': filename,
                            'download_url': url_for('download_file', filename=filename),
                            'track_number': track_num + 1
                        })
                        
                        logger.info(f"Track {track_num + 1} generated successfully: {filename}")
                        
                    except Exception as e:
                        logger.error(f"Error during music generation for track {track_num + 1}: {str(e)}")
                        update_progress(generation_id, 0, f"Error generating track {track_num + 1}: {str(e)}")
                        return jsonify({'error': f'Generation failed for track {track_num + 1}: {str(e)}'}), 500
                else:
                    update_progress(generation_id, 0, "Model not loaded")
                    return jsonify({'error': 'Model not loaded. Please restart the application.'}), 500
                
            except Exception as e:
                logger.error(f"Error in track generation loop: {str(e)}")
                update_progress(generation_id, 0, f"Error: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        # Final progress update
        update_progress(generation_id, 100, "Generation complete!")
        
        # Return success with all generated files
        message = f'Generated {track_count} track(s) successfully!' + (' (Demo Mode)' if music_generator["type"] == "demo" else '')
        if reference_analysis:
            message += ' (with reference audio inspiration)'
        
        return jsonify({
            'success': True,
            'files': generated_files,
            'message': message,
            'track_count': track_count,
            'generation_id': generation_id
        })
            
    except Exception as e:
        logger.error(f"Error in generate_music: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/progress/<generation_id>')
def get_progress_route(generation_id):
    """Get progress for a specific generation"""
    progress_data = get_progress(generation_id)
    return jsonify(progress_data)

@app.route('/download/<filename>')
def download_file(filename):
    """Download generated music file"""
    try:
        filepath = os.path.join(GENERATED_MUSIC_DIR, filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Check if model is loaded"""
    model_type = music_generator.get("type", "unknown") if music_generator else "none"
    return jsonify({
        'model_loaded': music_generator is not None,
        'model_type': model_type,
        'cuda_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'audio_specs': {
            'format': 'WAV',
            'sample_rate': f'{TARGET_SAMPLE_RATE} Hz',
            'channels': f'{TARGET_CHANNELS} (Stereo)',
            'bit_depth': f'{TARGET_BIT_DEPTH}-bit'
        },
        'max_duration': 180  # 3 minutes in seconds
    })

if __name__ == '__main__':
    logger.info("Starting Music Generator Web App...")
    
    # Load model on startup
    model_loaded = load_model()
    
    if model_loaded:
        logger.info("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5002)
    else:
        logger.error("Failed to load model. Please check your installation.")
        print("Please run: pip install -r requirements.txt") 