import os
import uuid
from flask import Flask, render_template, request, jsonify, send_file, url_for
import torch
import torchaudio
import numpy as np
from datetime import datetime
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Create directories for generated music
GENERATED_MUSIC_DIR = 'generated_music'
if not os.path.exists(GENERATED_MUSIC_DIR):
    os.makedirs(GENERATED_MUSIC_DIR)

# Audio specifications
TARGET_SAMPLE_RATE = 44100  # 44.1 kHz
TARGET_CHANNELS = 2  # Stereo
TARGET_BIT_DEPTH = 16  # 16-bit

# Global variable to store the model
music_generator = None

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

def generate_demo_audio(prompt, duration=10):
    """Generate demo audio with synthesized sounds"""
    try:
        # Generate a simple synthesized audio based on prompt keywords
        t = np.linspace(0, duration, int(TARGET_SAMPLE_RATE * duration))
        
        # Create different tones based on prompt content
        if any(word in prompt.lower() for word in ['upbeat', 'dance', 'energetic', 'fast']):
            # Higher frequency for upbeat music
            freq = 440  # A4
            audio = np.sin(2 * np.pi * freq * t) * 0.3
            # Add some rhythm
            beat_freq = 2  # 2 beats per second
            envelope = (np.sin(2 * np.pi * beat_freq * t) + 1) * 0.5
            audio = audio * envelope
            
        elif any(word in prompt.lower() for word in ['ambient', 'calm', 'gentle', 'soft']):
            # Lower frequency for ambient music
            freq = 220  # A3
            audio = np.sin(2 * np.pi * freq * t) * 0.2
            # Add some gentle modulation
            mod_freq = 0.5
            modulation = np.sin(2 * np.pi * mod_freq * t) * 0.1 + 1
            audio = audio * modulation
            
        elif any(word in prompt.lower() for word in ['bass', 'heavy', 'dubstep']):
            # Very low frequency for bass-heavy music
            freq = 110  # A2
            audio = np.sin(2 * np.pi * freq * t) * 0.4
            # Add some distortion effect
            audio = np.tanh(audio * 3) * 0.3
            
        else:
            # Default electronic sound
            freq = 330  # E4
            audio = np.sin(2 * np.pi * freq * t) * 0.25
            # Add some harmonics
            audio += np.sin(2 * np.pi * freq * 2 * t) * 0.1
            audio += np.sin(2 * np.pi * freq * 0.5 * t) * 0.15
        
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
        
        return audio_tensor, TARGET_SAMPLE_RATE
        
    except Exception as e:
        logger.error(f"Error generating demo audio: {str(e)}")
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

@app.route('/generate', methods=['POST'])
def generate_music():
    """Generate music based on text prompt"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        duration = data.get('duration', 10)  # Default 10 seconds
        track_count = data.get('track_count', 1)  # Default 1 track
        
        if not prompt:
            return jsonify({'error': 'Please provide a text prompt'}), 400
        
        logger.info(f"Generating {track_count} track(s) for prompt: {prompt}, duration: {duration}s")
        
        generated_files = []
        
        # Generate multiple tracks
        for track_num in range(track_count):
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
                        
                        # Process the prompt
                        inputs = processor(
                            text=[prompt],
                            padding=True,
                            return_tensors="pt",
                        )
                        
                        # Calculate tokens based on duration (roughly 50 tokens per second)
                        max_new_tokens = int(duration * 50)
                        
                        # Generate audio
                        with torch.no_grad():
                            audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
                        
                        # Get sample rate from model
                        original_sample_rate = model.config.audio_encoder.sampling_rate
                        
                        # Convert to the right format
                        audio_tensor = audio_values[0].cpu()
                        
                        # Process audio to meet specifications
                        audio_tensor = process_audio_to_specs(audio_tensor, original_sample_rate)
                        
                        if audio_tensor is None:
                            return jsonify({'error': f'Failed to process audio for track {track_num + 1}'}), 500
                        
                    elif music_generator["type"] == "demo":
                        # Use demo generation
                        audio_tensor, original_sample_rate = generate_demo_audio(prompt, duration)
                        
                        if audio_tensor is None:
                            return jsonify({'error': 'Failed to generate demo audio'}), 500
                        
                        # Process audio to meet specifications
                        audio_tensor = process_audio_to_specs(audio_tensor, original_sample_rate)
                        
                        if audio_tensor is None:
                            return jsonify({'error': f'Failed to process demo audio for track {track_num + 1}'}), 500
                    
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
                    return jsonify({'error': f'Generation failed for track {track_num + 1}: {str(e)}'}), 500
            else:
                return jsonify({'error': 'Model not loaded. Please restart the application.'}), 500
        
        # Return success with all generated files
        message = f'Generated {track_count} track(s) successfully!' + (' (Demo Mode)' if music_generator["type"] == "demo" else '')
        
        return jsonify({
            'success': True,
            'files': generated_files,
            'message': message,
            'track_count': track_count
        })
            
    except Exception as e:
        logger.error(f"Error in generate_music: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
        }
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