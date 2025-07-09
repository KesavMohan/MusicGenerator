# Music Generator - AI Electronic Music Creation

A local web application that generates electronic music using AI models. Create up to 3-minute songs with optional reference audio inspiration!

## Features

- 🎵 **AI Music Generation**: Generate electronic music from text descriptions
- ⏱️ **Up to 3 Minutes**: Create songs up to 180 seconds in length
- 🎧 **Reference Audio Upload**: Upload songs to inspire the AI's generation
- 🎛️ **Multiple Tracks**: Generate 1-10 different variations
- 🎚️ **Professional Audio**: WAV format, 16-bit, 44.1kHz, stereo
- 🌐 **Modern Web Interface**: Beautiful, responsive UI
- 📱 **Mobile Friendly**: Works on all devices

## Audio Upload Feature

Upload reference audio files (WAV, MP3, FLAC, OGG, M4A) to inspire the AI's music generation. The system will:

- Analyze the uploaded audio's characteristics
- Extract tempo, loudness, and brightness information
- Use these features to enhance the generated music
- Provide detailed analysis feedback

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KesavMohan/MusicGenerator.git
   cd MusicGenerator
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   
   **On macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```
   
   **On Windows:**
   ```bash
   venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application

**Option 1: Using Python directly**
```bash
python app.py
```

**Option 2: Using the provided scripts**

**On macOS/Linux:**
```bash
./run.sh
```

**On Windows:**
```bash
run.bat
```

### Accessing the Application

1. Open your web browser
2. Navigate to: `http://localhost:5002`
3. Start generating music!

### How to Use

1. **Describe your music**: Enter a text description of the music you want to generate
2. **Set duration**: Choose between 5 seconds and 3 minutes (180 seconds)
3. **Select track count**: Generate 1-10 different variations
4. **Upload reference audio** (optional): Upload a song to inspire the generation
5. **Generate**: Click the generate button and wait for your music!

### Example Prompts

- "Upbeat electronic dance music with heavy bass and synth melodies"
- "Ambient electronic music with gentle pads and atmospheric sounds"
- "Fast-paced techno with driving beats and industrial sounds"
- "Chill downtempo electronic with smooth rhythms and warm tones"
- "Dubstep with heavy bass drops and electronic effects"
- "Synthwave with retro 80s style and melodic leads"

## Audio Specifications

All generated music is saved in professional quality:
- **Format**: WAV
- **Bit Depth**: 16-bit
- **Sample Rate**: 44.1 kHz
- **Channels**: Stereo
- **Max Duration**: 3 minutes (180 seconds)

## Technical Details

### Models Used

- **Primary**: Facebook's MusicGen model for high-quality generation
- **Fallback**: Demo audio generation with synthesized sounds
- **Reference Analysis**: Audio feature extraction for inspiration

### File Structure

```
MusicGenerator/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface
├── generated_music/       # Generated audio files
├── uploads/              # Temporary upload directory
├── run.sh               # macOS/Linux startup script
├── run.bat              # Windows startup script
└── README.md            # This file
```

### API Endpoints

- `GET /` - Main web interface
- `POST /generate` - Generate music from text prompt
- `POST /upload` - Upload reference audio for analysis
- `GET /download/<filename>` - Download generated audio files
- `GET /status` - Check system status and model availability

## Troubleshooting

### Common Issues

1. **Port 5002 already in use:**
   ```bash
   # Find and kill the process using port 5002
   lsof -ti:5002 | xargs kill -9
   ```

2. **Model loading errors:**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

3. **Audio generation fails:**
   - Check that you have sufficient disk space
   - Ensure the `generated_music` directory exists
   - Try generating shorter durations first

### System Requirements

- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: At least 2GB free space
- **Network**: Internet connection for initial model download

## Development

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Local Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run in debug mode
python app.py

# Access at http://localhost:5002
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Search existing issues on GitHub
3. Create a new issue with detailed information

---

**Happy Music Generation! 🎵** 