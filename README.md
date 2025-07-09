# 🎵 AI Music Generator

A web application that generates electronic music using the Tencent SongGeneration AI model from Hugging Face.

## Features

- **Text-to-Music Generation**: Generate music from text descriptions
- **Modern Web Interface**: Beautiful, responsive UI with real-time feedback
- **Audio Playback**: Play generated music directly in the browser
- **Download Support**: Download generated music as WAV files
- **Example Prompts**: Pre-built prompts for different music styles
- **Model Status Monitoring**: Real-time status of the AI model

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- At least 4GB of RAM (8GB+ recommended)
- Internet connection for initial model download

## Installation

1. **Clone or download this repository**
   ```bash
   cd "Music Generator"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the model** (This will happen automatically on first run)
   - The Tencent SongGeneration model (~2GB) will be downloaded from Hugging Face
   - This may take several minutes depending on your internet speed

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5002
   ```

3. **Generate music**:
   - Enter a text description of the music you want to generate
   - Set the duration (5-30 seconds)
   - Click "Generate Music"
   - Wait for the AI to process your request
   - Play the generated music in your browser or download it

## Example Prompts

Here are some example prompts to get you started:

- **Electronic Dance**: "A upbeat electronic dance track with synthesizers and heavy bass"
- **Ambient**: "Ambient electronic music with soft pads and gentle melodies"
- **Dubstep**: "Energetic dubstep with strong drops and electronic effects"
- **Lo-fi**: "Chill lo-fi electronic beats with vinyl crackle"
- **Techno**: "Dark techno with driving beats and industrial sounds"

## Technical Details

### Model Information
- **Model**: `tencent/SongGeneration`
- **Type**: Text-to-Audio generation
- **Architecture**: Language model-based framework
- **Output**: WAV audio files

### System Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 5GB free space for model and generated files
- **GPU**: Optional but recommended for faster generation

### Generated Files
- Generated music files are saved in the `generated_music/` directory
- Files are named with unique IDs to prevent conflicts
- Audio format: WAV, 16kHz sample rate

## Troubleshooting

### Common Issues

1. **Model loading fails**
   - Check your internet connection
   - Ensure you have enough disk space
   - Try restarting the application

2. **Generation takes too long**
   - Reduce the duration
   - Check if you have enough RAM
   - Consider using a GPU if available

3. **Out of memory errors**
   - Reduce the duration
   - Close other applications
   - Consider using a machine with more RAM

4. **Audio playback issues**
   - Try downloading the file and playing it with a media player
   - Check your browser's audio settings

### Error Messages

- **"Model not loaded"**: The AI model failed to load. Check the console for detailed error messages.
- **"Generation failed"**: The music generation process encountered an error. Try a different prompt or reduce the duration.
- **"Network error"**: Connection issue between browser and server. Refresh the page and try again.

## Performance Tips

1. **First run**: The initial model download may take time. Be patient!
2. **Generation time**: Expect 1-3 minutes for generation depending on your hardware
3. **Memory usage**: Close unnecessary applications to free up RAM
4. **GPU acceleration**: If you have a compatible GPU, it will be used automatically

## Limitations

- **Duration**: Limited to 5-30 seconds per generation
- **Style**: Optimized for electronic music genres
- **Quality**: Output quality depends on the input prompt specificity
- **Hardware**: Requires significant computational resources

## Privacy & Security

- All processing happens locally on your machine
- No data is sent to external servers (except for initial model download)
- Generated music files are stored locally
- No user data is collected or stored

## License

This project is for educational and personal use. Please check the Tencent SongGeneration model license for commercial usage restrictions.

## Support

If you encounter issues:

1. Check the console output for error messages
2. Ensure all dependencies are installed correctly
3. Verify your system meets the minimum requirements
4. Try reducing the generation duration

## Contributing

This is a standalone application. Feel free to modify and enhance it for your needs!

---

**Enjoy creating music with AI! 🎵** 