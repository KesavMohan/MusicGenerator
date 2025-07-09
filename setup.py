#!/usr/bin/env python3
"""
Setup script for AI Music Generator
This script helps users set up the application environment and run it.
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_venv():
    """Check if we're in a virtual environment"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def main():
    print("🎵 AI Music Generator Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if virtual environment exists
    venv_exists = os.path.exists('venv')
    in_venv = check_venv()
    
    if not in_venv and not venv_exists:
        print("📦 Creating virtual environment...")
        if not run_command(f"{sys.executable} -m venv venv", "Virtual environment creation"):
            sys.exit(1)
    
    # Activate virtual environment instructions
    if not in_venv:
        system = platform.system()
        if system == "Windows":
            activate_cmd = ".\\venv\\Scripts\\activate"
        else:
            activate_cmd = "source venv/bin/activate"
        
        print(f"⚠️  Please activate your virtual environment first:")
        print(f"   {activate_cmd}")
        print("   Then run this script again.")
        sys.exit(0)
    
    print("🔍 Virtual environment is active!")
    
    # Install requirements
    if not run_command("pip install --upgrade pip", "Pip upgrade"):
        print("⚠️  Pip upgrade failed, continuing anyway...")
    
    if not run_command("pip install -r requirements.txt", "Dependencies installation"):
        print("❌ Failed to install dependencies!")
        sys.exit(1)
    
    # Create necessary directories
    if not os.path.exists('generated_music'):
        os.makedirs('generated_music')
        print("✅ Created generated_music directory")
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 To start the application:")
    print("   python app.py")
    print("\n🌐 Then open your browser to:")
    print("   http://localhost:5002")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 