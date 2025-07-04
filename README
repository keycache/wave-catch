# Wave-Catch

A Python application for processing audio transcripts and incidents using LLM capabilities.

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd wave-catch
```

### 2. Set Up Virtual Environment
Create a virtual environment using Python 3.11.13 or higher:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration Setup
Rename the configuration template file:
```bash
mv constants.py.bak constants.py
```

Update the variables in `constants.py` with your specific values:
```python
BASE_TRANSCRIPT_FOLDER = "/your/actual/path/to/transcript/folder"
BASE_AUDIO_FOLDER = "/your/actual/path/to/audio/folder"
```

Replace the placeholder paths with your actual directory paths where:
- `BASE_TRANSCRIPT_FOLDER`: Directory containing transcript files
- `BASE_AUDIO_FOLDER`: Directory containing audio files

## Usage

### Local Execution (from project directory with activated environment)
```bash
python main.py --task incident
python main.py --task speech
python main.py --task incident --file .data/transcript/transcript.txt
```

### External Execution (from anywhere)
```bash
/fully-qualified/path-to/.venv/bin/python /fully-qualified/path/to/wave-catch/main.py --task incident|speech
```

## Configuration

### Configuration File
After renaming `constants.py.bak` to `constants.py`, ensure all required variables are properly set:
- `BASE_TRANSCRIPT_FOLDER`: Path to transcript files directory
- `BASE_AUDIO_FOLDER`: Path to audio files directory
- Audio processing parameters (SAMPLE_RATE, CHANNELS, etc.)

### LLM Setup
This application uses an OpenAI API compatible LLM. The current configuration assumes:
- Internally hosted LLM (no authentication required)
- OpenAI API compatible interface

### External API Configuration
If connecting to an externally hosted 3rd party OpenAI compatible API, update the code to include necessary authentication headers in the API requests.

## Tasks
- `incident`: Process incident-related transcripts
- `speech`: Process speech transcripts