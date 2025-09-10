# Audio Processing Feature

RAGAnything now supports audio content processing and knowledge graph insertion. This document introduces how to use the audio processing functionality.

## Features

- **Audio Content Analysis**: Supports intelligent analysis of audio files, including content type recognition, transcript analysis, etc.
- **Context-Aware Processing**: Can combine surrounding content to provide more accurate audio analysis
- **Knowledge Graph Integration**: Inserts audio content as entities into the knowledge graph, supporting subsequent retrieval and querying
- **Multiple Audio Formats**: Supports common audio formats (MP3, WAV, M4A, etc.)
- **Batch Processing**: Supports batch processing of multiple audio files

## Configuration Options

Added the following configuration to `RAGAnythingConfig`:

```python
enable_audio_processing: bool = True  # Enable audio processing
```

Can also be configured via environment variables:

```bash
export ENABLE_AUDIO_PROCESSING=true
```

## Audio Data Format

Audio content should contain the following fields:

```python
audio_data = {
    "audio_path": "/path/to/audio.mp3",        # Audio file path (required)
    "transcript": "Audio transcription text...",  # Audio transcription (optional)
    "duration": "00:05:30",                    # Audio duration (optional)
    "format": "MP3",                           # Audio format (optional)
    "description": "Audio content description"  # Audio description (optional)
}
```

## Usage

### 1. Basic Configuration

```python
from raganything import RAGAnything
from raganything.config import RAGAnythingConfig

# Configure RAGAnything
config = RAGAnythingConfig(
    working_dir="./rag_storage",
    enable_audio_processing=True,  # Enable audio processing
)

# Initialize
rag_anything = RAGAnything(
    config=config,
    llm_model_func=your_llm_function,
    audio_llm_func=your_audio_llm_function,  # Audio-specific LLM function
    embedding_func=your_embedding_function,
)
```

### 2. Insert Single Audio Content

```python
import json

# Prepare audio data
audio_data = {
    "audio_path": "/path/to/your/audio.mp3",
    "transcript": "This is the transcription of the audio...",
    "duration": "00:03:45",
    "format": "MP3",
    "description": "Educational lecture audio"
}

# Insert audio content
result = await rag_anything.insert_audio_content(
    audio_content=json.dumps(audio_data, ensure_ascii=False),
    entity_name="My Audio Lecture",
    file_path="lecture_audio"
)

print(f"Insertion successful: {result[1]['entity_name']}")
```

### 3. Batch Process Audio Content

```python
# Prepare multiple audio data
audio_batch = [
    {
        "audio_path": "/path/to/audio1.mp3",
        "transcript": "First audio transcription...",
        "duration": "00:05:30",
        "format": "MP3",
        "description": "Course part one"
    },
    {
        "audio_path": "/path/to/audio2.wav", 
        "transcript": "Second audio transcription...",
        "duration": "00:08:15",
        "format": "WAV",
        "description": "Course part two"
    }
]

# Batch insert
for i, audio_data in enumerate(audio_batch):
    result = await rag_anything.insert_audio_content(
        audio_content=json.dumps(audio_data, ensure_ascii=False),
        entity_name=f"Course Audio {i+1}",
        file_path=f"course_audio_{i+1}"
    )
    print(f"✓ Processed: {result[1]['entity_name']}")
```

### 4. Query Audio-Related Information

```python
# Query audio content
query = "What content was covered in the course?"
response = await rag_anything.aquery(query, mode="hybrid")
print(f"Query result: {response}")
```

## Context-Aware Processing

The audio processor supports context-aware analysis, providing more accurate analysis by combining surrounding content:

```python
# Set content source for context extraction
rag_anything.set_content_source_for_context(
    content_source=your_content_list,  # Complete content list containing audio
    content_format="minerU"  # Or other formats
)

# Configure context extraction parameters
rag_anything.update_context_config(
    context_window=2,          # Context from 2 pages before and after
    max_context_tokens=2000,   # Maximum context tokens
    include_headers=True,      # Include headers
    include_captions=True      # Include captions and descriptions
)
```

## Audio LLM Function

Audio processing uses a dedicated audio LLM function, similar to how vision processing uses a vision model:

```python
async def audio_llm_func(prompt, system_prompt=None, **kwargs):
    """
    Audio-specific LLM function for analyzing audio content
    
    This function should be capable of understanding audio-related prompts
    and providing detailed analysis of audio content including:
    - Speech recognition and transcription analysis
    - Audio quality assessment
    - Content type identification (speech, music, sound effects)
    - Emotional tone and atmosphere analysis
    """
    # Your audio LLM implementation here
    # e.g., OpenAI API with audio capabilities, specialized audio models, etc.
    pass

# Initialize RAGAnything with audio LLM function
rag_anything = RAGAnything(
    config=config,
    llm_model_func=your_general_llm_function,  # For general text analysis
    audio_llm_func=audio_llm_func,            # For audio-specific analysis
    embedding_func=your_embedding_function,
)
```

## Prompt Templates

Audio processing uses specialized prompt templates for analysis:

- `AUDIO_ANALYSIS_SYSTEM`: Audio analysis system prompt
- `audio_prompt`: Basic audio analysis prompt
- `audio_prompt_with_context`: Context-aware audio analysis prompt
- `audio_chunk`: Audio content chunk template
- `QUERY_AUDIO_ANALYSIS`: Audio query analysis prompt

## Example Code

The project provides two example files:

1. **Complete Example**: `examples/audio_processing_example.py`
   - Shows complete audio processing workflow
   - Includes batch processing and query functionality
   - Detailed configuration and error handling

2. **Simple Example**: `examples/simple_audio_insert_example.py`
   - Minimal audio insertion example
   - Quick start guide
   - Basic configuration and processing

## Running Examples

```bash
# Run complete example
python examples/audio_processing_example.py

# Run simple example
python examples/simple_audio_insert_example.py
```

## Notes

1. **File Paths**: Ensure audio file paths are correct and files exist
2. **Audio LLM Configuration**: Need to configure an audio LLM function that supports audio analysis
3. **Transcription Quality**: Providing accurate transcription text significantly improves analysis quality
4. **File Formats**: While multiple formats are supported, common audio formats are recommended
5. **Resource Cleanup**: Remember to call `finalize_storages()` after processing to clean up resources

## Error Handling

Common errors and solutions:

- **Audio file not found**: Check if file path is correct
- **Audio processing not enabled**: Ensure `enable_audio_processing=True`
- **Audio LLM function not configured**: Ensure a valid audio LLM function is provided
- **Permission issues**: Ensure read permissions for audio files

## Extended Features

The audio processor is built on `BaseModalProcessor` and supports:

- Custom prompt templates
- Context-aware analysis
- Batch processing mode
- Entity relationship extraction
- Vectorized storage and retrieval

Through these features, audio content can work with other modal content (text, images, tables, etc.) to build a unified knowledge graph.

## Architecture

```
RAGAnything
├── llm_model_func (for general text analysis)
├── audio_llm_func (for audio-specific analysis) 
├── vision_model_func (for image analysis)
└── embedding_func (for vectorization)

AudioModalProcessor
├── Uses audio_llm_func for audio analysis
├── Supports context-aware processing
├── Integrates with knowledge graph
└── Provides entity relationship extraction
```

The audio processor follows the same pattern as the image processor but uses the dedicated `audio_llm_func` for specialized audio content analysis.