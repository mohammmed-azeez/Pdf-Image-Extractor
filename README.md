# PDF Image Extractor

This project extracts meaningful images from PDF files, processes them, and generates metadata including tags and captions using AI models.

## Features
- Extracts images from PDF files
- Filters meaningful images using advanced image processing
- Enhances and crops images to focus on relevant content
- Generates descriptive tags using CLIP model
- Creates captions using Google's Gemini AI
- Saves images and metadata in organized structure

## Requirements
- Python 3.8+
- See requirements.txt for all dependencies

## Installation
1. Clone this repository
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Place your PDF file (named `chapter_4.pdf`) in the project directory
2. Run the script:
   ```
   python main.py
   ```
3. Extracted images will be saved in `extracted_images` folder
4. Metadata will be saved as `image_metadata.json` in the same folder

## Configuration
You can modify these parameters in `main.py`:
- `output_dir`: Change output directory
- `batch_size`: Adjust number of pages processed at once
- `gemini_api_key`: Set your Gemini API key

## Notes
- Requires Google Gemini API key for caption generation
- Processing large PDFs may take significant time
- Ensure sufficient disk space for extracted images