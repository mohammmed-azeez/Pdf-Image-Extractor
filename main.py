import fitz  # PyMuPDF for PDF processing
import os
import json
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import io
import google.generativeai as genai
import cv2
import aiofiles
import asyncio

class PDFImageExtractor:
    def __init__(self, pdf_path, output_dir='extracted_images', gemini_api_key='AIzaSyDptRo_34RiCU6LKCpvdxHHeWh43EY8axA', batch_size=10):
        """
        Initialize the PDF Image Extractor with batch processing capabilities
        
        :param pdf_path: Path to the PDF file
        :param output_dir: Directory to save extracted images
        :param gemini_api_key: API key for Google's Gemini model
        :param batch_size: Number of pages to process in each batch
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models lazily to optimize memory usage
        self.clip_model = None
        self.clip_processor = None
        self.gemini_model = None
        self.gemini_api_key = gemini_api_key
        
        # Thread-safe metadata storage
        self.image_metadata = []
        self.metadata_lock = None
        
    def _initialize_models(self):
        """
        Lazy initialization of models to optimize memory usage
        """
        if self.clip_model is None:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        if self.gemini_model is None:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro-vision')

    def _is_meaningful_image(self, img):
        """
        Determine if an image is meaningful using advanced image processing and deep learning
        
        :param img: PIL Image object
        :return: Tuple (Boolean indicating if image is meaningful, cropped PIL Image)
        """
        # Convert CMYK to RGB if necessary
        if img.mode == 'CMYK':
            img = img.convert('RGB')
            
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Enhanced image analysis using multiple techniques
        def analyze_image_quality():
            # Calculate image entropy for information content
            histogram = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            histogram = histogram[histogram > 0]
            probabilities = histogram / histogram.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities))
            
            # Analyze image sharpness using Laplacian variance
            laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            # Detect edges using Canny
            edges = cv2.Canny(img_gray, 100, 200)
            edge_density = np.mean(edges > 0)
            
            # Calculate local contrast using local standard deviation
            local_std = cv2.GaussianBlur(img_gray, (7, 7), 0)
            local_std = np.std(local_std)
            
            return entropy, sharpness, edge_density, local_std
        
        # Get image quality metrics
        entropy, sharpness, edge_density, local_std = analyze_image_quality()
        
        # Enhanced size and aspect ratio analysis
        height, width = img_gray.shape
        min_size = 100
        max_aspect_ratio = 5
        aspect_ratio = max(width/height, height/width)
        
        # Advanced content analysis
        white_threshold = 245
        non_white_ratio = np.mean(img_gray < white_threshold)
        
        # Multi-factor meaningful image detection
        is_meaningful = (
            entropy > 2.0 and
            width >= min_size and
            height >= min_size and
            aspect_ratio < max_aspect_ratio and
            (non_white_ratio > 0.08 or
             sharpness > 150 or
             edge_density > 0.1 or
             local_std > 20)
        )
        
        if is_meaningful:
            # Enhanced image segmentation
            def segment_image():
                # Apply adaptive thresholding
                thresh = cv2.adaptiveThreshold(
                    img_gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                
                # Noise reduction
                kernel = np.ones((3,3), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                
                # Find contours with hierarchy
                contours, hierarchy = cv2.findContours(
                    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                
                if not contours:
                    return None
                
                # Filter and sort contours
                valid_contours = []
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if area > 100:  # Minimum area threshold
                        x, y, w, h = cv2.boundingRect(contour)
                        if w > 20 and h > 20:  # Minimum dimension threshold
                            valid_contours.append((area, contour))
                
                if not valid_contours:
                    return None
                
                # Get the largest valid contour
                largest_contour = max(valid_contours, key=lambda x: x[0])[1]
                return cv2.boundingRect(largest_contour)
            
            # Get the bounding rectangle
            rect = segment_image()
            if rect:
                x, y, w, h = rect
                
                # Add intelligent padding based on image size
                padding = int(min(w, h) * 0.1)  # 10% of smaller dimension
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(width - x, w + 2 * padding)
                h = min(height - y, h + 2 * padding)
                
                # Ensure minimum dimensions and aspect ratio
                if w >= 100 and h >= 100 and max(w/h, h/w) <= max_aspect_ratio:
                    # Crop and enhance the image
                    cropped_cv = img_cv[y:y+h, x:x+w]
                    
                    # Apply subtle enhancement
                    enhanced_cv = cv2.detailEnhance(cropped_cv, sigma_s=10, sigma_r=0.15)
                    
                    # Convert back to PIL Image
                    enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2RGB))
                    return True, enhanced_pil
        
        return is_meaningful, img

    async def extract_images(self):
        """
        Extract images from the PDF using asynchronous batch processing
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        import threading
        import math
        
        # Thread-safe list for metadata
        self.metadata_lock = threading.Lock()
        
        # Open the PDF
        doc = fitz.open(self.pdf_path)
        total_pages = len(doc)
        
        # Calculate number of batches
        num_batches = math.ceil(total_pages / self.batch_size)
        
        # Create thread pool for CPU-bound tasks
        executor = ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 4))
        
        async def process_image(page_num, img_info, img_index):
            try:
                # Extract the image
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(image_bytes))
                
                # Process image in thread pool
                is_meaningful, processed_img = await asyncio.get_event_loop().run_in_executor(
                    executor, self._is_meaningful_image, img
                )
                
                if is_meaningful:
                    # Generate unique filename
                    filename = f"image_page{page_num+1}_{img_index}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    # Process image format
                    if processed_img.mode == 'CMYK':
                        processed_img = processed_img.convert('RGB')
                    elif processed_img.mode in ['RGBA', 'LA']:
                        background = Image.new('RGB', processed_img.size, (255, 255, 255))
                        background.paste(processed_img, mask=processed_img.split()[-1])
                        processed_img = background
                    elif processed_img.mode != 'RGB':
                        processed_img = processed_img.convert('RGB')
                    
                    # Save image in thread pool
                    await asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: processed_img.save(filepath, 'PNG', quality=95)
                    )
                    
                    # Generate tags and caption
                    tags, caption = await self._generate_tags(img, executor)
                    
                    # Thread-safe metadata update
                    with self.metadata_lock:
                        self.image_metadata.append({
                            "filename": filename,
                            "page": page_num + 1,
                            "tags": tags,
                            "caption": caption
                        })
                        
                        # Save metadata periodically to free memory
                        if len(self.image_metadata) >= 100:
                            await self._save_metadata()
                    
            except Exception as e:
                print(f"Error processing image on page {page_num + 1}: {e}")
        
        async def process_page(page_num):
            page = doc[page_num]
            images = page.get_images(full=True)
            tasks = []
            
            for img_index, img_info in enumerate(images):
                task = asyncio.create_task(process_image(page_num, img_info, img_index))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            print(f"Processed page {page_num + 1}/{total_pages}")
            
            # Free memory by clearing page contents
            page.clean_contents()
            del page
        
        # Process pages in batches
        async def process_batch(batch_num):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_pages)
            tasks = [process_page(i) for i in range(start_idx, end_idx)]
            await asyncio.gather(*tasks)
            print(f"Completed batch {batch_num + 1}/{num_batches}")
        
        # Process all batches
        async def process_all_batches():
            tasks = [process_batch(i) for i in range(num_batches)]
            await asyncio.gather(*tasks)
        
        try:
            # Process all batches
            await process_all_batches()
        finally:
            # Clean up
            executor.shutdown()
            doc.close()
            
            # Save any remaining metadata
            try:
                if self.image_metadata:
                    await self._save_metadata()
            except Exception as e:
                print(f"Error saving final metadata: {e}")
                # Attempt one more time with a delay
                try:
                    await asyncio.sleep(1)
                    await self._save_metadata()
                except Exception as e:
                    print(f"Failed to save metadata after retry: {e}")

    async def _generate_tags(self, img, executor):
        """
        Generate tags and caption for an image using CLIP and Gemini with async processing
        
        :param img: PIL Image object
        :param executor: ThreadPoolExecutor instance for CPU-bound tasks
        :return: Tuple (list of tags, caption string)
        """
        # Ensure models are initialized
        await asyncio.get_event_loop().run_in_executor(executor, self._initialize_models)
        
        async def get_clip_tags():
            # Prepare image for CLIP
            educational_concepts = [
                "educational diagram", "scientific illustration", "biological image",
                "chemical structure", "mathematical graph", "historical map",
                "anatomical drawing", "physics diagram", "geological chart",
                "statistical plot", "engineering schematic", "geographical map"
            ]
            
            inputs = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: self.clip_processor(
                    text=educational_concepts,
                    images=img,
                    return_tensors="pt",
                    padding=True
                )
            )
            
            def process_outputs():
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                return outputs
            
            outputs = await asyncio.get_event_loop().run_in_executor(executor, process_outputs)
            
            # Get top tags from CLIP
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            return [educational_concepts[i].replace(" ", "_")
                    for i in torch.topk(probs[0], k=3).indices.tolist()]
        
        async def get_gemini_caption():
            try:
                # Convert PIL image to bytes for Gemini with memory optimization
                def prepare_image():
                    if img.mode == 'CMYK':
                        img_rgb = img.convert('RGB')
                        img_resized = img_rgb.resize((800, 800), Image.Resampling.LANCZOS)
                    else:
                        img_resized = img.resize((800, 800), Image.Resampling.LANCZOS)
                    
                    img_byte_arr = io.BytesIO()
                    img_resized.save(img_byte_arr, format='PNG', optimize=True)
                    return img_byte_arr.getvalue()
                
                img_byte_arr = await asyncio.get_event_loop().run_in_executor(
                    executor, prepare_image
                )
                
                # Generate caption
                response = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: self.gemini_model.generate_content([
                        "Describe this educational image in a concise way. Focus on the main subject and its educational context.",
                        img_byte_arr
                    ])
                )
                caption = response.text
                
                # Extract keywords from caption
                caption_response = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: self.gemini_model.generate_content(
                        f"Extract 3-5 key subject-specific terms from this caption as tags: {caption}"
                    )
                )
                
                gemini_tags = [tag.strip().lower().replace(" ", "_") 
                              for tag in caption_response.text.split(',')]
                
                return gemini_tags, caption
            except Exception as e:
                print(f"Error generating Gemini caption: {e}")
                return [], ""
        
        # Process tags and caption concurrently
        clip_tags, (gemini_tags, caption) = await asyncio.gather(get_clip_tags(), get_gemini_caption())
        return clip_tags + gemini_tags, caption

    async def _save_metadata(self):
        """
        Save metadata to a JSON file with memory-efficient batch processing
        """
        metadata_file = os.path.join(self.output_dir, 'image_metadata.json')
        temp_file = metadata_file + '.tmp'
        
        try:
            # Write to a temporary file first
            async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                await f.write('[\n')
                for i, item in enumerate(self.image_metadata):
                    if i > 0:
                        await f.write(',\n')
                    await f.write(json.dumps(item, ensure_ascii=False, indent=4))
                await f.write('\n]')
                await f.flush()
            
            # Rename temp file to final file
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            os.rename(temp_file, metadata_file)
            
            # Clear metadata after successful save
            self.image_metadata = []
            print(f"Metadata saved to {metadata_file}")
            
        except Exception as e:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            print(f"Error saving metadata: {e}")
            raise


def main():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Specify the path to your PDF
    pdf_path = os.path.join(current_dir, 'chapter_4.pdf')
    
    # Create output directory
    output_dir = os.path.join(current_dir, 'extracted_images')
    
    print(f"Processing PDF: {pdf_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Create extractor
        extractor = PDFImageExtractor(pdf_path, output_dir)
        
        # Extract images using asyncio
        import asyncio
        asyncio.run(extractor.extract_images())
        
        print("Image extraction completed successfully!")
        
    except Exception as e:
        print(f"Error during image extraction: {e}")
        raise

if __name__ == "__main__":
    main()