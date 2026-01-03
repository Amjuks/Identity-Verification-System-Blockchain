"""
DID++ ML Engine
Multi-modal biometric processing for face, voice, and document embeddings.
Uses FaceNet (InceptionResnetV1) for face recognition and SpeechBrain ECAPA-TDNN for speaker verification.
"""

import os
# Fix for Windows symlink privilege issue with SpeechBrain
# Must be set BEFORE speechbrain is imported anywhere
os.environ['SPEECHBRAIN_LOCAL_STRATEGY'] = 'copy'

import io
import tempfile
import numpy as np
from typing import Tuple, Optional
import cv2
import librosa
import easyocr


class FaceProcessor:
    """
    Face embedding extraction using FaceNet (via facenet-pytorch).
    Produces 512-D embeddings that are highly discriminative.
    Uses MTCNN for face detection and InceptionResnetV1 pretrained on VGGFace2.
    """
    
    def __init__(self, output_dim: int = 512):
        self.output_dim = output_dim
        self.mtcnn = None
        self.resnet = None
        self._initialized = False
        self._device = None
    
    def _get_models(self):
        """Lazy initialization of FaceNet models with GPU support."""
        if not self._initialized:
            try:
                import torch
                from facenet_pytorch import MTCNN, InceptionResnetV1
                
                # Diagnostic logging for GPU availability
                cuda_available = torch.cuda.is_available()
                print(f"PyTorch version: {torch.__version__}")
                print(f"CUDA available: {cuda_available}")
                if cuda_available:
                    print(f"CUDA version: {torch.version.cuda}")
                    print(f"GPU device: {torch.cuda.get_device_name(0)}")
                
                # Use GPU if available
                self._device = torch.device('cuda' if cuda_available else 'cpu')
                
                # Initialize MTCNN for face detection
                # Returns face tensors ready for the recognition model
                self.mtcnn = MTCNN(
                    image_size=160,
                    margin=20,
                    min_face_size=20,
                    thresholds=[0.6, 0.7, 0.7],
                    factor=0.709,
                    post_process=True,
                    device=self._device,
                    keep_all=False  # Only keep the largest face
                )
                
                # Initialize InceptionResnetV1 for face embedding
                self.resnet = InceptionResnetV1(
                    pretrained='vggface2',
                    classify=False,
                    device=self._device
                ).eval()
                
                # Ensure model is on correct device
                self.resnet = self.resnet.to(self._device)
                
                self._initialized = True
                print(f"FaceNet initialized on device: {self._device}")
            except Exception as e:
                print(f"Failed to initialize FaceNet: {e}")
                import traceback
                traceback.print_exc()
                self._initialized = True  # Don't retry on failure
        return self.mtcnn, self.resnet
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better face detection.
        - Resize if too small
        - Convert color space if needed
        """
        if image is None:
            return None
        
        # Ensure image is in RGB format (MTCNN expects RGB)
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            # BGR to RGB (OpenCV loads as BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get dimensions
        h, w = image.shape[:2]
        
        # Resize if image is too small (face detection needs reasonable size)
        min_size = 160
        if h < min_size or w < min_size:
            scale = max(min_size / h, min_size / w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"Resized image from ({w}, {h}) to ({new_w}, {new_h})")
        
        return image
    
    def process(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Process face image and return 512-D FaceNet embedding.
        
        Args:
            image_bytes: Raw image bytes (JPEG)
            
        Returns:
            512-D float32 embedding or None if face not detected
        """
        try:
            import torch
            from PIL import Image
            
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                print("Failed to decode image")
                return None
            
            print(f"Original image size: {image.shape}")
            
            # Preprocess for better detection
            image = self.preprocess_image(image)
            
            if image is None:
                print("Image preprocessing failed")
                return None
            
            # Get models
            mtcnn, resnet = self._get_models()
            if mtcnn is None or resnet is None:
                print("FaceNet models not initialized")
                return None
            
            # Convert to PIL Image (MTCNN expects PIL Image or numpy array)
            pil_image = Image.fromarray(image)
            
            # Detect face and get aligned face tensor
            face_tensor = mtcnn(pil_image)
            
            if face_tensor is None:
                print(f"No face detected in image (size: {image.shape})")
                return None
            
            # Ensure tensor is on correct device and has batch dimension
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
            face_tensor = face_tensor.to(self._device)
            
            # Get embedding
            with torch.no_grad():
                embedding = resnet(face_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            print(f"Face detected! Embedding shape: {embedding.shape}")
            
            # Ensure it's float32 and normalized
            embedding = embedding.astype(np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            print(f"Face processing error: {e}")
            import traceback
            traceback.print_exc()
            return None


class VoiceProcessor:
    """
    Voice embedding extraction using SpeechBrain ECAPA-TDNN.
    Produces 192-D speaker embeddings that are highly discriminative.
    Falls back to MFCC if SpeechBrain fails.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.encoder = None
        self._initialized = False
        self._use_fallback = False
    
    def _get_encoder(self):
        """Lazy initialization of SpeechBrain speaker encoder."""
        if not self._initialized:
            try:
                # Check torchaudio compatibility first
                import torch
                import torchaudio
                
                # Try to check available backends (handle API changes)
                try:
                    if hasattr(torchaudio, 'list_audio_backends'):
                        backends = torchaudio.list_audio_backends()
                        print(f"Available audio backends: {backends}")
                except Exception:
                    pass
                
                from speechbrain.inference import EncoderClassifier
                # Import LocalStrategy for Windows symlink fix
                from speechbrain.utils.fetching import LocalStrategy
                
                # Use ECAPA-TDNN model for speaker embedding
                # Use COPY strategy to avoid Windows symlink privilege issues
                self.encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="data/speechbrain_models/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                    local_strategy=LocalStrategy.COPY
                )
                self._initialized = True
                print(f"SpeechBrain ECAPA-TDNN model loaded (device: {'cuda' if torch.cuda.is_available() else 'cpu'})")
            except ImportError as e:
                # Try alternative import path
                try:
                    import torch
                    from speechbrain.pretrained import EncoderClassifier
                    from speechbrain.utils.fetching import LocalStrategy
                    
                    self.encoder = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir="data/speechbrain_models/spkrec-ecapa-voxceleb",
                        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                        local_strategy=LocalStrategy.COPY
                    )
                    self._initialized = True
                    print("SpeechBrain loaded via pretrained import path")
                except Exception as e2:
                    print(f"Failed to initialize SpeechBrain: {e2}")
                    self._use_fallback = True
                    self._initialized = True
            except Exception as e:
                print(f"Failed to initialize SpeechBrain: {e}")
                import traceback
                traceback.print_exc()
                self._use_fallback = True
                self._initialized = True
        return self.encoder
    
    def process(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """
        Process voice audio and return 192-D speaker embedding.
        
        Includes enhanced preprocessing for better accuracy:
        - Audio normalization
        - Noise reduction
        - Voice activity detection (trimming silence)
        - Minimum 3-second requirement for better embeddings
        
        Args:
            audio_bytes: Raw audio bytes (WAV/WebM format)
            
        Returns:
            192-D float32 embedding or None on error
        """
        try:
            # Load audio using librosa (handles various formats)
            audio_io = io.BytesIO(audio_bytes)
            y, sr = librosa.load(audio_io, sr=self.sample_rate)
            
            if len(y) == 0:
                print("Empty audio")
                return None
            
            # ============ Enhanced Preprocessing ============
            
            # 1. Normalize audio to consistent volume level
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val * 0.95  # Normalize to 95% of max
            
            # 2. Remove DC offset
            y = y - np.mean(y)
            
            # 3. Trim silence from beginning and end (voice activity detection)
            # Use energy-based trimming
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            
            # Only use trimmed if it preserves reasonable amount of audio
            if len(y_trimmed) > self.sample_rate * 0.5:  # At least 0.5 second
                y = y_trimmed
                print(f"Audio trimmed: {len(y)/self.sample_rate:.2f}s of speech detected")
            
            # 4. Apply pre-emphasis filter to boost high frequencies (clearer speech)
            pre_emphasis = 0.97
            y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
            
            # 5. Minimum 3 seconds for reliable speaker embedding
            min_duration = 3 * self.sample_rate
            if len(y) < min_duration:
                # Repeat audio to reach minimum duration (better than padding with silence)
                repeats = int(np.ceil(min_duration / len(y)))
                y = np.tile(y, repeats)[:min_duration]
                print(f"Audio repeated to reach {min_duration/self.sample_rate:.1f}s minimum")
            
            # Re-normalize after preprocessing
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val * 0.95
            
            # Get encoder
            encoder = self._get_encoder()
            
            if encoder is None or self._use_fallback:
                print("Using MFCC fallback for voice embedding")
                return self._fallback_mfcc(y, sr)
            
            import torch
            
            # Move tensor to same device as model
            device = next(encoder.mods.parameters()).device
            
            # Convert to tensor (SpeechBrain expects [batch, time])
            audio_tensor = torch.tensor(y).unsqueeze(0).float().to(device)
            
            # Get embedding
            with torch.no_grad():
                embedding = encoder.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            print(f"Voice embedding extracted, shape: {embedding.shape}")
            
            # Ensure it's float32 and normalized
            embedding = embedding.astype(np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            print(f"Voice processing error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to MFCC
            try:
                audio_io = io.BytesIO(audio_bytes)
                y, sr = librosa.load(audio_io, sr=self.sample_rate)
                return self._fallback_mfcc(y, sr)
            except:
                return None
    
    def _fallback_mfcc(self, y: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Fallback MFCC embedding if SpeechBrain fails."""
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            
            # Compute statistics
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
            mfcc_delta2 = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)
            
            # Concatenate features (40*4 = 160, pad to 192)
            embedding = np.concatenate([mfcc_mean, mfcc_std, mfcc_delta, mfcc_delta2])
            
            # Pad to 192 dimensions to match ECAPA-TDNN
            if len(embedding) < 192:
                embedding = np.pad(embedding, (0, 192 - len(embedding)))
            else:
                embedding = embedding[:192]
            
            # L2 normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"MFCC fallback error: {e}")
            return None


class DocumentProcessor:
    """
    Document processing using EasyOCR and ArcFace.
    Extracts text and face embedding for combined embedding.
    """
    
    def __init__(self, output_dim: int = 512):
        self.output_dim = output_dim
        
        # Initialize EasyOCR reader with GPU support if available
        import torch
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        # Initialize face processor for document face detection
        self.face_processor = FaceProcessor()
        
        # Text embedding dimension
        self.text_dim = 128
    
    def extract_text(self, image_bytes: bytes) -> str:
        """Extract text from document using OCR."""
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return ""
        
        # Run OCR
        results = self.reader.readtext(image)
        
        # Combine all detected text
        text_parts = [result[1] for result in results]
        return " ".join(text_parts)
    
    def text_to_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to a simple embedding using character-level features.
        """
        if not text:
            return np.zeros(self.text_dim, dtype=np.float32)
        
        # Character frequency features
        text_lower = text.lower()
        char_freq = np.zeros(26 + 10, dtype=np.float32)  # a-z + 0-9
        
        for char in text_lower:
            if 'a' <= char <= 'z':
                char_freq[ord(char) - ord('a')] += 1
            elif '0' <= char <= '9':
                char_freq[26 + int(char)] += 1
        
        # Normalize
        if char_freq.sum() > 0:
            char_freq = char_freq / char_freq.sum()
        
        # Expand to text_dim using projection
        np.random.seed(43)
        projection = np.random.randn(36, self.text_dim).astype(np.float32)
        embedding = np.dot(char_freq, projection)
        
        # L2 normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding.astype(np.float32)
    
    def process(self, image_bytes: bytes) -> Tuple[Optional[np.ndarray], str]:
        """
        Process document and return embedding and extracted text.
        
        Args:
            image_bytes: Raw image bytes (JPEG)
            
        Returns:
            Tuple of (embedding, extracted text)
        """
        # Extract text
        text = self.extract_text(image_bytes)
        
        # Extract face from document using FaceNet
        face_embedding = self.face_processor.process(image_bytes)
        
        # Create text embedding
        text_embedding = self.text_to_embedding(text)
        
        if face_embedding is not None:
            # Combine face (512D) and text (128D) embeddings
            combined = np.concatenate([face_embedding, text_embedding])
        else:
            # Use text only, pad with zeros for face portion
            combined = np.concatenate([
                np.zeros(512, dtype=np.float32),
                text_embedding
            ])
        
        # L2 normalize final embedding
        combined = combined / (np.linalg.norm(combined) + 1e-8)
        
        return combined.astype(np.float32), text


class MLEngine:
    """Main ML engine combining all biometric processors."""
    
    def __init__(self):
        self.face_processor = FaceProcessor()
        self.voice_processor = VoiceProcessor()
        self.document_processor = DocumentProcessor()
    
    def process_face(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Process face image and return 512-D FaceNet embedding."""
        return self.face_processor.process(image_bytes)
    
    def process_voice(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Process voice audio and return 192-D speaker embedding."""
        return self.voice_processor.process(audio_bytes)
    
    def process_document(self, image_bytes: bytes) -> Tuple[Optional[np.ndarray], str]:
        """Process document and return embedding + text."""
        return self.document_processor.process(image_bytes)
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        # Handle different embedding sizes
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]
        
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot / (norm_a * norm_b))
    
    @staticmethod
    def text_overlap(text1: str, text2: str) -> float:
        """
        Compute text overlap score using Jaccard similarity.
        """
        if not text1 or not text2:
            return 0.0
        
        # Tokenize (simple whitespace split)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)


# Global ML engine instance
ml_engine = MLEngine()
