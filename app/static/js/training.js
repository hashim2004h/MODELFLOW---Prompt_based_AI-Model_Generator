/**
 * MODELFLOW Training Interface
 * Handles training data collection and model training
 */

class TrainingInterface {
    constructor() {
        this.webcamActive = false;
        this.capturedImages = [];
        this.currentClass = null;
        this.init();
    }
    
    init() {
        this.setupElements();
        this.attachEventListeners();
    }
    
    setupElements() {
        this.startWebcamBtn = document.getElementById('startWebcam');
        this.captureBtn = document.getElementById('captureImage');
        this.stopWebcamBtn = document.getElementById('stopWebcam');
        this.classInput = document.getElementById('classLabel');
        this.videoElement = document.getElementById('webcamVideo');
        this.canvasElement = document.getElementById('webcamCanvas');
        this.capturedImagesContainer = document.getElementById('capturedImages');
        this.startTrainingBtn = document.getElementById('startTraining');
        this.optionTabs = document.querySelectorAll('.option-tab');
    }
    
    attachEventListeners() {
        this.startWebcamBtn.addEventListener('click', () => this.startWebcam());
        this.captureBtn.addEventListener('click', () => this.captureImage());
        this.stopWebcamBtn.addEventListener('click', () => this.stopWebcam());
        this.startTrainingBtn.addEventListener('click', () => this.startTraining());
        
        // Tab switching
        this.optionTabs.forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e.target.closest('.option-tab')));
        });
        
        // Dropzone
        const dropzone = document.getElementById('dropzone');
        if (dropzone) {
            dropzone.addEventListener('click', () => document.getElementById('fileInput').click());
            dropzone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropzone.classList.add('dragover');
            });
            dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
            dropzone.addEventListener('drop', (e) => {
                e.preventDefault();
                this.handleFileUpload(e.dataTransfer.files);
            });
        }
        
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileUpload(e.target.files));
        }
    }
    
    switchTab(tab) {
        // Remove active from all tabs and contents
        document.querySelectorAll('.option-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        // Add active to clicked tab
        tab.classList.add('active');
        const tabId = tab.textContent.trim().replace(/\s+/g, '-').toLowerCase();
        const tabContent = document.getElementById(tabId + '-tab');
        if (tabContent) {
            tabContent.classList.add('active');
        }
    }
    
    async startWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 } }
            });
            
            this.videoElement.srcObject = stream;
            this.webcamActive = true;
            
            this.startWebcamBtn.disabled = true;
            this.captureBtn.disabled = false;
            this.stopWebcamBtn.disabled = false;
            
        } catch (error) {
            alert('Failed to access webcam: ' + error.message);
        }
    }
    
    captureImage() {
        const className = this.classInput.value.trim();
        if (!className) {
            alert('Please enter a class name first');
            return;
        }
        
        this.canvasElement.width = this.videoElement.videoWidth;
        this.canvasElement.height = this.videoElement.videoHeight;
        
        const ctx = this.canvasElement.getContext('2d');
        ctx.drawImage(this.videoElement, 0, 0);
        
        this.canvasElement.toBlob((blob) => {
            const imageUrl = URL.createObjectURL(blob);
            this.capturedImages.push({
                blob: blob,
                class: className,
                url: imageUrl
            });
            
            this.displayCapturedImage(imageUrl, className);
        });
    }
    
    displayCapturedImage(imageUrl, className) {
        const thumbnail = document.createElement('div');
        thumbnail.className = 'image-thumbnail';
        thumbnail.innerHTML = `
            <img src="${imageUrl}" alt="${className}">
            <button class="image-remove" onclick="training.removeImage(${this.capturedImages.length - 1})">
                <i class="fas fa-times"></i>
            </button>
            <small style="position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.7); padding: 0.25rem; text-align: center;">
                ${className}
            </small>
        `;
        this.capturedImagesContainer.appendChild(thumbnail);
    }
    
    removeImage(index) {
        this.capturedImages.splice(index, 1);
        this.refreshThumbnails();
    }
    
    refreshThumbnails() {
        this.capturedImagesContainer.innerHTML = '';
        this.capturedImages.forEach((img, idx) => {
            this.displayCapturedImage(img.url, img.class);
        });
    }
    
    stopWebcam() {
        const stream = this.videoElement.srcObject;
        stream.getTracks().forEach(track => track.stop());
        
        this.webcamActive = false;
        this.startWebcamBtn.disabled = false;
        this.captureBtn.disabled = true;
        this.stopWebcamBtn.disabled = true;
    }
    
    handleFileUpload(files) {
        // Handle multiple file uploads
        const uploadList = document.getElementById('uploadList');
        if (uploadList) {
            uploadList.innerHTML = Array.from(files).map((file, idx) => `
                <div class="upload-item">
                    <i class="fas fa-file"></i>
                    <span>${file.name}</span>
                    <span>${(file.size / 1024 / 1024).toFixed(2)} MB</span>
                </div>
            `).join('');
        }
    }
    
    async startTraining() {
        if (this.capturedImages.length === 0) {
            alert('Please capture or upload some images first');
            return;
        }
        
        const epochs = parseInt(document.getElementById('epochs').value);
        const batchSize = parseInt(document.getElementById('batchSize').value);
        const learningRate = parseFloat(document.getElementById('learningRate').value);
        
        const progressDiv = document.getElementById('trainingProgress');
        progressDiv.style.display = 'block';
        
        // Create FormData with images
        const formData = new FormData();
        this.capturedImages.forEach((img, idx) => {
            formData.append(`image_${idx}`, img.blob);
            formData.append(`class_${idx}`, img.class);
        });
        formData.append('epochs', epochs);
        formData.append('batch_size', batchSize);
        formData.append('learning_rate', learningRate);
        
        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.training_id) {
                this.monitorTraining(data.training_id);
            }
        } catch (error) {
            alert('Training failed: ' + error.message);
        }
    }
    
    async monitorTraining(trainingId) {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`/api/training-status/${trainingId}`);
                const data = await response.json();
                
                // Update UI
                const progress = (data.current_epoch / data.total_epochs) * 100;
                document.getElementById('progressFill').style.width = progress + '%';
                document.getElementById('currentEpoch').textContent = data.current_epoch;
                document.getElementById('currentLoss').textContent = data.metrics?.loss?.toFixed(4) || '--';
                document.getElementById('currentAccuracy').textContent = 
                    data.metrics?.accuracy ? (data.metrics.accuracy * 100).toFixed(2) + '%' : '--';
                
                if (data.status === 'completed') {
                    clearInterval(interval);
                    alert('Training completed!');
                    window.location.href = '/result';
                } else if (data.status === 'failed') {
                    clearInterval(interval);
                    alert('Training failed: ' + data.error);
                }
            } catch (error) {
                console.error('Error checking training status:', error);
            }
        }, 2000);
    }
}

// Global instance
const training = new TrainingInterface();
