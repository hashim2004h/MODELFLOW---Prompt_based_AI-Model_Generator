/**
 * MODELFLOW - File Upload Handler
 * Handles file uploads with drag-and-drop functionality
 */

class FileUploadHandler {
    constructor(dropzoneId, fileInputId) {
        this.dropzone = document.getElementById(dropzoneId);
        this.fileInput = document.getElementById(fileInputId);
        this.selectedFiles = [];
        this.init();
    }
    
    init() {
        if (!this.dropzone || !this.fileInput) {
            console.error('Upload elements not found');
            return;
        }
        
        // Click to upload
        this.dropzone.addEventListener('click', () => {
            this.fileInput.click();
        });
        
        // File selection
        this.fileInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });
        
        // Drag and drop
        this.dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropzone.classList.add('dragover');
        });
        
        this.dropzone.addEventListener('dragleave', () => {
            this.dropzone.classList.remove('dragover');
        });
        
        this.dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropzone.classList.remove('dragover');
            this.handleFiles(e.dataTransfer.files);
        });
    }
    
    handleFiles(files) {
        this.selectedFiles = Array.from(files);
        this.displayFiles();
        
        // Trigger custom event
        const event = new CustomEvent('filesSelected', {
            detail: { files: this.selectedFiles }
        });
        document.dispatchEvent(event);
    }
    
    displayFiles() {
        const filesList = document.getElementById('filesList');
        if (!filesList) return;
        
        filesList.innerHTML = '';
        
        if (this.selectedFiles.length === 0) {
            return;
        }
        
        const title = document.createElement('h3');
        title.textContent = 'Selected Files:';
        filesList.appendChild(title);
        
        this.selectedFiles.forEach((file, index) => {
            const fileItem = this.createFileItem(file, index);
            filesList.appendChild(fileItem);
        });
    }
    
    createFileItem(file, index) {
        const item = document.createElement('div');
        item.className = 'file-item';
        
        const fileInfo = document.createElement('span');
        fileInfo.innerHTML = `
            <i class="fas fa-file"></i> ${file.name}
        `;
        
        const fileSize = document.createElement('span');
        fileSize.className = 'file-size';
        fileSize.textContent = this.formatFileSize(file.size);
        
        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'btn-remove';
        removeBtn.innerHTML = '<i class="fas fa-times"></i>';
        removeBtn.addEventListener('click', () => {
            this.removeFile(index);
        });
        
        item.appendChild(fileInfo);
        item.appendChild(fileSize);
        item.appendChild(removeBtn);
        
        return item;
    }
    
    removeFile(index) {
        this.selectedFiles.splice(index, 1);
        this.displayFiles();
        
        // Trigger custom event
        const event = new CustomEvent('filesSelected', {
            detail: { files: this.selectedFiles }
        });
        document.dispatchEvent(event);
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }
    
    async uploadFiles(url, dataType) {
        if (this.selectedFiles.length === 0) {
            throw new Error('No files selected');
        }
        
        const results = [];
        
        for (const file of this.selectedFiles) {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('data_type', dataType);
            
            try {
                const response = await fetch(url, {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                results.push({ file: file.name, success: data.success, data });
                
            } catch (error) {
                results.push({ file: file.name, success: false, error: error.message });
            }
        }
        
        return results;
    }
    
    clearFiles() {
        this.selectedFiles = [];
        this.fileInput.value = '';
        this.displayFiles();
    }
    
    getFiles() {
        return this.selectedFiles;
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FileUploadHandler;
}
