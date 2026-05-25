/**
 * MODELFLOW - Webcam Handler
 * Handles webcam access and image capture
 */

class WebcamHandler {
    constructor(videoElementId, canvasElementId) {
        this.video = document.getElementById(videoElementId);
        this.canvas = document.getElementById(canvasElementId);
        this.stream = null;
        this.isActive = false;
        this.frameCount = 0;
        this.fps = 0;
        this.lastFrameTime = Date.now();
    }
    
    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
            
            if (this.video) {
                this.video.srcObject = this.stream;
                this.isActive = true;
                this.startFPSCounter();
                
                // Trigger event
                this.dispatchEvent('webcamStarted');
                
                return true;
            }
            
        } catch (error) {
            console.error('Error accessing webcam:', error);
            throw new Error('Failed to access webcam: ' + error.message);
        }
    }
    
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            if (this.video) {
                this.video.srcObject = null;
            }
            this.isActive = false;
            this.fps = 0;
            
            // Trigger event
            this.dispatchEvent('webcamStopped');
        }
    }
    
    capture() {
        if (!this.isActive || !this.video || !this.canvas) {
            throw new Error('Webcam not active');
        }
        
        // Set canvas size to match video
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        
        // Draw current frame
        const ctx = this.canvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0);
        
        // Update FPS
        this.updateFPS();
        
        return this.canvas;
    }
    
    captureAsBlob(mimeType = 'image/jpeg', quality = 0.95) {
        return new Promise((resolve, reject) => {
            try {
                this.capture();
                this.canvas.toBlob(
                    (blob) => {
                        if (blob) {
                            resolve(blob);
                        } else {
                            reject(new Error('Failed to create blob'));
                        }
                    },
                    mimeType,
                    quality
                );
            } catch (error) {
                reject(error);
            }
        });
    }
    
    captureAsDataURL(mimeType = 'image/jpeg', quality = 0.95) {
        try {
            this.capture();
            return this.canvas.toDataURL(mimeType, quality);
        } catch (error) {
            throw new Error('Failed to capture image: ' + error.message);
        }
    }
    
    startFPSCounter() {
        if (this.fpsInterval) {
            clearInterval(this.fpsInterval);
        }
        
        this.fpsInterval = setInterval(() => {
            if (this.isActive) {
                this.dispatchEvent('fpsUpdate', { fps: this.fps });
            }
        }, 1000);
    }
    
    updateFPS() {
        const currentTime = Date.now();
        const delta = currentTime - this.lastFrameTime;
        this.fps = Math.round(1000 / delta);
        this.frameCount++;
        this.lastFrameTime = currentTime;
    }
    
    dispatchEvent(eventName, detail = {}) {
        const event = new CustomEvent(eventName, { detail });
        document.dispatchEvent(event);
    }
    
    getStats() {
        return {
            isActive: this.isActive,
            fps: this.fps,
            frameCount: this.frameCount,
            videoWidth: this.video ? this.video.videoWidth : 0,
            videoHeight: this.video ? this.video.videoHeight : 0
        };
    }
    
    async takePhoto() {
        try {
            const blob = await this.captureAsBlob();
            const file = new File([blob], 'webcam_capture.jpg', { type: 'image/jpeg' });
            return file;
        } catch (error) {
            throw new Error('Failed to take photo: ' + error.message);
        }
    }
    
    startContinuousCapture(interval = 1000, callback) {
        if (this.captureInterval) {
            this.stopContinuousCapture();
        }
        
        this.captureInterval = setInterval(async () => {
            try {
                const blob = await this.captureAsBlob();
                if (callback) {
                    callback(blob);
                }
            } catch (error) {
                console.error('Continuous capture error:', error);
            }
        }, interval);
    }
    
    stopContinuousCapture() {
        if (this.captureInterval) {
            clearInterval(this.captureInterval);
            this.captureInterval = null;
        }
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebcamHandler;
}
