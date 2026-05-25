/**
 * MODELFLOW - Model Viewer
 * Display and interact with model information and results
 */

class ModelViewer {
    constructor() {
        this.currentModel = null;
        this.predictionHistory = [];
    }
    
    displayModelInfo(modelData) {
        this.currentModel = modelData;
        
        const container = document.getElementById('modelInfo');
        if (!container) return;
        
        container.innerHTML = `
            <div class="model-info-card">
                <h3>${modelData.name}</h3>
                <div class="model-details">
                    <div class="detail-item">
                        <strong>Architecture:</strong>
                        <span>${modelData.architecture}</span>
                    </div>
                    <div class="detail-item">
                        <strong>Task Type:</strong>
                        <span>${modelData.task_type}</span>
                    </div>
                    <div class="detail-item">
                        <strong>Framework:</strong>
                        <span>${modelData.framework}</span>
                    </div>
                    ${modelData.input_shape ? `
                    <div class="detail-item">
                        <strong>Input Shape:</strong>
                        <span>${modelData.input_shape.join(' × ')}</span>
                    </div>
                    ` : ''}
                    ${modelData.num_classes ? `
                    <div class="detail-item">
                        <strong>Classes:</strong>
                        <span>${modelData.num_classes}</span>
                    </div>
                    ` : ''}
                </div>
            </div>
        `;
    }
    
    displayPrediction(predictionData) {
        const container = document.getElementById('resultsArea');
        if (!container) return;
        
        this.predictionHistory.push({
            timestamp: new Date(),
            data: predictionData
        });
        
        container.innerHTML = this.createPredictionHTML(predictionData);
    }
    
    createPredictionHTML(data) {
        let html = '<div class="prediction-result">';
        
        // Main prediction
        html += `<h3>Prediction: <span class="highlight">${data.predicted_class_name || data.predicted_class || 'N/A'}</span></h3>`;
        
        // Confidence bar
        if (data.confidence !== undefined) {
            const confidencePercent = (data.confidence * 100).toFixed(2);
            html += `
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                </div>
                <p>Confidence: ${confidencePercent}%</p>
            `;
        }
        
        // All predictions
        if (data.all_predictions || data.all_probabilities) {
            const probs = data.all_predictions || data.all_probabilities;
            html += '<h4>All Predictions:</h4><div class="predictions-list">';
            
            probs.forEach((prob, idx) => {
                const percent = (prob * 100).toFixed(2);
                html += `
                    <div class="prediction-item">
                        <span>Class ${idx}</span>
                        <div class="prediction-bar">
                            <div class="bar-fill" style="width: ${percent}%"></div>
                            <span class="bar-label">${percent}%</span>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
        }
        
        // Additional metadata
        if (data.metadata) {
            html += '<div class="prediction-metadata">';
            html += '<h4>Additional Information:</h4>';
            for (const [key, value] of Object.entries(data.metadata)) {
                html += `<p><strong>${key}:</strong> ${value}</p>`;
            }
            html += '</div>';
        }
        
        html += '</div>';
        return html;
    }
    
    displayTrainingProgress(progressData) {
        // Update status
        const statusEl = document.getElementById('trainingStatus');
        if (statusEl) {
            statusEl.textContent = progressData.status;
        }
        
        // Update epoch
        const currentEpochEl = document.getElementById('currentEpoch');
        if (currentEpochEl) {
            currentEpochEl.textContent = progressData.current_epoch;
        }
        
        // Update metrics
        if (progressData.metrics) {
            const lossEl = document.getElementById('currentLoss');
            if (lossEl && progressData.metrics.loss !== undefined) {
                lossEl.textContent = progressData.metrics.loss.toFixed(4);
            }
            
            const accEl = document.getElementById('currentAccuracy');
            if (accEl && progressData.metrics.accuracy !== undefined) {
                accEl.textContent = (progressData.metrics.accuracy * 100).toFixed(2) + '%';
            }
        }
        
        // Update progress bar
        const progressBar = document.getElementById('progressBar');
        if (progressBar && progressData.total_epochs) {
            const percent = (progressData.current_epoch / progressData.total_epochs) * 100;
            progressBar.style.width = percent + '%';
        }
    }
    
    displayMetrics(metrics) {
        const metricsData = [
            { id: 'accuracy', value: metrics.accuracy },
            { id: 'precision', value: metrics.precision },
            { id: 'recall', value: metrics.recall },
            { id: 'f1score', value: metrics.f1_score }
        ];
        
        metricsData.forEach(metric => {
            const el = document.getElementById(metric.id);
            if (el && metric.value !== undefined) {
                el.textContent = (metric.value * 100).toFixed(2) + '%';
            }
        });
    }
    
    displayConfusionMatrix(matrix, classNames = null) {
        const container = document.getElementById('confusionMatrix');
        if (!container) return;
        
        let html = '<table class="cm-table"><thead><tr><th></th>';
        
        // Header row
        for (let i = 0; i < matrix.length; i++) {
            const label = classNames ? classNames[i] : `Class ${i}`;
            html += `<th>${label}</th>`;
        }
        html += '</tr></thead><tbody>';
        
        // Data rows
        matrix.forEach((row, i) => {
            const rowLabel = classNames ? classNames[i] : `Class ${i}`;
            html += `<tr><th>${rowLabel}</th>`;
            
            row.forEach((val, j) => {
                const isCorrect = i === j;
                const cellClass = isCorrect ? 'cm-correct' : 'cm-incorrect';
                html += `<td class="${cellClass}">${val}</td>`;
            });
            
            html += '</tr>';
        });
        
        html += '</tbody></table>';
        container.innerHTML = html;
    }
    
    clearResults() {
        const container = document.getElementById('resultsArea');
        if (container) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-chart-bar"></i>
                    <p>Results will appear here after prediction</p>
                </div>
            `;
        }
    }
    
    getPredictionHistory() {
        return this.predictionHistory;
    }
    
    exportHistory(format = 'json') {
        if (format === 'json') {
            const dataStr = JSON.stringify(this.predictionHistory, null, 2);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });
            this.downloadFile(dataBlob, 'prediction_history.json');
        } else if (format === 'csv') {
            let csv = 'Timestamp,Prediction,Confidence\n';
            this.predictionHistory.forEach(item => {
                csv += `${item.timestamp.toISOString()},${item.data.predicted_class},${item.data.confidence}\n`;
            });
            const csvBlob = new Blob([csv], { type: 'text/csv' });
            this.downloadFile(csvBlob, 'prediction_history.csv');
        }
    }
    
    downloadFile(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Create global instance
const modelViewer = new ModelViewer();

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ModelViewer;
}
