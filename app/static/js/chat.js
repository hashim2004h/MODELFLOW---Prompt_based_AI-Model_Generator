/**
 * MODELFLOW Chat Interface
 * Handles chat interactions and model recommendations
 */

class ModelflowChat {
    constructor() {
        this.messages = [];
        this.currentModel = null;
        this.isLoading = false;
        this.init();
    }
    
    init() {
        this.setupElements();
        this.attachEventListeners();
        this.loadChatHistory();
    }
    
    setupElements() {
        this.promptInput = document.getElementById('promptInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.messagesContainer = document.getElementById('messagesContainer');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.detailsPanel = document.getElementById('detailsPanel');
        this.panelContent = document.getElementById('panelContent');
        this.closePanel = document.getElementById('closePanel');
    }
    
    attachEventListeners() {
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        this.newChatBtn.addEventListener('click', () => this.newChat());
        this.closePanel.addEventListener('click', () => this.hideDetailsPanel());
        this.promptInput.addEventListener('input', () => this.autoGrowTextarea());
    }
    
    autoGrowTextarea() {
        this.promptInput.style.height = 'auto';
        this.promptInput.style.height = Math.min(this.promptInput.scrollHeight, 150) + 'px';
    }
    
    async sendMessage() {
        const prompt = this.promptInput.value.trim();
        if (!prompt || this.isLoading) return;
        
        // Add user message
        this.addMessage(prompt, 'user');
        this.promptInput.value = '';
        this.autoGrowTextarea();
        
        this.isLoading = true;
        this.sendBtn.disabled = true;
        this.sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        
        try {
            // Parse prompt
            const response = await fetch('/api/parse-prompt', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Add assistant response with model recommendations
                this.displayModelRecommendations(data);
            } else {
                this.addMessage('Error: ' + data.error, 'assistant');
            }
        } catch (error) {
            this.addMessage('Failed to process request: ' + error.message, 'assistant');
        } finally {
            this.isLoading = false;
            this.sendBtn.disabled = false;
            this.sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
    }
    
    displayModelRecommendations(data) {
        const taskType = data.task_type || 'Unknown';
        const inputType = data.input_type || 'Unknown';
        
        let html = `
            <div class="message assistant">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="message-text">
                        <p>✓ Understood! I've analyzed your task:</p>
                        <div style="margin: 1rem 0; padding: 1rem; background: rgba(99,102,241,0.1); border-radius: 8px;">
                            <p><strong>Task Type:</strong> ${taskType}</p>
                            <p><strong>Input Type:</strong> ${inputType}</p>
                        </div>
                        <p>Based on this, here are my recommended models:</p>
                    </div>
                </div>
            </div>
        `;
        
        this.messagesContainer.innerHTML += html;
        
        // Add model recommendations
        const modelHtml = `
            <div class="message assistant">
                <div class="message-avatar">
                    <i class="fas fa-lightbulb"></i>
                </div>
                <div class="message-content">
                    <div class="model-grid">
                        ${data.recommended_models.slice(0, 3).map(model => `
                            <div class="model-option" onclick="selectModel('${model.name}', '${taskType}')">
                                <h4>${model.name}</h4>
                                <p>${model.description}</p>
                                <p style="margin-top: 0.5rem; font-size: 0.8rem; color: var(--text-light);">
                                    Size: ${model.size} | Speed: ${model.speed}
                                </p>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
        
        this.messagesContainer.innerHTML += modelHtml;
        this.scrollToBottom();
    }
    
    addMessage(text, sender) {
        const messageHtml = `
            <div class="message ${sender}">
                <div class="message-avatar">
                    ${sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>'}
                </div>
                <div class="message-content">
                    <div class="message-text">
                        ${text}
                    </div>
                </div>
            </div>
        `;
        
        this.messagesContainer.innerHTML += messageHtml;
        this.scrollToBottom();
        this.messages.push({ text, sender, timestamp: new Date() });
    }
    
    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    showDetailsPanel(modelData) {
        this.detailsPanel.style.display = 'flex';
        this.panelContent.innerHTML = `
            <div class="panel-section">
                <h4>Model Info</h4>
                <div class="section-card">
                    <p><strong>${modelData.name}</strong></p>
                    <p style="margin-top: 0.5rem; color: var(--text-secondary);">${modelData.description}</p>
                </div>
            </div>
            <div class="panel-section">
                <h4>Specifications</h4>
                <div class="section-card">
                    <p>Size: ${modelData.size}</p>
                    <p>Speed: ${modelData.speed}</p>
                    <p>Accuracy: ${modelData.accuracy}</p>
                </div>
            </div>
            <div class="panel-section">
                <button class="btn-large btn-gradient" onclick="startTraining('${modelData.name}')">
                    <i class="fas fa-play"></i> Train This Model
                </button>
            </div>
        `;
    }
    
    hideDetailsPanel() {
        this.detailsPanel.style.display = 'none';
    }
    
    newChat() {
        this.messages = [];
        this.messagesContainer.innerHTML = `
            <div class="message assistant">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="message-text">
                        <p>👋 Hi! I'm your AI model builder assistant. Tell me what you'd like to create:</p>
                        <ul style="margin: 1rem 0 1rem 1.5rem; color: var(--text-secondary);">
                            <li>"Classify images of cats and dogs"</li>
                            <li>"Detect faces in photos"</li>
                            <li>"Analyze sentiment of product reviews"</li>
                            <li>"Predict house prices from features"</li>
                        </ul>
                        <p>What would you like to build today?</p>
                    </div>
                </div>
            </div>
        `;
        this.hideDetailsPanel();
    }
    
    loadChatHistory() {
        // Load from localStorage if available
        const saved = localStorage.getItem('modelflow_chat_history');
        if (saved) {
            this.messages = JSON.parse(saved);
            // Display saved messages
        }
    }
}

// Global functions
function selectModel(modelName, taskType) {
    window.location.href = `/train?model=${modelName}&task=${taskType}`;
}

function startTraining(modelName) {
    window.location.href = `/train?model=${modelName}`;
}

// Initialize chat
const chat = new ModelflowChat();
