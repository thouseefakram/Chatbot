// Audio Recognition Module
class AudioRecognition {
    constructor() {
        this.recognition = null;
        this.isListening = false;
        this.audioBtn = document.createElement('button');
        this.audioBtn.className = 'audio-btn';
        this.audioBtn.innerHTML = '<img src="/static/icons/icons8-square-play-button-50.png" alt="Start Recording" width="24" height="24">';
        this.audioBtn.title = 'Start voice recording';
        
        this.visualizer = document.createElement('div');
        this.visualizer.className = 'audio-visualizer';
        this.visualizer.style.display = 'none';
        
        // Create 5 bars for the visualizer
        for (let i = 0; i < 5; i++) {
            const bar = document.createElement('div');
            bar.className = 'audio-bar';
            bar.style.animationDelay = `${i * 0.1}s`;
            this.visualizer.appendChild(bar);
        }
        
        this.setupUI();
        this.initRecognition();
    }
    
    setupUI() {
        const inputContainer = document.querySelector('.input-container');
        inputContainer.appendChild(this.visualizer);
        inputContainer.appendChild(this.audioBtn);
        
        this.audioBtn.addEventListener('click', () => {
            if (this.isListening) {
                this.stopRecognition();
            } else {
                this.startRecognition();
            }
        });
    }
    
    initRecognition() {
        // Try to use Web Speech API first
        if ('webkitSpeechRecognition' in window) {
            this.recognition = new webkitSpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = true;
            this.recognition.lang = 'en-US';
            
            this.recognition.onstart = () => {
                this.isListening = true;
                this.audioBtn.innerHTML = '<img src="/static/icons/icons8-stop-circled-50.png" alt="Stop Recording" width="24" height="24">';
                this.audioBtn.classList.add('listening');
                this.visualizer.style.display = 'flex';
                userInput.placeholder = "Listening...";
            };
            
            this.recognition.onend = () => {
                if (this.isListening) { // Only reset if not manually stopped
                    this.resetUI();
                }
            };
            
            this.recognition.onerror = (event) => {
                console.error('Speech recognition error', event.error);
                appendMessage('bot', `Speech recognition error: ${event.error}`);
                this.resetUI();
            };
            
            this.recognition.onresult = (event) => {
                const transcript = Array.from(event.results)
                    .map(result => result[0])
                    .map(result => result.transcript)
                    .join('');
                
                userInput.value = transcript;
            };
        } else {
            // Fallback to Whisper API if Web Speech not available
            this.audioBtn.onclick = () => this.useWhisperAPI();
        }
    }
    
    startRecognition() {
        if (this.recognition) {
            this.recognition.start();
        }
    }
    
    stopRecognition() {
        if (this.recognition) {
            this.recognition.stop();
            this.resetUI();
            
            // If there's text in the input, send it automatically
            if (userInput.value.trim()) {
                sendMessage();
            }
        }
    }
    
    resetUI() {
        this.isListening = false;
        this.audioBtn.innerHTML = '<img src="/static/icons/icons8-square-play-button-50.png" alt="Start Recording" width="24" height="24">';
        this.audioBtn.classList.remove('listening');
        this.visualizer.style.display = 'none';
        userInput.placeholder = "Type a message...";
    }
    
    async useWhisperAPI() {
        try {
            if (this.isListening) {
                // Stop recording
                this.stopWhisperRecording();
                return;
            }
            
            // Start recording
            this.isListening = true;
            this.audioBtn.innerHTML = '<img src="/static/icons/icons8-stop-circled-50.png" alt="Stop Recording" width="24" height="24">';
            this.audioBtn.classList.add('listening');
            userInput.placeholder = "Listening...";
            
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            const audioChunks = [];
            
            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });
            
            mediaRecorder.addEventListener("stop", async () => {
                const audioBlob = new Blob(audioChunks);
                const formData = new FormData();
                formData.append("audio", audioBlob, "recording.webm");
                
                try {
                    const response = await fetch('http://127.0.0.1:8000/transcribe', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) throw new Error('Transcription failed');
                    
                    const data = await response.json();
                    userInput.value = data.text;
                    
                    // Auto-send if there's text
                    if (data.text.trim()) {
                        sendMessage();
                    }
                } catch (error) {
                    console.error('Whisper API error:', error);
                    appendMessage('bot', `Error in transcription: ${error.message}`);
                }
            });
            
            mediaRecorder.start();
            this.mediaRecorder = mediaRecorder;
            
        } catch (error) {
            console.error('Microphone access error:', error);
            appendMessage('bot', `Error accessing microphone: ${error.message}`);
            this.resetUI();
        }
    }
    
    stopWhisperRecording() {
        if (this.mediaRecorder && this.isListening) {
            this.mediaRecorder.stop();
            this.resetUI();
        }
    }
}

// Initialize audio recognition when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AudioRecognition();
});