// DOM Elements
const chatbotBody = document.getElementById('chatbot-body');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const dropDownButton = document.getElementById('drop-down-button');
const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file-input');
const pdfUploadBtn = document.getElementById('pdf-upload-btn');
const pdfInput = document.getElementById('pdf-input');

// State variables
let isGenerating = false;
let abortController = null;
let isUserScrolledUp = false;
let currentFiles = [];
let isUploading = false;

// Check if user is at bottom of chat
function isUserAtBottom() {
    const threshold = 10;
    return (
        chatbotBody.scrollTop + chatbotBody.clientHeight >=
        chatbotBody.scrollHeight - threshold
    );
}

// Toggle drop-down button visibility
function toggleDropDownButton() {
    if (isUserAtBottom()) {
        dropDownButton.classList.remove('visible');
        isUserScrolledUp = false;
    } else {
        dropDownButton.classList.add('visible');
        isUserScrolledUp = true;
    }
}

// Scroll to bottom when drop-down button is clicked
dropDownButton.addEventListener('click', () => {
    chatbotBody.scrollTop = chatbotBody.scrollHeight;
    isUserScrolledUp = false;
});

// Clear conversation memory on page load
async function clearMemory() {
    try {
        const response = await fetch('http://127.0.0.1:8000/clear-memory', {
            method: 'POST',
        });

        if (!response.ok) {
            throw new Error('Failed to clear memory');
        }

        const data = await response.json();
        console.log(data.status);
    } catch (error) {
        console.error('Error clearing memory:', error);
    }
}

// Parse markdown-like formatting
function parseMarkdown(text) {
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    text = text.replace(/\n/g, '<br>');
    text = text.replace(/```([\s\S]*?)```/g, '<pre class="code-block">$1</pre>');
    return text;
}

// Create copy button for code blocks
function createCopyButton(element, textToCopy) {
    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-btn';
    copyBtn.innerHTML = 'Copy';
    copyBtn.title = 'Copy to clipboard';
    
    copyBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        navigator.clipboard.writeText(textToCopy).then(() => {
            copyBtn.innerHTML = 'Copied!';
            setTimeout(() => {
                copyBtn.innerHTML = 'Copy';
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
        });
    });
    
    element.appendChild(copyBtn);
}

// Append message to chat
function appendMessage(sender, message, isTypingEffect = false) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    messageElement.classList.add(sender === 'user' ? 'user' : 'bot');

    const symbol = document.createElement('span');
    symbol.classList.add('symbol');
    symbol.innerHTML = sender === 'user' ? '&#128100;' : '&#129302;';
    messageElement.appendChild(symbol);

    const textContainer = document.createElement('div');
    textContainer.style.position = 'relative';
    textContainer.style.width = '100%';
    messageElement.appendChild(textContainer);

    const text = document.createElement('span');
    textContainer.appendChild(text);

    chatbotBody.appendChild(messageElement);

    if (!isUserScrolledUp) {
        chatbotBody.scrollTop = chatbotBody.scrollHeight;
    }

    if (!isTypingEffect) {
        text.innerHTML = parseMarkdown(message);
        
        if (sender === 'bot') {
            createCopyButton(textContainer, message);
        }
    }

    return { textElement: text, container: textContainer };
}

// Show loading indicator
function showLoadingIndicator() {
    const loadingMessage = document.createElement('div');
    loadingMessage.classList.add('message', 'bot');
    loadingMessage.innerHTML = `
        <span class="symbol">&#129302;</span>
        <div class="dot-loading">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    chatbotBody.appendChild(loadingMessage);

    if (!isUserScrolledUp) {
        chatbotBody.scrollTop = chatbotBody.scrollHeight;
    }

    return loadingMessage;
}

// Hide loading indicator
function hideLoadingIndicator(loadingMessage) {
    chatbotBody.removeChild(loadingMessage);
}

// Type message with typing effect
function typeMessage(element, message, delay = 10) {
    let index = 0;
    let fullMessage = '';
    return new Promise((resolve) => {
        function type() {
            if (index < message.length && isGenerating) {
                fullMessage += message[index];
                element.textElement.innerHTML = parseMarkdown(fullMessage);
                index++;

                if (!isUserScrolledUp) {
                    chatbotBody.scrollTop = chatbotBody.scrollHeight;
                }

                setTimeout(type, delay);
            } else {
                if (isGenerating) {
                    createCopyButton(element.container, message);
                }
                resolve();
            }
        }
        type();
    });
}

// Stop message generation
function stopGeneration() {
    isGenerating = false;
    if (abortController) {
        abortController.abort();
    }
    sendBtn.textContent = "Send";
    sendBtn.classList.remove("stop");
}

// Show file preview
function showFilePreview(file, url, type) {
    const previewContainer = document.createElement('div');
    previewContainer.className = 'file-preview-container';
    previewContainer.dataset.fileId = file.id;
    
    if (type === 'image') {
        const img = document.createElement('img');
        img.src = url;
        img.className = 'uploaded-file-preview';
        img.alt = 'Uploaded image preview';
        previewContainer.appendChild(img);
    } else {
        const icon = document.createElement('i');
        icon.className = 'fas fa-file-pdf';
        previewContainer.appendChild(icon);
        
        const nameSpan = document.createElement('span');
        nameSpan.textContent = file.name;
        previewContainer.appendChild(nameSpan);
    }
    
    const removeBtn = document.createElement('button');
    removeBtn.innerHTML = '×';
    removeBtn.className = 'remove-file-btn';
    removeBtn.title = 'Remove file';
    removeBtn.onclick = (e) => {
        e.stopPropagation();
        previewContainer.remove();
        currentFiles = currentFiles.filter(f => f.id !== file.id);
        // Re-enable upload button when file is removed
        uploadBtn.disabled = false;
        uploadBtn.classList.remove('disabled');
    };
    
    previewContainer.appendChild(removeBtn);
    
    const previewsContainer = document.querySelector('.previews-container') || createPreviewsContainer();
    previewsContainer.appendChild(previewContainer);
}


function createPreviewsContainer() {
    const container = document.createElement('div');
    container.className = 'previews-container';
    const footer = document.querySelector('.chatbot-footer');
    footer.parentNode.insertBefore(container, footer);
    return container;
}

// Handle file upload
async function handleFileUpload(file, isPdf = false) {
    if (isUploading || currentFiles.length > 0) {
        return; // Simply return without doing anything
    }

    isUploading = true;
    const uploadBtnIcon = uploadBtn.querySelector('i');
    const originalClass = uploadBtnIcon.className;
    
    try {
        // Show loading state on the upload button
        uploadBtnIcon.className = 'fas fa-spinner fa-spin';
        uploadBtn.disabled = true;
        uploadBtn.classList.add('disabled');
        
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://127.0.0.1:8000/upload-file', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Failed to upload ${isPdf ? 'PDF' : 'image'}`);
        }

        const data = await response.json();
        
        const fileObj = {
            id: `${isPdf ? 'pdf' : 'img'}-${Date.now()}`,
            type: data.file_type,
            filename: data.filename,
            url: data.file_url,
            file: file
        };
        currentFiles.push(fileObj);

        showFilePreview(fileObj, data.file_url, data.file_type);
        
        // Notify user of successful upload
        appendMessage('user', `[Uploaded ${data.file_type}: ${file.name}]`);
        
    } catch (error) {
        appendMessage('bot', `Error: ${error.message}`);
        console.error('Upload error:', error);
    } finally {
        // Restore upload button state
        uploadBtnIcon.className = originalClass;
        uploadBtn.disabled = currentFiles.length > 0; // Keep disabled if we have a file
        uploadBtn.classList.toggle('disabled', currentFiles.length > 0);
        isUploading = false;
        fileInput.value = '';
    }
}

// Send message to server
async function sendMessage() {
    // Prevent sending if currently uploading files
    if (isUploading) {
        appendMessage('bot', 'Please wait until file upload completes');
        return;
    }

    const message = userInput.value.trim();
    if (!message && currentFiles.length === 0) return;

    // Clear UI elements
    userInput.value = '';
    const previewsContainer = document.querySelector('.previews-container');
    if (previewsContainer) previewsContainer.remove();

    // Prepare files data for API
    const filesData = currentFiles.map(file => ({
        type: file.type,
        filename: file.filename
    }));
    currentFiles = []; // Clear current files

    // Display user message if exists
    if (message) appendMessage('user', message);

    // Set up loading state
    const loadingMessage = showLoadingIndicator();
    sendBtn.textContent = "Stop";
    sendBtn.classList.add("stop");
    isGenerating = true;
    abortController = new AbortController();

    try {
        // Determine if this is an extraction request
        const isExtraction = /extract|full text/i.test(message);
        
        // API request
        const response = await fetch('http://127.0.0.1:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                files: filesData,
                action: isExtraction ? "extract" : "analyze"
            }),
            signal: abortController.signal
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Server response error');
        }

        const data = await response.json();
        hideLoadingIndicator(loadingMessage);

        // Handle large documents
        if (data.full_text_length > 50000) {
            const warning = appendMessage('bot', 
                `Document truncated (${data.full_text_length} chars). Showing first 50000 chars.`);
            createCopyButton(warning.container, data.response);
            
            const continueBtn = document.createElement('button');
            continueBtn.className = 'continue-btn';
            continueBtn.textContent = 'Show More';
            continueBtn.onclick = () => fetchFullDocument(filesData[0].filename);
            warning.container.appendChild(continueBtn);
        }

        // Stream the response with typing effect
        const botMessage = appendMessage('bot', '', true);
        await typeMessage(botMessage, data.response);

    } catch (error) {
        if (error.name !== 'AbortError') {
            hideLoadingIndicator(loadingMessage);
            appendMessage('bot', `Error: ${error.message}`)
                .container.append(createCopyButtonElement(error.message));
            console.error('Chat error:', error);
        }
    } finally {
        // Reset all states
        isGenerating = false;
        abortController = null;
        sendBtn.textContent = "Send";
        sendBtn.classList.remove("stop", "uploading");
        sendBtn.disabled = false;
    }
}

// Helper function to create copy button element
function createCopyButtonElement(text) {
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.innerHTML = 'Copy';
    btn.onclick = () => navigator.clipboard.writeText(text);
    return btn;
}

async function fetchFullDocument(filename) {
    const loadingMessage = showLoadingIndicator();
    try {
        const response = await fetch('http://127.0.0.1:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: "extract full document",
                files: [{ type: "pdf", filename }],
                action: "extracts"
            })
        });

        if (!response.ok) throw new Error('Failed to get full document');
        
        const data = await response.json();
        hideLoadingIndicator(loadingMessage);
        
        // Create a scrollable container for large documents
        const container = document.createElement('div');
        container.className = 'document-container';
        container.textContent = data.response;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';
        messageDiv.innerHTML = '<span class="symbol">🤖</span>';
        messageDiv.appendChild(container);
        
        chatbotBody.appendChild(messageDiv);
        chatbotBody.scrollTop = chatbotBody.scrollHeight;
        
    } catch (error) {
        hideLoadingIndicator(loadingMessage);
        appendMessage('bot', `Error loading full document: ${error.message}`);
    }
}

// Event Listeners
chatbotBody.addEventListener('scroll', toggleDropDownButton);
window.addEventListener('load', clearMemory);

sendBtn.addEventListener('click', () => {
    if (isGenerating) {
        stopGeneration();
    } else {
        sendMessage();
    }
});

userInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        // Only send if not currently generating and send button is not disabled
        if (!isGenerating && !isUploading) {
            sendMessage();
        }
    }
});

uploadBtn.addEventListener('click', () => {
    if (!isUploading && currentFiles.length === 0) {
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
});

pdfUploadBtn.addEventListener('click', () => {
    if (!isUploading) {
        pdfInput.click();
    }
});

pdfInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileUpload(file, true);
    }
});