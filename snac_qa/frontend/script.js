
// Create audio visualizer bars
const visualizer = document.getElementById('audioVisualizer');
const BAR_COUNT = 40;
const bars = [];

for (let i = 0; i < BAR_COUNT; i++) {
    const bar = document.createElement('div');
    bar.className = 'visualizer-bar';
    visualizer.appendChild(bar);
    bars.push(bar);
}

let currentAudio = null;
const playButton = document.getElementById('playButton');
const downloadButton = document.getElementById('downloadButton');
const userText = document.getElementById('userText');

// Get temperature, duration, and emotion elements

const emotionSelect = document.getElementById('emotion');

// Update temperature display


// Update duration display


// Close the demo popup

function updateVisualizerBars(audio, bars) {
    if (!audio.paused) {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaElementSource(audio);
        const analyser = audioContext.createAnalyser();
        source.connect(analyser);
        analyser.connect(audioContext.destination);
        analyser.fftSize = 64;

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        function renderFrame() {
            if (!audio.paused) {
                analyser.getByteFrequencyData(dataArray);

                bars.forEach((bar, index) => {
                    let bar_strength = Math.abs((BAR_COUNT / 2) - index);
                    const barHeight = (dataArray[bar_strength] / 256) * 80;
                    bar.style.height = `${Math.max(barHeight, 4)}px`;
                    bar.classList.add('active');
                });

                requestAnimationFrame(renderFrame);
            } else {
                bars.forEach(bar => {
                    bar.classList.remove('active');
                    bar.style.height = '4px';
                });
            }
        }
        renderFrame();
    }
}

document.getElementById('textInput').addEventListener('keydown', function (event) {
    if (event.key === 'Enter') {
        const prompt = this.value.trim();
        if (prompt !== "") {


            // userText.textContent = `MY PROMPT: <${selectedValue}> ${prompt} </${selectedValue}>`;
            // document.getElementById("outputsGen").style.display = "block";
            // document.getElementById("numTokensSpan").textContent = document.getElementById("duration").value;
            // document.getElementById("timeSpan").textContent = Math.floor(document.getElementById("duration").value / 75);

// 
            // this.value = "";
            sendPostRequest(prompt);
        }
    }
});

"https://k66bdjupomoi7f-8080.proxy.runpod.net/inference";

function sendPostRequest(prompt) {
    const url = "https://mb2u57saut1wll-8080.proxy.runpod.net/inference-text";

    document.getElementById("textInput").classList.add("shimmer");


    const payload = {
        "prompt": prompt,
        "max_length": 150,
    };
    document.getElementById("output").style.display = "none";

    document.getElementById("outputSub").style.display = "none";
    document.getElementById("loaderContainer").style.display = "flex";
    fetch(url, {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById("output").style.display = "block";
            document.getElementById("loaderContainer").style.display = "none";

            document.getElementById("textInput").classList.remove("shimmer");
            console.log(data);
            document.getElementById("outputsGen").style.display = "none";
            if (data.numpy_audio && data.numpy_audio.length > 0 && data.numpy_audio[0].length > 0) {
                const audioUrl = convertFloat32ToWav(data.numpy_audio[0][0]);
                setupPlayButton(audioUrl);
                document.getElementById("textResponse").textContent = data.text_response;
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function sendAudioPostRequest(sample_list) {
    const url = "https://mb2u57saut1wll-8080.proxy.runpod.net/inference";

    document.getElementById("textInput").classList.add("shimmer");

    const payload = {
        "samples_list": sample_list,
        "max_length": 150,
    };
    document.getElementById("output").style.display = "none";

    document.getElementById("outputSub").style.display = "none";
    document.getElementById("loaderContainer").style.display = "flex";
    fetch(url, {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById("output").style.display = "block";
            document.getElementById("loaderContainer").style.display = "none";

            document.getElementById("textInput").classList.remove("shimmer");
            console.log(data);
            document.getElementById("outputsGen").style.display = "none";
            if (data.numpy_audio && data.numpy_audio.length > 0 && data.numpy_audio[0].length > 0) {
                const audioUrl = convertFloat32ToWav(data.numpy_audio[0][0]);
                setupPlayButton(audioUrl);
                document.getElementById("textResponse").textContent = data.text_response;
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

function setupPlayButton(audioUrl) {
    playButton.style.display = 'block';

    playButton.onclick = () => playAudio(audioUrl);
}

function playAudio(audioUrl) {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
    }

    const audioElement = new Audio(audioUrl);
    currentAudio = audioElement;
    audioElement.play();

    audioElement.addEventListener('play', () => {
        updateVisualizerBars(audioElement, bars);
    });

    audioElement.addEventListener('ended', () => {
        bars.forEach(bar => {
            bar.classList.remove('active');
            bar.style.height = '4px';
        });
    });
}

function convertFloat32ToWav(audioData) {
    const sampleRate = 24000;
    const numOfChannels = 1;
    const byteRate = sampleRate * numOfChannels * 2;
    const blockAlign = numOfChannels * 2;
    const buffer = new ArrayBuffer(44 + audioData.length * 2);
    const view = new DataView(buffer);

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + audioData.length * 2, true);
    writeString(view, 8, 'WAVE');

    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);

    writeString(view, 36, 'data');
    view.setUint32(40, audioData.length * 2, true);

    let offset = 44;
    for (let i = 0; i < audioData.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, audioData[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return URL.createObjectURL(new Blob([view], { type: 'audio/wav' }));
}

// Model Card Popup Functionality
const openModelCardButton = document.getElementById('openModelCard');
const openModelCardMobileButton = document.getElementById('openModelCardMobile');
const modelCard = document.getElementById('modelCard');
const closeModelCardButton = document.getElementById('closeModelCard');

// Function to open the model card
openModelCardButton.addEventListener('click', () => {
    modelCard.style.display = 'flex';
});

openModelCardMobileButton.addEventListener('click', () => {
    modelCard.style.display = 'flex';
});

// Function to close the model card
closeModelCardButton.addEventListener('click', () => {
    modelCard.style.display = 'none';
});

// Close the modal when clicking outside the content
window.addEventListener('click', (event) => {
    if (event.target === modelCard) {
        modelCard.style.display = 'none';
    }
});

// Removed the automatic opening of the model card on page load
/*
window.addEventListener('DOMContentLoaded', () => {
    modelCard.style.display = 'flex';
});
*/

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let analyserNode;
let animationFrame;
let waveformData = []; // New array to store all waveform data

const recordButton = document.getElementById('recordButton');
const micIcon = document.getElementById('micIcon');
const stopIcon = document.getElementById('stopIcon');
const status = document.getElementById('status');

recordButton.addEventListener('click', () => {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
});

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Set up audio context and analyzer
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 2048;
        source.connect(analyserNode);

        // Reset waveform data array
        waveformData = [];

        // Set up media recorder
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
          
            // Now you have raw PCM for each channel:
            const float32Array = audioBuffer.getChannelData(0);
            console.log("Total samples:", float32Array.length);
            const arrayfrom32 =  Array.from(float32Array);
            console.log(arrayfrom32)
            sendAudioPostRequest(arrayfrom32)
            // Send this PCM data wherever you’d like
          };
          

        // mediaRecorder.onstop = () => {
        //     const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        //     const audioUrl = URL.createObjectURL(audioBlob);

        //     // Print the total length of collected waveform data
        //     sendAudioPostRequest(waveformData);
        //     console.log('Total waveform samples collected:', waveformData.length);

        //     // Clean up
        //     stream.getTracks().forEach(track => track.stop());
        //     cancelAnimationFrame(animationFrame);
        // };

        // Start recording and visualization
        mediaRecorder.start();
        isRecording = true;
        updateUI(true);
        visualizeAudio();
    } catch (error) {
        console.error('Error accessing microphone:', error);
        status.textContent = 'Error accessing microphone';
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        updateUI(false);
    }
}

function visualizeAudio() {
    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
        animationFrame = requestAnimationFrame(draw);
        analyserNode.getByteTimeDomainData(dataArray);

        // Store the waveform data instead of logging it
        waveformData.push(...Array.from(dataArray));
    };

    draw();
}

function updateUI(recording) {
    recordButton.classList.toggle('recording', recording);
    micIcon.style.display = recording ? 'none' : 'block';
    stopIcon.style.display = recording ? 'block' : 'none';
}
