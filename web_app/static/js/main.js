document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const configSelect = document.getElementById("configId");
    const webcamVideo = document.getElementById("webcamVideo");
    const snapshotCanvas = document.getElementById("snapshotCanvas");
    const imagePreview = document.getElementById("imagePreview");
    const facePlaceholder = document.getElementById("facePlaceholder");
    const btnToggleWebcam = document.getElementById("btnToggleWebcam");
    const btnCapturePhoto = document.getElementById("btnCapturePhoto");
    const faceFileInput = document.getElementById("faceFileInput");
    const faceStatus = document.getElementById("faceStatus");

    const btnRecordMic = document.getElementById("btnRecordMic");
    const audioFileInput = document.getElementById("audioFileInput");
    const audioPlayback = document.getElementById("audioPlayback");
    const audioFileName = document.getElementById("audioFileName");
    const speechStatus = document.getElementById("speechStatus");

    const textInput = document.getElementById("textInput");
    const textStatus = document.getElementById("textStatus");
    const btnAnalyze = document.getElementById("btnAnalyze");

    const emotionIcon = document.getElementById("emotionIcon");
    const predictedEmotion = document.getElementById("predictedEmotion");
    const confidenceBar = document.getElementById("confidenceBar");
    const confidencePct = document.getElementById("confidencePct");
    const inferenceBadge = document.getElementById("inferenceBadge");

    const faceContribBar = document.getElementById("faceContribBar");
    const speechContribBar = document.getElementById("speechContribBar");
    const textContribBar = document.getElementById("textContribBar");
    const faceContribPct = document.getElementById("faceContribPct");
    const speechContribPct = document.getElementById("speechContribPct");
    const textContribPct = document.getElementById("textContribPct");

    const btnToggleBenchmarks = document.getElementById("btnToggleBenchmarks");
    const benchmarkContainer = document.getElementById("benchmarkTableContainer");
    const benchmarkTableBody = document.querySelector("#benchmarkTable tbody");

    let webcamStream = null;
    let capturedFaceBase64 = null;
    let recordedAudioBlob = null;
    let mediaRecorder = null;
    let audioChunks = [];

    const EMOTION_EMOJIS = {
        "Anger": "😠",
        "Disgust": "🤢",
        "Fear": "😨",
        "Happiness": "😄",
        "Sadness": "😢",
        "Surprise": "😲",
        "Neutral": "😐"
    };

    // 1. Webcam Handling
    btnToggleWebcam.addEventListener("click", async () => {
        if (webcamStream) {
            webcamStream.getTracks().forEach(t => t.stop());
            webcamStream = null;
            webcamVideo.style.display = "none";
            btnToggleWebcam.textContent = "Start Webcam";
            btnCapturePhoto.style.display = "none";
            if (!capturedFaceBase64 && !imagePreview.src) {
                facePlaceholder.style.display = "block";
            }
        } else {
            try {
                webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
                webcamVideo.srcObject = webcamStream;
                webcamVideo.style.display = "block";
                facePlaceholder.style.display = "none";
                imagePreview.style.display = "none";
                btnToggleWebcam.textContent = "Stop Webcam";
                btnCapturePhoto.style.display = "inline-flex";
            } catch (err) {
                alert("Camera access failed: " + err.message);
            }
        }
    });

    btnCapturePhoto.addEventListener("click", () => {
        if (!webcamStream) return;
        snapshotCanvas.width = webcamVideo.videoWidth || 320;
        snapshotCanvas.height = webcamVideo.videoHeight || 240;
        const ctx = snapshotCanvas.getContext("2d");
        ctx.drawImage(webcamVideo, 0, 0, snapshotCanvas.width, snapshotCanvas.height);
        capturedFaceBase64 = snapshotCanvas.toDataURL("image/jpeg");

        imagePreview.src = capturedFaceBase64;
        imagePreview.style.display = "block";
        webcamVideo.style.display = "none";
        btnCapturePhoto.style.display = "none";
        btnToggleWebcam.textContent = "Retake Webcam";

        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;

        faceStatus.textContent = "Ready (Captured)";
        faceStatus.classList.add("ready");
    });

    faceFileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (evt) => {
                imagePreview.src = evt.target.result;
                imagePreview.style.display = "block";
                facePlaceholder.style.display = "none";
                if (webcamVideo) webcamVideo.style.display = "none";
                capturedFaceBase64 = null; // file input takes precedence
                faceStatus.textContent = "Ready (File)";
                faceStatus.classList.add("ready");
            };
            reader.readAsDataURL(file);
        }
    });

    // 2. Microphone Recording
    btnRecordMic.addEventListener("click", async () => {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
            btnRecordMic.textContent = "Record Mic (3s)";
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = () => {
                recordedAudioBlob = new Blob(audioChunks, { type: "audio/wav" });
                audioPlayback.src = URL.createObjectURL(recordedAudioBlob);
                audioPlayback.style.display = "block";
                audioFileName.textContent = "Recorded audio clip ready (3s)";
                speechStatus.textContent = "Ready (Mic)";
                speechStatus.classList.add("ready");
                stream.getTracks().forEach(t => t.stop());
            };

            mediaRecorder.start();
            btnRecordMic.textContent = "Recording... (Click to stop)";
            speechStatus.textContent = "Recording...";

            // Auto-stop after 3 seconds
            setTimeout(() => {
                if (mediaRecorder && mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                    btnRecordMic.textContent = "Record Mic (3s)";
                }
            }, 3000);

        } catch (err) {
            alert("Microphone access failed: " + err.message);
        }
    });

    audioFileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            recordedAudioBlob = null;
            audioPlayback.src = URL.createObjectURL(file);
            audioPlayback.style.display = "block";
            audioFileName.textContent = file.name;
            speechStatus.textContent = "Ready (File)";
            speechStatus.classList.add("ready");
        }
    });

    // 3. Text Prompts
    textInput.addEventListener("input", () => {
        if (textInput.value.trim().length > 0) {
            textStatus.textContent = "Ready";
            textStatus.classList.add("ready");
        } else {
            textStatus.textContent = "Idle";
            textStatus.classList.remove("ready");
        }
    });

    document.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", () => {
            textInput.value = chip.getAttribute("data-text");
            textStatus.textContent = "Ready";
            textStatus.classList.add("ready");
        });
    });

    // 4. Run Analysis
    btnAnalyze.addEventListener("click", async () => {
        const formData = new FormData();
        formData.append("config_id", configSelect.value);
        formData.append("text", textInput.value);

        if (capturedFaceBase64) {
            formData.append("face_base64", capturedFaceBase64);
        } else if (faceFileInput.files.length > 0) {
            formData.append("face_file", faceFileInput.files[0]);
        }

        if (recordedAudioBlob) {
            formData.append("audio_file", recordedAudioBlob, "recording.wav");
        } else if (audioFileInput.files.length > 0) {
            formData.append("audio_file", audioFileInput.files[0]);
        }

        btnAnalyze.disabled = true;
        btnAnalyze.innerHTML = "<span>⏳</span> Computing Cross-Modal Fusion...";
        inferenceBadge.textContent = "Processing";

        try {
            const resp = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });
            const data = await resp.json();

            if (data.status === "success") {
                renderResults(data.data);
            } else {
                alert("Error during prediction: " + data.message);
            }
        } catch (err) {
            alert("Network / inference failure: " + err.message);
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.innerHTML = '<span class="btn-icon">⚡</span> Run Multimodal Fusion Inference';
            inferenceBadge.textContent = "Complete";
        }
    });

    function renderResults(result) {
        // Emotion & Confidence
        const emotion = result.predicted_emotion;
        predictedEmotion.textContent = emotion;
        emotionIcon.textContent = EMOTION_EMOJIS[emotion] || "🎭";

        const conf = result.confidence;
        confidencePct.textContent = `${conf}%`;
        confidenceBar.style.width = `${conf}%`;

        // Modality Contribution Weights
        const contrib = result.modality_contributions;
        faceContribPct.textContent = `${contrib.facial}%`;
        faceContribBar.style.width = `${contrib.facial}%`;

        speechContribPct.textContent = `${contrib.speech}%`;
        speechContribBar.style.width = `${contrib.speech}%`;

        textContribPct.textContent = `${contrib.text}%`;
        textContribBar.style.width = `${contrib.text}%`;

        // Softmax Class Probabilities
        const probs = result.class_probabilities;
        document.querySelectorAll(".class-prob-item").forEach(item => {
            const emo = item.getAttribute("data-emotion");
            if (probs[emo] !== undefined) {
                const val = (probs[emo] * 100).toFixed(1);
                item.querySelector(".cpi-val").textContent = `${val}%`;
                item.querySelector(".cpi-fill").style.width = `${val}%`;
                if (emo === emotion) {
                    item.style.borderColor = "var(--accent-blue)";
                } else {
                    item.style.borderColor = "rgba(255, 255, 255, 0.05)";
                }
            }
        });
    }

    // 5. Benchmarks Drawer
    btnToggleBenchmarks.addEventListener("click", async () => {
        if (benchmarkContainer.style.display === "none") {
            benchmarkContainer.style.display = "block";
            btnToggleBenchmarks.textContent = "▲ Hide Table I: Experimental Benchmarks";

            try {
                const resp = await fetch("/api/benchmarks");
                const res = await resp.json();
                if (res.status === "success") {
                    benchmarkTableBody.innerHTML = "";
                    const b = res.benchmarks;
                    Object.keys(b).sort().forEach(key => {
                        const row = b[key];
                        const tr = document.createElement("tr");
                        const acc = (row.accuracy * 100).toFixed(2);
                        const f1 = (row.macro_f1 * 100).toFixed(2);
                        const mods = (row.modalities || []).join(" + ");
                        tr.innerHTML = `
                            <td><strong>${key}</strong></td>
                            <td>${row.name}</td>
                            <td>${mods}</td>
                            <td>${acc}%</td>
                            <td>${f1}%</td>
                        `;
                        benchmarkTableBody.appendChild(tr);
                    });
                }
            } catch (err) {
                console.error("Could not fetch benchmarks", err);
            }
        } else {
            benchmarkContainer.style.display = "none";
            btnToggleBenchmarks.textContent = "📊 View Table I: Experimental Benchmarks (E1 - E7)";
        }
    });
});
