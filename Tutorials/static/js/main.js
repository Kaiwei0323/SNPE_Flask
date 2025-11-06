// JavaScript to toggle RTSP and Webcam input visibility
function toggleInputFields() {
    var videoSource = document.getElementById("video_source").value;
    var rtspInput = document.getElementById("rtsp-url");
    var webcamIdxInput = document.getElementById("webcam-idx");
    
    if (videoSource === "RTSP") {
        rtspInput.style.display = "block"; // Show RTSP URL input
        webcamIdxInput.style.display = "none"; // Hide Webcam index input
    } else if (videoSource === "/dev/video") {
        rtspInput.style.display = "none"; // Hide RTSP URL input
        webcamIdxInput.style.display = "block"; // Show Webcam index input
    } else {
        rtspInput.style.display = "none"; // Hide RTSP URL input
        webcamIdxInput.style.display = "none"; // Hide Webcam index input
    }
}

// Trigger the function on page load to check the default value
window.onload = toggleInputFields;

