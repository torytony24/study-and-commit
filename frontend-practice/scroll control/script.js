const imageList = [];
for (let i = 1; i <= 20; i++) {
  imageList.push(`https://picsum.photos/300/300?random=${i}`);
}


const gallery = document.querySelector('#gallery');
const sentinel = document.querySelector('#sentinel');

let currentIndex = 0;
const count = 12;

function loadImages() {

  const fragment = document.createDocumentFragment();

  for (let i = 0; i < count; i++) {
      const imageIdx = currentIndex % imageList.length; 
      
      const img = document.createElement('img');
      img.src = imageList[imageIdx]; 
      img.classList.add('gallery-item');
      img.loading = "lazy"; 
      
      fragment.appendChild(img);
      currentIndex++;
    }

  gallery.appendChild(fragment);
}

const observer = new IntersectionObserver((entries) => {
  if (entries[0].isIntersecting) {
    loadImages();
  }
}, { threshold: 0.1 });


observer.observe(sentinel);


const videoElement = document.querySelector('.input_video');

const canvasElement = document.querySelector('.output_canvas');
const canvasCtx = canvasElement.getContext('2d');

let targetSpeed = 0;
let currentSpeed = 0;
const lerpFactor = 0.08;

function lerp(start, end, factor) {
    return start + (end - start) * factor;
}

const hands = new Hands({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1, 
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

hands.onResults((results) => {
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const hand = results.multiHandLandmarks[0];
      
      const thumb = hand[4];
      const index = hand[8];

      const distance = Math.sqrt(
          Math.pow(thumb.x - index.x, 2) + 
          Math.pow(thumb.y - index.y, 2)
      );

    if (distance > 0.05) {
      targetSpeed = distance * 200;
    } 
    else {
      targetSpeed = 0;
    }

    const thumbX = thumb.x * canvasElement.width;
    const thumbY = thumb.y * canvasElement.height;
    const indexX = index.x * canvasElement.width;
    const indexY = index.y * canvasElement.height;

    canvasCtx.strokeStyle = '#00FF00';
    canvasCtx.lineWidth = 4;
    canvasCtx.lineCap = 'round';

    canvasCtx.beginPath();
    canvasCtx.moveTo(thumbX, thumbY);
    canvasCtx.lineTo(indexX, indexY);
    canvasCtx.stroke();

    canvasCtx.fillStyle = 'red';
    canvasCtx.beginPath(); canvasCtx.arc(thumbX, thumbY, 6, 0, 2 * Math.PI); canvasCtx.fill();
    canvasCtx.beginPath(); canvasCtx.arc(indexX, indexY, 6, 0, 2 * Math.PI); canvasCtx.fill();

  } 

  else {
    targetSpeed = 0;
  }


});

function smoothScroll() {
    currentSpeed = lerp(currentSpeed, targetSpeed, lerpFactor);
    
    if (Math.abs(currentSpeed) > 0.1) {
        window.scrollBy(0, currentSpeed);
    }
    
    requestAnimationFrame(smoothScroll);
}

smoothScroll();

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({image: videoElement});
    },
    width: 640,
    height: 480
});

camera.start();