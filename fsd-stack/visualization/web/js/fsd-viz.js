/**
 * Tesla FSD Developer Mode Visualization
 *
 * Real-time visualization of FSD neural network outputs.
 */

class FSDVisualization {
    constructor() {
        this.cameraCanvas = document.getElementById('camera-canvas');
        this.overlayCanvas = document.getElementById('overlay-canvas');
        this.bevCanvas = document.getElementById('bev-canvas');

        this.cameraCtx = this.cameraCanvas.getContext('2d');
        this.overlayCtx = this.overlayCanvas.getContext('2d');
        this.bevCtx = this.bevCanvas.getContext('2d');

        this.setupCanvases();
        this.startSimulation();
    }

    setupCanvases() {
        // Camera canvas
        const camRect = this.cameraCanvas.parentElement.getBoundingClientRect();
        this.cameraCanvas.width = camRect.width;
        this.cameraCanvas.height = camRect.height;
        this.overlayCanvas.width = camRect.width;
        this.overlayCanvas.height = camRect.height;

        // BEV canvas
        const bevRect = this.bevCanvas.parentElement.getBoundingClientRect();
        this.bevCanvas.width = bevRect.width - 30;
        this.bevCanvas.height = bevRect.height - 45;

        // Handle resize
        window.addEventListener('resize', () => this.setupCanvases());
    }

    startSimulation() {
        this.frame = 0;
        this.detections = this.generateDetections();
        this.trajectory = this.generateTrajectory();

        this.animate();
    }

    generateDetections() {
        return [
            { type: 'car', x: 0.5, y: 0.5, w: 0.15, h: 0.2, dist: 15.3, conf: 0.92 },
            { type: 'car', x: 0.75, y: 0.55, w: 0.1, h: 0.12, dist: 25.8, conf: 0.88 },
            { type: 'pedestrian', x: 0.2, y: 0.6, w: 0.05, h: 0.15, dist: 8.2, conf: 0.95 },
            { type: 'traffic_light', x: 0.5, y: 0.15, w: 0.04, h: 0.1, dist: 45.0, state: 'green' },
        ];
    }

    generateTrajectory() {
        const points = [];
        for (let i = 0; i < 20; i++) {
            points.push({
                x: 0.5 + Math.sin(i * 0.1) * 0.02,
                y: 0.9 - i * 0.04
            });
        }
        return points;
    }

    animate() {
        this.frame++;

        // Update simulation
        this.updateSimulation();

        // Render
        this.renderCamera();
        this.renderOverlay();
        this.renderBEV();
        this.updateUI();

        requestAnimationFrame(() => this.animate());
    }

    updateSimulation() {
        // Simulate slight variations
        const steering = Math.sin(this.frame * 0.02) * 10;
        const throttle = 30 + Math.sin(this.frame * 0.05) * 10;
        const speed = 60 + Math.sin(this.frame * 0.03) * 8;

        // Update steering wheel rotation
        const wheel = document.getElementById('steering-wheel');
        wheel.style.transform = `rotate(${steering}deg)`;

        // Update displays
        document.getElementById('speed').textContent = Math.round(speed);
        document.getElementById('steering-val').textContent = `${steering.toFixed(1)}°`;
        document.getElementById('throttle-val').textContent = `${Math.round(throttle)}%`;

        // Update bars
        document.getElementById('steering-bar').style.width = `${50 + steering}%`;
        document.getElementById('throttle-bar').style.width = `${throttle}%`;
        document.getElementById('throttle-pedal').style.height = `${throttle}%`;

        // Update FPS
        document.getElementById('fps').textContent = `FPS: ${Math.round(60 + Math.random() * 5)}`;
        document.getElementById('inference-time').textContent =
            `Inference: ${(20 + Math.random() * 10).toFixed(1)}ms`;
    }

    renderCamera() {
        const ctx = this.cameraCtx;
        const w = this.cameraCanvas.width;
        const h = this.cameraCanvas.height;

        // Sky gradient
        const sky = ctx.createLinearGradient(0, 0, 0, h * 0.5);
        sky.addColorStop(0, '#1a2a3a');
        sky.addColorStop(1, '#2a3a4a');
        ctx.fillStyle = sky;
        ctx.fillRect(0, 0, w, h * 0.5);

        // Road
        ctx.fillStyle = '#333';
        ctx.beginPath();
        ctx.moveTo(0, h);
        ctx.lineTo(w, h);
        ctx.lineTo(w * 0.65, h * 0.4);
        ctx.lineTo(w * 0.35, h * 0.4);
        ctx.fill();

        // Lane markings
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 3;
        ctx.setLineDash([20, 20]);

        // Center lane
        ctx.beginPath();
        ctx.moveTo(w * 0.5, h);
        ctx.lineTo(w * 0.5, h * 0.4);
        ctx.stroke();

        // Left lane
        ctx.beginPath();
        ctx.moveTo(w * 0.3, h);
        ctx.lineTo(w * 0.4, h * 0.4);
        ctx.stroke();

        // Right lane
        ctx.beginPath();
        ctx.moveTo(w * 0.7, h);
        ctx.lineTo(w * 0.6, h * 0.4);
        ctx.stroke();

        ctx.setLineDash([]);
    }

    renderOverlay() {
        const ctx = this.overlayCtx;
        const w = this.overlayCanvas.width;
        const h = this.overlayCanvas.height;

        ctx.clearRect(0, 0, w, h);

        // Draw detections
        for (const det of this.detections) {
            const x = det.x * w - det.w * w / 2;
            const y = det.y * h - det.h * h / 2;
            const bw = det.w * w;
            const bh = det.h * h;

            // Color based on type
            let color = '#3e6ae1';
            if (det.type === 'pedestrian') color = '#f5a623';
            if (det.type === 'traffic_light') {
                color = det.state === 'green' ? '#17b06b' :
                        det.state === 'red' ? '#e82127' : '#f5a623';
            }

            // Draw box
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, bw, bh);

            // Draw label
            const label = `${det.type} ${det.dist.toFixed(1)}m`;
            ctx.fillStyle = color;
            ctx.fillRect(x, y - 20, ctx.measureText(label).width + 10, 18);
            ctx.fillStyle = '#000';
            ctx.font = '12px Arial';
            ctx.fillText(label, x + 5, y - 6);
        }

        // Draw trajectory
        ctx.strokeStyle = 'rgba(23, 176, 107, 0.6)';
        ctx.lineWidth = 4;
        ctx.beginPath();
        for (let i = 0; i < this.trajectory.length; i++) {
            const p = this.trajectory[i];
            const x = p.x * w;
            const y = p.y * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    renderBEV() {
        const ctx = this.bevCtx;
        const w = this.bevCanvas.width;
        const h = this.bevCanvas.height;

        // Background
        ctx.fillStyle = '#0d0d12';
        ctx.fillRect(0, 0, w, h);

        // Grid
        ctx.strokeStyle = '#222';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 10; i++) {
            ctx.beginPath();
            ctx.moveTo(i * w / 10, 0);
            ctx.lineTo(i * w / 10, h);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, i * h / 10);
            ctx.lineTo(w, i * h / 10);
            ctx.stroke();
        }

        // Ego vehicle
        const egoX = w / 2;
        const egoY = h * 0.75;

        ctx.fillStyle = '#3e6ae1';
        ctx.beginPath();
        ctx.moveTo(egoX, egoY - 15);
        ctx.lineTo(egoX - 8, egoY + 10);
        ctx.lineTo(egoX + 8, egoY + 10);
        ctx.fill();

        // Draw other vehicles
        const vehicles = [
            { x: 0.55, y: 0.4, type: 'car' },
            { x: 0.3, y: 0.55, type: 'car' },
            { x: 0.45, y: 0.6, type: 'pedestrian' },
        ];

        for (const v of vehicles) {
            const vx = v.x * w;
            const vy = v.y * h;

            if (v.type === 'car') {
                ctx.fillStyle = 'rgba(100, 200, 255, 0.8)';
                ctx.fillRect(vx - 6, vy - 10, 12, 20);
            } else {
                ctx.fillStyle = 'rgba(245, 166, 35, 0.8)';
                ctx.beginPath();
                ctx.arc(vx, vy, 5, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Draw planned path
        ctx.strokeStyle = '#17b06b';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(egoX, egoY);
        ctx.lineTo(egoX + Math.sin(this.frame * 0.02) * 10, egoY - 50);
        ctx.lineTo(egoX + Math.sin(this.frame * 0.02) * 15, egoY - 100);
        ctx.lineTo(egoX + Math.sin(this.frame * 0.02) * 20, egoY - 150);
        ctx.stroke();

        // Range circles
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        for (const r of [50, 100, 150]) {
            ctx.beginPath();
            ctx.arc(egoX, egoY, r, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    updateUI() {
        // Update detection list periodically
        if (this.frame % 30 === 0) {
            const list = document.getElementById('detection-list');
            // Could dynamically update detection list here
        }
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    new FSDVisualization();
});
