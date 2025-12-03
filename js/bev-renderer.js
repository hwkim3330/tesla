/**
 * Bird's Eye View (BEV) Renderer
 * Tesla-style top-down visualization of detected objects and planned path
 */

class BEVRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.resize();

        // BEV parameters
        this.scale = 3; // meters per pixel
        this.egoPosition = { x: 0.5, y: 0.8 }; // Ego vehicle position in canvas

        // Scene objects
        this.vehicles = [];
        this.laneLines = [];
        this.plannedPath = [];

        // Animation
        this.time = 0;
        this.animate();

        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = rect.width * dpr;
        this.canvas.height = (rect.height - 30) * dpr;
        this.ctx.scale(dpr, dpr);
        this.width = rect.width;
        this.height = rect.height - 30;
    }

    update(autopilotState) {
        this.time += 0.016;

        const steering = autopilotState?.steering || 0;
        const detections = autopilotState?.detections || [];

        // Update vehicles from detections
        this.vehicles = detections
            .filter(d => d.type === 'car')
            .map(d => ({
                x: (d.position?.x || 0.5) * this.width,
                y: this.height * 0.3 - (d.distance || 30) * 2,
                width: 20,
                height: 35,
                speed: d.speed || 60
            }));

        // Generate lane lines
        this.generateLaneLines(steering);

        // Generate planned path
        this.generatePlannedPath(steering);
    }

    generateLaneLines(steering) {
        this.laneLines = [];

        const egoX = this.width * this.egoPosition.x;
        const laneWidth = 60; // pixels

        // Generate curved lane lines based on steering
        for (let lane = -1; lane <= 1; lane++) {
            const points = [];
            for (let y = this.height; y > 0; y -= 10) {
                const progress = 1 - y / this.height;
                const curvature = steering * progress * progress * 3;
                const x = egoX + lane * laneWidth + curvature;
                points.push({ x, y });
            }
            this.laneLines.push({
                points,
                type: lane === 0 ? 'dashed' : 'solid',
                color: lane === 0 ? '#f5a623' : '#3e6ae1'
            });
        }
    }

    generatePlannedPath(steering) {
        this.plannedPath = [];
        const egoX = this.width * this.egoPosition.x;
        const egoY = this.height * this.egoPosition.y;

        for (let i = 0; i < 50; i++) {
            const progress = i / 50;
            const y = egoY - progress * this.height * 0.7;
            const curvature = steering * progress * progress * 2;
            const x = egoX + curvature;

            this.plannedPath.push({ x, y, alpha: 1 - progress * 0.7 });
        }
    }

    draw() {
        // Clear with gradient background
        const gradient = this.ctx.createLinearGradient(0, 0, 0, this.height);
        gradient.addColorStop(0, '#0a0a12');
        gradient.addColorStop(1, '#1a1a2e');
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Draw grid
        this.drawGrid();

        // Draw lane lines
        this.drawLaneLines();

        // Draw planned path
        this.drawPlannedPath();

        // Draw detected vehicles
        this.drawVehicles();

        // Draw ego vehicle
        this.drawEgoVehicle();

        // Draw distance markers
        this.drawDistanceMarkers();

        // Draw radar arcs
        this.drawRadarArcs();
    }

    drawGrid() {
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        this.ctx.lineWidth = 1;

        // Vertical lines
        for (let x = 0; x < this.width; x += 30) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.height);
            this.ctx.stroke();
        }

        // Horizontal lines
        for (let y = 0; y < this.height; y += 30) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.width, y);
            this.ctx.stroke();
        }
    }

    drawLaneLines() {
        this.laneLines.forEach(lane => {
            this.ctx.strokeStyle = lane.color;
            this.ctx.lineWidth = 2;

            if (lane.type === 'dashed') {
                this.ctx.setLineDash([15, 15]);
            } else {
                this.ctx.setLineDash([]);
            }

            this.ctx.beginPath();
            lane.points.forEach((point, i) => {
                const alpha = 1 - (1 - point.y / this.height) * 0.8;
                if (i === 0) {
                    this.ctx.moveTo(point.x, point.y);
                } else {
                    this.ctx.lineTo(point.x, point.y);
                }
            });
            this.ctx.stroke();
            this.ctx.setLineDash([]);
        });
    }

    drawPlannedPath() {
        if (this.plannedPath.length < 2) return;

        // Draw path as gradient line
        this.ctx.lineWidth = 4;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';

        // Create gradient
        const gradient = this.ctx.createLinearGradient(
            this.plannedPath[0].x, this.plannedPath[0].y,
            this.plannedPath[this.plannedPath.length - 1].x,
            this.plannedPath[this.plannedPath.length - 1].y
        );
        gradient.addColorStop(0, 'rgba(23, 176, 107, 0.8)');
        gradient.addColorStop(1, 'rgba(23, 176, 107, 0.1)');

        this.ctx.strokeStyle = gradient;
        this.ctx.beginPath();
        this.plannedPath.forEach((point, i) => {
            if (i === 0) {
                this.ctx.moveTo(point.x, point.y);
            } else {
                this.ctx.lineTo(point.x, point.y);
            }
        });
        this.ctx.stroke();

        // Draw path points
        this.plannedPath.forEach((point, i) => {
            if (i % 5 !== 0) return;
            this.ctx.fillStyle = `rgba(23, 176, 107, ${point.alpha})`;
            this.ctx.beginPath();
            this.ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }

    drawVehicles() {
        this.vehicles.forEach(vehicle => {
            // Vehicle shadow
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            this.ctx.fillRect(
                vehicle.x - vehicle.width / 2 + 3,
                vehicle.y - vehicle.height / 2 + 3,
                vehicle.width,
                vehicle.height
            );

            // Vehicle body
            const gradient = this.ctx.createLinearGradient(
                vehicle.x, vehicle.y - vehicle.height / 2,
                vehicle.x, vehicle.y + vehicle.height / 2
            );
            gradient.addColorStop(0, '#4a90d9');
            gradient.addColorStop(1, '#2563eb');

            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.roundRect(
                vehicle.x - vehicle.width / 2,
                vehicle.y - vehicle.height / 2,
                vehicle.width,
                vehicle.height,
                4
            );
            this.ctx.fill();

            // Windshield
            this.ctx.fillStyle = '#1a365d';
            this.ctx.fillRect(
                vehicle.x - vehicle.width / 2 + 3,
                vehicle.y - vehicle.height / 2 + 4,
                vehicle.width - 6,
                10
            );

            // Taillights
            this.ctx.fillStyle = '#ef4444';
            this.ctx.fillRect(vehicle.x - vehicle.width / 2 + 2, vehicle.y + vehicle.height / 2 - 4, 4, 3);
            this.ctx.fillRect(vehicle.x + vehicle.width / 2 - 6, vehicle.y + vehicle.height / 2 - 4, 4, 3);

            // Distance label
            const distance = Math.round((this.height * this.egoPosition.y - vehicle.y) / 2);
            this.ctx.fillStyle = '#fff';
            this.ctx.font = '10px sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(`${distance}m`, vehicle.x, vehicle.y + vehicle.height / 2 + 15);
        });
    }

    drawEgoVehicle() {
        const x = this.width * this.egoPosition.x;
        const y = this.height * this.egoPosition.y;
        const width = 24;
        const height = 40;

        // Glow effect
        const glow = this.ctx.createRadialGradient(x, y, 0, x, y, 50);
        glow.addColorStop(0, 'rgba(23, 176, 107, 0.3)');
        glow.addColorStop(1, 'transparent');
        this.ctx.fillStyle = glow;
        this.ctx.beginPath();
        this.ctx.arc(x, y, 50, 0, Math.PI * 2);
        this.ctx.fill();

        // Vehicle body
        const gradient = this.ctx.createLinearGradient(x, y - height / 2, x, y + height / 2);
        gradient.addColorStop(0, '#e82127');
        gradient.addColorStop(1, '#991b1b');

        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.roundRect(x - width / 2, y - height / 2, width, height, 5);
        this.ctx.fill();

        // Roof
        this.ctx.fillStyle = '#7f1d1d';
        this.ctx.fillRect(x - width / 2 + 3, y - height / 2 + 8, width - 6, 14);

        // Windshield
        this.ctx.fillStyle = '#1f2937';
        this.ctx.fillRect(x - width / 2 + 3, y - height / 2 + 4, width - 6, 6);

        // Tesla logo indicator
        this.ctx.fillStyle = '#fff';
        this.ctx.font = 'bold 8px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('T', x, y + 3);
    }

    drawDistanceMarkers() {
        const egoY = this.height * this.egoPosition.y;
        const distances = [10, 20, 30, 50];

        this.ctx.font = '9px sans-serif';
        this.ctx.fillStyle = '#666';
        this.ctx.textAlign = 'right';

        distances.forEach(d => {
            const y = egoY - d * 4;
            if (y < 0) return;

            // Dashed line
            this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            this.ctx.setLineDash([5, 5]);
            this.ctx.beginPath();
            this.ctx.moveTo(10, y);
            this.ctx.lineTo(this.width - 10, y);
            this.ctx.stroke();
            this.ctx.setLineDash([]);

            // Label
            this.ctx.fillText(`${d}m`, this.width - 5, y + 3);
        });
    }

    drawRadarArcs() {
        const x = this.width * this.egoPosition.x;
        const y = this.height * this.egoPosition.y;

        // Pulsing radar effect
        const pulse = (Math.sin(this.time * 3) + 1) / 2;

        for (let i = 1; i <= 3; i++) {
            const radius = i * 40;
            const alpha = (0.1 - i * 0.02) * (1 + pulse * 0.5);

            this.ctx.strokeStyle = `rgba(62, 106, 225, ${alpha})`;
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius, Math.PI * 1.2, Math.PI * 1.8);
            this.ctx.stroke();
        }

        // Scanning line
        const scanAngle = Math.PI * 1.2 + (this.time % 2) / 2 * Math.PI * 0.6;
        this.ctx.strokeStyle = 'rgba(23, 176, 107, 0.5)';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
        this.ctx.lineTo(
            x + Math.cos(scanAngle) * 120,
            y + Math.sin(scanAngle) * 120
        );
        this.ctx.stroke();
    }

    roundRect(x, y, w, h, r) {
        this.ctx.moveTo(x + r, y);
        this.ctx.lineTo(x + w - r, y);
        this.ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        this.ctx.lineTo(x + w, y + h - r);
        this.ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        this.ctx.lineTo(x + r, y + h);
        this.ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        this.ctx.lineTo(x, y + r);
        this.ctx.quadraticCurveTo(x, y, x + r, y);
    }

    animate() {
        const state = window.autopilot?.getState();
        this.update(state);
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.bevRenderer = new BEVRenderer('bev-canvas');
});
