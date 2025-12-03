// Tesla FSD Visualization
// Simulates Tesla's Full Self-Driving developer visualization

class TeslaFSDVisualization {
    constructor() {
        this.canvas = document.getElementById('bev-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.objectsContainer = document.getElementById('objects-container');

        // Simulation state
        this.speed = 65;
        this.setSpeed = 70;
        this.time = 0;

        // Detected objects
        this.vehicles = [];
        this.pedestrians = [];
        this.trafficLights = [];
        this.speedSigns = [];

        // Lead vehicle
        this.leadVehicle = {
            distance: 45,
            speed: 62,
            ttc: 12.5
        };

        // Traffic light state
        this.currentLightState = 'green';
        this.lightTimer = 8;

        this.init();
    }

    init() {
        this.setupToggleButtons();
        this.generateInitialObjects();
        this.startSimulation();
    }

    setupToggleButtons() {
        document.querySelectorAll('.toggle-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
            });
        });
    }

    generateInitialObjects() {
        // Generate vehicles
        this.vehicles = [
            { id: 1, x: 50, y: 35, width: 80, height: 50, distance: 45, type: 'sedan' },
            { id: 2, x: 75, y: 25, width: 60, height: 35, distance: 72, type: 'suv' },
            { id: 3, x: 20, y: 40, width: 70, height: 45, distance: 35, type: 'truck' }
        ];

        // Generate pedestrians
        this.pedestrians = [
            { id: 1, x: 85, y: 50, width: 25, height: 50, distance: 28 }
        ];

        // Generate traffic lights
        this.trafficLights = [
            { id: 1, x: 45, y: 15, state: 'green', distance: 85, timer: 12 },
            { id: 2, x: 60, y: 12, state: 'red', distance: 120, timer: 8 }
        ];

        this.renderCameraObjects();
    }

    renderCameraObjects() {
        this.objectsContainer.innerHTML = '';

        // Render vehicles
        this.vehicles.forEach(v => {
            const div = document.createElement('div');
            div.className = 'detected-object vehicle';
            div.style.cssText = `
                left: ${v.x}%;
                bottom: ${v.y}%;
                width: ${v.width}px;
                height: ${v.height}px;
                transform: translateX(-50%);
            `;
            div.innerHTML = `
                <span class="object-label">VEHICLE</span>
                <span class="distance-tag">${v.distance}m</span>
            `;
            this.objectsContainer.appendChild(div);
        });

        // Render pedestrians
        this.pedestrians.forEach(p => {
            const div = document.createElement('div');
            div.className = 'detected-object pedestrian';
            div.style.cssText = `
                left: ${p.x}%;
                bottom: ${p.y}%;
                width: ${p.width}px;
                height: ${p.height}px;
                transform: translateX(-50%);
            `;
            div.innerHTML = `
                <span class="object-label">PEDESTRIAN</span>
                <span class="distance-tag">${p.distance}m</span>
            `;
            this.objectsContainer.appendChild(div);
        });

        // Render traffic lights
        this.trafficLights.forEach(tl => {
            const div = document.createElement('div');
            div.className = 'traffic-light';
            div.style.cssText = `
                left: ${tl.x}%;
                top: ${tl.y}%;
                transform: translateX(-50%);
            `;
            div.innerHTML = `
                <div class="traffic-light-box">
                    <div class="light red ${tl.state === 'red' ? 'active' : ''}"></div>
                    <div class="light yellow ${tl.state === 'yellow' ? 'active' : ''}"></div>
                    <div class="light green ${tl.state === 'green' ? 'active' : ''}"></div>
                </div>
                <div class="traffic-light-info">
                    ${tl.distance}m | ${tl.timer}s
                </div>
            `;
            this.objectsContainer.appendChild(div);
        });

        // Update counts
        document.getElementById('vehicle-count').textContent = this.vehicles.length;
        document.getElementById('pedestrian-count').textContent = this.pedestrians.length;
        document.getElementById('traffic-light-count').textContent = this.trafficLights.length;
    }

    drawBEV() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        // Clear
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, w, h);

        // Draw grid
        ctx.strokeStyle = 'rgba(62, 106, 225, 0.1)';
        ctx.lineWidth = 1;

        // Horizontal lines
        for (let y = 0; y < h; y += 20) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        // Vertical lines
        for (let x = 0; x < w; x += 20) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }

        // Draw road lanes
        ctx.strokeStyle = 'rgba(62, 106, 225, 0.5)';
        ctx.lineWidth = 2;
        ctx.setLineDash([10, 10]);

        // Left lane
        ctx.beginPath();
        ctx.moveTo(w * 0.25, 0);
        ctx.lineTo(w * 0.25, h);
        ctx.stroke();

        // Center lane
        ctx.beginPath();
        ctx.moveTo(w * 0.5, 0);
        ctx.lineTo(w * 0.5, h);
        ctx.stroke();

        // Right lane
        ctx.beginPath();
        ctx.moveTo(w * 0.75, 0);
        ctx.lineTo(w * 0.75, h);
        ctx.stroke();

        ctx.setLineDash([]);

        // Draw ego vehicle (bottom center)
        this.drawEgoVehicle(w / 2, h - 40);

        // Draw predicted path
        ctx.fillStyle = 'rgba(23, 176, 107, 0.2)';
        ctx.beginPath();
        ctx.moveTo(w / 2 - 15, h - 60);
        ctx.lineTo(w / 2 + 15, h - 60);
        ctx.lineTo(w / 2 + 10, 50);
        ctx.lineTo(w / 2 - 10, 50);
        ctx.closePath();
        ctx.fill();

        // Draw other vehicles
        this.vehicles.forEach((v, i) => {
            const bevX = w * (0.3 + i * 0.2);
            const bevY = h * 0.2 + i * 60;
            this.drawVehicle(bevX, bevY, v.type, '#3e6ae1');
        });

        // Draw pedestrians
        this.pedestrians.forEach((p, i) => {
            const bevX = w * 0.8;
            const bevY = h * 0.5;
            this.drawPedestrian(bevX, bevY);
        });

        // Draw traffic light indicators
        this.trafficLights.forEach((tl, i) => {
            const bevX = w * (0.4 + i * 0.2);
            const bevY = 30;
            this.drawTrafficLightBEV(bevX, bevY, tl.state);
        });

        // Draw distance markers
        ctx.fillStyle = '#666';
        ctx.font = '10px Arial';
        ctx.textAlign = 'right';
        for (let d = 20; d <= 100; d += 20) {
            const y = h - (d / 100) * (h - 80);
            ctx.fillText(`${d}m`, w - 5, y);

            ctx.strokeStyle = 'rgba(255,255,255,0.1)';
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w - 25, y);
            ctx.stroke();
        }
    }

    drawEgoVehicle(x, y) {
        const ctx = this.ctx;

        // Car body
        ctx.fillStyle = '#17b06b';
        ctx.beginPath();
        ctx.roundRect(x - 12, y - 25, 24, 50, 5);
        ctx.fill();

        // Windshield
        ctx.fillStyle = '#0d7a4a';
        ctx.beginPath();
        ctx.roundRect(x - 8, y - 20, 16, 15, 3);
        ctx.fill();

        // Direction indicator
        ctx.fillStyle = '#17b06b';
        ctx.beginPath();
        ctx.moveTo(x, y - 35);
        ctx.lineTo(x - 6, y - 28);
        ctx.lineTo(x + 6, y - 28);
        ctx.closePath();
        ctx.fill();
    }

    drawVehicle(x, y, type, color) {
        const ctx = this.ctx;

        ctx.fillStyle = color;
        ctx.globalAlpha = 0.8;

        let width = 20;
        let height = 40;

        if (type === 'truck') {
            width = 24;
            height = 55;
        } else if (type === 'suv') {
            width = 22;
            height = 45;
        }

        ctx.beginPath();
        ctx.roundRect(x - width / 2, y - height / 2, width, height, 4);
        ctx.fill();

        // Glow effect
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;

        ctx.globalAlpha = 1;
    }

    drawPedestrian(x, y) {
        const ctx = this.ctx;

        ctx.fillStyle = '#f5a623';
        ctx.globalAlpha = 0.8;

        // Body
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();

        // Glow
        ctx.shadowColor = '#f5a623';
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;

        ctx.globalAlpha = 1;
    }

    drawTrafficLightBEV(x, y, state) {
        const ctx = this.ctx;

        const colors = {
            red: '#ff0000',
            yellow: '#ffff00',
            green: '#00ff00'
        };

        ctx.fillStyle = colors[state];
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fill();

        ctx.shadowColor = colors[state];
        ctx.shadowBlur = 15;
        ctx.fill();
        ctx.shadowBlur = 0;
    }

    updateSimulation() {
        this.time += 0.1;

        // Update speed (simulate slight variations)
        this.speed = 65 + Math.sin(this.time * 0.5) * 3;
        document.getElementById('current-speed').textContent = Math.round(this.speed);

        // Update lead vehicle
        this.leadVehicle.distance = 45 + Math.sin(this.time * 0.3) * 5;
        this.leadVehicle.speed = 62 + Math.sin(this.time * 0.4) * 4;
        this.leadVehicle.ttc = this.leadVehicle.distance / Math.abs(this.speed - this.leadVehicle.speed + 0.1);

        document.getElementById('lead-distance').textContent = `${Math.round(this.leadVehicle.distance)}m`;
        document.getElementById('lead-speed').textContent = `${Math.round(this.leadVehicle.speed)} km/h`;
        document.getElementById('ttc').textContent = `${this.leadVehicle.ttc.toFixed(1)}s`;

        // Update traffic light timer
        this.lightTimer -= 0.1;
        if (this.lightTimer <= 0) {
            this.lightTimer = 8 + Math.random() * 5;
            const states = ['red', 'yellow', 'green'];
            const currentIndex = states.indexOf(this.currentLightState);
            this.currentLightState = states[(currentIndex + 1) % 3];

            this.trafficLights[0].state = this.currentLightState;
            this.trafficLights[0].timer = Math.round(this.lightTimer);
        }

        const lightColors = { red: 'RED', yellow: 'YLW', green: 'GRN' };
        document.getElementById('next-light').textContent = `${lightColors[this.currentLightState]} ${Math.round(this.lightTimer)}s`;
        document.getElementById('next-light').className = `info-value ${this.currentLightState === 'green' ? 'green' : this.currentLightState === 'yellow' ? 'yellow' : 'red'}`;

        // Update neural network activity (simulated)
        document.getElementById('nn-vision').style.width = `${75 + Math.sin(this.time) * 15}%`;
        document.getElementById('nn-planning').style.width = `${70 + Math.sin(this.time * 1.2) * 20}%`;
        document.getElementById('nn-control').style.width = `${85 + Math.sin(this.time * 0.8) * 10}%`;
        document.getElementById('nn-prediction').style.width = `${65 + Math.sin(this.time * 1.5) * 25}%`;

        // Move objects slightly
        this.vehicles.forEach((v, i) => {
            v.y = 35 + Math.sin(this.time + i) * 3;
            v.distance = 45 + i * 15 + Math.sin(this.time * 0.5 + i) * 5;
        });

        this.pedestrians.forEach((p, i) => {
            p.x = 85 + Math.sin(this.time * 0.3) * 3;
        });

        this.trafficLights.forEach((tl, i) => {
            tl.timer = Math.max(1, Math.round(this.lightTimer - i * 2));
        });

        this.renderCameraObjects();
        this.drawBEV();
    }

    startSimulation() {
        this.drawBEV();

        setInterval(() => {
            this.updateSimulation();
        }, 100);
    }
}

// Canvas roundRect polyfill
if (!CanvasRenderingContext2D.prototype.roundRect) {
    CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
        if (w < 2 * r) r = w / 2;
        if (h < 2 * r) r = h / 2;
        this.moveTo(x + r, y);
        this.arcTo(x + w, y, x + w, y + h, r);
        this.arcTo(x + w, y + h, x, y + h, r);
        this.arcTo(x, y + h, x, y, r);
        this.arcTo(x, y, x + w, y, r);
        return this;
    };
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.fsd = new TeslaFSDVisualization();
});
