/**
 * FSD V12 Driving Simulation
 * Simulates End-to-End neural network driving
 */

class FSDSimulation {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.resize();

        // Vehicle state
        this.egoVehicle = {
            x: 0,
            y: 0,
            speed: 65,
            steering: 0,
            acceleration: 0.3,
            brake: 0
        };

        // Neural network output (simulated)
        this.nnOutput = {
            steering: 0,
            acceleration: 0.3,
            brake: 0
        };

        // Other vehicles
        this.vehicles = [];
        this.generateVehicles();

        // Road
        this.roadOffset = 0;

        // Traffic light
        this.trafficLight = {
            state: 'green',
            distance: 85,
            timer: 12
        };

        // Animation
        this.time = 0;
        this.animate();

        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        this.width = rect.width;
        this.height = rect.height;
    }

    generateVehicles() {
        this.vehicles = [
            { x: 0, z: 45, speed: 62, lane: 0, type: 'sedan' },
            { x: 50, z: 80, speed: 58, lane: 1, type: 'suv' },
            { x: -50, z: 120, speed: 55, lane: -1, type: 'truck' }
        ];
    }

    update() {
        this.time += 0.016;

        // Simulate neural network output
        this.simulateNeuralNetwork();

        // Update ego vehicle
        this.egoVehicle.speed = 65 + Math.sin(this.time * 0.5) * 5;
        this.egoVehicle.steering = Math.sin(this.time * 0.3) * 3;

        // Update road offset (simulating forward movement)
        this.roadOffset += this.egoVehicle.speed * 0.01;
        if (this.roadOffset > 50) this.roadOffset = 0;

        // Update other vehicles
        this.vehicles.forEach((v, i) => {
            v.z -= (this.egoVehicle.speed - v.speed) * 0.016;
            if (v.z < -20) {
                v.z = 150 + Math.random() * 50;
                v.speed = 50 + Math.random() * 20;
            }
        });

        // Update traffic light
        this.trafficLight.timer -= 0.016;
        if (this.trafficLight.timer <= 0) {
            const states = ['red', 'yellow', 'green'];
            const idx = states.indexOf(this.trafficLight.state);
            this.trafficLight.state = states[(idx + 1) % 3];
            this.trafficLight.timer = this.trafficLight.state === 'yellow' ? 3 : 10 + Math.random() * 5;
        }
        this.trafficLight.distance = 85 - (this.time * 10) % 100;
        if (this.trafficLight.distance < 5) this.trafficLight.distance = 85;

        // Update UI
        this.updateUI();
    }

    simulateNeuralNetwork() {
        // Simulated neural network decision making
        // In reality, this would be the output of a massive transformer

        // Steering: smooth lane centering
        this.nnOutput.steering = Math.sin(this.time * 0.5) * 5 + 50;

        // Acceleration: based on lead vehicle distance
        const leadDistance = this.vehicles[0] ? this.vehicles[0].z : 100;
        if (leadDistance < 30) {
            this.nnOutput.acceleration = 0;
            this.nnOutput.brake = 50;
        } else if (leadDistance < 50) {
            this.nnOutput.acceleration = 20;
            this.nnOutput.brake = 10;
        } else {
            this.nnOutput.acceleration = 30 + Math.sin(this.time) * 10;
            this.nnOutput.brake = 5;
        }

        // React to traffic light
        if (this.trafficLight.state === 'red' && this.trafficLight.distance < 50) {
            this.nnOutput.brake = 70;
            this.nnOutput.acceleration = 0;
        }
    }

    updateUI() {
        // Speed
        const speedEl = document.getElementById('speed-value');
        if (speedEl) {
            speedEl.textContent = Math.round(this.egoVehicle.speed);
        }

        // Control bars
        const steeringBar = document.getElementById('steering-bar');
        const accelBar = document.getElementById('accel-bar');
        const brakeBar = document.getElementById('brake-bar');

        if (steeringBar) steeringBar.style.width = `${this.nnOutput.steering}%`;
        if (accelBar) accelBar.style.width = `${this.nnOutput.acceleration}%`;
        if (brakeBar) brakeBar.style.width = `${this.nnOutput.brake}%`;

        // Attention bars
        const attRoad = document.getElementById('att-road');
        const attVehicles = document.getElementById('att-vehicles');
        const attSigns = document.getElementById('att-signs');

        if (attRoad) attRoad.style.width = `${75 + Math.sin(this.time) * 15}%`;
        if (attVehicles) attVehicles.style.width = `${50 + Math.sin(this.time * 1.3) * 20}%`;
        if (attSigns) attSigns.style.width = `${30 + Math.sin(this.time * 0.7) * 25}%`;
    }

    draw() {
        // Clear
        this.ctx.fillStyle = '#0a0a0f';
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Draw road
        this.drawRoad();

        // Draw vehicles
        this.drawVehicles();

        // Draw ego vehicle
        this.drawEgoVehicle();

        // Draw predicted path
        this.drawPredictedPath();

        // Draw attention visualization
        this.drawAttention();

        // Draw traffic light indicator
        this.drawTrafficLight();
    }

    drawRoad() {
        const centerX = this.width / 2;
        const horizonY = this.height * 0.35;

        // Sky gradient
        const skyGradient = this.ctx.createLinearGradient(0, 0, 0, horizonY);
        skyGradient.addColorStop(0, '#0a0a1a');
        skyGradient.addColorStop(1, '#1a1a2e');
        this.ctx.fillStyle = skyGradient;
        this.ctx.fillRect(0, 0, this.width, horizonY);

        // Road surface
        this.ctx.fillStyle = '#1a1a20';
        this.ctx.beginPath();
        this.ctx.moveTo(centerX - 30, horizonY);
        this.ctx.lineTo(centerX + 30, horizonY);
        this.ctx.lineTo(this.width, this.height);
        this.ctx.lineTo(0, this.height);
        this.ctx.closePath();
        this.ctx.fill();

        // Lane lines
        const laneOffsets = [-40, 0, 40];
        laneOffsets.forEach((offset, idx) => {
            const isDashed = idx === 1;

            for (let d = 0; d < 100; d += 10) {
                const depth = d + this.roadOffset % 10;
                const scale = Math.max(0.01, 1 - depth / 100);
                const y = horizonY + (this.height - horizonY) * (1 - scale);
                const x = centerX + offset * scale * 5;

                if (isDashed && Math.floor((d + this.roadOffset) / 5) % 2 === 0) continue;

                const lineWidth = 2 * scale;
                const lineLength = 20 * scale;

                this.ctx.strokeStyle = `rgba(62, 106, 225, ${scale * 0.7})`;
                this.ctx.lineWidth = lineWidth;
                this.ctx.beginPath();
                this.ctx.moveTo(x, y);
                this.ctx.lineTo(x, y + lineLength);
                this.ctx.stroke();
            }
        });
    }

    drawVehicles() {
        const centerX = this.width / 2;
        const horizonY = this.height * 0.35;

        this.vehicles.forEach(vehicle => {
            if (vehicle.z < 0 || vehicle.z > 150) return;

            const scale = Math.max(0.1, 1 - vehicle.z / 150);
            const y = horizonY + (this.height - horizonY) * (1 - scale * scale);
            const x = centerX + vehicle.x * scale * 3;

            // Vehicle body
            const width = 60 * scale;
            const height = 40 * scale;

            // Glow
            const gradient = this.ctx.createRadialGradient(
                x, y, 0,
                x, y, width
            );
            gradient.addColorStop(0, 'rgba(62, 106, 225, 0.3)');
            gradient.addColorStop(1, 'transparent');
            this.ctx.fillStyle = gradient;
            this.ctx.fillRect(x - width, y - height / 2, width * 2, height);

            // Body
            this.ctx.fillStyle = '#3e6ae1';
            this.ctx.beginPath();
            this.ctx.roundRect(x - width / 2, y - height / 2, width, height, 5 * scale);
            this.ctx.fill();

            // Distance label
            this.ctx.fillStyle = 'white';
            this.ctx.font = `${10 + scale * 4}px sans-serif`;
            this.ctx.textAlign = 'center';
            this.ctx.fillText(`${Math.round(vehicle.z)}m`, x, y + height / 2 + 15 * scale);

            // Bounding box
            this.ctx.strokeStyle = 'rgba(62, 106, 225, 0.5)';
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(x - width / 2 - 5, y - height / 2 - 5, width + 10, height + 10);
        });
    }

    drawEgoVehicle() {
        // Simple ego vehicle indicator at bottom
        const centerX = this.width / 2;
        const y = this.height - 30;

        // Hood shape
        this.ctx.fillStyle = '#17b06b';
        this.ctx.beginPath();
        this.ctx.moveTo(centerX - 40, this.height);
        this.ctx.lineTo(centerX + 40, this.height);
        this.ctx.lineTo(centerX + 30, y);
        this.ctx.lineTo(centerX - 30, y);
        this.ctx.closePath();
        this.ctx.fill();
    }

    drawPredictedPath() {
        const centerX = this.width / 2;
        const horizonY = this.height * 0.35;

        // Predicted path (green area)
        const gradient = this.ctx.createLinearGradient(0, this.height, 0, horizonY);
        gradient.addColorStop(0, 'rgba(23, 176, 107, 0.4)');
        gradient.addColorStop(0.5, 'rgba(23, 176, 107, 0.1)');
        gradient.addColorStop(1, 'transparent');

        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();

        // Path curves based on steering
        const steeringOffset = this.egoVehicle.steering * 0.5;

        this.ctx.moveTo(centerX - 20, this.height);
        this.ctx.quadraticCurveTo(
            centerX + steeringOffset, this.height * 0.6,
            centerX + steeringOffset * 0.5, horizonY + 50
        );
        this.ctx.lineTo(centerX + steeringOffset * 0.5 + 10, horizonY + 50);
        this.ctx.quadraticCurveTo(
            centerX + steeringOffset + 10, this.height * 0.6,
            centerX + 20, this.height
        );
        this.ctx.closePath();
        this.ctx.fill();
    }

    drawAttention() {
        // Attention heatmap overlay
        const gradient = this.ctx.createRadialGradient(
            this.width / 2, this.height * 0.5, 0,
            this.width / 2, this.height * 0.5, this.width * 0.4
        );

        const alpha = 0.1 + Math.sin(this.time * 2) * 0.05;
        gradient.addColorStop(0, `rgba(155, 89, 182, ${alpha})`);
        gradient.addColorStop(0.5, `rgba(155, 89, 182, ${alpha * 0.5})`);
        gradient.addColorStop(1, 'transparent');

        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.width, this.height);
    }

    drawTrafficLight() {
        if (this.trafficLight.distance > 100) return;

        const centerX = this.width / 2;
        const scale = Math.max(0.3, 1 - this.trafficLight.distance / 100);
        const y = this.height * 0.35 + (this.height * 0.2) * (1 - scale);

        // Traffic light box
        const boxWidth = 20 * scale;
        const boxHeight = 50 * scale;
        const x = centerX + 80 * scale;

        this.ctx.fillStyle = '#222';
        this.ctx.fillRect(x - boxWidth / 2, y, boxWidth, boxHeight);

        // Lights
        const colors = {
            red: this.trafficLight.state === 'red' ? '#ff0000' : '#330000',
            yellow: this.trafficLight.state === 'yellow' ? '#ffff00' : '#333300',
            green: this.trafficLight.state === 'green' ? '#00ff00' : '#003300'
        };

        const lightRadius = 5 * scale;
        const lightPositions = [
            { color: colors.red, y: y + boxHeight * 0.2 },
            { color: colors.yellow, y: y + boxHeight * 0.5 },
            { color: colors.green, y: y + boxHeight * 0.8 }
        ];

        lightPositions.forEach(light => {
            this.ctx.fillStyle = light.color;
            this.ctx.beginPath();
            this.ctx.arc(x, light.y, lightRadius, 0, Math.PI * 2);
            this.ctx.fill();

            // Glow for active light
            if (light.color.startsWith('#ff') || light.color === '#00ff00') {
                const glow = this.ctx.createRadialGradient(
                    x, light.y, 0,
                    x, light.y, lightRadius * 3
                );
                glow.addColorStop(0, light.color);
                glow.addColorStop(1, 'transparent');
                this.ctx.fillStyle = glow;
                this.ctx.beginPath();
                this.ctx.arc(x, light.y, lightRadius * 3, 0, Math.PI * 2);
                this.ctx.fill();
            }
        });
    }

    animate() {
        this.update();
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.fsdSim = new FSDSimulation('sim-canvas');
});
