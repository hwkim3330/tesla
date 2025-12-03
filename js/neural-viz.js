/**
 * Neural Network Activity Visualization
 * Real-time visualization of neural network layers and activations
 */

class NeuralNetworkViz {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.resize();

        // Network architecture (NVIDIA-style)
        this.layers = [
            { name: 'Input', neurons: 8, type: 'input', color: '#3e6ae1' },
            { name: 'Conv1', neurons: 24, type: 'conv', color: '#9b59b6' },
            { name: 'Conv2', neurons: 36, type: 'conv', color: '#8e44ad' },
            { name: 'Conv3', neurons: 48, type: 'conv', color: '#2980b9' },
            { name: 'Conv4', neurons: 64, type: 'conv', color: '#1abc9c' },
            { name: 'FC1', neurons: 100, type: 'dense', color: '#27ae60' },
            { name: 'FC2', neurons: 50, type: 'dense', color: '#17b06b' },
            { name: 'Out', neurons: 1, type: 'output', color: '#e82127' }
        ];

        // Activation states
        this.activations = this.layers.map(l =>
            new Array(Math.min(l.neurons, 12)).fill(0).map(() => Math.random())
        );

        // Connection weights (for visualization)
        this.connections = [];

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

        // Update activations based on autopilot state
        const steeringNorm = (autopilotState?.steering || 0) / 25 + 0.5;

        this.activations.forEach((layer, i) => {
            layer.forEach((_, j) => {
                // Create wave-like activation patterns
                const wave = Math.sin(this.time * 3 + i * 0.5 + j * 0.3);
                const signal = Math.sin(this.time * 5 - i * 0.8);

                // Mix base activation with steering influence
                const baseActivation = (wave * 0.3 + 0.5) + (signal * 0.2);
                const steeringInfluence = (1 - i / this.layers.length) * steeringNorm * 0.3;

                this.activations[i][j] = Math.max(0, Math.min(1,
                    this.activations[i][j] * 0.7 + (baseActivation + steeringInfluence) * 0.3
                ));
            });
        });
    }

    draw() {
        // Clear
        this.ctx.fillStyle = '#0a0a0f';
        this.ctx.fillRect(0, 0, this.width, this.height);

        const padding = 30;
        const layerSpacing = (this.width - padding * 2) / (this.layers.length - 1);

        // Draw connections first (behind neurons)
        this.drawConnections(padding, layerSpacing);

        // Draw layers
        this.layers.forEach((layer, layerIdx) => {
            const x = padding + layerIdx * layerSpacing;
            const displayNeurons = Math.min(layer.neurons, 12);
            const neuronSpacing = (this.height - 40) / (displayNeurons + 1);

            // Draw neurons
            for (let i = 0; i < displayNeurons; i++) {
                const y = 20 + (i + 1) * neuronSpacing;
                const activation = this.activations[layerIdx][i];
                const radius = 4 + activation * 4;

                // Glow effect
                const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, radius * 3);
                gradient.addColorStop(0, layer.color + Math.floor(activation * 200).toString(16).padStart(2, '0'));
                gradient.addColorStop(1, 'transparent');

                this.ctx.fillStyle = gradient;
                this.ctx.beginPath();
                this.ctx.arc(x, y, radius * 3, 0, Math.PI * 2);
                this.ctx.fill();

                // Neuron core
                this.ctx.fillStyle = layer.color;
                this.ctx.globalAlpha = 0.3 + activation * 0.7;
                this.ctx.beginPath();
                this.ctx.arc(x, y, radius, 0, Math.PI * 2);
                this.ctx.fill();
                this.ctx.globalAlpha = 1;
            }

            // Layer label
            this.ctx.fillStyle = '#666';
            this.ctx.font = '9px sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(layer.name, x, this.height - 5);
        });

        // Draw signal flow particles
        this.drawSignalFlow(padding, layerSpacing);
    }

    drawConnections(padding, layerSpacing) {
        for (let layerIdx = 0; layerIdx < this.layers.length - 1; layerIdx++) {
            const x1 = padding + layerIdx * layerSpacing;
            const x2 = padding + (layerIdx + 1) * layerSpacing;

            const neurons1 = Math.min(this.layers[layerIdx].neurons, 12);
            const neurons2 = Math.min(this.layers[layerIdx + 1].neurons, 12);

            const spacing1 = (this.height - 40) / (neurons1 + 1);
            const spacing2 = (this.height - 40) / (neurons2 + 1);

            // Draw subset of connections
            const maxConnections = 30;
            let connDrawn = 0;

            for (let i = 0; i < neurons1 && connDrawn < maxConnections; i++) {
                for (let j = 0; j < neurons2 && connDrawn < maxConnections; j++) {
                    if (Math.random() > 0.7) continue; // Skip some connections

                    const y1 = 20 + (i + 1) * spacing1;
                    const y2 = 20 + (j + 1) * spacing2;

                    const activation = (this.activations[layerIdx][i] + this.activations[layerIdx + 1][j]) / 2;

                    this.ctx.strokeStyle = `rgba(62, 106, 225, ${activation * 0.3})`;
                    this.ctx.lineWidth = 0.5;
                    this.ctx.beginPath();
                    this.ctx.moveTo(x1, y1);
                    this.ctx.lineTo(x2, y2);
                    this.ctx.stroke();

                    connDrawn++;
                }
            }
        }
    }

    drawSignalFlow(padding, layerSpacing) {
        // Animated particles flowing through network
        const numParticles = 5;

        for (let p = 0; p < numParticles; p++) {
            const t = (this.time * 0.5 + p * 0.2) % 1;
            const layerProgress = t * (this.layers.length - 1);
            const layerIdx = Math.floor(layerProgress);
            const layerT = layerProgress - layerIdx;

            if (layerIdx >= this.layers.length - 1) continue;

            const x1 = padding + layerIdx * layerSpacing;
            const x2 = padding + (layerIdx + 1) * layerSpacing;
            const x = x1 + (x2 - x1) * layerT;

            const neurons1 = Math.min(this.layers[layerIdx].neurons, 12);
            const spacing = (this.height - 40) / (neurons1 + 1);
            const neuronIdx = Math.floor(p / numParticles * neurons1);
            const y = 20 + (neuronIdx + 1) * spacing;

            // Draw particle
            const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, 8);
            gradient.addColorStop(0, '#17b06b');
            gradient.addColorStop(1, 'transparent');

            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 8, 0, Math.PI * 2);
            this.ctx.fill();
        }
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
    window.neuralViz = new NeuralNetworkViz('nn-canvas');
});
