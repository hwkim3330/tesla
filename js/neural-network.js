/**
 * Neural Network Visualization
 * Visualizes the internal structure of the End-to-End network
 */

class NeuralNetworkViz {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.resize();

        // Network structure
        this.layers = [
            { name: 'Input', neurons: 8, color: '#3e6ae1' },      // 8 cameras
            { name: 'Backbone', neurons: 12, color: '#9b59b6' },  // Feature extraction
            { name: 'Temporal', neurons: 10, color: '#8e44ad' },  // Temporal processing
            { name: 'Transformer', neurons: 16, color: '#17b06b' }, // Main transformer
            { name: 'World Model', neurons: 12, color: '#1abc9c' }, // World model
            { name: 'Policy', neurons: 8, color: '#e74c3c' },     // Policy head
            { name: 'Output', neurons: 3, color: '#e82127' }      // Steering, accel, brake
        ];

        // Animation state
        this.time = 0;
        this.connections = [];
        this.activeConnections = [];

        this.generateConnections();
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

    generateConnections() {
        this.connections = [];

        for (let l = 0; l < this.layers.length - 1; l++) {
            const layer1 = this.layers[l];
            const layer2 = this.layers[l + 1];

            for (let i = 0; i < layer1.neurons; i++) {
                for (let j = 0; j < layer2.neurons; j++) {
                    // Random connection probability
                    if (Math.random() > 0.3) {
                        this.connections.push({
                            layer1: l,
                            neuron1: i,
                            layer2: l + 1,
                            neuron2: j,
                            weight: Math.random()
                        });
                    }
                }
            }
        }
    }

    getNeuronPosition(layerIdx, neuronIdx) {
        const layer = this.layers[layerIdx];
        const layerSpacing = this.width / (this.layers.length + 1);
        const x = layerSpacing * (layerIdx + 1);

        const neuronSpacing = this.height / (layer.neurons + 1);
        const y = neuronSpacing * (neuronIdx + 1);

        return { x, y };
    }

    draw() {
        // Clear canvas
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Draw connections
        this.drawConnections();

        // Draw neurons
        this.drawNeurons();

        // Draw layer labels
        this.drawLabels();
    }

    drawConnections() {
        // Draw all connections with low opacity
        this.ctx.lineWidth = 0.5;

        for (const conn of this.connections) {
            const pos1 = this.getNeuronPosition(conn.layer1, conn.neuron1);
            const pos2 = this.getNeuronPosition(conn.layer2, conn.neuron2);

            // Animate signal flow
            const phase = (this.time * 2 + conn.layer1 * 0.5 + conn.neuron1 * 0.1) % 1;
            const alpha = Math.sin(phase * Math.PI) * 0.3;

            const gradient = this.ctx.createLinearGradient(pos1.x, pos1.y, pos2.x, pos2.y);
            const color1 = this.layers[conn.layer1].color;
            const color2 = this.layers[conn.layer2].color;

            gradient.addColorStop(0, color1 + Math.floor(alpha * 255).toString(16).padStart(2, '0'));
            gradient.addColorStop(1, color2 + Math.floor(alpha * 255).toString(16).padStart(2, '0'));

            this.ctx.strokeStyle = gradient;
            this.ctx.beginPath();
            this.ctx.moveTo(pos1.x, pos1.y);
            this.ctx.lineTo(pos2.x, pos2.y);
            this.ctx.stroke();
        }

        // Draw active signal paths
        this.drawActivePaths();
    }

    drawActivePaths() {
        const pathCount = 5;

        for (let p = 0; p < pathCount; p++) {
            const offset = (this.time * 0.5 + p * 0.2) % 1;
            const layerFloat = offset * (this.layers.length - 1);
            const currentLayer = Math.floor(layerFloat);
            const progress = layerFloat - currentLayer;

            if (currentLayer < this.layers.length - 1) {
                // Random neuron in current and next layer
                const neuron1 = Math.floor(Math.random() * this.layers[currentLayer].neurons);
                const neuron2 = Math.floor(Math.random() * this.layers[currentLayer + 1].neurons);

                const pos1 = this.getNeuronPosition(currentLayer, neuron1);
                const pos2 = this.getNeuronPosition(currentLayer + 1, neuron2);

                // Signal position
                const signalX = pos1.x + (pos2.x - pos1.x) * progress;
                const signalY = pos1.y + (pos2.y - pos1.y) * progress;

                // Draw signal
                const gradient = this.ctx.createRadialGradient(
                    signalX, signalY, 0,
                    signalX, signalY, 15
                );
                gradient.addColorStop(0, '#ffffff');
                gradient.addColorStop(0.5, this.layers[currentLayer].color);
                gradient.addColorStop(1, 'transparent');

                this.ctx.fillStyle = gradient;
                this.ctx.beginPath();
                this.ctx.arc(signalX, signalY, 15, 0, Math.PI * 2);
                this.ctx.fill();
            }
        }
    }

    drawNeurons() {
        for (let l = 0; l < this.layers.length; l++) {
            const layer = this.layers[l];

            for (let n = 0; n < layer.neurons; n++) {
                const pos = this.getNeuronPosition(l, n);

                // Activation animation
                const activation = 0.5 + 0.5 * Math.sin(this.time * 3 + l + n * 0.5);
                const radius = 4 + activation * 3;

                // Glow
                const gradient = this.ctx.createRadialGradient(
                    pos.x, pos.y, 0,
                    pos.x, pos.y, radius * 3
                );
                gradient.addColorStop(0, layer.color);
                gradient.addColorStop(0.5, layer.color + '44');
                gradient.addColorStop(1, 'transparent');

                this.ctx.fillStyle = gradient;
                this.ctx.beginPath();
                this.ctx.arc(pos.x, pos.y, radius * 3, 0, Math.PI * 2);
                this.ctx.fill();

                // Neuron core
                this.ctx.fillStyle = layer.color;
                this.ctx.beginPath();
                this.ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
                this.ctx.fill();
            }
        }
    }

    drawLabels() {
        this.ctx.fillStyle = '#666';
        this.ctx.font = '10px sans-serif';
        this.ctx.textAlign = 'center';

        for (let l = 0; l < this.layers.length; l++) {
            const layer = this.layers[l];
            const pos = this.getNeuronPosition(l, 0);

            this.ctx.fillText(layer.name, pos.x, this.height - 10);
        }
    }

    animate() {
        this.time += 0.016;
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.nnViz = new NeuralNetworkViz('nn-canvas');
});
