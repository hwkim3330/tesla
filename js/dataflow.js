/**
 * Data Flow Visualization
 * Shows the flow of data through the End-to-End network
 */

class DataFlowViz {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.resize();

        // Pipeline stages
        this.stages = [
            {
                name: 'Photons (Camera)',
                icon: '📹',
                color: '#3e6ae1',
                x: 0.1,
                items: ['Front Main', 'Front Narrow', 'Front Wide', 'Left Pillar', 'Right Pillar', 'Side Left', 'Side Right', 'Rear']
            },
            {
                name: 'Feature Extraction',
                icon: '🔍',
                color: '#9b59b6',
                x: 0.25,
                items: ['RegNet Backbone', 'Multi-scale Features', 'Spatial Encoding']
            },
            {
                name: 'Temporal Processing',
                icon: '⏱️',
                color: '#8e44ad',
                x: 0.4,
                items: ['Video Memory', 'Motion Analysis', 'Temporal Attention']
            },
            {
                name: 'Transformer',
                icon: '🧠',
                color: '#17b06b',
                x: 0.55,
                items: ['Self-Attention', 'Cross-Attention', 'World Understanding']
            },
            {
                name: 'World Model',
                icon: '🌍',
                color: '#1abc9c',
                x: 0.7,
                items: ['Future Prediction', 'Scene Simulation', 'Risk Assessment']
            },
            {
                name: 'Control Output',
                icon: '🎮',
                color: '#e82127',
                x: 0.85,
                items: ['Steering', 'Acceleration', 'Brake']
            }
        ];

        // Particles for data flow
        this.particles = [];
        this.generateParticles();

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

    generateParticles() {
        this.particles = [];

        for (let i = 0; i < 50; i++) {
            this.particles.push({
                x: Math.random(),
                y: 0.3 + Math.random() * 0.4,
                speed: 0.001 + Math.random() * 0.002,
                size: 2 + Math.random() * 3,
                stageIdx: Math.floor(Math.random() * this.stages.length),
                offset: Math.random()
            });
        }
    }

    update() {
        this.time += 0.016;

        // Update particles
        this.particles.forEach(p => {
            p.x += p.speed;

            // Wrap around
            if (p.x > 1) {
                p.x = 0;
                p.y = 0.3 + Math.random() * 0.4;
            }

            // Find current stage
            for (let i = 0; i < this.stages.length; i++) {
                if (p.x < this.stages[i].x) {
                    p.stageIdx = i;
                    break;
                }
                p.stageIdx = this.stages.length - 1;
            }
        });
    }

    draw() {
        // Clear
        this.ctx.fillStyle = '#0a0a0f';
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Draw connection lines between stages
        this.drawConnections();

        // Draw stages
        this.drawStages();

        // Draw particles
        this.drawParticles();

        // Draw labels
        this.drawLabels();
    }

    drawConnections() {
        for (let i = 0; i < this.stages.length - 1; i++) {
            const stage1 = this.stages[i];
            const stage2 = this.stages[i + 1];

            const x1 = stage1.x * this.width + 40;
            const x2 = stage2.x * this.width - 40;
            const y = this.height / 2;

            // Draw flowing line
            const gradient = this.ctx.createLinearGradient(x1, 0, x2, 0);
            gradient.addColorStop(0, stage1.color);
            gradient.addColorStop(1, stage2.color);

            this.ctx.strokeStyle = gradient;
            this.ctx.lineWidth = 3;
            this.ctx.globalAlpha = 0.3;

            this.ctx.beginPath();
            this.ctx.moveTo(x1, y);

            // Curved connection
            const cp1x = x1 + (x2 - x1) * 0.4;
            const cp2x = x1 + (x2 - x1) * 0.6;
            const wave = Math.sin(this.time * 2 + i) * 20;

            this.ctx.bezierCurveTo(
                cp1x, y + wave,
                cp2x, y - wave,
                x2, y
            );

            this.ctx.stroke();
            this.ctx.globalAlpha = 1;

            // Arrow
            this.ctx.fillStyle = stage2.color;
            this.ctx.beginPath();
            this.ctx.moveTo(x2 + 10, y);
            this.ctx.lineTo(x2, y - 6);
            this.ctx.lineTo(x2, y + 6);
            this.ctx.closePath();
            this.ctx.fill();
        }
    }

    drawStages() {
        this.stages.forEach((stage, idx) => {
            const x = stage.x * this.width;
            const y = this.height / 2;
            const radius = 35;

            // Glow
            const glowGradient = this.ctx.createRadialGradient(x, y, 0, x, y, radius * 2);
            glowGradient.addColorStop(0, stage.color + '44');
            glowGradient.addColorStop(1, 'transparent');
            this.ctx.fillStyle = glowGradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius * 2, 0, Math.PI * 2);
            this.ctx.fill();

            // Node
            this.ctx.fillStyle = stage.color;
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius, 0, Math.PI * 2);
            this.ctx.fill();

            // Pulse animation
            const pulseRadius = radius + 5 + Math.sin(this.time * 3 + idx) * 5;
            this.ctx.strokeStyle = stage.color;
            this.ctx.lineWidth = 2;
            this.ctx.globalAlpha = 0.5 - Math.sin(this.time * 3 + idx) * 0.3;
            this.ctx.beginPath();
            this.ctx.arc(x, y, pulseRadius, 0, Math.PI * 2);
            this.ctx.stroke();
            this.ctx.globalAlpha = 1;

            // Icon
            this.ctx.font = '24px sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(stage.icon, x, y);

            // Stage name
            this.ctx.font = 'bold 12px sans-serif';
            this.ctx.fillStyle = '#fff';
            this.ctx.fillText(stage.name, x, y - radius - 20);

            // Items
            this.ctx.font = '10px sans-serif';
            this.ctx.fillStyle = '#666';
            stage.items.forEach((item, i) => {
                const itemY = y + radius + 25 + i * 18;
                const alpha = 0.3 + Math.sin(this.time * 2 + i * 0.5) * 0.3;
                this.ctx.globalAlpha = alpha;
                this.ctx.fillText(item, x, itemY);
            });
            this.ctx.globalAlpha = 1;
        });
    }

    drawParticles() {
        this.particles.forEach(p => {
            const x = p.x * this.width;
            const y = p.y * this.height + Math.sin(this.time * 5 + p.offset * 10) * 30;

            // Get color from current stage
            const stage = this.stages[Math.min(p.stageIdx, this.stages.length - 1)];
            const nextStage = this.stages[Math.min(p.stageIdx + 1, this.stages.length - 1)];

            // Blend color based on position
            const progress = (p.x - stage.x) / (nextStage.x - stage.x || 1);

            // Particle
            const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, p.size * 2);
            gradient.addColorStop(0, '#ffffff');
            gradient.addColorStop(0.3, stage.color);
            gradient.addColorStop(1, 'transparent');

            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, p.size * 2, 0, Math.PI * 2);
            this.ctx.fill();

            // Trail
            this.ctx.strokeStyle = stage.color + '44';
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.moveTo(x, y);
            this.ctx.lineTo(x - 30 * p.speed * 100, y);
            this.ctx.stroke();
        });
    }

    drawLabels() {
        // Title
        this.ctx.font = 'bold 16px sans-serif';
        this.ctx.fillStyle = '#888';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
            'Data Flow: Every 50ms, 1.2B parameters process 8 camera feeds',
            this.width / 2, 30
        );

        // Stats at bottom
        this.ctx.font = '12px sans-serif';
        this.ctx.fillStyle = '#444';
        this.ctx.textAlign = 'left';

        const stats = [
            `Input: ${8} cameras × 1280×960`,
            `Processing: ~100 TFLOPS`,
            `Latency: 50ms`,
            `Output: Steering, Acceleration, Brake`
        ];

        stats.forEach((stat, i) => {
            this.ctx.fillText(stat, 20, this.height - 20 - (stats.length - 1 - i) * 20);
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
    window.dataFlowViz = new DataFlowViz('dataflow-canvas');
});
