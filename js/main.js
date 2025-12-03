/**
 * Tesla FSD V12 - Main Application
 * Hero animation and general interactions
 */

class HeroAnimation {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.resize();

        // Neural network nodes for background
        this.nodes = [];
        this.connections = [];

        this.generateNetwork();

        this.time = 0;
        this.animate();

        window.addEventListener('resize', () => {
            this.resize();
            this.generateNetwork();
        });
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.width = this.canvas.width;
        this.height = this.canvas.height;
    }

    generateNetwork() {
        this.nodes = [];
        this.connections = [];

        const nodeCount = Math.floor((this.width * this.height) / 15000);

        for (let i = 0; i < nodeCount; i++) {
            this.nodes.push({
                x: Math.random() * this.width,
                y: Math.random() * this.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                radius: 1 + Math.random() * 2,
                pulse: Math.random() * Math.PI * 2
            });
        }

        // Generate connections
        for (let i = 0; i < this.nodes.length; i++) {
            for (let j = i + 1; j < this.nodes.length; j++) {
                const dist = this.getDistance(this.nodes[i], this.nodes[j]);
                if (dist < 150) {
                    this.connections.push({
                        from: i,
                        to: j,
                        dist: dist
                    });
                }
            }
        }
    }

    getDistance(a, b) {
        return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
    }

    update() {
        this.time += 0.016;

        // Update nodes
        this.nodes.forEach(node => {
            node.x += node.vx;
            node.y += node.vy;
            node.pulse += 0.05;

            // Bounce off edges
            if (node.x < 0 || node.x > this.width) node.vx *= -1;
            if (node.y < 0 || node.y > this.height) node.vy *= -1;

            // Keep in bounds
            node.x = Math.max(0, Math.min(this.width, node.x));
            node.y = Math.max(0, Math.min(this.height, node.y));
        });

        // Update connections
        this.connections = [];
        for (let i = 0; i < this.nodes.length; i++) {
            for (let j = i + 1; j < this.nodes.length; j++) {
                const dist = this.getDistance(this.nodes[i], this.nodes[j]);
                if (dist < 150) {
                    this.connections.push({
                        from: i,
                        to: j,
                        dist: dist
                    });
                }
            }
        }
    }

    draw() {
        // Clear with fade effect
        this.ctx.fillStyle = 'rgba(10, 10, 15, 0.1)';
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Draw connections
        this.connections.forEach(conn => {
            const from = this.nodes[conn.from];
            const to = this.nodes[conn.to];
            const alpha = (1 - conn.dist / 150) * 0.3;

            this.ctx.strokeStyle = `rgba(62, 106, 225, ${alpha})`;
            this.ctx.lineWidth = 0.5;
            this.ctx.beginPath();
            this.ctx.moveTo(from.x, from.y);
            this.ctx.lineTo(to.x, to.y);
            this.ctx.stroke();
        });

        // Draw nodes
        this.nodes.forEach(node => {
            const pulse = Math.sin(node.pulse) * 0.5 + 0.5;
            const radius = node.radius + pulse;

            // Glow
            const gradient = this.ctx.createRadialGradient(
                node.x, node.y, 0,
                node.x, node.y, radius * 3
            );
            gradient.addColorStop(0, `rgba(62, 106, 225, ${0.5 + pulse * 0.3})`);
            gradient.addColorStop(1, 'transparent');

            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, radius * 3, 0, Math.PI * 2);
            this.ctx.fill();

            // Core
            this.ctx.fillStyle = `rgba(255, 255, 255, ${0.5 + pulse * 0.5})`;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
            this.ctx.fill();
        });

        // Occasional signal pulse
        if (Math.random() < 0.02) {
            const startNode = this.nodes[Math.floor(Math.random() * this.nodes.length)];
            this.drawSignalPulse(startNode);
        }
    }

    drawSignalPulse(node) {
        const gradient = this.ctx.createRadialGradient(
            node.x, node.y, 0,
            node.x, node.y, 50
        );
        gradient.addColorStop(0, 'rgba(23, 176, 107, 0.8)');
        gradient.addColorStop(0.5, 'rgba(23, 176, 107, 0.2)');
        gradient.addColorStop(1, 'transparent');

        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, 50, 0, Math.PI * 2);
        this.ctx.fill();
    }

    animate() {
        this.update();
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
}

// Architecture node interactions
function initArchitectureInteractions() {
    const nodes = document.querySelectorAll('.arch-node');

    nodes.forEach(node => {
        node.addEventListener('mouseenter', () => {
            node.style.transform = 'translateY(-10px) scale(1.05)';
        });

        node.addEventListener('mouseleave', () => {
            node.style.transform = 'translateY(0) scale(1)';
        });
    });
}

// Smooth scroll
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Scroll animations
function initScrollAnimations() {
    const sections = document.querySelectorAll('.section');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, {
        threshold: 0.1
    });

    sections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(50px)';
        section.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
        observer.observe(section);
    });
}

// Initialize counter animations
function initCounters() {
    const counters = [
        { id: 'param-count', target: 1.2, suffix: 'B', decimals: 1 },
        { id: 'flops', target: 100, suffix: '', decimals: 0, prefix: '~' },
        { id: 'latency', target: 50, suffix: 'ms', decimals: 0 },
        { id: 'fps', target: 20, suffix: '', decimals: 0 }
    ];

    counters.forEach(counter => {
        const el = document.getElementById(counter.id);
        if (!el) return;

        let current = 0;
        const step = counter.target / 50;
        const interval = setInterval(() => {
            current += step;
            if (current >= counter.target) {
                current = counter.target;
                clearInterval(interval);
            }

            el.textContent = (counter.prefix || '') +
                current.toFixed(counter.decimals) +
                counter.suffix;
        }, 30);
    });
}

// Initialize everything
document.addEventListener('DOMContentLoaded', () => {
    // Hero animation
    window.heroAnim = new HeroAnimation('hero-canvas');

    // Interactions
    initArchitectureInteractions();
    initSmoothScroll();

    // Delay scroll animations to avoid initial flash
    setTimeout(() => {
        initScrollAnimations();
        initCounters();
    }, 100);

    console.log('Tesla FSD V12 Visualization loaded');
    console.log('End-to-End Neural Network Architecture');
});
