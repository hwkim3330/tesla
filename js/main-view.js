/**
 * Main View - 3D Road Visualization with Three.js
 * Combines camera feed simulation with road rendering
 */

class MainView {
    constructor() {
        this.canvas = document.getElementById('main-canvas');
        this.cameraCanvas = document.getElementById('camera-canvas');
        if (!this.canvas) return;

        this.initThree();
        this.initCameraSimulation();
        this.initDetectionList();
        this.initControls();

        this.time = 0;
        this.animate();
    }

    initThree() {
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.Fog(0x0a0a0f, 50, 300);

        // Camera
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
        this.camera.position.set(0, 5, 10);
        this.camera.lookAt(0, 0, -50);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setClearColor(0x0a0a0f);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(0, 50, 50);
        this.scene.add(directionalLight);

        // Create road and environment
        this.createRoad();
        this.createLaneLines();
        this.createEnvironment();
        this.createVehicles();

        // Handle resize
        window.addEventListener('resize', () => this.onResize());
    }

    createRoad() {
        // Road surface
        const roadGeometry = new THREE.PlaneGeometry(12, 500);
        const roadMaterial = new THREE.MeshStandardMaterial({
            color: 0x1a1a1a,
            roughness: 0.9
        });
        this.road = new THREE.Mesh(roadGeometry, roadMaterial);
        this.road.rotation.x = -Math.PI / 2;
        this.road.position.y = 0;
        this.road.position.z = -200;
        this.scene.add(this.road);

        // Road edges (curbs)
        const curbGeometry = new THREE.BoxGeometry(0.3, 0.2, 500);
        const curbMaterial = new THREE.MeshStandardMaterial({ color: 0x333333 });

        const leftCurb = new THREE.Mesh(curbGeometry, curbMaterial);
        leftCurb.position.set(-6, 0.1, -200);
        this.scene.add(leftCurb);

        const rightCurb = new THREE.Mesh(curbGeometry, curbMaterial);
        rightCurb.position.set(6, 0.1, -200);
        this.scene.add(rightCurb);
    }

    createLaneLines() {
        this.laneLines = [];

        // Center dashed line
        const dashMaterial = new THREE.MeshBasicMaterial({ color: 0xf5a623 });
        for (let z = 0; z > -500; z -= 8) {
            const dashGeometry = new THREE.PlaneGeometry(0.15, 4);
            const dash = new THREE.Mesh(dashGeometry, dashMaterial);
            dash.rotation.x = -Math.PI / 2;
            dash.position.set(0, 0.01, z);
            this.scene.add(dash);
            this.laneLines.push(dash);
        }

        // Side lane lines (solid)
        const solidMaterial = new THREE.MeshBasicMaterial({ color: 0x3e6ae1 });
        [-4, 4].forEach(x => {
            const lineGeometry = new THREE.PlaneGeometry(0.15, 500);
            const line = new THREE.Mesh(lineGeometry, solidMaterial);
            line.rotation.x = -Math.PI / 2;
            line.position.set(x, 0.01, -200);
            this.scene.add(line);
        });
    }

    createEnvironment() {
        // Trees/poles on sides
        const poleGeometry = new THREE.CylinderGeometry(0.1, 0.15, 8);
        const poleMaterial = new THREE.MeshStandardMaterial({ color: 0x444444 });

        for (let z = 0; z > -400; z -= 30) {
            [-8, 8].forEach(x => {
                const pole = new THREE.Mesh(poleGeometry, poleMaterial);
                pole.position.set(x, 4, z);
                this.scene.add(pole);

                // Light on pole
                const lightGeometry = new THREE.BoxGeometry(1, 0.3, 0.3);
                const lightMaterial = new THREE.MeshBasicMaterial({ color: 0xffaa00 });
                const light = new THREE.Mesh(lightGeometry, lightMaterial);
                light.position.set(x > 0 ? x - 0.5 : x + 0.5, 7.5, z);
                this.scene.add(light);
            });
        }

        // Ground plane
        const groundGeometry = new THREE.PlaneGeometry(200, 500);
        const groundMaterial = new THREE.MeshStandardMaterial({
            color: 0x0d1f0d,
            roughness: 1
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -0.1;
        ground.position.z = -200;
        this.scene.add(ground);

        // Skybox gradient
        const skyGeometry = new THREE.SphereGeometry(400, 32, 32);
        const skyMaterial = new THREE.ShaderMaterial({
            uniforms: {
                topColor: { value: new THREE.Color(0x0a0a15) },
                bottomColor: { value: new THREE.Color(0x1a1a30) }
            },
            vertexShader: `
                varying vec3 vWorldPosition;
                void main() {
                    vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                    vWorldPosition = worldPosition.xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 topColor;
                uniform vec3 bottomColor;
                varying vec3 vWorldPosition;
                void main() {
                    float h = normalize(vWorldPosition).y;
                    gl_FragColor = vec4(mix(bottomColor, topColor, max(h, 0.0)), 1.0);
                }
            `,
            side: THREE.BackSide
        });
        const sky = new THREE.Mesh(skyGeometry, skyMaterial);
        this.scene.add(sky);
    }

    createVehicles() {
        this.otherVehicles = [];

        // Create simple car meshes
        const createCar = (color) => {
            const group = new THREE.Group();

            // Body
            const bodyGeometry = new THREE.BoxGeometry(2, 1, 4);
            const bodyMaterial = new THREE.MeshStandardMaterial({ color });
            const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
            body.position.y = 0.7;
            group.add(body);

            // Roof
            const roofGeometry = new THREE.BoxGeometry(1.8, 0.6, 2);
            const roofMaterial = new THREE.MeshStandardMaterial({ color: 0x111111 });
            const roof = new THREE.Mesh(roofGeometry, roofMaterial);
            roof.position.y = 1.5;
            roof.position.z = -0.3;
            group.add(roof);

            // Wheels
            const wheelGeometry = new THREE.CylinderGeometry(0.3, 0.3, 0.2, 16);
            const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x111111 });

            [[-0.9, -1.3], [-0.9, 1.3], [0.9, -1.3], [0.9, 1.3]].forEach(([x, z]) => {
                const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial);
                wheel.rotation.z = Math.PI / 2;
                wheel.position.set(x, 0.3, z);
                group.add(wheel);
            });

            // Headlights
            const lightGeometry = new THREE.BoxGeometry(0.3, 0.2, 0.1);
            const lightMaterial = new THREE.MeshBasicMaterial({ color: 0xffffcc });
            [[-0.6, -2], [0.6, -2]].forEach(([x, z]) => {
                const headlight = new THREE.Mesh(lightGeometry, lightMaterial);
                headlight.position.set(x, 0.7, z);
                group.add(headlight);
            });

            // Taillights
            const tailMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
            [[-0.6, 2], [0.6, 2]].forEach(([x, z]) => {
                const taillight = new THREE.Mesh(lightGeometry, tailMaterial);
                taillight.position.set(x, 0.7, z);
                group.add(taillight);
            });

            return group;
        };

        // Lead vehicle
        const leadCar = createCar(0x3b82f6);
        leadCar.position.set(0, 0, -40);
        this.scene.add(leadCar);
        this.otherVehicles.push({ mesh: leadCar, baseZ: -40, lane: 0 });

        // Side vehicles
        const sideCar1 = createCar(0x666666);
        sideCar1.position.set(-3.5, 0, -60);
        this.scene.add(sideCar1);
        this.otherVehicles.push({ mesh: sideCar1, baseZ: -60, lane: -1 });

        const sideCar2 = createCar(0x888888);
        sideCar2.position.set(3.5, 0, -80);
        this.scene.add(sideCar2);
        this.otherVehicles.push({ mesh: sideCar2, baseZ: -80, lane: 1 });
    }

    initCameraSimulation() {
        this.cameraCtx = this.cameraCanvas.getContext('2d');
        this.cameraCanvas.width = 280;
        this.cameraCanvas.height = 180;
    }

    drawCameraFeed() {
        const ctx = this.cameraCtx;
        const w = this.cameraCanvas.width;
        const h = this.cameraCanvas.height;

        // Sky gradient
        const skyGradient = ctx.createLinearGradient(0, 0, 0, h * 0.4);
        skyGradient.addColorStop(0, '#0a0a15');
        skyGradient.addColorStop(1, '#1a1a30');
        ctx.fillStyle = skyGradient;
        ctx.fillRect(0, 0, w, h * 0.4);

        // Road
        ctx.fillStyle = '#1a1a1a';
        ctx.beginPath();
        ctx.moveTo(0, h * 0.4);
        ctx.lineTo(w, h * 0.4);
        ctx.lineTo(w, h);
        ctx.lineTo(0, h);
        ctx.fill();

        // Vanishing point
        const vpX = w / 2 + (window.autopilot?.steeringAngle || 0) * 2;
        const vpY = h * 0.4;

        // Lane lines
        ctx.strokeStyle = '#3e6ae1';
        ctx.lineWidth = 2;

        // Left lane line
        ctx.beginPath();
        ctx.moveTo(0, h);
        ctx.lineTo(vpX - 30, vpY);
        ctx.stroke();

        // Right lane line
        ctx.beginPath();
        ctx.moveTo(w, h);
        ctx.lineTo(vpX + 30, vpY);
        ctx.stroke();

        // Center dashed line
        ctx.strokeStyle = '#f5a623';
        ctx.setLineDash([10, 10]);
        ctx.beginPath();
        ctx.moveTo(w / 2, h);
        ctx.lineTo(vpX, vpY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Detection boxes
        const detections = window.autopilot?.detections || [];
        detections.forEach(d => {
            if (d.type === 'car' && d.position) {
                const boxW = 60 - d.distance * 0.5;
                const boxH = 40 - d.distance * 0.3;
                const x = d.position.x * w - boxW / 2;
                const y = d.position.y * h - boxH / 2;

                ctx.strokeStyle = '#3e6ae1';
                ctx.lineWidth = 2;
                ctx.strokeRect(x, y, boxW, boxH);

                ctx.fillStyle = 'rgba(62, 106, 225, 0.2)';
                ctx.fillRect(x, y, boxW, boxH);

                ctx.fillStyle = '#fff';
                ctx.font = '10px sans-serif';
                ctx.fillText(`${Math.round(d.distance)}m`, x, y - 5);
            }
        });

        // Lane detection overlay
        ctx.strokeStyle = 'rgba(23, 176, 107, 0.5)';
        ctx.lineWidth = 3;

        // Left lane boundary
        ctx.beginPath();
        ctx.moveTo(w * 0.1, h);
        ctx.quadraticCurveTo(vpX - 50, h * 0.6, vpX - 30, vpY + 10);
        ctx.stroke();

        // Right lane boundary
        ctx.beginPath();
        ctx.moveTo(w * 0.9, h);
        ctx.quadraticCurveTo(vpX + 50, h * 0.6, vpX + 30, vpY + 10);
        ctx.stroke();

        // Neural network processing indicator
        ctx.fillStyle = 'rgba(23, 176, 107, 0.8)';
        ctx.font = '9px sans-serif';
        ctx.fillText('NN Processing...', 10, 15);

        // Frame counter
        const frameNum = Math.floor(this.time * 20) % 1000;
        ctx.fillStyle = '#666';
        ctx.fillText(`Frame: ${frameNum}`, w - 70, 15);
    }

    initDetectionList() {
        this.detectionList = document.getElementById('detection-list');
    }

    updateDetectionList(detections) {
        if (!this.detectionList) return;

        const icons = {
            car: '🚗',
            person: '🚶',
            sign: '🚦',
            lane: '🛣️'
        };

        this.detectionList.innerHTML = detections.map(d => `
            <div class="detection-item">
                <div class="detection-icon ${d.type}">${icons[d.type] || '❓'}</div>
                <div class="detection-info">
                    <div class="detection-label">${d.label}</div>
                    <div class="detection-detail">${d.distance ? d.distance.toFixed(0) + 'm' : ''}</div>
                </div>
                <div class="detection-confidence">${(d.confidence * 100).toFixed(0)}%</div>
            </div>
        `).join('');
    }

    initControls() {
        // Button controls
        document.getElementById('btn-autopilot')?.addEventListener('click', () => {
            document.querySelectorAll('.control-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('btn-autopilot').classList.add('active');
        });

        document.getElementById('btn-webcam')?.addEventListener('click', () => {
            document.querySelectorAll('.control-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('btn-webcam').classList.add('active');
            this.startWebcam();
        });

        document.getElementById('btn-demo')?.addEventListener('click', () => {
            document.querySelectorAll('.control-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('btn-demo').classList.add('active');
        });
    }

    async startWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            // Could feed to neural network for real inference
            console.log('Webcam started');
        } catch (error) {
            console.error('Webcam error:', error);
        }
    }

    updateUI(state) {
        // Update steering display
        const steeringFill = document.getElementById('steering-fill');
        const steeringValue = document.getElementById('steering-value');
        if (steeringFill && steeringValue) {
            const steeringPercent = (state.steering / 25 + 1) * 50;
            steeringFill.style.width = `${steeringPercent}%`;
            steeringValue.textContent = `${state.steering.toFixed(1)}°`;
        }

        // Update throttle display
        const throttleFill = document.getElementById('throttle-fill');
        const throttleValue = document.getElementById('throttle-value');
        if (throttleFill && throttleValue) {
            throttleFill.style.width = `${state.throttle * 100}%`;
            throttleValue.textContent = `${(state.throttle * 100).toFixed(0)}%`;
        }

        // Update brake display
        const brakeFill = document.getElementById('brake-fill');
        const brakeValue = document.getElementById('brake-value');
        if (brakeFill && brakeValue) {
            brakeFill.style.width = `${state.brake * 100}%`;
            brakeValue.textContent = `${(state.brake * 100).toFixed(0)}%`;
        }

        // Update speed
        const speedDisplay = document.getElementById('speed-display');
        if (speedDisplay) {
            speedDisplay.textContent = Math.round(state.speed);
        }

        // Update steering wheel
        const steeringWheel = document.getElementById('steering-wheel');
        if (steeringWheel) {
            steeringWheel.style.transform = `rotate(${state.steering * 2}deg)`;
        }

        // Update inference time
        const inferenceTime = document.getElementById('inference-time');
        if (inferenceTime) {
            inferenceTime.textContent = `${state.inferenceTime.toFixed(1)}ms`;
        }

        // Update detection list
        if (state.detections) {
            this.updateDetectionList(state.detections);
        }
    }

    update() {
        this.time += 0.016;

        // Get autopilot state
        const state = window.autopilot?.simulatePrediction() || {
            steering: 0,
            throttle: 0.3,
            brake: 0,
            speed: 65,
            inferenceTime: 12,
            detections: []
        };

        // Update camera position based on steering
        this.camera.position.x = state.steering * 0.05;
        this.camera.rotation.y = -state.steering * 0.01;

        // Animate other vehicles
        this.otherVehicles.forEach((v, i) => {
            // Relative motion
            v.mesh.position.z = v.baseZ + Math.sin(this.time * 0.5 + i) * 10;
            v.mesh.position.x = v.lane * 3.5 + Math.sin(this.time * 0.3 + i * 2) * 0.5;
        });

        // Animate lane lines (road movement illusion)
        this.laneLines.forEach((line, i) => {
            line.position.z = ((line.position.z + state.speed * 0.01) % 8) - 4;
        });

        // Update UI
        this.updateUI(state);
    }

    onResize() {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    animate() {
        this.update();
        this.drawCameraFeed();
        this.renderer.render(this.scene, this.camera);
        requestAnimationFrame(() => this.animate());
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.mainView = new MainView();
});
