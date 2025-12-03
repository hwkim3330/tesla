// Tesla FSD 3D Visualization
// Three.js + Mapbox GL JS Integration

class TeslaFSD3D {
    constructor() {
        // Mapbox token (public demo token)
        this.mapboxToken = 'pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NXVycTA2emYycXBndHRqcmZ3N3gifQ.rJcFIG214AriISLbB6B5aw';

        // Seoul Gangnam coordinates
        this.startLocation = {
            lng: 127.0276,
            lat: 37.4979,
            bearing: 0
        };

        // Simulation state
        this.time = 0;
        this.speed = 65;
        this.setSpeed = 70;
        this.currentBearing = 0;
        this.routeProgress = 0;

        // Traffic light state
        this.trafficLightState = 'green';
        this.trafficLightTimer = 12;
        this.trafficLightDistance = 85;

        // Detected objects
        this.vehicles = [];
        this.pedestrians = [];
        this.trafficLights = [];

        // Lead vehicle
        this.leadVehicle = {
            distance: 45,
            speed: 62,
            ttc: 12.5
        };

        // Three.js scenes
        this.mainScene = null;
        this.bevScene = null;
        this.mainCamera = null;
        this.bevCamera = null;
        this.mainRenderer = null;
        this.bevRenderer = null;

        // 3D Objects
        this.egoVehicle3D = null;
        this.otherVehicles3D = [];
        this.pedestrians3D = [];
        this.buildings3D = [];
        this.roadMesh = null;

        // View mode
        this.viewMode = '3d'; // '3d', 'top', 'follow'

        this.init();
    }

    async init() {
        await this.initMap();
        this.initThreeJS();
        this.initBEVScene();
        this.generateSimulatedObjects();
        this.setupEventListeners();
        this.startSimulation();

        // Hide loading screen
        setTimeout(() => {
            document.getElementById('loading-screen').classList.add('hidden');
        }, 2000);
    }

    async initMap() {
        mapboxgl.accessToken = this.mapboxToken;

        this.map = new mapboxgl.Map({
            container: 'map',
            style: 'mapbox://styles/mapbox/dark-v11',
            center: [this.startLocation.lng, this.startLocation.lat],
            zoom: 18,
            pitch: 60,
            bearing: this.startLocation.bearing,
            antialias: true
        });

        await new Promise(resolve => {
            this.map.on('load', () => {
                // Add 3D buildings
                this.map.addLayer({
                    'id': '3d-buildings',
                    'source': 'composite',
                    'source-layer': 'building',
                    'filter': ['==', 'extrude', 'true'],
                    'type': 'fill-extrusion',
                    'minzoom': 15,
                    'paint': {
                        'fill-extrusion-color': [
                            'interpolate',
                            ['linear'],
                            ['get', 'height'],
                            0, '#1a1a2e',
                            50, '#16213e',
                            100, '#0f3460',
                            200, '#1a1a40'
                        ],
                        'fill-extrusion-height': ['get', 'height'],
                        'fill-extrusion-base': ['get', 'min_height'],
                        'fill-extrusion-opacity': 0.8
                    }
                });

                // Add route line
                this.addRouteLine();

                resolve();
            });
        });
    }

    addRouteLine() {
        // Simulated route through Gangnam
        const routeCoords = [
            [127.0276, 37.4979],
            [127.0286, 37.4989],
            [127.0296, 37.4999],
            [127.0306, 37.5009],
            [127.0316, 37.5019],
            [127.0326, 37.5029],
            [127.0336, 37.5039]
        ];

        this.routeCoords = routeCoords;

        this.map.addSource('route', {
            'type': 'geojson',
            'data': {
                'type': 'Feature',
                'properties': {},
                'geometry': {
                    'type': 'LineString',
                    'coordinates': routeCoords
                }
            }
        });

        // Route glow effect
        this.map.addLayer({
            'id': 'route-glow',
            'type': 'line',
            'source': 'route',
            'layout': {
                'line-join': 'round',
                'line-cap': 'round'
            },
            'paint': {
                'line-color': '#17b06b',
                'line-width': 12,
                'line-opacity': 0.3,
                'line-blur': 3
            }
        });

        // Route line
        this.map.addLayer({
            'id': 'route-line',
            'type': 'line',
            'source': 'route',
            'layout': {
                'line-join': 'round',
                'line-cap': 'round'
            },
            'paint': {
                'line-color': '#17b06b',
                'line-width': 4,
                'line-opacity': 0.9
            }
        });
    }

    initThreeJS() {
        const canvas = document.getElementById('three-canvas');
        const container = canvas.parentElement;

        // Scene
        this.mainScene = new THREE.Scene();

        // Camera
        this.mainCamera = new THREE.PerspectiveCamera(
            60,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );
        this.mainCamera.position.set(0, 5, 10);
        this.mainCamera.lookAt(0, 0, 0);

        // Renderer
        this.mainRenderer = new THREE.WebGLRenderer({
            canvas: canvas,
            alpha: true,
            antialias: true
        });
        this.mainRenderer.setSize(container.clientWidth, container.clientHeight);
        this.mainRenderer.setPixelRatio(window.devicePixelRatio);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        this.mainScene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 20, 10);
        this.mainScene.add(directionalLight);

        // Create ego vehicle
        this.createEgoVehicle();

        // Handle resize
        window.addEventListener('resize', () => this.onResize());
    }

    initBEVScene() {
        const canvas = document.getElementById('bev-three-canvas');
        const container = canvas.parentElement;

        // Scene
        this.bevScene = new THREE.Scene();
        this.bevScene.background = new THREE.Color(0x0a0a0f);

        // Orthographic camera for top-down view
        const aspect = container.clientWidth / container.clientHeight;
        const viewSize = 100;
        this.bevCamera = new THREE.OrthographicCamera(
            -viewSize * aspect / 2,
            viewSize * aspect / 2,
            viewSize / 2,
            -viewSize / 2,
            0.1,
            1000
        );
        this.bevCamera.position.set(0, 100, 0);
        this.bevCamera.lookAt(0, 0, 0);

        // Renderer
        this.bevRenderer = new THREE.WebGLRenderer({
            canvas: canvas,
            antialias: true
        });
        this.bevRenderer.setSize(container.clientWidth, container.clientHeight);
        this.bevRenderer.setPixelRatio(window.devicePixelRatio);

        // Grid
        this.createBEVGrid();

        // Road lanes
        this.createBEVRoad();

        // Ego vehicle
        this.createBEVEgoVehicle();

        // Predicted path
        this.createBEVPredictedPath();
    }

    createEgoVehicle() {
        const group = new THREE.Group();

        // Car body
        const bodyGeom = new THREE.BoxGeometry(2, 0.8, 4.5);
        const bodyMat = new THREE.MeshPhongMaterial({
            color: 0x17b06b,
            shininess: 100
        });
        const body = new THREE.Mesh(bodyGeom, bodyMat);
        body.position.y = 0.6;
        group.add(body);

        // Roof
        const roofGeom = new THREE.BoxGeometry(1.6, 0.6, 2);
        const roof = new THREE.Mesh(roofGeom, bodyMat);
        roof.position.set(0, 1.2, -0.3);
        group.add(roof);

        // Windshield (dark glass)
        const glassGeom = new THREE.BoxGeometry(1.5, 0.5, 0.1);
        const glassMat = new THREE.MeshPhongMaterial({
            color: 0x1a1a2e,
            transparent: true,
            opacity: 0.8
        });
        const windshield = new THREE.Mesh(glassGeom, glassMat);
        windshield.position.set(0, 1.1, 0.75);
        windshield.rotation.x = Math.PI * 0.15;
        group.add(windshield);

        // Wheels
        const wheelGeom = new THREE.CylinderGeometry(0.35, 0.35, 0.2, 16);
        const wheelMat = new THREE.MeshPhongMaterial({ color: 0x333333 });

        const wheelPositions = [
            [-0.9, 0.35, 1.3],
            [0.9, 0.35, 1.3],
            [-0.9, 0.35, -1.3],
            [0.9, 0.35, -1.3]
        ];

        wheelPositions.forEach(pos => {
            const wheel = new THREE.Mesh(wheelGeom, wheelMat);
            wheel.rotation.z = Math.PI / 2;
            wheel.position.set(...pos);
            group.add(wheel);
        });

        // Headlights
        const headlightGeom = new THREE.BoxGeometry(0.3, 0.15, 0.05);
        const headlightMat = new THREE.MeshBasicMaterial({ color: 0xffffcc });

        const leftHeadlight = new THREE.Mesh(headlightGeom, headlightMat);
        leftHeadlight.position.set(-0.6, 0.5, 2.28);
        group.add(leftHeadlight);

        const rightHeadlight = new THREE.Mesh(headlightGeom, headlightMat);
        rightHeadlight.position.set(0.6, 0.5, 2.28);
        group.add(rightHeadlight);

        // Tesla glow effect
        const glowMat = new THREE.MeshBasicMaterial({
            color: 0x17b06b,
            transparent: true,
            opacity: 0.3
        });
        const glowGeom = new THREE.BoxGeometry(2.2, 0.9, 4.7);
        const glow = new THREE.Mesh(glowGeom, glowMat);
        glow.position.y = 0.6;
        group.add(glow);

        this.egoVehicle3D = group;
        this.mainScene.add(group);
    }

    createBEVGrid() {
        const gridHelper = new THREE.GridHelper(200, 20, 0x1a3a5c, 0x0d1f2d);
        gridHelper.rotation.x = 0;
        this.bevScene.add(gridHelper);

        // Distance markers
        const markerMat = new THREE.LineBasicMaterial({ color: 0x3e6ae1, transparent: true, opacity: 0.5 });

        for (let d = 20; d <= 100; d += 20) {
            const points = [];
            points.push(new THREE.Vector3(-50, 0.1, -d + 50));
            points.push(new THREE.Vector3(50, 0.1, -d + 50));

            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, markerMat);
            this.bevScene.add(line);
        }
    }

    createBEVRoad() {
        // Road surface
        const roadGeom = new THREE.PlaneGeometry(20, 200);
        const roadMat = new THREE.MeshBasicMaterial({
            color: 0x1a1a20,
            transparent: true,
            opacity: 0.8
        });
        const road = new THREE.Mesh(roadGeom, roadMat);
        road.rotation.x = -Math.PI / 2;
        road.position.y = 0.05;
        this.bevScene.add(road);

        // Lane markings
        const laneMat = new THREE.MeshBasicMaterial({ color: 0x3e6ae1 });

        // Left lane
        const leftLaneGeom = new THREE.PlaneGeometry(0.3, 200);
        const leftLane = new THREE.Mesh(leftLaneGeom, laneMat);
        leftLane.rotation.x = -Math.PI / 2;
        leftLane.position.set(-8, 0.1, 0);
        this.bevScene.add(leftLane);

        // Center lanes (dashed)
        for (let i = -95; i < 100; i += 10) {
            const dashGeom = new THREE.PlaneGeometry(0.2, 4);
            const dash = new THREE.Mesh(dashGeom, laneMat);
            dash.rotation.x = -Math.PI / 2;
            dash.position.set(-3, 0.1, i);
            this.bevScene.add(dash);

            const dash2 = dash.clone();
            dash2.position.set(3, 0.1, i);
            this.bevScene.add(dash2);
        }

        // Right lane
        const rightLane = leftLane.clone();
        rightLane.position.set(8, 0.1, 0);
        this.bevScene.add(rightLane);
    }

    createBEVEgoVehicle() {
        const group = new THREE.Group();

        // Vehicle body
        const bodyGeom = new THREE.BoxGeometry(3, 0.5, 5);
        const bodyMat = new THREE.MeshBasicMaterial({ color: 0x17b06b });
        const body = new THREE.Mesh(bodyGeom, bodyMat);
        body.position.y = 0.5;
        group.add(body);

        // Direction arrow
        const arrowShape = new THREE.Shape();
        arrowShape.moveTo(0, 1.5);
        arrowShape.lineTo(-1, 0);
        arrowShape.lineTo(1, 0);
        arrowShape.lineTo(0, 1.5);

        const arrowGeom = new THREE.ShapeGeometry(arrowShape);
        const arrowMat = new THREE.MeshBasicMaterial({ color: 0x17b06b, side: THREE.DoubleSide });
        const arrow = new THREE.Mesh(arrowGeom, arrowMat);
        arrow.rotation.x = -Math.PI / 2;
        arrow.position.set(0, 1.1, -4);
        group.add(arrow);

        // Glow effect
        const glowMat = new THREE.MeshBasicMaterial({
            color: 0x17b06b,
            transparent: true,
            opacity: 0.3
        });
        const glowGeom = new THREE.BoxGeometry(4, 0.6, 6);
        const glow = new THREE.Mesh(glowGeom, glowMat);
        glow.position.y = 0.5;
        group.add(glow);

        group.position.set(0, 0, 40);
        this.bevEgoVehicle = group;
        this.bevScene.add(group);
    }

    createBEVPredictedPath() {
        const pathShape = new THREE.Shape();
        pathShape.moveTo(-2, 40);
        pathShape.lineTo(2, 40);
        pathShape.lineTo(1.5, -50);
        pathShape.lineTo(-1.5, -50);
        pathShape.lineTo(-2, 40);

        const pathGeom = new THREE.ShapeGeometry(pathShape);
        const pathMat = new THREE.MeshBasicMaterial({
            color: 0x17b06b,
            transparent: true,
            opacity: 0.2,
            side: THREE.DoubleSide
        });

        const path = new THREE.Mesh(pathGeom, pathMat);
        path.rotation.x = -Math.PI / 2;
        path.position.y = 0.08;

        this.bevPredictedPath = path;
        this.bevScene.add(path);
    }

    createVehicle3D(type = 'sedan', color = 0x3e6ae1) {
        const group = new THREE.Group();

        let width = 2;
        let height = 1;
        let length = 4;

        if (type === 'truck') {
            width = 2.4;
            height = 1.5;
            length = 6;
        } else if (type === 'suv') {
            width = 2.2;
            height = 1.3;
            length = 4.5;
        }

        // Body
        const bodyGeom = new THREE.BoxGeometry(width, height * 0.6, length);
        const bodyMat = new THREE.MeshPhongMaterial({ color: color });
        const body = new THREE.Mesh(bodyGeom, bodyMat);
        body.position.y = height * 0.3;
        group.add(body);

        // Roof
        if (type !== 'truck') {
            const roofGeom = new THREE.BoxGeometry(width * 0.8, height * 0.4, length * 0.5);
            const roof = new THREE.Mesh(roofGeom, bodyMat);
            roof.position.set(0, height * 0.7, -length * 0.1);
            group.add(roof);
        }

        // Glow
        const glowMat = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.2
        });
        const glowGeom = new THREE.BoxGeometry(width + 0.2, height * 0.7, length + 0.2);
        const glow = new THREE.Mesh(glowGeom, glowMat);
        glow.position.y = height * 0.3;
        group.add(glow);

        return group;
    }

    createPedestrian3D() {
        const group = new THREE.Group();

        // Body (cylinder)
        const bodyGeom = new THREE.CylinderGeometry(0.3, 0.3, 1.2, 8);
        const bodyMat = new THREE.MeshPhongMaterial({ color: 0xf5a623 });
        const body = new THREE.Mesh(bodyGeom, bodyMat);
        body.position.y = 0.8;
        group.add(body);

        // Head (sphere)
        const headGeom = new THREE.SphereGeometry(0.25, 8, 8);
        const head = new THREE.Mesh(headGeom, bodyMat);
        head.position.y = 1.65;
        group.add(head);

        // Glow
        const glowMat = new THREE.MeshBasicMaterial({
            color: 0xf5a623,
            transparent: true,
            opacity: 0.3
        });
        const glowGeom = new THREE.SphereGeometry(0.6, 8, 8);
        const glow = new THREE.Mesh(glowGeom, glowMat);
        glow.position.y = 1;
        group.add(glow);

        return group;
    }

    createTrafficLight3D(state = 'green') {
        const group = new THREE.Group();

        // Pole
        const poleGeom = new THREE.CylinderGeometry(0.1, 0.1, 4, 8);
        const poleMat = new THREE.MeshPhongMaterial({ color: 0x333333 });
        const pole = new THREE.Mesh(poleGeom, poleMat);
        pole.position.y = 2;
        group.add(pole);

        // Box
        const boxGeom = new THREE.BoxGeometry(0.6, 1.5, 0.4);
        const boxMat = new THREE.MeshPhongMaterial({ color: 0x1a1a1a });
        const box = new THREE.Mesh(boxGeom, boxMat);
        box.position.y = 4.5;
        group.add(box);

        // Lights
        const lightGeom = new THREE.SphereGeometry(0.15, 16, 16);

        const colors = {
            red: state === 'red' ? 0xff0000 : 0x330000,
            yellow: state === 'yellow' ? 0xffff00 : 0x333300,
            green: state === 'green' ? 0x00ff00 : 0x003300
        };

        const redLight = new THREE.Mesh(lightGeom, new THREE.MeshBasicMaterial({ color: colors.red }));
        redLight.position.set(0, 4.9, 0.22);
        group.add(redLight);

        const yellowLight = new THREE.Mesh(lightGeom, new THREE.MeshBasicMaterial({ color: colors.yellow }));
        yellowLight.position.set(0, 4.5, 0.22);
        group.add(yellowLight);

        const greenLight = new THREE.Mesh(lightGeom, new THREE.MeshBasicMaterial({ color: colors.green }));
        greenLight.position.set(0, 4.1, 0.22);
        group.add(greenLight);

        group.userData = { state, redLight, yellowLight, greenLight };

        return group;
    }

    generateSimulatedObjects() {
        // Generate vehicles
        this.vehicles = [
            { id: 1, x: 0, z: 30, type: 'sedan', distance: 45 },
            { id: 2, x: 5, z: 50, type: 'suv', distance: 72 },
            { id: 3, x: -5, z: 25, type: 'truck', distance: 35 },
            { id: 4, x: -8, z: 60, type: 'sedan', distance: 85 }
        ];

        // Create 3D vehicles for BEV
        this.vehicles.forEach(v => {
            const vehicle3D = this.createVehicle3D(v.type, 0x3e6ae1);
            vehicle3D.position.set(v.x, 0, 40 - v.z);
            vehicle3D.scale.set(0.6, 0.6, 0.6);
            this.bevScene.add(vehicle3D);
            this.otherVehicles3D.push({ mesh: vehicle3D, data: v });
        });

        // Generate pedestrians
        this.pedestrians = [
            { id: 1, x: 12, z: 35, distance: 28 },
            { id: 2, x: -15, z: 45, distance: 40 }
        ];

        this.pedestrians.forEach(p => {
            const ped3D = this.createPedestrian3D();
            ped3D.position.set(p.x, 0, 40 - p.z);
            ped3D.scale.set(0.8, 0.8, 0.8);
            this.bevScene.add(ped3D);
            this.pedestrians3D.push({ mesh: ped3D, data: p });
        });

        // Generate traffic lights
        this.trafficLights = [
            { id: 1, x: 8, z: 70, state: 'green', distance: 85, timer: 12 }
        ];

        this.trafficLights.forEach(tl => {
            const tl3D = this.createTrafficLight3D(tl.state);
            tl3D.position.set(tl.x, 0, 40 - tl.z);
            tl3D.scale.set(0.5, 0.5, 0.5);
            this.bevScene.add(tl3D);
            this.trafficLights3D = this.trafficLights3D || [];
            this.trafficLights3D.push({ mesh: tl3D, data: tl });
        });

        this.updateDetectionUI();
    }

    setupEventListeners() {
        // View mode buttons
        document.getElementById('btn-3d').addEventListener('click', () => this.setViewMode('3d'));
        document.getElementById('btn-top').addEventListener('click', () => this.setViewMode('top'));
        document.getElementById('btn-follow').addEventListener('click', () => this.setViewMode('follow'));

        // BEV toggle buttons
        document.querySelectorAll('.toggle-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
            });
        });
    }

    setViewMode(mode) {
        this.viewMode = mode;

        // Update button states
        document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById(`btn-${mode}`).classList.add('active');

        // Update map view
        switch (mode) {
            case '3d':
                this.map.easeTo({
                    pitch: 60,
                    zoom: 18,
                    duration: 1000
                });
                break;
            case 'top':
                this.map.easeTo({
                    pitch: 0,
                    zoom: 17,
                    duration: 1000
                });
                break;
            case 'follow':
                this.map.easeTo({
                    pitch: 75,
                    zoom: 19,
                    duration: 1000
                });
                break;
        }
    }

    updateSimulation() {
        this.time += 0.016; // ~60fps

        // Update speed
        this.speed = 65 + Math.sin(this.time * 0.5) * 5;
        document.getElementById('current-speed').textContent = Math.round(this.speed);

        // Update bearing and move along route
        this.routeProgress += 0.0001;
        if (this.routeProgress > 1) this.routeProgress = 0;

        // Calculate position along route
        if (this.routeCoords && this.routeCoords.length > 1) {
            const totalPoints = this.routeCoords.length - 1;
            const currentIndex = Math.floor(this.routeProgress * totalPoints);
            const nextIndex = Math.min(currentIndex + 1, totalPoints);
            const t = (this.routeProgress * totalPoints) % 1;

            const currentPoint = this.routeCoords[currentIndex];
            const nextPoint = this.routeCoords[nextIndex];

            const lng = currentPoint[0] + (nextPoint[0] - currentPoint[0]) * t;
            const lat = currentPoint[1] + (nextPoint[1] - currentPoint[1]) * t;

            // Calculate bearing
            const dLng = nextPoint[0] - currentPoint[0];
            const dLat = nextPoint[1] - currentPoint[1];
            this.currentBearing = Math.atan2(dLng, dLat) * 180 / Math.PI;

            // Update map position smoothly
            this.map.easeTo({
                center: [lng, lat],
                bearing: this.currentBearing,
                duration: 100,
                easing: t => t
            });
        }

        // Update lead vehicle
        this.leadVehicle.distance = 45 + Math.sin(this.time * 0.3) * 8;
        this.leadVehicle.speed = 62 + Math.sin(this.time * 0.4) * 5;
        this.leadVehicle.ttc = this.leadVehicle.distance / Math.abs(this.speed - this.leadVehicle.speed + 0.1);

        document.getElementById('lead-distance').textContent = `${Math.round(this.leadVehicle.distance)}m`;
        document.getElementById('lead-speed').textContent = `${Math.round(this.leadVehicle.speed)} km/h`;
        document.getElementById('ttc').textContent = `${this.leadVehicle.ttc.toFixed(1)}s`;

        // Update traffic light
        this.trafficLightTimer -= 0.016;
        if (this.trafficLightTimer <= 0) {
            const states = ['red', 'yellow', 'green'];
            const currentIndex = states.indexOf(this.trafficLightState);
            this.trafficLightState = states[(currentIndex + 1) % 3];
            this.trafficLightTimer = this.trafficLightState === 'yellow' ? 3 : 10 + Math.random() * 5;

            // Update traffic light UI
            this.updateTrafficLightUI();
        }

        this.trafficLightDistance = 85 - (this.time % 85);
        if (this.trafficLightDistance < 5) this.trafficLightDistance = 85;

        document.getElementById('tl-distance').textContent = `${Math.round(this.trafficLightDistance)}m`;
        document.getElementById('tl-timer').textContent = `${Math.round(this.trafficLightTimer)}s remaining`;

        const lightColors = { red: 'RED', yellow: 'YLW', green: 'GRN' };
        const nextLightEl = document.getElementById('next-light');
        nextLightEl.textContent = `${lightColors[this.trafficLightState]} ${Math.round(this.trafficLightTimer)}s`;
        nextLightEl.className = `info-value ${this.trafficLightState}`;

        // Update neural network activity
        document.getElementById('nn-vision').style.width = `${75 + Math.sin(this.time) * 15}%`;
        document.getElementById('nn-planning').style.width = `${70 + Math.sin(this.time * 1.2) * 20}%`;
        document.getElementById('nn-control').style.width = `${85 + Math.sin(this.time * 0.8) * 10}%`;
        document.getElementById('nn-prediction').style.width = `${65 + Math.sin(this.time * 1.5) * 25}%`;

        // Update route info
        const remainingDist = 5.2 - (this.routeProgress * 5.2);
        const eta = Math.ceil(remainingDist / (this.speed / 60));
        document.getElementById('route-distance').textContent = `${remainingDist.toFixed(1)} km remaining`;
        document.getElementById('route-eta').textContent = `${eta} min`;

        // Update objects positions
        this.updateObjects();

        // Render 3D scenes
        this.render();
    }

    updateTrafficLightUI() {
        // Update HUD traffic light
        document.getElementById('tl-red').classList.toggle('active', this.trafficLightState === 'red');
        document.getElementById('tl-yellow').classList.toggle('active', this.trafficLightState === 'yellow');
        document.getElementById('tl-green').classList.toggle('active', this.trafficLightState === 'green');

        // Update 3D traffic lights
        if (this.trafficLights3D) {
            this.trafficLights3D.forEach(tl => {
                const colors = {
                    red: this.trafficLightState === 'red' ? 0xff0000 : 0x330000,
                    yellow: this.trafficLightState === 'yellow' ? 0xffff00 : 0x333300,
                    green: this.trafficLightState === 'green' ? 0x00ff00 : 0x003300
                };

                tl.mesh.userData.redLight.material.color.setHex(colors.red);
                tl.mesh.userData.yellowLight.material.color.setHex(colors.yellow);
                tl.mesh.userData.greenLight.material.color.setHex(colors.green);
            });
        }
    }

    updateObjects() {
        // Update vehicle positions
        this.otherVehicles3D.forEach((v, i) => {
            const data = this.vehicles[i];
            data.z = data.z + Math.sin(this.time + i) * 0.1;
            data.distance = 30 + i * 15 + Math.sin(this.time * 0.5 + i) * 5;

            v.mesh.position.z = 40 - data.z + Math.sin(this.time + i) * 2;
            v.mesh.position.x = data.x + Math.sin(this.time * 0.3 + i) * 0.5;
        });

        // Update pedestrian positions
        this.pedestrians3D.forEach((p, i) => {
            const data = this.pedestrians[i];
            p.mesh.position.x = data.x + Math.sin(this.time * 0.5 + i * 2) * 2;
        });
    }

    updateDetectionUI() {
        document.getElementById('vehicle-count').textContent = this.vehicles.length;
        document.getElementById('pedestrian-count').textContent = this.pedestrians.length;
        document.getElementById('traffic-light-count').textContent = this.trafficLights.length;
    }

    render() {
        // Render main scene (overlay on map)
        if (this.mainRenderer && this.mainScene && this.mainCamera) {
            // Rotate ego vehicle based on simulation
            if (this.egoVehicle3D) {
                this.egoVehicle3D.rotation.y = Math.sin(this.time * 0.2) * 0.05;
            }
            this.mainRenderer.render(this.mainScene, this.mainCamera);
        }

        // Render BEV scene
        if (this.bevRenderer && this.bevScene && this.bevCamera) {
            // Animate BEV ego vehicle
            if (this.bevEgoVehicle) {
                this.bevEgoVehicle.rotation.y = Math.sin(this.time * 0.3) * 0.03;
            }

            // Pulse predicted path
            if (this.bevPredictedPath) {
                this.bevPredictedPath.material.opacity = 0.15 + Math.sin(this.time * 2) * 0.1;
            }

            this.bevRenderer.render(this.bevScene, this.bevCamera);
        }
    }

    onResize() {
        // Resize main canvas
        const mainCanvas = document.getElementById('three-canvas');
        const mainContainer = mainCanvas.parentElement;

        this.mainCamera.aspect = mainContainer.clientWidth / mainContainer.clientHeight;
        this.mainCamera.updateProjectionMatrix();
        this.mainRenderer.setSize(mainContainer.clientWidth, mainContainer.clientHeight);

        // Resize BEV canvas
        const bevCanvas = document.getElementById('bev-three-canvas');
        const bevContainer = bevCanvas.parentElement;

        const aspect = bevContainer.clientWidth / bevContainer.clientHeight;
        const viewSize = 100;
        this.bevCamera.left = -viewSize * aspect / 2;
        this.bevCamera.right = viewSize * aspect / 2;
        this.bevCamera.updateProjectionMatrix();
        this.bevRenderer.setSize(bevContainer.clientWidth, bevContainer.clientHeight);

        // Resize map
        this.map.resize();
    }

    startSimulation() {
        const animate = () => {
            this.updateSimulation();
            requestAnimationFrame(animate);
        };
        animate();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.teslaFSD = new TeslaFSD3D();
});
