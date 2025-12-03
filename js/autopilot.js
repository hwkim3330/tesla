/**
 * Tesla Autopilot - Steering Angle Prediction using Neural Network
 * Inspired by NVIDIA's End-to-End Learning for Self-Driving Cars
 * and akshaybahadur21/Autopilot
 */

class AutopilotNN {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.inputShape = [66, 200, 3]; // NVIDIA model input shape
        this.steeringAngle = 0;
        this.throttle = 0.3;
        this.brake = 0;
        this.inferenceTime = 0;

        // Simulation state
        this.simulationMode = true;
        this.time = 0;

        // Road parameters for simulation
        this.roadCurvature = 0;
        this.roadOffset = 0;

        // Vehicle state
        this.speed = 65;
        this.position = { x: 0, y: 0 };

        // Detection results
        this.detections = [];

        this.initModel();
    }

    async initModel() {
        try {
            // Create a simple CNN model for steering prediction
            // This mimics the NVIDIA architecture
            this.model = tf.sequential();

            // Normalization layer
            this.model.add(tf.layers.rescaling({
                scale: 1/127.5,
                offset: -1,
                inputShape: this.inputShape
            }));

            // Convolutional layers (NVIDIA architecture)
            this.model.add(tf.layers.conv2d({
                filters: 24, kernelSize: 5, strides: 2, activation: 'elu'
            }));
            this.model.add(tf.layers.conv2d({
                filters: 36, kernelSize: 5, strides: 2, activation: 'elu'
            }));
            this.model.add(tf.layers.conv2d({
                filters: 48, kernelSize: 5, strides: 2, activation: 'elu'
            }));
            this.model.add(tf.layers.conv2d({
                filters: 64, kernelSize: 3, activation: 'elu'
            }));
            this.model.add(tf.layers.conv2d({
                filters: 64, kernelSize: 3, activation: 'elu'
            }));

            // Flatten and dense layers
            this.model.add(tf.layers.flatten());
            this.model.add(tf.layers.dropout({ rate: 0.5 }));
            this.model.add(tf.layers.dense({ units: 100, activation: 'elu' }));
            this.model.add(tf.layers.dense({ units: 50, activation: 'elu' }));
            this.model.add(tf.layers.dense({ units: 10, activation: 'elu' }));
            this.model.add(tf.layers.dense({ units: 1 })); // Steering output

            this.model.compile({
                optimizer: 'adam',
                loss: 'meanSquaredError'
            });

            this.isModelLoaded = true;
            console.log('Autopilot Neural Network initialized');
            console.log('Model summary:');
            this.model.summary();

        } catch (error) {
            console.error('Failed to initialize model:', error);
            this.isModelLoaded = false;
        }
    }

    // Preprocess image for neural network
    preprocessImage(imageData) {
        return tf.tidy(() => {
            // Convert to tensor
            let tensor = tf.browser.fromPixels(imageData);

            // Crop to region of interest (bottom half where road is)
            const height = tensor.shape[0];
            const cropHeight = Math.floor(height * 0.4);
            tensor = tensor.slice([height - cropHeight, 0, 0], [cropHeight, tensor.shape[1], 3]);

            // Resize to model input shape
            tensor = tf.image.resizeBilinear(tensor, [this.inputShape[0], this.inputShape[1]]);

            // Add batch dimension
            tensor = tensor.expandDims(0);

            return tensor;
        });
    }

    // Run inference on image
    async predict(imageData) {
        if (!this.isModelLoaded || !imageData) {
            return this.simulatePrediction();
        }

        const startTime = performance.now();

        try {
            const input = this.preprocessImage(imageData);
            const prediction = this.model.predict(input);
            const steeringValue = (await prediction.data())[0];

            this.inferenceTime = performance.now() - startTime;
            this.steeringAngle = Math.max(-1, Math.min(1, steeringValue)) * 25; // Scale to degrees

            // Clean up tensors
            input.dispose();
            prediction.dispose();

        } catch (error) {
            console.error('Prediction error:', error);
            return this.simulatePrediction();
        }

        return {
            steering: this.steeringAngle,
            throttle: this.throttle,
            brake: this.brake,
            inferenceTime: this.inferenceTime
        };
    }

    // Simulate realistic steering prediction for demo
    simulatePrediction() {
        this.time += 0.016;

        // Simulate road curvature changes
        this.roadCurvature = Math.sin(this.time * 0.3) * 15 +
                           Math.sin(this.time * 0.7) * 8 +
                           Math.sin(this.time * 1.2) * 3;

        // Add some noise for realism
        const noise = (Math.random() - 0.5) * 2;

        // Steering follows road curvature with some delay (like real NN)
        this.steeringAngle = this.steeringAngle * 0.85 + (this.roadCurvature + noise) * 0.15;

        // Speed-dependent throttle/brake
        const targetSpeed = 65 + Math.sin(this.time * 0.2) * 10;
        if (this.speed < targetSpeed) {
            this.throttle = Math.min(0.6, (targetSpeed - this.speed) * 0.05);
            this.brake = 0;
        } else {
            this.throttle = 0.1;
            this.brake = Math.min(0.3, (this.speed - targetSpeed) * 0.03);
        }

        // Update speed
        this.speed += (this.throttle - this.brake) * 2;
        this.speed = Math.max(30, Math.min(120, this.speed));

        this.inferenceTime = 8 + Math.random() * 8; // 8-16ms simulated

        // Generate detections
        this.generateDetections();

        return {
            steering: this.steeringAngle,
            throttle: this.throttle,
            brake: this.brake,
            inferenceTime: this.inferenceTime,
            speed: this.speed,
            detections: this.detections
        };
    }

    generateDetections() {
        // Generate realistic detection data
        this.detections = [];

        // Lead vehicle
        if (Math.random() > 0.3) {
            this.detections.push({
                type: 'car',
                label: 'Lead Vehicle',
                distance: 25 + Math.random() * 40,
                confidence: 0.92 + Math.random() * 0.07,
                position: { x: 0.5, y: 0.4 },
                speed: this.speed - 5 + Math.random() * 10
            });
        }

        // Adjacent vehicles
        if (Math.random() > 0.5) {
            this.detections.push({
                type: 'car',
                label: 'Adjacent Vehicle',
                distance: 15 + Math.random() * 25,
                confidence: 0.88 + Math.random() * 0.1,
                position: { x: Math.random() > 0.5 ? 0.2 : 0.8, y: 0.45 }
            });
        }

        // Lane lines (always detected)
        this.detections.push({
            type: 'lane',
            label: 'Lane Markings',
            confidence: 0.95 + Math.random() * 0.04,
            laneOffset: (Math.random() - 0.5) * 0.3
        });

        // Traffic signs occasionally
        if (Math.random() > 0.7) {
            const signs = ['Speed Limit 60', 'Speed Limit 80', 'Yield', 'Stop Ahead'];
            this.detections.push({
                type: 'sign',
                label: signs[Math.floor(Math.random() * signs.length)],
                distance: 50 + Math.random() * 100,
                confidence: 0.85 + Math.random() * 0.12
            });
        }

        // Pedestrians rarely
        if (Math.random() > 0.9) {
            this.detections.push({
                type: 'person',
                label: 'Pedestrian',
                distance: 30 + Math.random() * 50,
                confidence: 0.78 + Math.random() * 0.15,
                position: { x: Math.random() > 0.5 ? 0.1 : 0.9, y: 0.5 }
            });
        }
    }

    getState() {
        return {
            steering: this.steeringAngle,
            throttle: this.throttle,
            brake: this.brake,
            speed: this.speed,
            inferenceTime: this.inferenceTime,
            detections: this.detections,
            roadCurvature: this.roadCurvature
        };
    }
}

// Global instance
window.autopilot = new AutopilotNN();
