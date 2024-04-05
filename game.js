//code by derby ;)

class FluidSimulation {
	constructor(gridSize, diffusion, viscosity, dt) {
		this.size = gridSize;
		this.viscosity = viscosity;
		this.diffusion = diffusion;
		this.dt = dt; // time step
		// Dye concentration
		this.dye = new Float32Array(this.size * this.size);
		// Current velocity fields
		this.velocityX = new Float32Array(this.size * this.size);
		this.velocityY = new Float32Array(this.size * this.size);

		// Velocity fields from the previous timestep
		this.velocityX0 = new Float32Array(this.size * this.size);
		this.velocityY0 = new Float32Array(this.size * this.size);

		// Pressure and divergence fields
		this.pressure = new Float32Array(this.size * this.size);
		this.divergence = new Float32Array(this.size * this.size);
	}
	addForce(x, y, forceX, forceY) {
		const index = x + y * this.size;
		this.velocityX[index] += forceX;
		this.velocityY[index] += forceY;
	}
	addDye(x, y, amount) {
		const index = x + y * this.size;
		this.dye[index] += amount;
	}

	updatePreviousVelocities() {
		this.velocityX0.set(this.velocityX);
		this.velocityY0.set(this.velocityY);
	}

	set_bnd(b, x, N) {
		for (let i = 1; i < N - 1; i++) {
			x[i] = b === 1 ? -x[i + N] : x[i + N]; // Bottom
			x[(N - 1) * N + i] = b === 1 ? -x[(N - 2) * N + i] : x[(N - 2) * N + i]; // Top
		}

		for (let j = 1; j < N - 1; j++) {
			x[j * N] = b === 2 ? -x[j * N + 1] : x[j * N + 1]; // Left
			x[j * N + N - 1] = b === 2 ? -x[j * N + N - 2] : x[j * N + N - 2]; // Right
		}

		// Corner cells average adjacent sides
		x[0] = 0.5 * (x[1] + x[N]);
		x[N - 1] = 0.5 * (x[N - 2] + x[2 * N - 1]);
		x[(N - 1) * N] = 0.5 * (x[(N - 2) * N] + x[(N - 1) * N + 1]);
		x[N * N - 1] = 0.5 * (x[N * N - 2] + x[(N - 1) * N - 1]);
	}

	// Step 1: Advection
	/*
    Explanation
    Variables: i0, i1, j0, j1 are the indices for the corners of the cell from which a particle is advected. s0, s1, t0, t1 are the interpolation weights.
    Looping Through Grid: We iterate through each cell in the grid, excluding the boundary cells (which are handled separately to enforce boundary conditions).
    Backward Tracing: For each cell, we compute where the fluid in the current cell came from by tracing backwards along the velocity field (tmp1, tmp2 are temporary variables for this purpose).
    Bounds Checking: We ensure that the traced-back position doesn't go outside the fluid domain.
    Interpolation: We linearly interpolate the value from the four surrounding cells (i0, i1, j0, j1) based on the traced-back position (x, y). This gives us the advected value for the cell.
    Boundary Conditions: set_bnd is a function you'd need to implement to handle boundary conditions appropriately, ensuring that fluid velocity behaves correctly at the edges of the simulation domain.


    */

	advect(b, d, d0, velocityX, velocityY, dt) {
		let i0, i1, j0, j1;
		const N = this.size;
		const dtx = dt * (this.size - 2);
		const dty = dt * (this.size - 2);

		let s0, s1, t0, t1;
		let tmp1, tmp2, x, y;

		let ifloat, jfloat;
		let i, j;

		for (j = 1, jfloat = 1; j < N - 1; j++, jfloat++) {
			for (i = 1, ifloat = 1; i < N - 1; i++, ifloat++) {
				// Calculate the backward trace position
				tmp1 = dtx * velocityX[j * N + i];
				tmp2 = dty * velocityY[j * N + i];
				x = ifloat - tmp1;
				y = jfloat - tmp2;

				// Ensure the backward trace stays within bounds
				if (x < 0.5) x = 0.5;
				if (x > N + 0.5) x = N + 0.5;
				i0 = Math.floor(x);
				i1 = i0 + 1;

				if (y < 0.5) y = 0.5;
				if (y > N + 0.5) y = N + 0.5;
				j0 = Math.floor(y);
				j1 = j0 + 1;

				// Linear interpolation coefficients
				s1 = x - i0;
				s0 = 1 - s1;
				t1 = y - j0;
				t0 = 1 - t1;

				// Interpolate the value at the new position
				d[j * N + i] =
					s0 * (t0 * d0[j0 * N + i0] + t1 * d0[j1 * N + i0]) +
					s1 * (t0 * d0[j0 * N + i1] + t1 * d0[j1 * N + i1]);
			}
		}

		this.set_bnd(b, d, N);
	}

	// Step 2: Diffusion (Viscosity)
	/*

    Notes on diffuse
    This method updates the x array, which represents the current state of the quantity being diffused (e.g., the x-component of velocity).
    x0 is the state of x at the previous time step, serving as the initial condition for the diffusion equation.
    The diffusion rate diff and the time step dt are used to calculate the coefficient a, which influences how strongly diffusion affects the quantity.
    The method iteratively updates each cell to the average of its neighbors, adjusted by the original state and the diffusion coefficient, using the Gauss-Seidel method.
    After updating all cells, boundary conditions are applied through set_bnd to ensure the simulation respects physical constraints at the domain's edges.

    */

	diffuse(b, x, x0, diff, dt) {
		let i, j;
		const a = dt * diff * (this.size - 2) * (this.size - 2);
		const N = this.size;

		// Perform iterations for Gauss-Seidel relaxation
		for (let k = 0; k < 20; k++) {
			for (j = 1; j < N - 1; j++) {
				for (i = 1; i < N - 1; i++) {
					x[j * N + i] =
						(x0[j * N + i] +
							a *
								(x[(j - 1) * N + i] +
									x[(j + 1) * N + i] +
									x[j * N + i - 1] +
									x[j * N + i + 1])) /
						(1 + 4 * a);
				}
			}
			this.set_bnd(b, x, N);
		}
	}

	// Step 3: Calculate Pressure
	/*

    Explanation
    Divergence Calculation: The first loop calculates the divergence of the velocity field. The divergence at each cell is approximated by the differences in velocity across neighboring cells, indicating how much fluid is diverging from or converging into the cell.
    Pressure Solving: After initializing the pressure field to zero and applying boundary conditions, a series of iterations (using a simple relaxation method) solves the Poisson equation for pressure based on the divergence field. This effectively spreads out the divergence in the velocity field across the pressure field.
    Applying Pressure Gradient: The final loop adjusts the velocity field by subtracting the pressure gradient. This step corrects velocities to ensure they collectively do not imply any compression or expansion of fluid—making the velocity field divergence-free.
    Boundary Conditions: set_bnd is used to apply appropriate boundary conditions to both the divergence, pressure, and corrected velocity fields, ensuring the simulation remains physically consistent.

    */

	project(velocityX, velocityY, p, div) {
		let i, j;
		const N = this.size;

		// Compute divergence of velocity field
		for (j = 1; j < N - 1; j++) {
			for (i = 1; i < N - 1; i++) {
				div[j * N + i] =
					(-0.5 *
						(velocityX[j * N + i + 1] -
							velocityX[j * N + i - 1] +
							velocityY[(j + 1) * N + i] -
							velocityY[(j - 1) * N + i])) /
					N;
				p[j * N + i] = 0;
			}
		}
		this.set_bnd(0, div, N); // Apply boundary conditions to divergence
		this.set_bnd(0, p, N); // Apply boundary conditions to pressure

		// Solve for pressure
		for (let k = 0; k < 20; k++) {
			for (j = 1; j < N - 1; j++) {
				for (i = 1; i < N - 1; i++) {
					p[j * N + i] =
						(div[j * N + i] +
							p[(j - 1) * N + i] +
							p[(j + 1) * N + i] +
							p[j * N + i - 1] +
							p[j * N + i + 1]) /
						4;
				}
			}
			this.set_bnd(0, p, N); // Apply boundary conditions to pressure
		}

		// Subtract pressure gradient from velocity field
		for (j = 1; j < N - 1; j++) {
			for (i = 1; i < N - 1; i++) {
				velocityX[j * N + i] -= 0.5 * (p[j * N + i + 1] - p[j * N + i - 1]) * N;
				velocityY[j * N + i] -=
					0.5 * (p[(j + 1) * N + i] - p[(j - 1) * N + i]) * N;
			}
		}
		this.set_bnd(1, velocityX, N); // Apply boundary conditions to velocityX
		this.set_bnd(2, velocityY, N); // Apply boundary conditions to velocityY
	}

	// Simulation step
	simulate() {
		// Example method to update 'previous step' velocities
		this.updatePreviousVelocities();
		this.diffuse(1, this.velocityX0, this.velocityX, this.viscosity, this.dt);
		this.diffuse(2, this.velocityY0, this.velocityY, this.viscosity, this.dt);

		this.project(
			this.velocityX0,
			this.velocityY0,
			this.velocityX,
			this.velocityY
		);
		// Advect dye
		const newDye = new Float32Array(this.size * this.size);
		this.advect(0, newDye, this.dye, this.velocityX, this.velocityY, this.dt);
		this.dye.set(newDye);
		this.advect(
			1,
			this.velocityX,
			this.velocityX0,
			this.velocityX0,
			this.velocityY0,
			this.dt
		);
		this.advect(
			2,
			this.velocityY,
			this.velocityY0,
			this.velocityX0,
			this.velocityY0,
			this.dt
		);

		this.project(
			this.velocityX,
			this.velocityY,
			this.pressure,
			this.divergence
		);
	}
}
class Particle {
	constructor(x, y) {
		this.x = x;
		this.y = y;
	}
}

class FluidGame {
	constructor(size, diffusion, viscosity, dt, particleCount) {
		this.simulation = new FluidSimulation(size, diffusion, viscosity, dt);
		this.size = size;
		this.outputElement = document.createElement("pre");
		this.lastRenderTime = 0;
		this.renderInterval = 10; // milliseconds
		this.frameId = null;
		this.particleCount = particleCount;

		this.initCanvas();
		this.initParticles(this.particleCount);
		this.initMouseHandling();
	}

	initParticles(particleCount) {
		this.particles = [];
		for (let i = 0; i < particleCount; i++) {
			const x = Math.random() * this.size;
			const y = Math.random() * this.size;
			this.particles.push(new Particle(x, y));
		}
	}
	reinitializeParticles() {
		this.initParticles(this.particleCount);
	}

	initCanvas() {
		this.canvas = document.getElementById("fluidCanvas");
		this.ctx = this.canvas.getContext("2d");
		this.canvas.width = 512; // Set canvas size
		this.canvas.height = 512;
		this.scale = this.canvas.width / this.size;
	}

	initMouseHandling() {
		document.addEventListener("mousedown", this.handleMouseDown.bind(this));
		document.addEventListener("mousemove", this.handleMouseMove.bind(this));
		document.addEventListener("mouseup", this.handleMouseUp.bind(this));

		this.mouseDown = false;
		this.lastMouseX = null;
		this.lastMouseY = null;
	}

	handleMouseDown(event) {
		this.mouseDown = true;
		this.addForceAndDye(event);
	}
	handleMouseUp(event) {
		this.mouseDown = false;
	}

	handleMouseMove(event) {
		const rect = this.canvas.getBoundingClientRect();
		const scaleX = this.size / rect.width;
		const scaleY = this.size / rect.height;
		this.mouseX = (event.clientX - rect.left) * scaleX;
		this.mouseY = (event.clientY - rect.top) * scaleY;

		if (this.mouseDown) {
			this.addForceAndDye(event);
		}
	}

	start() {
		this.visualizationMode = document.getElementById("visualizationMode").value;
		requestAnimationFrame(this.loop.bind(this));
	}
	advectParticles() {
		this.particles.forEach((particle) => {
			const x = Math.floor(particle.x);
			const y = Math.floor(particle.y);

			if (x >= 0 && x < this.size && y >= 0 && y < this.size) {
				const index = x + y * this.size;
				particle.x += this.simulation.velocityX[index];
				particle.y += this.simulation.velocityY[index];
			}

			// Simple wrapping boundary condition
			particle.x = (particle.x + this.size) % this.size;
			particle.y = (particle.y + this.size) % this.size;
		});
	}

	addForceAndDye(event) {
		const rect = this.canvas.getBoundingClientRect();
		const scaleX = this.size / rect.width;
		const scaleY = this.size / rect.height;
		const x = Math.floor((event.clientX - rect.left) * scaleX);
		const y = Math.floor((event.clientY - rect.top) * scaleY);
		const forceMagnitude = 5; // Adjust as needed
		const forceDirection = Math.random() * 2 * Math.PI;
		const forceX = Math.cos(forceDirection) * forceMagnitude;
		const forceY = Math.sin(forceDirection) * forceMagnitude;
		this.simulation.addForce(x, y, forceX, forceY);
		this.simulation.addDye(x, y, 100); // Adjust dye amount as needed
	}
	updateStats() {
		let totalPressure = 0,
			totalVelocity = 0;
		for (let i = 0; i < this.simulation.pressure.length; i++) {
			totalPressure += this.simulation.pressure[i];
			totalVelocity += Math.sqrt(
				Math.pow(this.simulation.velocityX[i], 2) +
					Math.pow(this.simulation.velocityY[i], 2)
			);
		}
		let avgPressure = (totalPressure / this.simulation.pressure.length) * 1000;
		let avgVelocity = totalVelocity / this.simulation.pressure.length;

		document.getElementById("avgPressure").textContent = avgPressure.toFixed(2);
		document.getElementById("avgVelocity").textContent = avgVelocity.toFixed(2);
	}

	loop = () => {
		this.simulation.simulate();
		this.advectParticles();
		this.render();
		this.updateStats(); // Update the stats in each animation frame
		this.frameId = requestAnimationFrame(this.loop);
	};
	stop() {
		if (this.frameId) {
			cancelAnimationFrame(this.frameId);
			this.frameId = null;

			// Optionally, clear the canvas or reset the simulation state
			this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
		}
	}
	render() {
		this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); // Clear the canvas first
		switch (this.visualizationMode) {
			case "dye":
				this.renderDyeConcentration();
				break;
			case "velocity":
				this.renderVelocityField();
				break;
			case "pressure":
				this.renderPressureField();
				break;
			case "flowLines":
				this.renderFlowFieldLines();
				break;
			case "particles":
				this.renderParticles();
				break;
			case "particlesAndVelocity": // New combined visualization mode
				this.renderVelocityField();
				this.renderParticles();
				break;
			case "particlesAndPressure": // New combined visualization mode
				this.renderPressureField();
				this.renderParticles();
				break;
			case "particlesAndPressureAndVelocity": // New combined visualization mode
				this.renderPressureField();
				this.renderVelocityField();
				this.renderParticles();

				break;
			case "combined":
				this.renderDyeConcentration();
				this.renderVelocityField();
				break;
		}
	}

	renderDyeConcentration() {
		// Render dye concentration
		for (let j = 0; j < this.size; j++) {
			for (let i = 0; i < this.size; i++) {
				const index = i + j * this.size;
				const concentration = this.normalizeConcentration(
					this.simulation.dye[index]
				);
				const color = this.getGradientColor(concentration);
				this.ctx.fillStyle = color;
				this.ctx.fillRect(
					i * this.scale,
					j * this.scale,
					this.scale,
					this.scale
				);
			}
		}
	}
	renderParticles() {
		this.ctx.fillStyle = "black"; // Particle color
		this.particles.forEach((particle) => {
			this.ctx.beginPath();
			this.ctx.arc(
				particle.x * this.scale,
				particle.y * this.scale,
				2,
				0,
				2 * Math.PI
			); // Adjust size as needed
			this.ctx.fill();
		});
	}

	renderVelocityField() {
		// Velocity visualization
		for (let j = 0; j < this.size; j++) {
			for (let i = 0; i < this.size; i++) {
				const index = i + j * this.size;
				const velocityX = this.simulation.velocityX[index];
				const velocityY = this.simulation.velocityY[index];
				// Draw velocity arrows
				const startX = i * this.scale + this.scale / 2;
				const startY = j * this.scale + this.scale / 2;
				const endX = startX + velocityX * 10; // Scale for visibility
				const endY = startY + velocityY * 10;
				this.drawArrow(this.ctx, startX, startY, endX, endY, "red");
			}
		}
	}
	// Example normalization function
	normalizeConcentration(value) {
		// Normalize based on your simulation's typical concentration range
		return Math.min(value / 255, 1);
	}
	// Map dye concentration to color
	getGradientColor(concentration) {
		// This creates a simple blue to red gradient for low to high concentration
		let red = Math.floor(255 * concentration);
		let blue = 255 - red;
		return `rgba(${red}, 0, ${blue}, 0.7)`;
	}
	renderPressureField() {
		let minPressure = Infinity;
		let maxPressure = -Infinity;

		// First pass to find min and max pressure values
		for (let i = 0; i < this.simulation.pressure.length; i++) {
			const pressure = this.simulation.pressure[i];
			if (pressure < minPressure) minPressure = pressure;
			if (pressure > maxPressure) maxPressure = pressure;
		}

		// Avoid division by zero in case minPressure equals maxPressure
		if (minPressure === maxPressure) {
			minPressure -= 1;
			maxPressure += 1;
		}

		// Second pass to render with normalized pressure values
		for (let j = 0; j < this.size; j++) {
			for (let i = 0; i < this.size; i++) {
				const index = i + j * this.size;
				const pressure = this.simulation.pressure[index];
				const normalizedPressure =
					(pressure - minPressure) / (maxPressure - minPressure);
				const color = this.pressureToColor(normalizedPressure); // Convert pressure to color
				this.ctx.fillStyle = color;
				this.ctx.fillRect(
					i * this.scale,
					j * this.scale,
					this.scale,
					this.scale
				);
			}
		}
	}
	pressureToColor(normalizedPressure) {
		// Map the normalized pressure to a 0-360 hue range (you can choose any range you like)
		let hue = (1 - normalizedPressure) * 240; // 0 (high pressure, red) to 240 (low pressure, blue)
		return `hsl(${hue}, 100%, 50%)`;
	}
	renderFlowFieldLines() {
		const step = Math.max(Math.floor(this.size / 20), 1); // Adjust the density of flow lines
		for (let j = 0; j < this.size; j += step) {
			for (let i = 0; i < this.size; i += step) {
				const index = i + j * this.size;
				const velocityX = this.simulation.velocityX[index];
				const velocityY = this.simulation.velocityY[index];
				const startX = i * this.scale + this.scale / 2;
				const startY = j * this.scale + this.scale / 2;
				const endX = startX + velocityX * 2; // Scale to control arrow length
				const endY = startY + velocityY * 2;
				this.drawArrow(this.ctx, startX, startY, endX, endY, "blue"); // Use a distinct color for flow lines
			}
		}
	}

	calculateVorticity() {
		let vorticity = new Float32Array(this.size * this.size);
		for (let j = 1; j < this.size - 1; j++) {
			for (let i = 1; i < this.size - 1; i++) {
				const index = i + j * this.size;
				const velocityXRight = this.simulation.velocityX[index + 1];
				const velocityXLeft = this.simulation.velocityX[index - 1];
				const velocityYUp = this.simulation.velocityY[index - this.size];
				const velocityYDown = this.simulation.velocityY[index + this.size];
				// Simple approximation of curl
				vorticity[index] =
					velocityYDown - velocityYUp - (velocityXRight - velocityXLeft);
			}
		}
		return vorticity;
	}

	// Helper function to draw an arrow representing velocity vector
	drawArrow(ctx, fromX, fromY, toX, toY) {
		ctx.beginPath();
		ctx.moveTo(fromX, fromY);
		ctx.lineTo(toX, toY);
		// Optional: add code to draw arrow head
		ctx.strokeStyle = "red"; // Velocity vector color
		ctx.stroke();
	}

	// Simple example function to map concentration to a color
	concentrationToColor(concentration) {
		// This is a simple example; you can implement more complex gradients
		const alpha = Math.min(concentration / 255, 1);
		return `rgba(${255 * alpha}, ${0}, ${(1 - alpha) * 255}, 1)`;
	}

	concentrationToChar(concentration) {
		// Convert dye concentration to an appropriate ASCII character
		if (concentration > 50) return "O"; // High concentration
		if (concentration > 20) return "+"; // Medium concentration
		if (concentration > 5) return "."; // Low concentration
		return " "; // No concentration
	}
	magnitudeToChar(magnitude) {
		// Convert magnitude to an appropriate ASCII character
		if (magnitude > 0.5) return "#";
		if (magnitude > 0.25) return "*";
		if (magnitude > 0.1) return ".";
		return " ";
	}

	// Implement methods to handle user input here
	// For example, adding forces based on mouse position or keyboard input
}
