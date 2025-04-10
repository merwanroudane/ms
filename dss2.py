import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from scipy.stats import norm, binom, poisson, expon, uniform, beta, probplot
import time
from io import BytesIO

st.set_page_config(layout="wide", page_title="Visualizing Statistical Concepts with Animations")

# English LTR support (Adjusted from RTL)
st.markdown(
	"""
	<style>
	.ltr {
		direction: ltr;
		text-align: left;
		font-family: 'Arial', sans-serif;
	}
	.stTabs [data-baseweb="tab-list"] {
		gap: 24px;
	}
	.stTabs [data-baseweb="tab"] {
		height: 50px;
		white-space: pre-wrap;
		font-size: 16px;
		font-weight: 500;
		direction: ltr; /* Changed from rtl */
	}
	.highlight {
		background-color: #f0f8ff;
		padding: 10px;
		border-radius: 5px;
		border-left: 4px solid #4682b4; /* Changed from border-right */
		margin: 10px 0;
	}
	</style>
	""", unsafe_allow_html=True
)

st.markdown('<h1>Interactive Statistical Distributions Library</h1>', unsafe_allow_html=True)
st.markdown(
	'<div>An interactive tool to understand statistical distributions and important theorems like the Central Limit Theorem and the Law of Large Numbers</div>',
	unsafe_allow_html=True)

tabs = st.tabs(["Central Limit Theorem", "Law of Large Numbers", "Discrete Probability Distributions",
				"Continuous Probability Distributions"])

with tabs[0]:
	st.markdown('<h2>Central Limit Theorem (CLT)</h2>', unsafe_allow_html=True)

	st.markdown("""
    <div class="highlight">
    <strong>Central Limit Theorem:</strong> States that when drawing sufficient samples from any distribution with a defined mean and variance,
    the distribution of the sample means will approximate a normal distribution, regardless of the original distribution's shape.
    </div>
    """, unsafe_allow_html=True)

	st.markdown('<h3>CLT Simulation</h3>', unsafe_allow_html=True)

	col1, col2 = st.columns([1, 1])

	with col1:
		st.markdown('<div>Select the original distribution type:</div>', unsafe_allow_html=True)
		source_dist = st.selectbox(
			"Source Distribution",
			["Exponential", "Uniform", "Binomial"],
			index=0,
			label_visibility="collapsed"
		)

		st.markdown('<div>Sample Size:</div>', unsafe_allow_html=True)
		sample_size = st.slider("Sample Size", min_value=1, max_value=100, value=30, step=1,
								label_visibility="collapsed")

		st.markdown('<div>Number of Samples:</div>', unsafe_allow_html=True)
		num_samples = st.slider("Number of Samples", min_value=100, max_value=5000, value=2000, step=100,
								label_visibility="collapsed")

	with col2:
		st.markdown('<div>Information about the Central Limit Theorem:</div>', unsafe_allow_html=True)
		st.markdown("""
        <div>
        <ul>
          <li>As the sample size increases, the sampling distribution approaches the normal distribution more closely.</li>
          <li>Even with non-normal original distributions, the theorem's effect is clearly visible.</li>
          <li>Understanding this theorem helps in constructing confidence intervals and performing hypothesis tests.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

	if st.button("Run Simulation", key="clt_sim"):
		# Generate data based on selected distribution
		if source_dist == "Exponential":
			data = np.random.exponential(1, size=num_samples * sample_size)
			dist_title = "Exponential Distribution"
		elif source_dist == "Uniform":
			data = np.random.uniform(0, 1, size=num_samples * sample_size)
			dist_title = "Uniform Distribution"
		else:  # Binomial
			data = np.random.binomial(10, 0.5, size=num_samples * sample_size)
			dist_title = "Binomial Distribution"

		# Reshape data to compute sample means
		data_reshaped = data.reshape(num_samples, sample_size)
		sample_means = np.mean(data_reshaped, axis=1)

		# Create animation frames
		frames = []
		progress = st.progress(0)

		# For plotting
		bins = 30
		hist_color = 'skyblue'
		curve_color = 'red'

		N_steps = min(100, num_samples)  # Number of frames to show
		step_size = max(1, num_samples // N_steps)

		# Create animated plot with Plotly
		sample_means_frames = []
		for i in range(0, num_samples, step_size):
			current_means = sample_means[:i + 1]
			if len(current_means) > 1:
				mu = np.mean(current_means)
				sigma = np.std(current_means)

				# Histogram data
				counts, bin_edges = np.histogram(current_means, bins=bins, density=True)
				bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

				# Normal curve data
				x_curve = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
				y_curve = norm.pdf(x_curve, mu, sigma)

				sample_means_frames.append(
					go.Frame(
						data=[
							go.Bar(x=bin_centers, y=counts, marker_color=hist_color, name='Histogram'),
							go.Scatter(x=x_curve, y=y_curve, mode='lines', line=dict(color=curve_color),
									   name='Normal Distribution')
						],
						layout=go.Layout(
							title_text=f"Number of Samples: {i + 1}"
						)
					)
				)

			progress.progress((i + step_size) / num_samples)

		# Initial frame
		if len(sample_means) > 0:
			initial_means = sample_means[:1]
			initial_mu = np.mean(initial_means)
			initial_sigma = 0.1  # Initial arbitrary sigma

			counts_init, bin_edges_init = np.histogram(initial_means, bins=bins, density=True)
			bin_centers_init = (bin_edges_init[:-1] + bin_edges_init[1:]) / 2

			x_curve_init = np.linspace(initial_mu - 4 * initial_sigma, initial_mu + 4 * initial_sigma, 100)
			y_curve_init = norm.pdf(x_curve_init, initial_mu, initial_sigma)

			# Create figure
			fig = go.Figure(
				data=[
					go.Bar(x=bin_centers_init, y=counts_init, marker_color=hist_color, name='Histogram'),
					go.Scatter(x=x_curve_init, y=y_curve_init, mode='lines', line=dict(color=curve_color),
							   name='Normal Distribution')
				],
				layout=go.Layout(
					title=f"CLT Simulation: {dist_title}",
					xaxis=dict(title="Sample Mean"),
					yaxis=dict(title="Density"),
					updatemenus=[{
						"type": "buttons",
						"buttons": [{
							"label": "Play",
							"method": "animate",
							"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]
						}]
					}]
				),
				frames=sample_means_frames
			)

			st.plotly_chart(fig, use_container_width=True)

			# Final distribution comparison
			st.markdown('<h3>Final Distribution vs. Normal Distribution</h3>', unsafe_allow_html=True)

			fig2, ax = plt.subplots(figsize=(10, 6))
			ax.hist(sample_means, bins=bins, density=True, alpha=0.7, color=hist_color)

			# Fit normal distribution
			mu = np.mean(sample_means)
			sigma = np.std(sample_means)
			x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
			ax.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal Distribution')

			ax.set_title('Final Distribution of Sample Means', fontsize=14)
			ax.set_xlabel('Sample Mean')
			ax.set_ylabel('Density')
			ax.legend()
			plt.tight_layout()

			# Display statistics
			st.pyplot(fig2)

			col1, col2, col3 = st.columns(3)
			with col1:
				st.metric("Theoretical Mean", f"{np.mean(data):.4f}")
			with col2:
				st.metric("Calculated Mean", f"{mu:.4f}")
			with col3:
				st.metric("Standard Deviation", f"{sigma:.4f}")

		progress.empty()

with tabs[1]:
	st.markdown('<h2>Law of Large Numbers (LLN)</h2>', unsafe_allow_html=True)

	st.markdown("""
    <div class="highlight">
    <strong>Law of Large Numbers:</strong> States that as the sample size increases, the sample mean approaches the true population mean.
    The more trials performed, the closer the empirical average gets to the expected value.
    </div>
    """, unsafe_allow_html=True)

	col1, col2 = st.columns([1, 1])

	with col1:
		st.markdown('<div>Select the distribution type:</div>', unsafe_allow_html=True)
		lln_dist = st.selectbox(
			"LLN Distribution",
			["Exponential", "Uniform", "Poisson"],
			index=0,
			label_visibility="collapsed"
		)

		st.markdown('<div>Maximum Number of Samples:</div>', unsafe_allow_html=True)
		max_samples = st.slider("Max Samples", min_value=1000, max_value=10000, value=5000, step=1000,
								label_visibility="collapsed")

	with col2:
		st.markdown('<div>Information about the Law of Large Numbers:</div>', unsafe_allow_html=True)
		st.markdown("""
        <div>
        <ul>
          <li>This law explains how sample values converge to the true population values as the sample size grows.</li>
          <li>It is fundamental to many statistical applications like survey analysis and studies.</li>
          <li>It allows us to predict with increasing accuracy as more data becomes available.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

	if st.button("Run Simulation", key="lln_sim"):
		# Generate data based on selected distribution
		if lln_dist == "Exponential":
			true_mean = 1.0
			data = np.random.exponential(true_mean, size=max_samples)
			dist_title = "Exponential Distribution"
		elif lln_dist == "Uniform":
			true_mean = 0.5
			data = np.random.uniform(0, 1, size=max_samples)
			dist_title = "Uniform Distribution"
		else:  # Poisson
			true_mean = 5.0
			data = np.random.poisson(true_mean, size=max_samples)
			dist_title = "Poisson Distribution"

		# Calculate running mean
		running_mean = np.cumsum(data) / np.arange(1, max_samples + 1)

		# Create animation using Plotly
		fig = go.Figure()

		# Add true mean line
		fig.add_trace(go.Scatter(
			x=[1, max_samples],
			y=[true_mean, true_mean],
			mode='lines',
			name='Expected Value',
			line=dict(color='red', width=2, dash='dash')
		))

		# Add initial running mean
		fig.add_trace(go.Scatter(
			x=[1],
			y=[data[0]],
			mode='lines',
			name='Running Mean',
			line=dict(color='blue', width=2)
		))

		# Create frames for animation
		frames = []
		step_size = max(1, max_samples // 100)  # For smoother animation

		for i in range(step_size, max_samples, step_size):
			frames.append(
				go.Frame(
					data=[
						go.Scatter(
							x=[1, max_samples],
							y=[true_mean, true_mean],
							mode='lines',
							name='Expected Value',
							line=dict(color='red', width=2, dash='dash')
						),
						go.Scatter(
							x=np.arange(1, i + 1),
							y=running_mean[:i],
							mode='lines',
							name='Running Mean',
							line=dict(color='blue', width=2)
						)
					],
					traces=[0, 1],
					name=f'Frame {i}'
				)
			)

		# Set up layout with animation controls
		fig.update_layout(
			title=f"LLN Simulation: {dist_title}",
			xaxis=dict(title="Number of Samples", type="log"),
			yaxis=dict(title="Mean"),
			updatemenus=[{
				"type": "buttons",
				"buttons": [{
					"label": "Play",
					"method": "animate",
					"args": [None, {"frame": {"duration": 30, "redraw": True}, "fromcurrent": True}]
				}]
			}]
		)

		fig.frames = frames
		st.plotly_chart(fig, use_container_width=True)

		# Display final results
		st.markdown('<h3>Final Results</h3>', unsafe_allow_html=True)
		col1, col2, col3 = st.columns(3)
		with col1:
			st.metric("True Mean", f"{true_mean:.4f}")
		with col2:
			st.metric("Calculated Mean", f"{running_mean[-1]:.4f}")
		with col3:
			st.metric("Difference", f"{abs(true_mean - running_mean[-1]):.4f}")

		# Show final convergence chart
		fig2, ax = plt.subplots(figsize=(10, 6))

		# Plot differences from mean
		differences = np.abs(running_mean - true_mean)
		ax.plot(np.arange(1, max_samples + 1), differences, color='green')
		ax.set_title('Difference Between Running Mean and Expected Value', fontsize=14)
		ax.set_xlabel('Number of Samples')
		ax.set_ylabel('Absolute Difference')
		ax.set_xscale('log')
		plt.tight_layout()

		st.pyplot(fig2)

with tabs[2]:
	st.markdown('<h2>Discrete Probability Distributions</h2>', unsafe_allow_html=True)

	discrete_dist = st.selectbox(
		"Select Discrete Distribution",
		["Binomial Distribution", "Poisson Distribution", "Geometric Distribution"]
	)

	col1, col2 = st.columns([1, 2])

	if discrete_dist == "Binomial Distribution":
		with col1:
			st.markdown(
				'<div class="highlight">The Binomial distribution describes the number of successes in a fixed number of independent trials, with a constant probability of success in each trial.</div>',
				unsafe_allow_html=True)

			st.markdown('<div>Number of Trials (n):</div>', unsafe_allow_html=True)
			n_trials = st.slider("n", min_value=1, max_value=50, value=10, step=1, label_visibility="collapsed")

			st.markdown('<div>Probability of Success (p):</div>', unsafe_allow_html=True)
			p_success = st.slider("p", min_value=0.0, max_value=1.0, value=0.5, step=0.05, label_visibility="collapsed")

			st.markdown(f"""
            <div>
            <strong>Binomial Distribution Properties:</strong><br>
            Mean = n × p = {n_trials * p_success:.2f}<br>
            Variance = n × p × (1-p) = {n_trials * p_success * (1 - p_success):.2f}<br>
            Standard Deviation = {np.sqrt(n_trials * p_success * (1 - p_success)):.2f}
            </div>
            """, unsafe_allow_html=True)

			n_sim = st.slider("Number of Simulations:", min_value=100, max_value=10000, value=1000, step=100)

		with col2:
			if st.button("Show Probability Mass Function (PMF)", key="binom_pmf"):
				# Create PMF plot
				x = np.arange(0, n_trials + 1)
				pmf = binom.pmf(x, n_trials, p_success)

				fig, ax = plt.subplots(figsize=(10, 6))
				ax.bar(x, pmf, alpha=0.7)
				ax.set_xlabel('Number of Successes')
				ax.set_ylabel('Probability')
				ax.set_title(f'PMF for Binomial Distribution: n={n_trials}, p={p_success}')
				ax.grid(alpha=0.3)

				st.pyplot(fig)

			if st.button("Run Distribution Simulation", key="binom_sim"):
				# Simulate binomial samples
				samples = np.random.binomial(n_trials, p_success, size=n_sim)

				# Create animation of build-up
				num_frames = 50
				step_size = max(1, n_sim // num_frames)

				fig = go.Figure()

				# Add theoretical PMF
				x_theory = np.arange(0, n_trials + 1)
				pmf_theory = binom.pmf(x_theory, n_trials, p_success)

				fig.add_trace(go.Bar(
					x=x_theory,
					y=pmf_theory,
					name='Theoretical Distribution',
					marker_color='rgba(255, 0, 0, 0.6)',
					opacity=0.6
				))

				# Initial histogram (first few samples)
				initial_samples = samples[:step_size]
				hist_data = np.histogram(initial_samples, bins=np.arange(-0.5, n_trials + 1.5), density=True)[0]

				fig.add_trace(go.Bar(
					x=x_theory,
					y=hist_data,
					name='Simulated Samples',
					marker_color='rgba(0, 0, 255, 0.6)',
					opacity=0.6
				))

				# Create frames
				frames = []
				for i in range(step_size, n_sim + 1, step_size):
					current_samples = samples[:i]
					hist_data = np.histogram(current_samples, bins=np.arange(-0.5, n_trials + 1.5), density=True)[0]

					frames.append(go.Frame(
						data=[
							go.Bar(
								x=x_theory,
								y=pmf_theory,
								name='Theoretical Distribution',
								marker_color='rgba(255, 0, 0, 0.6)',
								opacity=0.6
							),
							go.Bar(
								x=x_theory,
								y=hist_data,
								name='Simulated Samples',
								marker_color='rgba(0, 0, 255, 0.6)',
								opacity=0.6
							)
						],
						name=f'Frame {i}'
					))

				fig.frames = frames

				fig.update_layout(
					title=f'Binomial Distribution Simulation: n={n_trials}, p={p_success}',
					xaxis=dict(title='Number of Successes', tickmode='linear'),
					yaxis=dict(title='Probability Density'),
					barmode='overlay',
					updatemenus=[{
						"type": "buttons",
						"buttons": [{
							"label": "Play",
							"method": "animate",
							"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]
						}]
					}]
				)

				st.plotly_chart(fig, use_container_width=True)

				# Display statistics comparison
				st.markdown('<h3>Comparison between Theoretical and Simulated Values</h3>', unsafe_allow_html=True)

				col1, col2, col3 = st.columns(3)
				with col1:
					st.metric("Theoretical Mean", f"{n_trials * p_success:.4f}",
							  delta=f"{np.mean(samples) - n_trials * p_success:.4f}")
				with col2:
					st.metric("Simulated Mean", f"{np.mean(samples):.4f}")
				with col3:
					st.metric("Theoretical Standard Deviation", f"{np.sqrt(n_trials * p_success * (1 - p_success)):.4f}",
							  delta=f"{np.std(samples) - np.sqrt(n_trials * p_success * (1 - p_success)):.4f}")

	elif discrete_dist == "Poisson Distribution":
		with col1:
			st.markdown(
				'<div class="highlight">The Poisson distribution describes the number of events occurring in a fixed interval of time or space, when these events are independent and occur at a constant average rate.</div>',
				unsafe_allow_html=True)

			st.markdown('<div>Rate (λ):</div>', unsafe_allow_html=True)
			lambda_param = st.slider("lambda", min_value=0.1, max_value=20.0, value=5.0, step=0.1,
									 label_visibility="collapsed")

			st.markdown(f"""
            <div>
            <strong>Poisson Distribution Properties:</strong><br>
            Mean = λ = {lambda_param:.2f}<br>
            Variance = λ = {lambda_param:.2f}<br>
            Standard Deviation = √λ = {np.sqrt(lambda_param):.2f}
            </div>
            """, unsafe_allow_html=True)

			n_sim = st.slider("Number of Simulations:", min_value=100, max_value=10000, value=1000, step=100,
							  key="poisson_sim_count")

		with col2:
			if st.button("Show Probability Mass Function (PMF)", key="poisson_pmf"):
				# Create PMF plot
				x_max = int(lambda_param * 3 + 10)  # Use a reasonable range for x
				x = np.arange(0, x_max)
				pmf = poisson.pmf(x, lambda_param)

				fig, ax = plt.subplots(figsize=(10, 6))
				ax.bar(x, pmf, alpha=0.7)
				ax.set_xlabel('Number of Events')
				ax.set_ylabel('Probability')
				ax.set_title(f'PMF for Poisson Distribution: λ={lambda_param}')
				ax.grid(alpha=0.3)

				st.pyplot(fig)

			if st.button("Run Distribution Simulation", key="poisson_sim"):
				# Simulate Poisson samples
				samples = np.random.poisson(lambda_param, size=n_sim)

				# Create animation of build-up
				num_frames = 50
				step_size = max(1, n_sim // num_frames)

				# Define range for histogram
				x_max = int(max(np.max(samples), lambda_param * 3) + 1)
				x_range = np.arange(0, x_max)

				# Theoretical PMF
				pmf_theory = poisson.pmf(x_range, lambda_param)

				fig = go.Figure()

				# Add theoretical PMF
				fig.add_trace(go.Bar(
					x=x_range,
					y=pmf_theory,
					name='Theoretical Distribution',
					marker_color='rgba(255, 0, 0, 0.6)',
					opacity=0.6
				))

				# Initial histogram (first few samples)
				initial_samples = samples[:step_size]
				hist_data = np.histogram(initial_samples, bins=np.arange(-0.5, x_max + 0.5), density=True)[0]

				fig.add_trace(go.Bar(
					x=x_range,
					y=hist_data,
					name='Simulated Samples',
					marker_color='rgba(0, 0, 255, 0.6)',
					opacity=0.6
				))

				# Create frames
				frames = []
				for i in range(step_size, n_sim + 1, step_size):
					current_samples = samples[:i]
					hist_data = np.histogram(current_samples, bins=np.arange(-0.5, x_max + 0.5), density=True)[0]

					frames.append(go.Frame(
						data=[
							go.Bar(
								x=x_range,
								y=pmf_theory,
								name='Theoretical Distribution',
								marker_color='rgba(255, 0, 0, 0.6)',
								opacity=0.6
							),
							go.Bar(
								x=x_range,
								y=hist_data,
								name='Simulated Samples',
								marker_color='rgba(0, 0, 255, 0.6)',
								opacity=0.6
							)
						],
						name=f'Frame {i}'
					))

				fig.frames = frames

				fig.update_layout(
					title=f'Poisson Distribution Simulation: λ={lambda_param}',
					xaxis=dict(title='Number of Events', tickmode='linear'),
					yaxis=dict(title='Probability Density'),
					barmode='overlay',
					updatemenus=[{
						"type": "buttons",
						"buttons": [{
							"label": "Play",
							"method": "animate",
							"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]
						}]
					}]
				)

				st.plotly_chart(fig, use_container_width=True)

				# Display statistics comparison
				st.markdown('<h3>Comparison between Theoretical and Simulated Values</h3>', unsafe_allow_html=True)

				col1, col2, col3 = st.columns(3)
				with col1:
					st.metric("Theoretical Mean", f"{lambda_param:.4f}",
							  delta=f"{np.mean(samples) - lambda_param:.4f}")
				with col2:
					st.metric("Simulated Mean", f"{np.mean(samples):.4f}")
				with col3:
					st.metric("Theoretical Standard Deviation", f"{np.sqrt(lambda_param):.4f}",
							  delta=f"{np.std(samples) - np.sqrt(lambda_param):.4f}")

	else:  # Geometric distribution
		with col1:
			st.markdown(
				'<div class="highlight">The Geometric distribution describes the number of trials needed to get the first success, with a constant probability of success in each trial.</div>',
				unsafe_allow_html=True)

			st.markdown('<div>Probability of Success (p):</div>', unsafe_allow_html=True)
			p_success = st.slider("p_geom", min_value=0.05, max_value=0.95, value=0.25, step=0.05,
								  label_visibility="collapsed")

			st.markdown(f"""
            <div>
            <strong>Geometric Distribution Properties:</strong><br>
            Mean = 1/p = {1 / p_success:.2f}<br>
            Variance = (1-p)/p² = {(1 - p_success) / (p_success ** 2):.2f}<br>
            Standard Deviation = {np.sqrt((1 - p_success) / (p_success ** 2)):.2f}
            </div>
            """, unsafe_allow_html=True)

			n_sim = st.slider("Number of Simulations:", min_value=100, max_value=10000, value=1000, step=100,
							  key="geom_sim_count")

		with col2:
			if st.button("Show Probability Mass Function (PMF)", key="geom_pmf"):
				# Calculate appropriate range for x
				x_max = int(min(5 / p_success, 50))  # Limit to reasonable range
				x = np.arange(1, x_max + 1)

				# Calculate PMF using scipy.stats.geom (k is number of failures *before* success)
				# To match the definition "number of trials *needed*", we use k-1 for scipy's geom
				pmf = [(1 - p_success) ** (k - 1) * p_success for k in x] # Manual calculation for trials needed

				fig, ax = plt.subplots(figsize=(10, 6))
				ax.bar(x, pmf, alpha=0.7)
				ax.set_xlabel('Number of Trials until First Success')
				ax.set_ylabel('Probability')
				ax.set_title(f'PMF for Geometric Distribution: p={p_success}')
				ax.grid(alpha=0.3)

				st.pyplot(fig)

			if st.button("Run Distribution Simulation", key="geom_sim"):
				# Simulate geometric samples (np.random.geometric gives trials needed)
				samples = np.random.geometric(p_success, size=n_sim)

				# Create animation of build-up
				num_frames = 50
				step_size = max(1, n_sim // num_frames)

				# Define range for histogram
				x_max = int(min(np.percentile(samples, 99), 4 / p_success, 50)) # Limit range
				x_range = np.arange(1, x_max + 1)

				# Theoretical PMF
				pmf_theory = [(1 - p_success) ** (k - 1) * p_success for k in x_range]

				fig = go.Figure()

				# Add theoretical PMF
				fig.add_trace(go.Bar(
					x=x_range,
					y=pmf_theory,
					name='Theoretical Distribution',
					marker_color='rgba(255, 0, 0, 0.6)',
					opacity=0.6
				))

				# Initial histogram (first few samples)
				initial_samples = samples[:step_size]
				hist_data = np.histogram(initial_samples, bins=np.arange(0.5, x_max + 1.5), density=True)[0]

				fig.add_trace(go.Bar(
					x=x_range,
					y=hist_data,
					name='Simulated Samples',
					marker_color='rgba(0, 0, 255, 0.6)',
					opacity=0.6
				))

				# Create frames
				frames = []
				for i in range(step_size, n_sim + 1, step_size):
					current_samples = samples[:i]
					hist_data = np.histogram(current_samples, bins=np.arange(0.5, x_max + 1.5), density=True)[0]

					frames.append(go.Frame(
						data=[
							go.Bar(
								x=x_range,
								y=pmf_theory,
								name='Theoretical Distribution',
								marker_color='rgba(255, 0, 0, 0.6)',
								opacity=0.6
							),
							go.Bar(
								x=x_range,
								y=hist_data,
								name='Simulated Samples',
								marker_color='rgba(0, 0, 255, 0.6)',
								opacity=0.6
							)
						],
						name=f'Frame {i}'
					))

				fig.frames = frames

				fig.update_layout(
					title=f'Geometric Distribution Simulation: p={p_success}',
					xaxis=dict(title='Number of Trials until First Success', tickmode='linear'),
					yaxis=dict(title='Probability Density'),
					barmode='overlay',
					updatemenus=[{
						"type": "buttons",
						"buttons": [{
							"label": "Play",
							"method": "animate",
							"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]
						}]
					}]
				)

				st.plotly_chart(fig, use_container_width=True)

				# Display statistics comparison
				st.markdown('<h3>Comparison between Theoretical and Simulated Values</h3>', unsafe_allow_html=True)

				col1, col2, col3 = st.columns(3)
				with col1:
					st.metric("Theoretical Mean", f"{1 / p_success:.4f}",
							  delta=f"{np.mean(samples) - 1 / p_success:.4f}")
				with col2:
					st.metric("Simulated Mean", f"{np.mean(samples):.4f}")
				with col3:
					st.metric("Theoretical Standard Deviation", f"{np.sqrt((1 - p_success) / (p_success ** 2)):.4f}",
							  delta=f"{np.std(samples) - np.sqrt((1 - p_success) / (p_success ** 2)):.4f}")

with tabs[3]:
	st.markdown('<h2>Continuous Probability Distributions</h2>', unsafe_allow_html=True)

	continuous_dist = st.selectbox(
		"Select Continuous Distribution",
		["Normal Distribution", "Exponential Distribution", "Uniform Distribution"]
	)

	col1, col2 = st.columns([1, 2])

	if continuous_dist == "Normal Distribution":
		with col1:
			st.markdown(
				'<div class="highlight">The Normal (or Gaussian) distribution is one of the most important distributions in statistics and appears in many natural phenomena.</div>',
				unsafe_allow_html=True)

			st.markdown('<div>Mean (μ):</div>', unsafe_allow_html=True)
			mu = st.slider("mu", min_value=-10.0, max_value=10.0, value=0.0, step=0.5, label_visibility="collapsed")

			st.markdown('<div>Standard Deviation (σ):</div>', unsafe_allow_html=True)
			sigma = st.slider("sigma", min_value=0.1, max_value=5.0, value=1.0, step=0.1, label_visibility="collapsed")

			st.markdown(f"""
            <div>
            <strong>Normal Distribution Properties:</strong><br>
            Mean = μ = {mu:.2f}<br>
            Variance = σ² = {sigma ** 2:.2f}<br>
            Standard Deviation = σ = {sigma:.2f}
            </div>
            """, unsafe_allow_html=True)

			n_sim = st.slider("Number of Simulations:", min_value=100, max_value=10000, value=1000, step=100,
							  key="normal_sim_count")

		with col2:
			if st.button("Show Probability Density Function (PDF)", key="normal_pdf"):
				# Create PDF plot
				x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
				pdf = norm.pdf(x, mu, sigma)

				fig, ax = plt.subplots(figsize=(10, 6))
				ax.plot(x, pdf, 'r-', lw=2)
				ax.fill_between(x, pdf, alpha=0.3, color='skyblue')
				ax.set_xlabel('Value')
				ax.set_ylabel('Probability Density')
				ax.set_title(f'PDF for Normal Distribution: μ={mu}, σ={sigma}')
				ax.grid(alpha=0.3)

				st.pyplot(fig)

			if st.button("Run Distribution Simulation", key="normal_sim"):
				# Simulate normal samples
				samples = np.random.normal(mu, sigma, size=n_sim)

				# Create animation of build-up
				num_frames = 50
				step_size = max(1, n_sim // num_frames)

				# Define range for histogram
				x_min, x_max = mu - 4 * sigma, mu + 4 * sigma
				x_range = np.linspace(x_min, x_max, 100)

				# Theoretical PDF
				pdf_theory = norm.pdf(x_range, mu, sigma)

				fig = go.Figure()

				# Add theoretical PDF
				fig.add_trace(go.Scatter(
					x=x_range,
					y=pdf_theory,
					mode='lines',
					name='Theoretical Distribution',
					line=dict(color='red', width=2)
				))

				# Initial histogram (first few samples)
				initial_samples = samples[:step_size]
				hist_data, bin_edges = np.histogram(initial_samples, bins=30,
													range=(x_min, x_max), density=True)
				bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

				fig.add_trace(go.Bar(
					x=bin_centers,
					y=hist_data,
					name='Simulated Samples',
					marker_color='rgba(0, 0, 255, 0.6)',
					opacity=0.6
				))

				# Create frames
				frames = []
				for i in range(step_size, n_sim + 1, step_size):
					current_samples = samples[:i]
					hist_data, _ = np.histogram(current_samples, bins=30,
												range=(x_min, x_max), density=True)

					frames.append(go.Frame(
						data=[
							go.Scatter(
								x=x_range,
								y=pdf_theory,
								mode='lines',
								name='Theoretical Distribution',
								line=dict(color='red', width=2)
							),
							go.Bar(
								x=bin_centers,
								y=hist_data,
								name='Simulated Samples',
								marker_color='rgba(0, 0, 255, 0.6)',
								opacity=0.6
							)
						],
						name=f'Frame {i}'
					))

				fig.frames = frames

				fig.update_layout(
					title=f'Normal Distribution Simulation: μ={mu}, σ={sigma}',
					xaxis=dict(title='Value'),
					yaxis=dict(title='Probability Density'),
					updatemenus=[{
						"type": "buttons",
						"buttons": [{
							"label": "Play",
							"method": "animate",
							"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]
						}]
					}]
				)

				st.plotly_chart(fig, use_container_width=True)

				# Display statistics comparison
				st.markdown('<h3>Comparison between Theoretical and Simulated Values</h3>', unsafe_allow_html=True)

				col1, col2, col3 = st.columns(3)
				with col1:
					st.metric("Theoretical Mean", f"{mu:.4f}",
							  delta=f"{np.mean(samples) - mu:.4f}")
				with col2:
					st.metric("Simulated Mean", f"{np.mean(samples):.4f}")
				with col3:
					st.metric("Theoretical Standard Deviation", f"{sigma:.4f}",
							  delta=f"{np.std(samples) - sigma:.4f}")

				# QQ Plot
				st.markdown('<h3>QQ Plot for Normality Check</h3>', unsafe_allow_html=True)

				fig_qq, ax_qq = plt.subplots(figsize=(8, 8))
				probplot(samples, dist="norm", plot=ax_qq)
				ax_qq.set_title('QQ Plot for Normality Check')

				st.pyplot(fig_qq)

	elif continuous_dist == "Exponential Distribution":
		with col1:
			st.markdown(
				'<div class="highlight">The Exponential distribution describes the time intervals between successive events that occur at a constant and independent rate.</div>',
				unsafe_allow_html=True)

			st.markdown('<div>Event Rate (λ):</div>', unsafe_allow_html=True)
			lambda_param = st.slider("lambda_exp", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
									 label_visibility="collapsed")

			st.markdown(f"""
            <div>
            <strong>Exponential Distribution Properties:</strong><br>
            Mean = 1/λ = {1 / lambda_param:.2f}<br>
            Variance = 1/λ² = {1 / (lambda_param ** 2):.2f}<br>
            Standard Deviation = 1/λ = {1 / lambda_param:.2f}
            </div>
            """, unsafe_allow_html=True)

			n_sim = st.slider("Number of Simulations:", min_value=100, max_value=10000, value=1000, step=100,
							  key="exp_sim_count")

		with col2:
			if st.button("Show Probability Density Function (PDF)", key="exp_pdf"):
				# Create PDF plot
				# Scipy uses scale = 1/lambda
				scale_param = 1 / lambda_param
				x = np.linspace(0, 5 * scale_param, 1000)
				pdf = expon.pdf(x, scale=scale_param)

				fig, ax = plt.subplots(figsize=(10, 6))
				ax.plot(x, pdf, 'r-', lw=2)
				ax.fill_between(x, pdf, alpha=0.3, color='skyblue')
				ax.set_xlabel('Value')
				ax.set_ylabel('Probability Density')
				ax.set_title(f'PDF for Exponential Distribution: λ={lambda_param}')
				ax.grid(alpha=0.3)

				st.pyplot(fig)

			if st.button("Run Distribution Simulation", key="exp_sim"):
				# Simulate exponential samples
				scale_param = 1 / lambda_param
				samples = np.random.exponential(scale=scale_param, size=n_sim)

				# Create animation of build-up
				num_frames = 50
				step_size = max(1, n_sim // num_frames)

				# Define range for histogram (use 95th percentile to avoid extreme values)
				x_max = max(np.percentile(samples, 98), 5 * scale_param) # Adjust range based on samples and scale
				x_range = np.linspace(0, x_max, 100)

				# Theoretical PDF
				pdf_theory = expon.pdf(x_range, scale=scale_param)

				fig = go.Figure()

				# Add theoretical PDF
				fig.add_trace(go.Scatter(
					x=x_range,
					y=pdf_theory,
					mode='lines',
					name='Theoretical Distribution',
					line=dict(color='red', width=2)
				))

				# Initial histogram (first few samples)
				initial_samples = samples[:step_size]
				hist_data, bin_edges = np.histogram(initial_samples, bins=30,
													range=(0, x_max), density=True)
				bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

				fig.add_trace(go.Bar(
					x=bin_centers,
					y=hist_data,
					name='Simulated Samples',
					marker_color='rgba(0, 0, 255, 0.6)',
					opacity=0.6
				))

				# Create frames
				frames = []
				for i in range(step_size, n_sim + 1, step_size):
					current_samples = samples[:i]
					hist_data, _ = np.histogram(current_samples, bins=30,
												range=(0, x_max), density=True)

					frames.append(go.Frame(
						data=[
							go.Scatter(
								x=x_range,
								y=pdf_theory,
								mode='lines',
								name='Theoretical Distribution',
								line=dict(color='red', width=2)
							),
							go.Bar(
								x=bin_centers,
								y=hist_data,
								name='Simulated Samples',
								marker_color='rgba(0, 0, 255, 0.6)',
								opacity=0.6
							)
						],
						name=f'Frame {i}'
					))

				fig.frames = frames

				fig.update_layout(
					title=f'Exponential Distribution Simulation: λ={lambda_param}',
					xaxis=dict(title='Value'),
					yaxis=dict(title='Probability Density'),
					updatemenus=[{
						"type": "buttons",
						"buttons": [{
							"label": "Play",
							"method": "animate",
							"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]
						}]
					}]
				)

				st.plotly_chart(fig, use_container_width=True)

				# Display statistics comparison
				st.markdown('<h3>Comparison between Theoretical and Simulated Values</h3>', unsafe_allow_html=True)

				theoretical_mean = 1 / lambda_param
				theoretical_std = 1 / lambda_param

				col1, col2, col3 = st.columns(3)
				with col1:
					st.metric("Theoretical Mean", f"{theoretical_mean:.4f}",
							  delta=f"{np.mean(samples) - theoretical_mean:.4f}")
				with col2:
					st.metric("Simulated Mean", f"{np.mean(samples):.4f}")
				with col3:
					st.metric("Theoretical Standard Deviation", f"{theoretical_std:.4f}",
							  delta=f"{np.std(samples) - theoretical_std:.4f}")

				# Memory-less property visualization
				st.markdown('<h3>Memoryless Property of the Exponential Distribution</h3>', unsafe_allow_html=True)

				# Create chart to illustrate memoryless property
				scale_param = 1/lambda_param
				t_vals = np.linspace(0, 5 * scale_param, 100)
				s = scale_param # Point to show conditional probability, e.g., the mean

				fig_mem, ax_mem = plt.subplots(figsize=(10, 6))

				# Plot PDF
				ax_mem.plot(t_vals, expon.pdf(t_vals, scale=scale_param), 'b-', lw=2, label='PDF')

				# Show conditional probability
				ax_mem.axvline(x=s, color='g', linestyle='--', label=f'Time s = {s:.2f}')

				# Show original tail probability P(X > s)
				x_fill1 = np.linspace(s, 5 * scale_param, 50)
				y_fill1 = expon.pdf(x_fill1, scale=scale_param)
				# P(X > s) = exp(-lambda * s) = exp(-s / scale)
				prob_x_gt_s = np.exp(-s / scale_param)
				ax_mem.fill_between(x_fill1, y_fill1, alpha=0.3, color='red',
								label=f'P(X > {s:.2f}) = {prob_x_gt_s:.4f}')

				# Show conditional probability P(X > s+t | X > s) = P(X > t)
				t_extra = scale_param # Let t be one mean lifetime
				ax_mem.axvline(x=s + t_extra, color='purple', linestyle='--', label=f'Time s+t = {s + t_extra:.2f}')
				# P(X > t) = exp(-t / scale)
				prob_x_gt_t = np.exp(-t_extra / scale_param)
				st.markdown(f"<div class='ltr'>Illustrative calculation: P(X > {t_extra:.2f}) = {prob_x_gt_t:.4f}. This should equal P(X > {s+t_extra:.2f} | X > {s:.2f}).</div>", unsafe_allow_html=True)


				ax_mem.set_xlabel('Time')
				ax_mem.set_ylabel('Probability Density')
				ax_mem.set_title('Memoryless Property: P(X > s+t | X > s) = P(X > t)')
				ax_mem.legend()
				ax_mem.grid(alpha=0.3)

				st.pyplot(fig_mem)

				# Explanation
				st.markdown("""
                <div class="highlight">
                <strong>Memoryless Property:</strong> The Exponential distribution is the only continuous distribution with the memoryless property.
                This means the probability of waiting an additional time t, given that you have already waited for time s, is the same as the probability of waiting for time t from the beginning:
                <br><br>
                P(X > s+t | X > s) = P(X > t)
                </div>
                """, unsafe_allow_html=True)

	else:  # Uniform distribution
		with col1:
			st.markdown(
				'<div class="highlight">The Uniform distribution represents equal probabilities for all values within a specified range.</div>',
				unsafe_allow_html=True)

			st.markdown('<div>Minimum Value (a):</div>', unsafe_allow_html=True)
			a_param = st.slider("a_unif", min_value=-10.0, max_value=10.0, value=0.0, step=0.5,
								label_visibility="collapsed")

			st.markdown('<div>Maximum Value (b):</div>', unsafe_allow_html=True)
			# Ensure b > a
			min_b = a_param + 0.1
			default_b = max(min_b, a_param + 1.0)
			b_param = st.slider("b_unif", min_value=min_b, max_value=a_param + 20.0, value=default_b,
								step=0.5, label_visibility="collapsed")

			st.markdown(f"""
            <div>
            <strong>Uniform Distribution Properties:</strong><br>
            Mean = (a+b)/2 = {(a_param + b_param) / 2:.2f}<br>
            Variance = (b-a)²/12 = {((b_param - a_param) ** 2) / 12:.2f}<br>
            Standard Deviation = {np.sqrt(((b_param - a_param) ** 2) / 12):.2f}
            </div>
            """, unsafe_allow_html=True)

			n_sim = st.slider("Number of Simulations:", min_value=100, max_value=10000, value=1000, step=100,
							  key="unif_sim_count")

		with col2:
			if st.button("Show Probability Density Function (PDF)", key="unif_pdf"):
				# Create PDF plot
				width = b_param - a_param
				x = np.linspace(a_param - 0.1 * width, b_param + 0.1 * width, 1000)
				pdf = uniform.pdf(x, loc=a_param, scale=width)

				fig, ax = plt.subplots(figsize=(10, 6))
				ax.plot(x, pdf, 'r-', lw=2)
				# Fill only between a and b where pdf is non-zero
				x_fill = np.linspace(a_param, b_param, 100)
				pdf_fill = uniform.pdf(x_fill, loc=a_param, scale=width)
				ax.fill_between(x_fill, pdf_fill, alpha=0.3, color='skyblue')
				ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
				ax.set_xlabel('Value')
				ax.set_ylabel('Probability Density')
				ax.set_title(f'PDF for Uniform Distribution: a={a_param}, b={b_param}')
				ax.grid(alpha=0.3)

				st.pyplot(fig)

			if st.button("Run Distribution Simulation", key="unif_sim"):
				# Simulate uniform samples
				samples = np.random.uniform(a_param, b_param, size=n_sim)

				# Create animation of build-up
				num_frames = 50
				step_size = max(1, n_sim // num_frames)

				# Define range for histogram and plot
				width = b_param - a_param
				x_min, x_max = a_param - 0.1 * width, b_param + 0.1 * width
				x_range = np.linspace(x_min, x_max, 100)

				# Theoretical PDF
				pdf_theory = uniform.pdf(x_range, loc=a_param, scale=width)

				fig = go.Figure()

				# Add theoretical PDF
				fig.add_trace(go.Scatter(
					x=x_range,
					y=pdf_theory,
					mode='lines',
					name='Theoretical Distribution',
					line=dict(color='red', width=2),
					# Fill between a and b
					fill='tozeroy', fillcolor='rgba(255,0,0,0.1)',
					# Define shape only between a and b
					x0=a_param, x1=b_param, y0=0, y1=1/width if width > 0 else 0
				))


				# Initial histogram (first few samples)
				initial_samples = samples[:step_size]
				# Use range [a, b] for histogram bins for better comparison
				hist_data, bin_edges = np.histogram(initial_samples, bins=30,
													range=(a_param, b_param), density=True)
				bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

				fig.add_trace(go.Bar(
					x=bin_centers,
					y=hist_data,
					name='Simulated Samples',
					marker_color='rgba(0, 0, 255, 0.6)',
					opacity=0.6
				))

				# Create frames
				frames = []
				for i in range(step_size, n_sim + 1, step_size):
					current_samples = samples[:i]
					hist_data, _ = np.histogram(current_samples, bins=30,
												range=(a_param, b_param), density=True)

					frames.append(go.Frame(
						data=[
							go.Scatter( # Keep theoretical PDF static
								x=x_range, y=pdf_theory, mode='lines', name='Theoretical Distribution',
								line=dict(color='red', width=2),
                                fill='tozeroy', fillcolor='rgba(255,0,0,0.1)',
                                x0=a_param, x1=b_param, y0=0, y1=1/width if width > 0 else 0
							),
							go.Bar( # Update histogram bar data
								x=bin_centers,
								y=hist_data,
								name='Simulated Samples',
								marker_color='rgba(0, 0, 255, 0.6)',
								opacity=0.6
							)
						],
						# Specify which traces are updated in the frame
                        # Trace 0 is Scatter, Trace 1 is Bar
                        traces=[1], # Only update the Bar trace
						name=f'Frame {i}'
					))

				fig.frames = frames

				fig.update_layout(
					title=f'Uniform Distribution Simulation: a={a_param}, b={b_param}',
					xaxis=dict(title='Value', range=[x_min, x_max]), # Set x-axis range
					yaxis=dict(title='Probability Density'),
                    barmode='overlay', # Ensure bar overlays scatter properly
					updatemenus=[{
						"type": "buttons",
						"buttons": [{
							"label": "Play",
							"method": "animate",
							# Args for animate: [None] targets all frames
							"args": [None, {"frame": {"duration": 50, "redraw": True},
											"fromcurrent": True, "mode": "immediate"}]
						}]
					}]
				)
                # Set initial y-axis range appropriately
				fig.update_yaxes(range=[0, 1.5 / width if width > 0 else 1])

				st.plotly_chart(fig, use_container_width=True)

				# Display statistics comparison
				st.markdown('<h3>Comparison between Theoretical and Simulated Values</h3>', unsafe_allow_html=True)

				expected_mean = (a_param + b_param) / 2
				expected_std = np.sqrt((b_param - a_param) ** 2 / 12)

				col1, col2, col3 = st.columns(3)
				with col1:
					st.metric("Theoretical Mean", f"{expected_mean:.4f}",
							  delta=f"{np.mean(samples) - expected_mean:.4f}")
				with col2:
					st.metric("Simulated Mean", f"{np.mean(samples):.4f}")
				with col3:
					st.metric("Theoretical Standard Deviation", f"{expected_std:.4f}",
							  delta=f"{np.std(samples) - expected_std:.4f}")

				# Uniform order statistics
				st.markdown('<h3>Order Statistics for Uniform Distribution</h3>', unsafe_allow_html=True)

				st.markdown("""
                <div class="highlight">
                Order statistics for the Uniform distribution have interesting properties. If we take a sample of n values from the Uniform [0,1] distribution and sort them in ascending order,
                the i-th order statistic follows a Beta distribution with parameters (i, n-i+1).
                </div>
                """, unsafe_allow_html=True)

				# Demonstrate with a small sample
				order_n = 5
				unif_samples = np.random.uniform(0, 1, size=(1000, order_n))
				ordered_samples = np.sort(unif_samples, axis=1)

				fig_order, axs_order = plt.subplots(1, order_n, figsize=(15, 4), sharey=True)

				for i in range(order_n):
					axs_order[i].hist(ordered_samples[:, i], bins=20, density=True, alpha=0.6)
					# Add Beta PDF
					x_beta = np.linspace(0.01, 0.99, 100) # Avoid 0 and 1 for Beta PDF if params < 1
					# Parameters for Beta(i, n-i+1) -> alpha = i+1, beta = n-(i+1)+1 = n-i
					alpha_beta = i + 1
					beta_beta = order_n - i
					axs_order[i].plot(x_beta, beta.pdf(x_beta, alpha_beta, beta_beta), 'r-', lw=2)
					axs_order[i].set_title(f'Order Statistic {i + 1}')
					axs_order[i].set_xlabel('Value')
					if i == 0:
						axs_order[i].set_ylabel('Density')

				plt.tight_layout()
				st.pyplot(fig_order)

				st.markdown("""
                <div>
                The plot above shows the distribution of order statistics for the Uniform [0,1] distribution for a sample size of 5.
                The red line represents the theoretical distribution (Beta distribution) for each order statistic.
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="margin-top: 50px; padding: 20px; border-top: 1px solid #ccc;">
<h2>About This Application</h2>
<p>This application is designed to help teach and understand fundamental statistical concepts through animation and interactivity.
It covers the following concepts:</p>
<ul>
  <li>Central Limit Theorem (CLT)</li>
  <li>Law of Large Numbers (LLN)</li>
  <li>Discrete Distributions (Binomial, Poisson, Geometric)</li>
  <li>Continuous Distributions (Normal, Exponential, Uniform)</li>
</ul>
<p>This application can be used as a teaching aid or as a resource for self-study by students.</p>
</div>
""", unsafe_allow_html=True)