# --- START OF FILE dss.py ---

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from scipy.stats import norm, binom, poisson, expon, uniform, beta, probplot
import time
from io import BytesIO

st.set_page_config(layout="wide", page_title="Visualizing Statistical Concepts") # Updated title

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
	'<div>An interactive tool to understand statistical distributions, key theorems, properties, and convergences.</div>', # Updated description
	unsafe_allow_html=True)

# Updated tab list
tabs = st.tabs([
    "Central Limit Theorem",
    "Law of Large Numbers",
    "Distribution Properties & Convergences", # NEW TAB
    "Discrete Distributions Examples", # Renamed
    "Continuous Distributions Examples" # Renamed
])

# --- Tab 0: Central Limit Theorem ---
with tabs[0]:
	st.markdown('<h2>Central Limit Theorem (CLT)</h2>', unsafe_allow_html=True)

	st.markdown("""
    <div class="highlight">
    <strong>Central Limit Theorem:</strong> States that when drawing sufficient samples from any distribution with a defined mean and variance,
    the distribution of the sample means will approximate a normal distribution, regardless of the original distribution's shape.
    </div>
    """, unsafe_allow_html=True)

	st.markdown('<h3>CLT Simulation</h3>', unsafe_allow_html=True)

	col1_clt, col2_clt = st.columns([1, 1])

	with col1_clt:
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

	with col2_clt:
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
			# Use scale=1 for standard exponential mean 1
			data = np.random.exponential(scale=1.0, size=num_samples * sample_size)
			dist_title = "Exponential Distribution"
			theoretical_mean_pop = 1.0 # For standard exponential
		elif source_dist == "Uniform":
			data = np.random.uniform(0, 1, size=num_samples * sample_size)
			dist_title = "Uniform Distribution"
			theoretical_mean_pop = 0.5 # For U(0,1)
		else:  # Binomial
			# Parameters n=10, p=0.5
			data = np.random.binomial(10, 0.5, size=num_samples * sample_size)
			dist_title = "Binomial Distribution"
			theoretical_mean_pop = 10 * 0.5 # n*p

		# Reshape data to compute sample means
		data_reshaped = data.reshape(num_samples, sample_size)
		sample_means = np.mean(data_reshaped, axis=1)

		# Create animation frames
		frames_clt = []
		progress_clt = st.progress(0)

		# For plotting
		bins_clt = 30
		hist_color_clt = 'skyblue'
		curve_color_clt = 'red'

		N_steps_clt = min(100, num_samples)  # Number of frames to show
		step_size_clt = max(1, num_samples // N_steps_clt)

		# Create animated plot with Plotly
		sample_means_frames_clt = []
		# Define plot range based on final distribution +/- 4 std dev
		final_mu_clt = np.mean(sample_means)
		final_sigma_clt = np.std(sample_means)
		plot_xmin = final_mu_clt - 4.5 * final_sigma_clt if final_sigma_clt > 0 else final_mu_clt - 1
		plot_xmax = final_mu_clt + 4.5 * final_sigma_clt if final_sigma_clt > 0 else final_mu_clt + 1


		for i in range(0, num_samples, step_size_clt):
			current_means = sample_means[:i + 1]
			if len(current_means) > 1:
				mu_frame = np.mean(current_means)
				sigma_frame = np.std(current_means)

				# Histogram data
				counts, bin_edges = np.histogram(current_means, bins=bins_clt, range=(plot_xmin, plot_xmax), density=True)
				bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

				# Normal curve data
				if sigma_frame > 1e-6: # Avoid issues with zero std dev early on
				    x_curve = np.linspace(plot_xmin, plot_xmax, 200)
				    y_curve = norm.pdf(x_curve, mu_frame, sigma_frame)
				else:
				    x_curve = np.array([])
				    y_curve = np.array([])


				sample_means_frames_clt.append(
					go.Frame(
						data=[
							go.Bar(x=bin_centers, y=counts, marker_color=hist_color_clt, name='Histogram'),
							go.Scatter(x=x_curve, y=y_curve, mode='lines', line=dict(color=curve_color_clt),
									   name='Normal Approximation')
						],
						layout=go.Layout(
							title_text=f"CLT Simulation: {dist_title} (Samples: {i + 1})" # Updated title in frame
						)
					)
				)

			progress_clt.progress(min(1.0, (i + step_size_clt) / num_samples))

		# Initial frame
		if len(sample_means) > 0:
			initial_means = sample_means[:step_size_clt] if num_samples >= step_size_clt else sample_means[:1]
			if len(initial_means) > 0:
				initial_mu = np.mean(initial_means)
				initial_sigma = np.std(initial_means) if np.std(initial_means) > 1e-6 else 0.1 # Avoid zero sigma

				counts_init, bin_edges_init = np.histogram(initial_means, bins=bins_clt, range=(plot_xmin, plot_xmax), density=True)
				bin_centers_init = (bin_edges_init[:-1] + bin_edges_init[1:]) / 2

				x_curve_init = np.linspace(plot_xmin, plot_xmax, 200)
				y_curve_init = norm.pdf(x_curve_init, initial_mu, initial_sigma)

				# Create figure
				fig_clt = go.Figure(
					data=[
						go.Bar(x=bin_centers_init, y=counts_init, marker_color=hist_color_clt, name='Histogram'),
						go.Scatter(x=x_curve_init, y=y_curve_init, mode='lines', line=dict(color=curve_color_clt),
								name='Normal Approximation')
					],
					layout=go.Layout(
						title=f"CLT Simulation: {dist_title}",
						xaxis=dict(title="Sample Mean", range=[plot_xmin, plot_xmax]), # Set range
						yaxis=dict(title="Density"),
						updatemenus=[{
							"type": "buttons",
                            "direction": "left",
                            "pad": {"r": 10, "t": 10},
                            "showactive": True,
                            "x": 0.1,
                            "xanchor": "left",
                            "y": 1.15,
                            "yanchor": "top",
							"buttons": [{
								"label": "Play",
								"method": "animate",
								"args": [None, {"frame": {"duration": 50, "redraw": True},
                                                "fromcurrent": True, "transition": {"duration": 0},
                                                "mode": "immediate"}]
							}]
						}]
					),
					frames=sample_means_frames_clt
				)
                # Set initial y-axis range dynamically based on the final approx height
				max_y = np.max(norm.pdf(np.linspace(final_mu_clt-0.5*final_sigma_clt, final_mu_clt+0.5*final_sigma_clt, 5), final_mu_clt, final_sigma_clt)) if final_sigma_clt > 0 else 1
				fig_clt.update_yaxes(range=[0, max_y * 1.5])


				st.plotly_chart(fig_clt, use_container_width=True)

				# Final distribution comparison
				st.markdown('<h3>Final Distribution vs. Normal Distribution</h3>', unsafe_allow_html=True)

				fig2_clt, ax2_clt = plt.subplots(figsize=(10, 6))
				ax2_clt.hist(sample_means, bins=bins_clt, density=True, alpha=0.7, color=hist_color_clt, label='Sample Means Histogram')

				# Fit normal distribution
				mu_fit = np.mean(sample_means)
				sigma_fit = np.std(sample_means)
				x_fit = np.linspace(mu_fit - 4 * sigma_fit, mu_fit + 4 * sigma_fit, 100)
				ax2_clt.plot(x_fit, norm.pdf(x_fit, mu_fit, sigma_fit), 'r-', lw=2, label='Fitted Normal PDF')

				ax2_clt.set_title('Final Distribution of Sample Means vs. Normal', fontsize=14)
				ax2_clt.set_xlabel('Sample Mean')
				ax2_clt.set_ylabel('Density')
				ax2_clt.legend()
				plt.tight_layout()

				# Display statistics
				st.pyplot(fig2_clt)

				# Calculate theoretical standard deviation of sample means (Std Error)
				if source_dist == "Exponential":
					theoretical_var_pop = 1.0 # Var(Exp(1)) = 1/lambda^2 = 1
				elif source_dist == "Uniform":
					theoretical_var_pop = 1/12 # Var(U(0,1)) = (b-a)^2/12 = 1/12
				else: # Binomial
					theoretical_var_pop = 10 * 0.5 * (1 - 0.5) # n*p*(1-p)

				theoretical_std_err = np.sqrt(theoretical_var_pop / sample_size) if sample_size > 0 else 0

				col1_res, col2_res, col3_res, col4_res = st.columns(4)
				with col1_res:
					st.metric("Population Mean", f"{theoretical_mean_pop:.4f}")
				with col2_res:
					st.metric("Mean of Sample Means", f"{mu_fit:.4f}", delta=f"{mu_fit - theoretical_mean_pop:.4f}")
				with col3_res:
					st.metric("Theoretical Std Error", f"{theoretical_std_err:.4f}")
				with col4_res:
					st.metric("Std Dev of Sample Means", f"{sigma_fit:.4f}", delta=f"{sigma_fit - theoretical_std_err:.4f}")

			else:
				st.warning("Could not generate initial frame. Need at least one sample.")

		else:
			st.warning("No samples generated.")

		progress_clt.empty()

# --- Tab 1: Law of Large Numbers ---
with tabs[1]:
	st.markdown('<h2>Law of Large Numbers (LLN)</h2>', unsafe_allow_html=True)

	st.markdown("""
    <div class="highlight">
    <strong>Law of Large Numbers:</strong> States that as the sample size increases, the sample mean approaches the true population mean.
    The more trials performed, the closer the empirical average gets to the expected value.
    </div>
    """, unsafe_allow_html=True)

	col1_lln, col2_lln = st.columns([1, 1])

	with col1_lln:
		st.markdown('<div>Select the distribution type:</div>', unsafe_allow_html=True)
		lln_dist = st.selectbox(
			"LLN Distribution",
			["Exponential", "Uniform", "Poisson"],
			index=0,
			label_visibility="collapsed"
		)

		st.markdown('<div>Maximum Number of Samples:</div>', unsafe_allow_html=True)
		max_samples_lln = st.slider("Max Samples", min_value=100, max_value=10000, value=5000, step=100,
								label_visibility="collapsed")

	with col2_lln:
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
			data_lln = np.random.exponential(true_mean, size=max_samples_lln)
			dist_title_lln = "Exponential Distribution"
		elif lln_dist == "Uniform":
			true_mean = 0.5
			data_lln = np.random.uniform(0, 1, size=max_samples_lln)
			dist_title_lln = "Uniform Distribution"
		else:  # Poisson
			true_mean = 5.0
			data_lln = np.random.poisson(true_mean, size=max_samples_lln)
			dist_title_lln = "Poisson Distribution"

		# Calculate running mean
		running_mean = np.cumsum(data_lln) / np.arange(1, max_samples_lln + 1)

		# Create animation using Plotly
		fig_lln = go.Figure()

		# Add true mean line
		fig_lln.add_trace(go.Scatter(
			x=[1, max_samples_lln],
			y=[true_mean, true_mean],
			mode='lines',
			name='Expected Value',
			line=dict(color='red', width=2, dash='dash')
		))

		# Add initial running mean trace (will be updated by frames)
		fig_lln.add_trace(go.Scatter(
			x=[1], # Start with first point
			y=[running_mean[0]],
			mode='lines',
			name='Running Mean',
			line=dict(color='blue', width=2)
		))

		# Create frames for animation
		frames_lln = []
		step_size_lln = max(1, max_samples_lln // 100)  # For smoother animation
		# Ensure step_size > 0
		if step_size_lln == 0: step_size_lln = 1

		for i in range(step_size_lln, max_samples_lln + 1, step_size_lln):
			# Ensure index doesn't exceed bounds
			current_index = min(i, max_samples_lln)
			frames_lln.append(
				go.Frame(
                    # Update the data for the running mean trace (index 1)
					data=[
                        go.Scatter(
							x=np.arange(1, current_index + 1),
							y=running_mean[:current_index]
                        )
					],
					traces=[1], # Specify which trace to update (the running mean trace)
					name=f'Frame {current_index}' # Frame name
				)
			)

		# Set up layout with animation controls
		fig_lln.update_layout(
			title=f"LLN Simulation: {dist_title_lln}",
			xaxis=dict(title="Number of Samples", type="log", range=[0, np.log10(max_samples_lln)]), # Log scale
			yaxis=dict(title="Mean"),
			updatemenus=[{
				"type": "buttons",
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "x": 0.1,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
				"buttons": [{
					"label": "Play",
					"method": "animate",
                    # args[1] defines animation settings
					"args": [None, {"frame": {"duration": 30, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 0},
                                    "mode": "immediate"}]
				}]
			}]
		)
        # Dynamically set y-axis range based on data variation
		min_y = min(np.min(running_mean), true_mean)
		max_y = max(np.max(running_mean), true_mean)
		padding = (max_y - min_y) * 0.1
		fig_lln.update_yaxes(range=[min_y - padding, max_y + padding])


		fig_lln.frames = frames_lln
		st.plotly_chart(fig_lln, use_container_width=True)

		# Display final results
		st.markdown('<h3>Final Results</h3>', unsafe_allow_html=True)
		col1_lln_res, col2_lln_res, col3_lln_res = st.columns(3)
		final_calculated_mean = running_mean[-1]
		with col1_lln_res:
			st.metric("True Mean", f"{true_mean:.4f}")
		with col2_lln_res:
			st.metric("Calculated Mean (Final)", f"{final_calculated_mean:.4f}")
		with col3_lln_res:
			st.metric("Final Difference", f"{abs(true_mean - final_calculated_mean):.4f}")

		# Show final convergence chart (Difference Plot)
		fig2_lln, ax2_lln = plt.subplots(figsize=(10, 6))

		# Plot differences from mean
		differences = np.abs(running_mean - true_mean)
		ax2_lln.plot(np.arange(1, max_samples_lln + 1), differences, color='green')
		ax2_lln.set_title('Absolute Difference Between Running Mean and Expected Value', fontsize=14)
		ax2_lln.set_xlabel('Number of Samples (Log Scale)')
		ax2_lln.set_ylabel('Absolute Difference')
		ax2_lln.set_xscale('log')
		ax2_lln.grid(True, alpha=0.3)
		plt.tight_layout()

		st.pyplot(fig2_lln)

# --- Tab 2: Distribution Properties & Convergences ---
with tabs[2]:
    st.markdown('<h2>Explore Distribution Properties and Convergences</h2>', unsafe_allow_html=True)

    explore_mode = st.radio(
        "Select Exploration Mode:",
        ("Individual Distribution Properties", "Distribution Convergences"),
        horizontal=True,
        key="explore_mode_radio"
    )

    st.divider()

    if explore_mode == "Individual Distribution Properties":
        st.markdown('<h3>Individual Distribution Properties</h3>', unsafe_allow_html=True)
        st.markdown("Select a distribution and adjust its parameters to see how its shape, mean, and variance change.")

        dist_options_prop = ["Normal", "Binomial", "Poisson", "Exponential", "Uniform"]
        selected_dist_prop = st.selectbox("Choose Distribution", dist_options_prop, key="dist_prop_select")

        col_params_prop, col_plot_prop = st.columns([1, 2])

        # Use matplotlib for interactive plot updates without full page reload
        fig_prop, ax_prop = plt.subplots(figsize=(10, 6))

        with col_params_prop:
            st.markdown("#### Parameters")
            mean_val, var_val = 0, 0 # Initialize
            plot_title = "Distribution Plot"

            if selected_dist_prop == "Normal":
                mu_prop = st.slider("Mean (μ)", -10.0, 10.0, 0.0, 0.5, key="mu_prop")
                sigma_prop = st.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0, 0.1, key="sigma_prop")
                mean_val = mu_prop
                var_val = sigma_prop**2
                x_norm = np.linspace(mu_prop - 4*sigma_prop, mu_prop + 4*sigma_prop, 500)
                y_norm = norm.pdf(x_norm, mu_prop, sigma_prop)
                ax_prop.plot(x_norm, y_norm, 'b-', lw=2)
                ax_prop.fill_between(x_norm, y_norm, alpha=0.2, color='blue')
                ax_prop.set_xlabel('Value')
                ax_prop.set_ylabel('Probability Density')
                plot_title = f'Normal PDF: μ={mu_prop:.1f}, σ={sigma_prop:.1f}'

            elif selected_dist_prop == "Binomial":
                n_prop = st.slider("Number of Trials (n)", 1, 100, 20, 1, key="n_prop")
                p_prop = st.slider("Probability of Success (p)", 0.01, 0.99, 0.5, 0.01, key="p_prop")
                mean_val = n_prop * p_prop
                var_val = n_prop * p_prop * (1 - p_prop)
                x_binom = np.arange(0, n_prop + 1)
                y_binom = binom.pmf(x_binom, n_prop, p_prop)
                ax_prop.bar(x_binom, y_binom, alpha=0.7, color='green', width=0.8)
                ax_prop.set_xlabel('Number of Successes')
                ax_prop.set_ylabel('Probability')
                # Adjust x-axis limits and ticks for clarity
                ax_prop.set_xlim(-0.5, n_prop + 0.5)
                if n_prop <= 20:
                    ax_prop.set_xticks(x_binom)
                else:
                    ax_prop.set_xticks(np.linspace(0, n_prop, min(n_prop + 1, 11), dtype=int))
                plot_title = f'Binomial PMF: n={n_prop}, p={p_prop:.2f}'

            elif selected_dist_prop == "Poisson":
                lambda_prop = st.slider("Rate (λ)", 0.1, 30.0, 5.0, 0.1, key="lambda_prop")
                mean_val = lambda_prop
                var_val = lambda_prop
                # Determine plot range dynamically
                x_max_poisson = int(max(10, lambda_prop + 4 * np.sqrt(lambda_prop))) + 1
                x_poisson = np.arange(0, x_max_poisson)
                y_poisson = poisson.pmf(x_poisson, lambda_prop)
                ax_prop.bar(x_poisson, y_poisson, alpha=0.7, color='purple', width=0.8)
                ax_prop.set_xlabel('Number of Events')
                ax_prop.set_ylabel('Probability')
                ax_prop.set_xlim(-0.5, x_max_poisson - 0.5)
                if x_max_poisson <= 20:
                    ax_prop.set_xticks(x_poisson)
                else:
                     ax_prop.set_xticks(np.linspace(0, x_max_poisson -1 , min(x_max_poisson, 11), dtype=int))
                plot_title = f'Poisson PMF: λ={lambda_prop:.1f}'

            elif selected_dist_prop == "Exponential":
                lambda_exp_prop = st.slider("Rate (λ)", 0.1, 5.0, 1.0, 0.1, key="lambda_exp_prop")
                scale_exp_prop = 1 / lambda_exp_prop
                mean_val = scale_exp_prop
                var_val = scale_exp_prop**2
                # Determine plot range dynamically
                x_max_exp = max(1.0, 6 * scale_exp_prop) # Show tail up to ~6 means
                x_exp = np.linspace(0, x_max_exp, 500)
                y_exp = expon.pdf(x_exp, scale=scale_exp_prop)
                ax_prop.plot(x_exp, y_exp, 'r-', lw=2)
                ax_prop.fill_between(x_exp, y_exp, alpha=0.2, color='red')
                ax_prop.set_xlabel('Value')
                ax_prop.set_ylabel('Probability Density')
                ax_prop.set_xlim(left=0)
                plot_title = f'Exponential PDF: λ={lambda_exp_prop:.1f} (Scale={scale_exp_prop:.2f})'

            elif selected_dist_prop == "Uniform":
                 a_prop = st.slider("Minimum (a)", -10.0, 10.0, 0.0, 0.5, key="a_prop")
                 # Ensure b > a slightly
                 min_b_u_prop = a_prop + 0.1
                 default_b_u_prop = max(min_b_u_prop, a_prop + 1.0)
                 b_prop = st.slider("Maximum (b)", min_value=min_b_u_prop, max_value=a_prop + 20.0, value=default_b_u_prop, step=0.1, key="b_prop")
                 width_prop = b_prop - a_prop
                 mean_val = (a_prop + b_prop) / 2
                 var_val = width_prop**2 / 12 if width_prop > 0 else 0
                 # Determine plot range dynamically
                 plot_range = width_prop * 0.2 # Padding on each side
                 x_unif = np.linspace(a_prop - plot_range, b_prop + plot_range, 500)
                 y_unif = uniform.pdf(x_unif, loc=a_prop, scale=width_prop)
                 ax_prop.plot(x_unif, y_unif, 'orange', lw=2)
                 # Fill only between a and b
                 x_fill_u = np.linspace(a_prop, b_prop, 100)
                 y_fill_u = uniform.pdf(x_fill_u, loc=a_prop, scale=width_prop)
                 ax_prop.fill_between(x_fill_u, y_fill_u, alpha=0.2, color='orange')
                 ax_prop.set_ylim(bottom=0)
                 ax_prop.set_xlabel('Value')
                 ax_prop.set_ylabel('Probability Density')
                 plot_title = f'Uniform PDF: a={a_prop:.1f}, b={b_prop:.1f}'

            st.markdown("#### Properties")
            st.metric("Mean", f"{mean_val:.3f}")
            st.metric("Variance", f"{var_val:.3f}")

        with col_plot_prop:
            st.markdown(f"#### {plot_title}")
            ax_prop.set_title(plot_title)
            ax_prop.grid(True, alpha=0.3)
            # Set dynamic y-limits for better visualization
            if selected_dist_prop in ["Normal", "Exponential", "Uniform"]:
                 y_max_plot = ax_prop.get_ylim()[1] # Get auto-scaled max
                 ax_prop.set_ylim(0, y_max_plot * 1.1) # Add padding
            elif selected_dist_prop in ["Binomial", "Poisson"]:
                 y_max_plot = np.max(ax_prop.containers[0].datavalues) if ax_prop.containers else 0.1
                 ax_prop.set_ylim(0, y_max_plot * 1.15) # Add padding

            st.pyplot(fig_prop)


    elif explore_mode == "Distribution Convergences":
        st.markdown('<h3>Distribution Convergences</h3>', unsafe_allow_html=True)
        st.markdown("Visualize how one distribution approximates another under certain limiting conditions.")

        convergence_options = [
            "Binomial → Poisson (Large n, Small p, n*p=λ)",
            "Binomial → Normal (Large n)",
            "Poisson → Normal (Large λ)"
        ]
        selected_conv = st.selectbox("Choose Convergence", convergence_options, key="conv_select")

        st.markdown("#### Set Parameters & Animation Controls")
        col_params_conv, col_anim_conv = st.columns([1, 2])

        # --- Binomial to Poisson ---
        if selected_conv == "Binomial → Poisson (Large n, Small p, n*p=λ)":
            with col_params_conv:
                st.markdown("""
                <div class="highlight">
                <b>Condition:</b> Binomial(n, p) approaches Poisson(λ) as n → ∞ and p → 0, such that n*p remains constant (λ).
                </div>""", unsafe_allow_html=True)
                lambda_target_b2p = st.slider("Target λ (n*p)", 1.0, 30.0, 10.0, 0.5, key="lambda_b2p")
                n_start_b2p = st.slider("Starting n", max(5, int(lambda_target_b2p)), 100, max(10, int(lambda_target_b2p)), 5, key="n_start_b2p") # Start n >= lambda
                n_end_b2p = st.slider("Ending n", n_start_b2p + 10, 500, 100, 10, key="n_end_b2p")
                frames_count_b2p = st.slider("Animation Frames", 10, 100, 40, 5, key="frames_b2p")

            if st.button("Generate Binomial to Poisson Animation", key="b2p_anim"):
                progress_b2p = st.progress(0)
                frames_list_b2p = []
                n_values_b2p = np.linspace(n_start_b2p, n_end_b2p, frames_count_b2p, dtype=int)

                # Determine consistent x-range based on Poisson mean + std dev
                max_x_poisson = int(lambda_target_b2p + 5 * np.sqrt(lambda_target_b2p)) + 1
                x_axis_b2p = np.arange(0, max_x_poisson)
                pmf_poisson_target = poisson.pmf(x_axis_b2p, lambda_target_b2p)
                max_y_val = np.max(pmf_poisson_target) * 1.2 # Base max y on Poisson

                for i, n_current in enumerate(n_values_b2p):
                    p_current = lambda_target_b2p / n_current
                    if p_current > 1.0: continue # Skip if p > 1

                    pmf_binom = binom.pmf(x_axis_b2p, n_current, p_current)
                    max_y_val = max(max_y_val, np.max(pmf_binom) * 1.1) # Adjust max_y if needed

                    frames_list_b2p.append(go.Frame(
                        data=[
                            go.Bar(x=x_axis_b2p, y=pmf_binom, name=f'Binomial(n={n_current}, p={p_current:.3f})', marker_color='blue', opacity=0.7),
                            go.Bar(x=x_axis_b2p, y=pmf_poisson_target, name=f'Poisson(λ={lambda_target_b2p})', marker_color='red', opacity=0.7)
                        ],
                        name=f'n={n_current}', # Frame name
                        layout=go.Layout(yaxis=dict(range=[0, max_y_val])) # Update yaxis range per frame if needed (or set once)
                    ))
                    progress_b2p.progress((i + 1) / frames_count_b2p)

                # Initial figure state
                p_initial_b2p = lambda_target_b2p / n_start_b2p
                pmf_binom_initial = binom.pmf(x_axis_b2p, n_start_b2p, p_initial_b2p)
                fig_conv_b2p = go.Figure(
                    data=[
                        go.Bar(x=x_axis_b2p, y=pmf_binom_initial, name=f'Binomial(n={n_start_b2p}, p={p_initial_b2p:.3f})', marker_color='blue', opacity=0.7),
                        go.Bar(x=x_axis_b2p, y=pmf_poisson_target, name=f'Poisson(λ={lambda_target_b2p})', marker_color='red', opacity=0.7)
                    ],
                    layout=go.Layout(
                        title=f'Binomial to Poisson Convergence (Target λ={lambda_target_b2p})',
                        xaxis=dict(title="Number of Events/Successes", range=[-0.5, max_x_poisson - 0.5]),
                        yaxis=dict(title="Probability", range=[0, max_y_val]), # Set consistent y range
                        barmode='overlay',
                        updatemenus=[{"type": "buttons",
                                      "direction": "left", "pad": {"r": 10, "t": 10}, "showactive": True, "x": 0.1, "xanchor": "left", "y": 1.15, "yanchor": "top",
                                      "buttons": [{"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True, "transition": {"duration": 50}, "mode": "immediate"}]}]}]
                    ),
                    frames=frames_list_b2p
                )
                with col_anim_conv:
                    st.plotly_chart(fig_conv_b2p, use_container_width=True)
                progress_b2p.empty()

        # --- Binomial to Normal ---
        elif selected_conv == "Binomial → Normal (Large n)":
             with col_params_conv:
                st.markdown("""
                <div class="highlight">
                <b>Condition:</b> Binomial(n, p) approaches Normal(μ=np, σ²=np(1-p)) as n → ∞. The approximation is good when np > 5 and n(1-p) > 5.
                </div>""", unsafe_allow_html=True)
                p_fixed_b2n = st.slider("Fixed p", 0.05, 0.95, 0.5, 0.05, key="p_fixed_b2n")
                n_start_b2n = st.slider("Starting n", 10, 100, 20, 5, key="n_start_b2n")
                n_end_b2n = st.slider("Ending n", n_start_b2n + 20, 1000, 200, 10, key="n_end_b2n")
                frames_count_b2n = st.slider("Animation Frames", 10, 100, 40, 5, key="frames_b2n")

             if st.button("Generate Binomial to Normal Animation", key="b2n_anim"):
                progress_b2n = st.progress(0)
                frames_list_b2n = []
                n_values_b2n = np.linspace(n_start_b2n, n_end_b2n, frames_count_b2n, dtype=int)

                # Determine a dynamic x-range covering the distributions during animation
                mu_end_b2n = n_end_b2n * p_fixed_b2n
                sigma_end_b2n = np.sqrt(n_end_b2n * p_fixed_b2n * (1 - p_fixed_b2n)) if n_end_b2n * p_fixed_b2n * (1 - p_fixed_b2n) > 0 else 0.1
                x_min_b2n = int(max(0, mu_end_b2n - 4.5 * sigma_end_b2n))
                x_max_b2n = int(min(n_end_b2n, mu_end_b2n + 4.5 * sigma_end_b2n)) + 1
                x_axis_disc_b2n = np.arange(x_min_b2n, x_max_b2n) # For Binomial PMF bars
                x_axis_cont_b2n = np.linspace(x_min_b2n -0.5, x_max_b2n-0.5, 400) # For Normal PDF curve

                # Determine max y value for consistent axis
                max_y_val_b2n = 0.0

                for i, n_current in enumerate(n_values_b2n):
                    mu_current = n_current * p_fixed_b2n
                    var_current = n_current * p_fixed_b2n * (1 - p_fixed_b2n)
                    sigma_current = np.sqrt(var_current) if var_current > 0 else 0.1 # Avoid sigma=0

                    pmf_binom = binom.pmf(x_axis_disc_b2n, n_current, p_fixed_b2n)
                    pdf_norm = norm.pdf(x_axis_cont_b2n, mu_current, sigma_current)
                    max_y_val_b2n = max(max_y_val_b2n, np.max(pdf_norm) * 1.1, np.max(pmf_binom) * 1.1) # Track max height

                    frames_list_b2n.append(go.Frame(
                        data=[
                            go.Bar(x=x_axis_disc_b2n, y=pmf_binom, name=f'Binomial(n={n_current})', marker_color='blue', opacity=0.6, width=1), # Width 1 for continuity illusion
                            go.Scatter(x=x_axis_cont_b2n, y=pdf_norm, name=f'Normal(μ={mu_current:.1f}, σ={sigma_current:.1f})', line=dict(color='red', width=3))
                        ],
                        name=f'n={n_current}',
                        layout=go.Layout(yaxis=dict(range=[0, max_y_val_b2n]))
                    ))
                    progress_b2n.progress((i + 1) / frames_count_b2n)

                # Initial figure state
                n_initial_b2n = n_values_b2n[0]
                mu_initial_b2n = n_initial_b2n * p_fixed_b2n
                sigma_initial_b2n = np.sqrt(n_initial_b2n * p_fixed_b2n * (1 - p_fixed_b2n)) if n_initial_b2n * p_fixed_b2n * (1-p_fixed_b2n)>0 else 0.1
                pmf_binom_initial = binom.pmf(x_axis_disc_b2n, n_initial_b2n, p_fixed_b2n)
                pdf_norm_initial = norm.pdf(x_axis_cont_b2n, mu_initial_b2n, sigma_initial_b2n)
                max_y_val_b2n = max(max_y_val_b2n, np.max(pdf_norm_initial)*1.1, np.max(pmf_binom_initial)*1.1) # Final check on max_y

                fig_conv_b2n = go.Figure(
                    data=[
                        go.Bar(x=x_axis_disc_b2n, y=pmf_binom_initial, name=f'Binomial(n={n_initial_b2n})', marker_color='blue', opacity=0.6, width=1),
                        go.Scatter(x=x_axis_cont_b2n, y=pdf_norm_initial, name=f'Normal(μ={mu_initial_b2n:.1f}, σ={sigma_initial_b2n:.1f})', line=dict(color='red', width=3))
                    ],
                    layout=go.Layout(
                        title=f'Binomial to Normal Convergence (Fixed p={p_fixed_b2n})',
                        xaxis=dict(title="Number of Successes", range=[x_min_b2n - 0.5, x_max_b2n - 0.5]),
                        yaxis=dict(title="Probability / Density", range=[0, max_y_val_b2n]),
                        barmode='overlay',
                        updatemenus=[{"type": "buttons",
                                      "direction": "left", "pad": {"r": 10, "t": 10}, "showactive": True, "x": 0.1, "xanchor": "left", "y": 1.15, "yanchor": "top",
                                      "buttons": [{"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True, "transition": {"duration": 50}, "mode": "immediate"}]}]}]
                    ),
                    frames=frames_list_b2n
                )
                with col_anim_conv:
                    st.plotly_chart(fig_conv_b2n, use_container_width=True)
                progress_b2n.empty()

        # --- Poisson to Normal ---
        elif selected_conv == "Poisson → Normal (Large λ)":
            with col_params_conv:
                st.markdown("""
                <div class="highlight">
                <b>Condition:</b> Poisson(λ) approaches Normal(μ=λ, σ²=λ) as λ → ∞. The approximation is good for λ > 10 or 20.
                </div>""", unsafe_allow_html=True)
                lambda_start_p2n = st.slider("Starting λ", 5.0, 50.0, 10.0, 1.0, key="lambda_start_p2n")
                lambda_end_p2n = st.slider("Ending λ", lambda_start_p2n + 10, 200.0, 50.0, 5.0, key="lambda_end_p2n")
                frames_count_p2n = st.slider("Animation Frames", 10, 100, 40, 5, key="frames_p2n")

            if st.button("Generate Poisson to Normal Animation", key="p2n_anim"):
                progress_p2n = st.progress(0)
                frames_list_p2n = []
                lambda_values_p2n = np.linspace(lambda_start_p2n, lambda_end_p2n, frames_count_p2n)

                # Determine a dynamic x-range covering the distributions during animation
                mu_end_p2n = lambda_end_p2n
                sigma_end_p2n = np.sqrt(lambda_end_p2n)
                x_min_p2n = int(max(0, mu_end_p2n - 4.5 * sigma_end_p2n))
                x_max_p2n = int(mu_end_p2n + 4.5 * sigma_end_p2n) + 1
                x_axis_disc_p2n = np.arange(x_min_p2n, x_max_p2n) # For Poisson PMF bars
                x_axis_cont_p2n = np.linspace(x_min_p2n - 0.5, x_max_p2n - 0.5, 400) # For Normal PDF curve

                # Determine max y value for consistent axis
                max_y_val_p2n = 0.0

                for i, lambda_current in enumerate(lambda_values_p2n):
                    mu_current_p2n = lambda_current
                    sigma_current_p2n = np.sqrt(lambda_current)

                    pmf_poisson = poisson.pmf(x_axis_disc_p2n, lambda_current)
                    pdf_norm = norm.pdf(x_axis_cont_p2n, mu_current_p2n, sigma_current_p2n)
                    max_y_val_p2n = max(max_y_val_p2n, np.max(pdf_norm) * 1.1, np.max(pmf_poisson) * 1.1) # Track max height

                    frames_list_p2n.append(go.Frame(
                        data=[
                            go.Bar(x=x_axis_disc_p2n, y=pmf_poisson, name=f'Poisson(λ={lambda_current:.1f})', marker_color='purple', opacity=0.6, width=1), # Width 1
                            go.Scatter(x=x_axis_cont_p2n, y=pdf_norm, name=f'Normal(μ={mu_current_p2n:.1f}, σ={sigma_current_p2n:.1f})', line=dict(color='red', width=3))
                        ],
                        name=f'λ={lambda_current:.1f}',
                        layout=go.Layout(yaxis=dict(range=[0, max_y_val_p2n]))
                    ))
                    progress_p2n.progress((i + 1) / frames_count_p2n)

                # Initial figure state
                lambda_initial_p2n = lambda_values_p2n[0]
                mu_initial_p2n = lambda_initial_p2n
                sigma_initial_p2n = np.sqrt(lambda_initial_p2n)
                pmf_poisson_initial = poisson.pmf(x_axis_disc_p2n, lambda_initial_p2n)
                pdf_norm_initial = norm.pdf(x_axis_cont_p2n, mu_initial_p2n, sigma_initial_p2n)
                max_y_val_p2n = max(max_y_val_p2n, np.max(pdf_norm_initial) * 1.1, np.max(pmf_poisson_initial) * 1.1) # Final check

                fig_conv_p2n = go.Figure(
                    data=[
                        go.Bar(x=x_axis_disc_p2n, y=pmf_poisson_initial, name=f'Poisson(λ={lambda_initial_p2n:.1f})', marker_color='purple', opacity=0.6, width=1),
                        go.Scatter(x=x_axis_cont_p2n, y=pdf_norm_initial, name=f'Normal(μ={mu_initial_p2n:.1f}, σ={sigma_initial_p2n:.1f})', line=dict(color='red', width=3))
                    ],
                    layout=go.Layout(
                        title=f'Poisson to Normal Convergence',
                        xaxis=dict(title="Number of Events", range=[x_min_p2n - 0.5, x_max_p2n - 0.5]),
                        yaxis=dict(title="Probability / Density", range=[0, max_y_val_p2n]),
                        barmode='overlay',
                        updatemenus=[{"type": "buttons",
                                      "direction": "left", "pad": {"r": 10, "t": 10}, "showactive": True, "x": 0.1, "xanchor": "left", "y": 1.15, "yanchor": "top",
                                      "buttons": [{"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True, "transition": {"duration": 50}, "mode": "immediate"}]}]}]
                    ),
                    frames=frames_list_p2n
                )
                with col_anim_conv:
                    st.plotly_chart(fig_conv_p2n, use_container_width=True)
                progress_p2n.empty()

# --- Tab 3: Discrete Distributions Examples --- (Original Tab 2)
with tabs[3]:
	st.markdown('<h2>Discrete Probability Distribution Examples</h2>', unsafe_allow_html=True) # Updated Title

	discrete_dist = st.selectbox(
		"Select Discrete Distribution Example",
		["Binomial Distribution", "Poisson Distribution", "Geometric Distribution"],
        key="discrete_select_example"
	)

	col1_disc, col2_disc = st.columns([1, 2])

	if discrete_dist == "Binomial Distribution":
		with col1_disc:
			st.markdown(
				'<div class="highlight">The Binomial distribution describes the number of successes in a fixed number of independent trials, with a constant probability of success in each trial.</div>',
				unsafe_allow_html=True)

			st.markdown('<div>Number of Trials (n):</div>', unsafe_allow_html=True)
			n_trials_disc = st.slider("n", min_value=1, max_value=50, value=10, step=1, label_visibility="collapsed", key="n_trials_disc")

			st.markdown('<div>Probability of Success (p):</div>', unsafe_allow_html=True)
			p_success_disc = st.slider("p", min_value=0.01, max_value=0.99, value=0.5, step=0.01, label_visibility="collapsed", key="p_success_disc")

			mean_binom_disc = n_trials_disc * p_success_disc
			var_binom_disc = n_trials_disc * p_success_disc * (1 - p_success_disc)
			std_binom_disc = np.sqrt(var_binom_disc)
			st.markdown(f"""
            <div>
            <strong>Binomial Distribution Properties:</strong><br>
            Mean = n × p = {mean_binom_disc:.2f}<br>
            Variance = n × p × (1-p) = {var_binom_disc:.2f}<br>
            Standard Deviation = {std_binom_disc:.2f}
            </div>
            """, unsafe_allow_html=True)

			n_sim_disc_binom = st.slider("Number of Simulations:", min_value=100, max_value=10000, value=1000, step=100, key="n_sim_disc_binom")

		with col2_disc:
			st.markdown("#### Visualization and Simulation")
			if st.button("Show Probability Mass Function (PMF)", key="binom_pmf"):
				# Create PMF plot
				x_binom_pmf = np.arange(0, n_trials_disc + 1)
				pmf_binom_vals = binom.pmf(x_binom_pmf, n_trials_disc, p_success_disc)

				fig_binom_pmf, ax_binom_pmf = plt.subplots(figsize=(10, 6))
				ax_binom_pmf.bar(x_binom_pmf, pmf_binom_vals, alpha=0.7, color='blue')
				ax_binom_pmf.set_xlabel('Number of Successes')
				ax_binom_pmf.set_ylabel('Probability')
				ax_binom_pmf.set_title(f'PMF for Binomial Distribution: n={n_trials_disc}, p={p_success_disc:.2f}')
				ax_binom_pmf.grid(alpha=0.3)
				ax_binom_pmf.set_xticks(x_binom_pmf) # Show all integer ticks if possible
				st.pyplot(fig_binom_pmf)

			if st.button("Run Distribution Simulation", key="binom_sim"):
				# Simulate binomial samples
				samples_binom = np.random.binomial(n_trials_disc, p_success_disc, size=n_sim_disc_binom)

				# Create animation of build-up using Plotly
				num_frames_binom = 50
				step_size_binom = max(1, n_sim_disc_binom // num_frames_binom)

				fig_binom_sim = go.Figure()

				# Theoretical PMF (for comparison)
				x_theory_binom = np.arange(0, n_trials_disc + 1)
				pmf_theory_binom = binom.pmf(x_theory_binom, n_trials_disc, p_success_disc)

				fig_binom_sim.add_trace(go.Bar(
					x=x_theory_binom,
					y=pmf_theory_binom,
					name='Theoretical PMF',
					marker_color='rgba(255, 0, 0, 0.5)', # Red, semi-transparent
					opacity=0.6
				))

				# Initial histogram (first few samples) - Normalized
				initial_samples_binom = samples_binom[:step_size_binom]
				hist_data_binom, bin_edges_binom = np.histogram(initial_samples_binom, bins=np.arange(-0.5, n_trials_disc + 1.5), density=True)
				# Adjust density to be comparable to PMF (multiply by bin width, which is 1)
				hist_data_binom = hist_data_binom * 1.0

				fig_binom_sim.add_trace(go.Bar(
					x=x_theory_binom, # Use same x-axis as PMF
					y=hist_data_binom,
					name='Simulated Samples (Normalized Freq)',
					marker_color='rgba(0, 0, 255, 0.6)', # Blue, semi-transparent
					opacity=0.6
				))

				# Create frames
				frames_binom_sim = []
				max_y_binom_sim = np.max(pmf_theory_binom) * 1.1 # Start with theoretical max

				for i in range(step_size_binom, n_sim_disc_binom + 1, step_size_binom):
					current_samples_binom = samples_binom[:i]
					hist_data_binom_frame, _ = np.histogram(current_samples_binom, bins=np.arange(-0.5, n_trials_disc + 1.5), density=True)
					hist_data_binom_frame = hist_data_binom_frame * 1.0 # Adjust density
					max_y_binom_sim = max(max_y_binom_sim, np.max(hist_data_binom_frame) * 1.1) # Update max y if needed

					frames_binom_sim.append(go.Frame(
						data=[
                            go.Bar( # Update the histogram data (trace 1)
                                y=hist_data_binom_frame
                            )
						],
                        traces=[1], # Specify only trace 1 needs updating
						name=f'Samples: {i}',
                        layout=go.Layout(yaxis=dict(range=[0, max_y_binom_sim])) # Update y range in frame
					))

				fig_binom_sim.frames = frames_binom_sim

				fig_binom_sim.update_layout(
					title=f'Binomial Simulation: n={n_trials_disc}, p={p_success_disc:.2f}',
					xaxis=dict(title='Number of Successes', tickmode='array', tickvals=x_theory_binom, range=[-0.5, n_trials_disc + 0.5]),
					yaxis=dict(title='Probability / Normalized Frequency', range=[0, max_y_binom_sim]),
					barmode='overlay',
					updatemenus=[{
						"type": "buttons",
						"direction": "left", "pad": {"r": 10, "t": 10}, "showactive": True, "x": 0.1, "xanchor": "left", "y": 1.15, "yanchor": "top",
						"buttons": [{
							"label": "Play",
							"method": "animate",
							"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"}]
						}]
					}]
				)

				st.plotly_chart(fig_binom_sim, use_container_width=True)

				# Display statistics comparison
				st.markdown('<h6>Comparison: Theoretical vs. Simulated</h6>', unsafe_allow_html=True)
				mean_sim_binom = np.mean(samples_binom)
				std_sim_binom = np.std(samples_binom)

				col1_binom, col2_binom, col3_binom = st.columns(3)
				with col1_binom:
					st.metric("Theoretical Mean", f"{mean_binom_disc:.4f}", delta=f"{mean_sim_binom - mean_binom_disc:.4f}")
				with col2_binom:
					st.metric("Simulated Mean", f"{mean_sim_binom:.4f}")
				with col3_binom:
					st.metric("Theoretical Std Dev", f"{std_binom_disc:.4f}", delta=f"{std_sim_binom - std_binom_disc:.4f}")

	elif discrete_dist == "Poisson Distribution":
		with col1_disc:
			st.markdown(
				'<div class="highlight">The Poisson distribution describes the number of events occurring in a fixed interval of time or space, when these events are independent and occur at a constant average rate.</div>',
				unsafe_allow_html=True)

			st.markdown('<div>Rate (λ):</div>', unsafe_allow_html=True)
			lambda_param_disc = st.slider("lambda", min_value=0.1, max_value=30.0, value=5.0, step=0.1,
									 label_visibility="collapsed", key="lambda_param_disc")

			mean_poisson_disc = lambda_param_disc
			var_poisson_disc = lambda_param_disc
			std_poisson_disc = np.sqrt(var_poisson_disc)
			st.markdown(f"""
            <div>
            <strong>Poisson Distribution Properties:</strong><br>
            Mean = λ = {mean_poisson_disc:.2f}<br>
            Variance = λ = {var_poisson_disc:.2f}<br>
            Standard Deviation = √λ = {std_poisson_disc:.2f}
            </div>
            """, unsafe_allow_html=True)

			n_sim_disc_poisson = st.slider("Number of Simulations:", min_value=100, max_value=10000, value=1000, step=100,
							  key="poisson_sim_count")

		with col2_disc:
			st.markdown("#### Visualization and Simulation")
			if st.button("Show Probability Mass Function (PMF)", key="poisson_pmf"):
				# Create PMF plot
				x_max_poisson_pmf = int(max(10, lambda_param_disc + 4 * np.sqrt(lambda_param_disc))) + 1 # Reasonable range
				x_poisson_pmf = np.arange(0, x_max_poisson_pmf)
				pmf_poisson_vals = poisson.pmf(x_poisson_pmf, lambda_param_disc)

				fig_poisson_pmf, ax_poisson_pmf = plt.subplots(figsize=(10, 6))
				ax_poisson_pmf.bar(x_poisson_pmf, pmf_poisson_vals, alpha=0.7, color='purple')
				ax_poisson_pmf.set_xlabel('Number of Events')
				ax_poisson_pmf.set_ylabel('Probability')
				ax_poisson_pmf.set_title(f'PMF for Poisson Distribution: λ={lambda_param_disc:.1f}')
				ax_poisson_pmf.grid(alpha=0.3)
				if x_max_poisson_pmf <= 20:
				    ax_poisson_pmf.set_xticks(x_poisson_pmf)
				else:
				    ax_poisson_pmf.set_xticks(np.linspace(0, x_max_poisson_pmf -1 , min(x_max_poisson_pmf, 11), dtype=int))

				st.pyplot(fig_poisson_pmf)

			if st.button("Run Distribution Simulation", key="poisson_sim"):
				# Simulate Poisson samples
				samples_poisson = np.random.poisson(lambda_param_disc, size=n_sim_disc_poisson)

				# Create animation of build-up
				num_frames_poisson = 50
				step_size_poisson = max(1, n_sim_disc_poisson // num_frames_poisson)

				# Define consistent range for histogram and PMF
				x_max_sim_poisson = int(max(np.max(samples_poisson), lambda_param_disc + 4 * np.sqrt(lambda_param_disc))) + 1
				x_range_poisson = np.arange(0, x_max_sim_poisson)

				# Theoretical PMF
				pmf_theory_poisson = poisson.pmf(x_range_poisson, lambda_param_disc)

				fig_poisson_sim = go.Figure()

				fig_poisson_sim.add_trace(go.Bar(
					x=x_range_poisson,
					y=pmf_theory_poisson,
					name='Theoretical PMF',
					marker_color='rgba(255, 0, 0, 0.5)',
					opacity=0.6
				))

				# Initial histogram (first few samples) - Normalized
				initial_samples_poisson = samples_poisson[:step_size_poisson]
				hist_data_poisson, bin_edges_poisson = np.histogram(initial_samples_poisson, bins=np.arange(-0.5, x_max_sim_poisson + 0.5), density=True)
				hist_data_poisson = hist_data_poisson * 1.0 # Adjust density

				fig_poisson_sim.add_trace(go.Bar(
					x=x_range_poisson,
					y=hist_data_poisson,
					name='Simulated Samples (Normalized Freq)',
					marker_color='rgba(0, 0, 255, 0.6)',
					opacity=0.6
				))

				# Create frames
				frames_poisson_sim = []
				max_y_poisson_sim = np.max(pmf_theory_poisson) * 1.1

				for i in range(step_size_poisson, n_sim_disc_poisson + 1, step_size_poisson):
					current_samples_poisson = samples_poisson[:i]
					hist_data_poisson_frame, _ = np.histogram(current_samples_poisson, bins=np.arange(-0.5, x_max_sim_poisson + 0.5), density=True)
					hist_data_poisson_frame = hist_data_poisson_frame * 1.0
					max_y_poisson_sim = max(max_y_poisson_sim, np.max(hist_data_poisson_frame) * 1.1)

					frames_poisson_sim.append(go.Frame(
						data=[go.Bar(y=hist_data_poisson_frame)],
                        traces=[1], # Update only trace 1
						name=f'Samples: {i}',
                        layout=go.Layout(yaxis=dict(range=[0, max_y_poisson_sim]))
					))

				fig_poisson_sim.frames = frames_poisson_sim

				fig_poisson_sim.update_layout(
					title=f'Poisson Simulation: λ={lambda_param_disc:.1f}',
					xaxis=dict(title='Number of Events', tickmode='auto', range=[-0.5, x_max_sim_poisson - 0.5]),
					yaxis=dict(title='Probability / Normalized Frequency', range=[0, max_y_poisson_sim]),
					barmode='overlay',
					updatemenus=[{
						"type": "buttons",
						"direction": "left", "pad": {"r": 10, "t": 10}, "showactive": True, "x": 0.1, "xanchor": "left", "y": 1.15, "yanchor": "top",
						"buttons": [{
							"label": "Play",
							"method": "animate",
							"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"}]
						}]
					}]
				)

				st.plotly_chart(fig_poisson_sim, use_container_width=True)

				# Display statistics comparison
				st.markdown('<h6>Comparison: Theoretical vs. Simulated</h6>', unsafe_allow_html=True)
				mean_sim_poisson = np.mean(samples_poisson)
				std_sim_poisson = np.std(samples_poisson)

				col1_poisson, col2_poisson, col3_poisson = st.columns(3)
				with col1_poisson:
					st.metric("Theoretical Mean", f"{mean_poisson_disc:.4f}",
							  delta=f"{mean_sim_poisson - mean_poisson_disc:.4f}")
				with col2_poisson:
					st.metric("Simulated Mean", f"{mean_sim_poisson:.4f}")
				with col3_poisson:
					st.metric("Theoretical Std Dev", f"{std_poisson_disc:.4f}",
							  delta=f"{std_sim_poisson - std_poisson_disc:.4f}")

	else:  # Geometric distribution
		with col1_disc:
			st.markdown(
				'<div class="highlight">The Geometric distribution describes the number of trials needed to get the first success, with a constant probability of success in each trial. (Note: numpy uses this definition)</div>',
				unsafe_allow_html=True)

			st.markdown('<div>Probability of Success (p):</div>', unsafe_allow_html=True)
			p_success_geom = st.slider("p_geom", min_value=0.01, max_value=0.99, value=0.25, step=0.01,
								  label_visibility="collapsed", key="p_success_geom")

			mean_geom_disc = 1 / p_success_geom if p_success_geom > 0 else float('inf')
			var_geom_disc = (1 - p_success_geom) / (p_success_geom ** 2) if p_success_geom > 0 else float('inf')
			std_geom_disc = np.sqrt(var_geom_disc) if p_success_geom > 0 else float('inf')
			st.markdown(f"""
            <div>
            <strong>Geometric Distribution Properties:</strong><br>
            Mean = 1/p = {mean_geom_disc:.2f}<br>
            Variance = (1-p)/p² = {var_geom_disc:.2f}<br>
            Standard Deviation = {std_geom_disc:.2f}
            </div>
            """, unsafe_allow_html=True)

			n_sim_disc_geom = st.slider("Number of Simulations:", min_value=100, max_value=10000, value=1000, step=100,
							  key="geom_sim_count")

		with col2_disc:
			st.markdown("#### Visualization and Simulation")
			if st.button("Show Probability Mass Function (PMF)", key="geom_pmf"):
				# Calculate appropriate range for x
				# Show up to percentile 99 or reasonable max
				x_max_geom_pmf = int(max(10, min(np.ceil(expon.ppf(0.99, scale=1/p_success_geom)), 60))) +1 if p_success_geom > 0 else 10
				x_geom_pmf = np.arange(1, x_max_geom_pmf) # Trials start from 1

				# Calculate PMF manually for trials needed def: P(X=k) = (1-p)^(k-1) * p
				pmf_geom_vals = [(1 - p_success_geom) ** (k - 1) * p_success_geom for k in x_geom_pmf] if p_success_geom > 0 else np.zeros_like(x_geom_pmf)

				fig_geom_pmf, ax_geom_pmf = plt.subplots(figsize=(10, 6))
				ax_geom_pmf.bar(x_geom_pmf, pmf_geom_vals, alpha=0.7, color='orange')
				ax_geom_pmf.set_xlabel('Number of Trials until First Success')
				ax_geom_pmf.set_ylabel('Probability')
				ax_geom_pmf.set_title(f'PMF for Geometric Distribution: p={p_success_geom:.2f}')
				ax_geom_pmf.grid(alpha=0.3)
				ax_geom_pmf.set_xticks(np.arange(1, x_max_geom_pmf, max(1, (x_max_geom_pmf-1)//10))) # Adjust ticks
				st.pyplot(fig_geom_pmf)

			if st.button("Run Distribution Simulation", key="geom_sim"):
				# Simulate geometric samples (numpy gives trials needed)
				samples_geom = np.random.geometric(p_success_geom, size=n_sim_disc_geom)

				# Create animation of build-up
				num_frames_geom = 50
				step_size_geom = max(1, n_sim_disc_geom // num_frames_geom)

				# Define range for histogram, considering outliers
				# Limit x-axis based on 99th percentile or a reasonable multiple of the mean
				x_max_sim_geom = int(max(10, min(np.percentile(samples_geom, 99.5), 5 / p_success_geom, 100))) + 1
				x_range_geom = np.arange(1, x_max_sim_geom) # Trials start from 1

				# Theoretical PMF for the plot range
				pmf_theory_geom = [(1 - p_success_geom) ** (k - 1) * p_success_geom for k in x_range_geom] if p_success_geom > 0 else np.zeros_like(x_range_geom)

				fig_geom_sim = go.Figure()

				fig_geom_sim.add_trace(go.Bar(
					x=x_range_geom,
					y=pmf_theory_geom,
					name='Theoretical PMF',
					marker_color='rgba(255, 0, 0, 0.5)',
					opacity=0.6
				))

				# Initial histogram (first few samples) - Normalized
				initial_samples_geom = samples_geom[:step_size_geom]
				# Bins should align with integers: [0.5, 1.5), [1.5, 2.5), ...
				hist_bins_geom = np.arange(0.5, x_max_sim_geom + 0.5)
				hist_data_geom, _ = np.histogram(initial_samples_geom, bins=hist_bins_geom, density=True)
				hist_data_geom = hist_data_geom * 1.0 # Adjust density

				fig_geom_sim.add_trace(go.Bar(
					x=x_range_geom, # Use integer x values for bars
					y=hist_data_geom,
					name='Simulated Samples (Normalized Freq)',
					marker_color='rgba(0, 0, 255, 0.6)',
					opacity=0.6
				))

				# Create frames
				frames_geom_sim = []
				max_y_geom_sim = np.max(pmf_theory_geom) * 1.1 if len(pmf_theory_geom) > 0 else 0.1

				for i in range(step_size_geom, n_sim_disc_geom + 1, step_size_geom):
					current_samples_geom = samples_geom[:i]
					hist_data_geom_frame, _ = np.histogram(current_samples_geom, bins=hist_bins_geom, density=True)
					hist_data_geom_frame = hist_data_geom_frame * 1.0
					max_y_geom_sim = max(max_y_geom_sim, np.max(hist_data_geom_frame) * 1.1)

					frames_geom_sim.append(go.Frame(
						data=[go.Bar(y=hist_data_geom_frame)],
                        traces=[1], # Update only trace 1
						name=f'Samples: {i}',
                        layout=go.Layout(yaxis=dict(range=[0, max_y_geom_sim]))
					))

				fig_geom_sim.frames = frames_geom_sim

				fig_geom_sim.update_layout(
					title=f'Geometric Simulation: p={p_success_geom:.2f}',
					xaxis=dict(title='Number of Trials until First Success', tickmode='auto', range=[0.5, x_max_sim_geom - 0.5]),
					yaxis=dict(title='Probability / Normalized Frequency', range=[0, max_y_geom_sim]),
					barmode='overlay',
					updatemenus=[{
						"type": "buttons",
						"direction": "left", "pad": {"r": 10, "t": 10}, "showactive": True, "x": 0.1, "xanchor": "left", "y": 1.15, "yanchor": "top",
						"buttons": [{
							"label": "Play",
							"method": "animate",
							"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"}]
						}]
					}]
				)

				st.plotly_chart(fig_geom_sim, use_container_width=True)

				# Display statistics comparison
				st.markdown('<h6>Comparison: Theoretical vs. Simulated</h6>', unsafe_allow_html=True)
				mean_sim_geom = np.mean(samples_geom)
				std_sim_geom = np.std(samples_geom)

				col1_geom, col2_geom, col3_geom = st.columns(3)
				with col1_geom:
					st.metric("Theoretical Mean", f"{mean_geom_disc:.4f}",
							  delta=f"{mean_sim_geom - mean_geom_disc:.4f}" if p_success_geom > 0 else "N/A")
				with col2_geom:
					st.metric("Simulated Mean", f"{mean_sim_geom:.4f}")
				with col3_geom:
					st.metric("Theoretical Std Dev", f"{std_geom_disc:.4f}",
							  delta=f"{std_sim_geom - std_geom_disc:.4f}" if p_success_geom > 0 else "N/A")

# --- Tab 4: Continuous Distributions Examples --- (Original Tab 3)
with tabs[4]:
	st.markdown('<h2>Continuous Probability Distribution Examples</h2>', unsafe_allow_html=True) # Updated title

	continuous_dist = st.selectbox(
		"Select Continuous Distribution Example",
		["Normal Distribution", "Exponential Distribution", "Uniform Distribution"],
        key="cont_select_example"
	)

	col1_cont, col2_cont = st.columns([1, 2])

	if continuous_dist == "Normal Distribution":
		with col1_cont:
			st.markdown(
				'<div class="highlight">The Normal (or Gaussian) distribution is one of the most important distributions in statistics and appears in many natural phenomena.</div>',
				unsafe_allow_html=True)

			st.markdown('<div>Mean (μ):</div>', unsafe_allow_html=True)
			mu_cont = st.slider("mu", min_value=-10.0, max_value=10.0, value=0.0, step=0.5, label_visibility="collapsed", key="mu_cont")

			st.markdown('<div>Standard Deviation (σ):</div>', unsafe_allow_html=True)
			sigma_cont = st.slider("sigma", min_value=0.1, max_value=5.0, value=1.0, step=0.1, label_visibility="collapsed", key="sigma_cont")

			mean_norm_cont = mu_cont
			var_norm_cont = sigma_cont ** 2
			st.markdown(f"""
            <div>
            <strong>Normal Distribution Properties:</strong><br>
            Mean = μ = {mean_norm_cont:.2f}<br>
            Variance = σ² = {var_norm_cont:.2f}<br>
            Standard Deviation = σ = {sigma_cont:.2f}
            </div>
            """, unsafe_allow_html=True)

			n_sim_cont_norm = st.slider("Number of Simulations:", min_value=100, max_value=10000, value=1000, step=100,
							  key="normal_sim_count")

		with col2_cont:
			st.markdown("#### Visualization and Simulation")
			if st.button("Show Probability Density Function (PDF)", key="normal_pdf"):
				# Create PDF plot
				x_norm_pdf = np.linspace(mu_cont - 4 * sigma_cont, mu_cont + 4 * sigma_cont, 1000)
				pdf_norm_vals = norm.pdf(x_norm_pdf, mu_cont, sigma_cont)

				fig_norm_pdf, ax_norm_pdf = plt.subplots(figsize=(10, 6))
				ax_norm_pdf.plot(x_norm_pdf, pdf_norm_vals, 'r-', lw=2)
				ax_norm_pdf.fill_between(x_norm_pdf, pdf_norm_vals, alpha=0.3, color='skyblue')
				ax_norm_pdf.set_xlabel('Value')
				ax_norm_pdf.set_ylabel('Probability Density')
				ax_norm_pdf.set_title(f'PDF for Normal Distribution: μ={mu_cont:.1f}, σ={sigma_cont:.1f}')
				ax_norm_pdf.grid(alpha=0.3)
				st.pyplot(fig_norm_pdf)

			if st.button("Run Distribution Simulation", key="normal_sim"):
				# Simulate normal samples
				samples_norm = np.random.normal(mu_cont, sigma_cont, size=n_sim_cont_norm)

				# Create animation of build-up
				num_frames_norm = 50
				step_size_norm = max(1, n_sim_cont_norm // num_frames_norm)

				# Define range for histogram and PDF plot
				x_min_norm_sim, x_max_norm_sim = mu_cont - 4 * sigma_cont, mu_cont + 4 * sigma_cont
				x_range_norm_sim = np.linspace(x_min_norm_sim, x_max_norm_sim, 400)
				bins_norm_sim = 50 # More bins for continuous

				# Theoretical PDF
				pdf_theory_norm = norm.pdf(x_range_norm_sim, mu_cont, sigma_cont)

				fig_norm_sim = go.Figure()

				# Add theoretical PDF
				fig_norm_sim.add_trace(go.Scatter(
					x=x_range_norm_sim,
					y=pdf_theory_norm,
					mode='lines',
					name='Theoretical PDF',
					line=dict(color='red', width=2)
				))

				# Initial histogram (first few samples) - Density = True
				initial_samples_norm = samples_norm[:step_size_norm]
				hist_data_norm, bin_edges_norm = np.histogram(initial_samples_norm, bins=bins_norm_sim,
													range=(x_min_norm_sim, x_max_norm_sim), density=True)
				bin_centers_norm = (bin_edges_norm[:-1] + bin_edges_norm[1:]) / 2

				fig_norm_sim.add_trace(go.Bar(
					x=bin_centers_norm,
					y=hist_data_norm,
					name='Simulated Samples (Density)',
					marker_color='rgba(0, 0, 255, 0.6)',
					opacity=0.6
				))

				# Create frames
				frames_norm_sim = []
				max_y_norm_sim = np.max(pdf_theory_norm) * 1.1

				for i in range(step_size_norm, n_sim_cont_norm + 1, step_size_norm):
					current_samples_norm = samples_norm[:i]
					hist_data_norm_frame, _ = np.histogram(current_samples_norm, bins=bins_norm_sim,
												range=(x_min_norm_sim, x_max_norm_sim), density=True)
					max_y_norm_sim = max(max_y_norm_sim, np.max(hist_data_norm_frame) * 1.1)

					frames_norm_sim.append(go.Frame(
						data=[go.Bar(y=hist_data_norm_frame)],
                        traces=[1], # Update only trace 1
						name=f'Samples: {i}',
                        layout=go.Layout(yaxis=dict(range=[0, max_y_norm_sim]))
					))

				fig_norm_sim.frames = frames_norm_sim

				fig_norm_sim.update_layout(
					title=f'Normal Simulation: μ={mu_cont:.1f}, σ={sigma_cont:.1f}',
					xaxis=dict(title='Value', range=[x_min_norm_sim, x_max_norm_sim]),
					yaxis=dict(title='Density', range=[0, max_y_norm_sim]),
                    barmode='overlay',
					updatemenus=[{
						"type": "buttons",
						"direction": "left", "pad": {"r": 10, "t": 10}, "showactive": True, "x": 0.1, "xanchor": "left", "y": 1.15, "yanchor": "top",
						"buttons": [{
							"label": "Play",
							"method": "animate",
							"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"}]
						}]
					}]
				)

				st.plotly_chart(fig_norm_sim, use_container_width=True)

				# Display statistics comparison
				st.markdown('<h6>Comparison: Theoretical vs. Simulated</h6>', unsafe_allow_html=True)
				mean_sim_norm = np.mean(samples_norm)
				std_sim_norm = np.std(samples_norm)

				col1_norm, col2_norm, col3_norm = st.columns(3)
				with col1_norm:
					st.metric("Theoretical Mean", f"{mean_norm_cont:.4f}",
							  delta=f"{mean_sim_norm - mean_norm_cont:.4f}")
				with col2_norm:
					st.metric("Simulated Mean", f"{mean_sim_norm:.4f}")
				with col3_norm:
					st.metric("Theoretical Std Dev", f"{sigma_cont:.4f}",
							  delta=f"{std_sim_norm - sigma_cont:.4f}")

				# QQ Plot
				st.markdown('<h6>QQ Plot for Normality Check</h6>', unsafe_allow_html=True)

				fig_qq, ax_qq = plt.subplots(figsize=(6, 6)) # Smaller QQ plot
				probplot(samples_norm, dist="norm", plot=ax_qq)
				ax_qq.set_title('QQ Plot vs Normal')
				st.pyplot(fig_qq)

	elif continuous_dist == "Exponential Distribution":
		with col1_cont:
			st.markdown(
				'<div class="highlight">The Exponential distribution describes the time intervals between successive events that occur at a constant and independent rate (λ). Scipy uses scale = 1/λ.</div>',
				unsafe_allow_html=True)

			st.markdown('<div>Event Rate (λ):</div>', unsafe_allow_html=True)
			lambda_param_cont = st.slider("lambda_exp", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
									 label_visibility="collapsed", key="lambda_param_cont")
			scale_param_cont = 1 / lambda_param_cont

			mean_exp_cont = scale_param_cont
			var_exp_cont = scale_param_cont ** 2
			std_exp_cont = scale_param_cont
			st.markdown(f"""
            <div>
            <strong>Exponential Distribution Properties:</strong><br>
            Mean = 1/λ = {mean_exp_cont:.2f}<br>
            Variance = 1/λ² = {var_exp_cont:.2f}<br>
            Standard Deviation = 1/λ = {std_exp_cont:.2f}
            </div>
            """, unsafe_allow_html=True)

			n_sim_cont_exp = st.slider("Number of Simulations:", min_value=100, max_value=10000, value=1000, step=100,
							  key="exp_sim_count")

		with col2_cont:
			st.markdown("#### Visualization and Simulation")
			if st.button("Show Probability Density Function (PDF)", key="exp_pdf"):
				# Create PDF plot
				x_max_exp_pdf = max(1.0, 6 * scale_param_cont) # Show ~6 means
				x_exp_pdf = np.linspace(0, x_max_exp_pdf, 1000)
				pdf_exp_vals = expon.pdf(x_exp_pdf, scale=scale_param_cont)

				fig_exp_pdf, ax_exp_pdf = plt.subplots(figsize=(10, 6))
				ax_exp_pdf.plot(x_exp_pdf, pdf_exp_vals, 'r-', lw=2)
				ax_exp_pdf.fill_between(x_exp_pdf, pdf_exp_vals, alpha=0.3, color='skyblue')
				ax_exp_pdf.set_xlabel('Value')
				ax_exp_pdf.set_ylabel('Probability Density')
				ax_exp_pdf.set_title(f'PDF for Exponential Distribution: λ={lambda_param_cont:.1f}')
				ax_exp_pdf.grid(alpha=0.3)
				ax_exp_pdf.set_xlim(left=0)
				st.pyplot(fig_exp_pdf)

			if st.button("Run Distribution Simulation", key="exp_sim"):
				# Simulate exponential samples
				samples_exp = np.random.exponential(scale=scale_param_cont, size=n_sim_cont_exp)

				# Create animation of build-up
				num_frames_exp = 50
				step_size_exp = max(1, n_sim_cont_exp // num_frames_exp)

				# Define range for histogram (use 99th percentile to avoid extreme values)
				x_max_exp_sim = max(1.0, np.percentile(samples_exp, 99.5), 6 * scale_param_cont)
				x_range_exp_sim = np.linspace(0, x_max_exp_sim, 400)
				bins_exp_sim = 50

				# Theoretical PDF
				pdf_theory_exp = expon.pdf(x_range_exp_sim, scale=scale_param_cont)

				fig_exp_sim = go.Figure()

				fig_exp_sim.add_trace(go.Scatter(
					x=x_range_exp_sim,
					y=pdf_theory_exp,
					mode='lines',
					name='Theoretical PDF',
					line=dict(color='red', width=2)
				))

				# Initial histogram (first few samples)
				initial_samples_exp = samples_exp[:step_size_exp]
				hist_data_exp, bin_edges_exp = np.histogram(initial_samples_exp, bins=bins_exp_sim,
													range=(0, x_max_exp_sim), density=True)
				bin_centers_exp = (bin_edges_exp[:-1] + bin_edges_exp[1:]) / 2

				fig_exp_sim.add_trace(go.Bar(
					x=bin_centers_exp,
					y=hist_data_exp,
					name='Simulated Samples (Density)',
					marker_color='rgba(0, 0, 255, 0.6)',
					opacity=0.6
				))

				# Create frames
				frames_exp_sim = []
				max_y_exp_sim = np.max(pdf_theory_exp) * 1.1

				for i in range(step_size_exp, n_sim_cont_exp + 1, step_size_exp):
					current_samples_exp = samples_exp[:i]
					hist_data_exp_frame, _ = np.histogram(current_samples_exp, bins=bins_exp_sim,
												range=(0, x_max_exp_sim), density=True)
					max_y_exp_sim = max(max_y_exp_sim, np.max(hist_data_exp_frame)*1.1)

					frames_exp_sim.append(go.Frame(
						data=[go.Bar(y=hist_data_exp_frame)],
                        traces=[1], # Update only trace 1
						name=f'Samples: {i}',
                        layout=go.Layout(yaxis=dict(range=[0, max_y_exp_sim]))
					))

				fig_exp_sim.frames = frames_exp_sim

				fig_exp_sim.update_layout(
					title=f'Exponential Simulation: λ={lambda_param_cont:.1f}',
					xaxis=dict(title='Value', range=[0, x_max_exp_sim]),
					yaxis=dict(title='Density', range=[0, max_y_exp_sim]),
                    barmode='overlay',
					updatemenus=[{
						"type": "buttons",
						"direction": "left", "pad": {"r": 10, "t": 10}, "showactive": True, "x": 0.1, "xanchor": "left", "y": 1.15, "yanchor": "top",
						"buttons": [{
							"label": "Play",
							"method": "animate",
							"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"}]
						}]
					}]
				)

				st.plotly_chart(fig_exp_sim, use_container_width=True)

				# Display statistics comparison
				st.markdown('<h6>Comparison: Theoretical vs. Simulated</h6>', unsafe_allow_html=True)
				mean_sim_exp = np.mean(samples_exp)
				std_sim_exp = np.std(samples_exp)

				col1_exp, col2_exp, col3_exp = st.columns(3)
				with col1_exp:
					st.metric("Theoretical Mean", f"{mean_exp_cont:.4f}",
							  delta=f"{mean_sim_exp - mean_exp_cont:.4f}")
				with col2_exp:
					st.metric("Simulated Mean", f"{mean_sim_exp:.4f}")
				with col3_exp:
					st.metric("Theoretical Std Dev", f"{std_exp_cont:.4f}",
							  delta=f"{std_sim_exp - std_exp_cont:.4f}")

				# Memory-less property visualization (Simplified static plot)
				st.markdown('<h6>Memoryless Property Illustration</h6>', unsafe_allow_html=True)
				t_vals_mem = np.linspace(0, 5 * scale_param_cont, 100)
				s_mem = scale_param_cont  # Point s = mean
				t_extra_mem = scale_param_cont # Additional time t = mean

				fig_mem, ax_mem = plt.subplots(figsize=(8, 5)) # Smaller plot
				ax_mem.plot(t_vals_mem, expon.pdf(t_vals_mem, scale=scale_param_cont), 'b-', lw=2, label='PDF')
				ax_mem.axvline(x=s_mem, color='g', linestyle='--', label=f'Time s = {s_mem:.2f}')
				# Fill P(X > s)
				x_fill1 = np.linspace(s_mem, 5 * scale_param_cont, 50)
				y_fill1 = expon.pdf(x_fill1, scale=scale_param_cont)
				prob_x_gt_s = np.exp(-s_mem / scale_param_cont)
				ax_mem.fill_between(x_fill1, y_fill1, alpha=0.3, color='red', label=f'P(X > {s_mem:.2f}) ≈ {prob_x_gt_s:.2f}')
				# Indicate s+t
				ax_mem.axvline(x=s_mem + t_extra_mem, color='purple', linestyle=':', label=f'Time s+t = {s_mem + t_extra_mem:.2f}')
				prob_x_gt_t = np.exp(-t_extra_mem / scale_param_cont)
				ax_mem.text(x_max_exp_sim*0.6, ax_mem.get_ylim()[1]*0.6, f'P(X > {t_extra_mem:.2f}) ≈ {prob_x_gt_t:.2f}\nShould equal\nP(X > s+t | X > s)',
							bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.4))

				ax_mem.set_xlabel('Time')
				ax_mem.set_ylabel('Density')
				ax_mem.set_title('Memoryless Property: P(X > s+t | X > s) = P(X > t)')
				ax_mem.legend(fontsize='small')
				ax_mem.grid(alpha=0.3)
				ax_mem.set_xlim(left=0)
				st.pyplot(fig_mem)

	else:  # Uniform distribution
		with col1_cont:
			st.markdown(
				'<div class="highlight">The Uniform distribution represents equal probabilities for all values within a specified range [a, b].</div>',
				unsafe_allow_html=True)

			st.markdown('<div>Minimum Value (a):</div>', unsafe_allow_html=True)
			a_param_cont = st.slider("a_unif", min_value=-10.0, max_value=10.0, value=0.0, step=0.1,
								label_visibility="collapsed", key="a_param_cont")

			st.markdown('<div>Maximum Value (b):</div>', unsafe_allow_html=True)
			min_b_cont = a_param_cont + 0.1
			default_b_cont = max(min_b_cont, a_param_cont + 1.0)
			b_param_cont = st.slider("b_unif", min_value=min_b_cont, max_value=a_param_cont + 20.0, value=default_b_cont,
								step=0.1, label_visibility="collapsed", key="b_param_cont")
			width_cont = b_param_cont - a_param_cont

			mean_unif_cont = (a_param_cont + b_param_cont) / 2
			var_unif_cont = (width_cont ** 2) / 12 if width_cont > 0 else 0
			std_unif_cont = np.sqrt(var_unif_cont)
			st.markdown(f"""
            <div>
            <strong>Uniform Distribution Properties:</strong><br>
            Mean = (a+b)/2 = {mean_unif_cont:.2f}<br>
            Variance = (b-a)²/12 = {var_unif_cont:.2f}<br>
            Standard Deviation = {std_unif_cont:.2f}
            </div>
            """, unsafe_allow_html=True)

			n_sim_cont_unif = st.slider("Number of Simulations:", min_value=100, max_value=10000, value=1000, step=100,
							  key="unif_sim_count")

		with col2_cont:
			st.markdown("#### Visualization and Simulation")
			if st.button("Show Probability Density Function (PDF)", key="unif_pdf"):
				# Create PDF plot
				plot_padding = width_cont * 0.1
				x_unif_pdf = np.linspace(a_param_cont - plot_padding, b_param_cont + plot_padding, 1000)
				pdf_unif_vals = uniform.pdf(x_unif_pdf, loc=a_param_cont, scale=width_cont)

				fig_unif_pdf, ax_unif_pdf = plt.subplots(figsize=(10, 6))
				ax_unif_pdf.plot(x_unif_pdf, pdf_unif_vals, 'r-', lw=2)
				# Fill between a and b
				x_fill_unif = np.linspace(a_param_cont, b_param_cont, 100)
				pdf_fill_unif = uniform.pdf(x_fill_unif, loc=a_param_cont, scale=width_cont)
				ax_unif_pdf.fill_between(x_fill_unif, pdf_fill_unif, alpha=0.3, color='skyblue')
				ax_unif_pdf.set_xlabel('Value')
				ax_unif_pdf.set_ylabel('Probability Density')
				ax_unif_pdf.set_title(f'PDF for Uniform Distribution: a={a_param_cont:.1f}, b={b_param_cont:.1f}')
				ax_unif_pdf.grid(alpha=0.3)
				ax_unif_pdf.set_ylim(bottom=0)
				st.pyplot(fig_unif_pdf)

			if st.button("Run Distribution Simulation", key="unif_sim"):
				# Simulate uniform samples
				samples_unif = np.random.uniform(a_param_cont, b_param_cont, size=n_sim_cont_unif)

				# Create animation of build-up
				num_frames_unif = 50
				step_size_unif = max(1, n_sim_cont_unif // num_frames_unif)

				# Define range for histogram and PDF
				plot_padding = width_cont * 0.1
				x_min_unif_sim = a_param_cont - plot_padding
				x_max_unif_sim = b_param_cont + plot_padding
				x_range_unif_sim = np.linspace(x_min_unif_sim, x_max_unif_sim, 400)
				bins_unif_sim = 40

				# Theoretical PDF
				pdf_theory_unif = uniform.pdf(x_range_unif_sim, loc=a_param_cont, scale=width_cont)
				max_y_unif_sim = (1 / width_cont) * 1.2 if width_cont > 0 else 1.0 # Expected height + padding

				fig_unif_sim = go.Figure()

				# Add theoretical PDF (as a shape for clarity)
				fig_unif_sim.add_shape(type="rect",
					x0=a_param_cont, y0=0, x1=b_param_cont, y1=1/width_cont if width_cont > 0 else 0,
					line=dict(color="Red", width=2), fillcolor="rgba(255,0,0,0.2)", layer="below",
                    name="Theoretical PDF")
				# Add a dummy scatter trace for the legend entry
				fig_unif_sim.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
												line=dict(color='Red', width=2),
												fillcolor='rgba(255,0,0,0.2)', fill='toself', # Match shape style
												name='Theoretical PDF'))


				# Initial histogram (first few samples)
				initial_samples_unif = samples_unif[:step_size_unif]
				hist_data_unif, bin_edges_unif = np.histogram(initial_samples_unif, bins=bins_unif_sim,
													range=(a_param_cont, b_param_cont), density=True) # Range [a,b]
				bin_centers_unif = (bin_edges_unif[:-1] + bin_edges_unif[1:]) / 2

				fig_unif_sim.add_trace(go.Bar(
					x=bin_centers_unif,
					y=hist_data_unif,
					name='Simulated Samples (Density)',
					marker_color='rgba(0, 0, 255, 0.6)',
					opacity=0.6
				))

				# Create frames
				frames_unif_sim = []

				for i in range(step_size_unif, n_sim_cont_unif + 1, step_size_unif):
					current_samples_unif = samples_unif[:i]
					hist_data_unif_frame, _ = np.histogram(current_samples_unif, bins=bins_unif_sim,
												range=(a_param_cont, b_param_cont), density=True)

					frames_unif_sim.append(go.Frame(
						data=[go.Bar(y=hist_data_unif_frame)],
                        # Update trace 2 (index 1 is dummy scatter, index 2 is Bar)
                        traces=[2],
						name=f'Samples: {i}'
					))

				fig_unif_sim.frames = frames_unif_sim

				fig_unif_sim.update_layout(
					title=f'Uniform Simulation: a={a_param_cont:.1f}, b={b_param_cont:.1f}',
					xaxis=dict(title='Value', range=[x_min_unif_sim, x_max_unif_sim]),
					yaxis=dict(title='Density', range=[0, max_y_unif_sim]),
                    barmode='overlay',
					updatemenus=[{
						"type": "buttons",
						"direction": "left", "pad": {"r": 10, "t": 10}, "showactive": True, "x": 0.1, "xanchor": "left", "y": 1.15, "yanchor": "top",
						"buttons": [{
							"label": "Play",
							"method": "animate",
							"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"}]
						}]
					}]
				)

				st.plotly_chart(fig_unif_sim, use_container_width=True)

				# Display statistics comparison
				st.markdown('<h6>Comparison: Theoretical vs. Simulated</h6>', unsafe_allow_html=True)
				mean_sim_unif = np.mean(samples_unif)
				std_sim_unif = np.std(samples_unif)

				col1_unif, col2_unif, col3_unif = st.columns(3)
				with col1_unif:
					st.metric("Theoretical Mean", f"{mean_unif_cont:.4f}",
							  delta=f"{mean_sim_unif - mean_unif_cont:.4f}")
				with col2_unif:
					st.metric("Simulated Mean", f"{mean_sim_unif:.4f}")
				with col3_unif:
					st.metric("Theoretical Std Dev", f"{std_unif_cont:.4f}",
							  delta=f"{std_sim_unif - std_unif_cont:.4f}")

				# Uniform order statistics (using Uniform [0,1] for simplicity)
				st.markdown('<h6>Order Statistics for Uniform[0,1]</h6>', unsafe_allow_html=True)

				order_n_unif = 5
				unif_samples_order = np.random.uniform(0, 1, size=(1000, order_n_unif))
				ordered_samples_unif = np.sort(unif_samples_order, axis=1)

				fig_order_unif, axs_order_unif = plt.subplots(1, order_n_unif, figsize=(15, 4), sharey=True)

				for i in range(order_n_unif):
					axs_order_unif[i].hist(ordered_samples_unif[:, i], bins=20, density=True, alpha=0.6, label='Simulated')
					# Add Beta PDF: Beta(i+1, n-i) for i-th statistic (0-indexed i)
					alpha_beta = i + 1
					beta_beta = order_n_unif - i
					x_beta = np.linspace(0.001, 0.999, 100) # Avoid exact 0, 1
					axs_order_unif[i].plot(x_beta, beta.pdf(x_beta, alpha_beta, beta_beta), 'r-', lw=2, label=f'Beta({alpha_beta},{beta_beta})')
					axs_order_unif[i].set_title(f'Order Stat {i + 1}')
					axs_order_unif[i].set_xlabel('Value')
					if i == 0:
						axs_order_unif[i].set_ylabel('Density')
					axs_order_unif[i].legend(fontsize='x-small')

				plt.tight_layout()
				st.pyplot(fig_order_unif)
				st.markdown("""
				<div>The i-th order statistic (sorted value) from a Uniform[0,1] sample of size n follows a Beta(i, n-i+1) distribution.</div>
				""", unsafe_allow_html=True)


# --- Footer ---
st.markdown("""
<hr>
<div style="margin-top: 30px; padding: 20px; border-top: 1px solid #ccc; text-align: center;">
<p>This application demonstrates key statistical concepts using interactive visualizations and simulations.</p>
<p>Concepts Covered: Central Limit Theorem, Law of Large Numbers, Properties and Convergences of Major Probability Distributions.</p>
</div>
""", unsafe_allow_html=True)

# --- END OF FILE dss.py ---