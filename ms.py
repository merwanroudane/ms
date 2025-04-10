import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.animation as animation
from io import BytesIO
import base64
from PIL import Image
import matplotlib

matplotlib.use('Agg')

# Set page configuration
st.set_page_config(
	page_title="Interactive Statistical Distributions",
	page_icon="📊",
	layout="wide",
	initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1565C0;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .property-name {
        font-weight: bold;
        color: #0277BD;
    }
    .formula {
        background-color: #E3F2FD;
        padding: 0.5rem;
        border-radius: 5px;
        font-family: monospace;
    }
    .note {
        background-color: #FFF9C4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 5px solid #FFC107;
    }
    .highlight {
        background-color: #E8F5E9;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<h1 class='main-header'>Interactive Statistical Distributions</h1>", unsafe_allow_html=True)
st.markdown("""
This interactive application helps you understand statistical distributions, their properties, 
relationships between them, and fundamental theorems in probability and statistics. Use this as 
a comprehensive guide for teaching and learning statistical concepts with animated visualizations.
""")

# Create sidebar for navigation
st.sidebar.title("Navigation")
main_sections = ["Introduction",
				 "Discrete Distributions",
				 "Continuous Distributions",
				 "Relationships Between Distributions",
				 "Limit Theorems & Convergence",
				 "Examples & Practice"]

selected_section = st.sidebar.radio("Go to", main_sections)


# Helper function to create animations
def create_animation(fig, frames, interval=200):
	anim = FuncAnimation(fig, frames=frames, interval=interval)
	buf = BytesIO()
	anim.save(buf, format='gif', writer='pillow', fps=5)
	buf.seek(0)
	data = base64.b64encode(buf.getvalue()).decode('utf-8')
	return f'<img src="data:image/gif;base64,{data}" alt="animation">'


# Helper function for LaTeX rendering
def latex(formula):
	return f"$${formula}$$"


# Introduction Section
if selected_section == "Introduction":
	st.markdown("<h2 class='sub-header'>Introduction to Probability Distributions</h2>", unsafe_allow_html=True)

	st.markdown("""
    A probability distribution is a mathematical function that provides the probabilities of occurrence of different 
    possible outcomes for an experiment. It is a description of a random phenomenon in terms of the probabilities of events.
    """)

	st.markdown("<h3 class='section-header'>Types of Distributions</h3>", unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("<h4>Discrete Distributions</h4>", unsafe_allow_html=True)
		st.markdown("""
        - Characterized by probability mass functions (PMF)
        - Deal with countable outcomes
        - Examples: Binomial, Poisson, Geometric
        """)

		# Generate simple discrete distribution example
		x = np.arange(0, 10)
		p = stats.binom.pmf(x, 10, 0.3)

		fig, ax = plt.subplots(figsize=(8, 4))
		ax.bar(x, p, width=0.4, alpha=0.7, color='skyblue', edgecolor='navy')
		ax.set_xlabel('X')
		ax.set_ylabel('Probability')
		ax.set_title('Binomial Distribution PMF (n=10, p=0.3)')
		ax.grid(alpha=0.3)
		st.pyplot(fig)

	with col2:
		st.markdown("<h4>Continuous Distributions</h4>", unsafe_allow_html=True)
		st.markdown("""
        - Characterized by probability density functions (PDF)
        - Deal with uncountable outcomes
        - Examples: Normal, Exponential, Uniform
        """)

		# Generate simple continuous distribution example
		x = np.linspace(-4, 4, 1000)
		p = stats.norm.pdf(x, 0, 1)

		fig, ax = plt.subplots(figsize=(8, 4))
		ax.plot(x, p, color='tomato', lw=2)
		ax.fill_between(x, p, alpha=0.3, color='tomato')
		ax.set_xlabel('X')
		ax.set_ylabel('Density')
		ax.set_title('Normal Distribution PDF (μ=0, σ=1)')
		ax.grid(alpha=0.3)
		st.pyplot(fig)

	st.markdown("<h3 class='section-header'>Key Concepts to Explore</h3>", unsafe_allow_html=True)

	st.markdown("""
    1. **Distribution Properties**: Mean, variance, moments, and other characteristics
    2. **Probability Calculation**: Using PMF, PDF, CDF
    3. **Transformation**: How random variables transform through functions
    4. **Convergence**: Limit theorems that describe the behavior of distributions
    5. **Relationships**: How distributions relate to each other
    """)

	st.markdown(
		"<div class='note'>This interactive application allows you to visualize these concepts dynamically through animations, making abstract statistical ideas more concrete for students.</div>",
		unsafe_allow_html=True)

# Discrete Distributions Section
elif selected_section == "Discrete Distributions":
	st.markdown("<h2 class='sub-header'>Discrete Probability Distributions</h2>", unsafe_allow_html=True)

	# Select distribution
	discrete_dist = st.selectbox(
		"Choose a discrete distribution:",
		["Bernoulli", "Binomial", "Geometric", "Negative Binomial", "Poisson", "Hypergeometric", "Uniform (Discrete)"]
	)

	if discrete_dist == "Bernoulli":
		st.markdown("<h3 class='section-header'>Bernoulli Distribution</h3>", unsafe_allow_html=True)

		# Parameters
		p = st.slider("Probability of success (p):", 0.01, 0.99, 0.5, 0.01)

		# Description
		st.markdown("""
        The Bernoulli distribution is the discrete probability distribution of a random variable which takes the value 1 with probability $p$ and the value 0 with probability $q = 1 - p$.
        """)

		# Properties
		st.markdown("<h4>Properties</h4>", unsafe_allow_html=True)
		col1, col2 = st.columns(2)

		with col1:
			st.markdown("<span class='property-name'>PMF:</span> $P(X = x) = p^x(1-p)^{1-x}$ for $x \in \{0, 1\}$",
						unsafe_allow_html=True)
			st.markdown(f"<span class='property-name'>Mean:</span> {latex('E[X] = p')}", unsafe_allow_html=True)
			st.markdown(f"<span class='property-name'>Variance:</span> {latex('Var[X] = p(1-p)')}",
						unsafe_allow_html=True)

		with col2:
			st.markdown("<span class='property-name'>MGF:</span> " + latex('M_X(t) = (1-p) + pe^t'),
						unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Skewness:</span> " + latex('\\frac{1-2p}{\\sqrt{p(1-p)}}'),
						unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Kurtosis:</span> " + latex('\\frac{1-6p(1-p)}{p(1-p)}'),
						unsafe_allow_html=True)

		# Visualization
		fig = plt.figure(figsize=(10, 6))

		x_values = [0, 1]
		pmf_values = [1 - p, p]

		plt.bar(x_values, pmf_values, color='skyblue', edgecolor='navy', alpha=0.7)
		plt.xticks(x_values)
		plt.xlabel('x')
		plt.ylabel('Probability')
		plt.title(f'Bernoulli Distribution (p={p})')
		plt.grid(alpha=0.3)
		st.pyplot(fig)

		# Animation showing how the PMF changes with p
		st.markdown("<h4>Interactive Animation: Effect of Probability p</h4>", unsafe_allow_html=True)

		fig_anim = go.Figure()

		# Add initial bars
		fig_anim.add_trace(go.Bar(
			x=[0, 1],
			y=[1 - p, p],
			marker_color=['#90CAF9', '#1E88E5'],
			name='PMF'
		))

		# Add slider for interactive animation
		fig_anim.update_layout(
			title='Bernoulli PMF with varying p',
			xaxis=dict(title='x', tickvals=[0, 1]),
			yaxis=dict(title='Probability', range=[0, 1]),
			sliders=[{
				'active': int(p * 100) - 1,
				'steps': [
					{
						'method': 'update',
						'args': [{'y': [[1 - i / 100, i / 100]]}],
						'label': f'{i / 100:.2f}'
					} for i in range(1, 100)
				],
				'pad': {'t': 50},
				'currentvalue': {'prefix': 'p = '},
				'len': 0.9,
				'x': 0.1,
				'xanchor': 'left'
			}]
		)

		st.plotly_chart(fig_anim, use_container_width=True)

		# Applications
		st.markdown("<h4>Applications</h4>", unsafe_allow_html=True)
		st.markdown("""
        The Bernoulli distribution is used to model:
        - Success/failure of a single experiment
        - Yes/no questions
        - Presence/absence of a characteristic
        - Any binary outcome
        """)

	elif discrete_dist == "Binomial":
		st.markdown("<h3 class='section-header'>Binomial Distribution</h3>", unsafe_allow_html=True)

		# Parameters
		col1, col2 = st.columns(2)
		with col1:
			n = st.slider("Number of trials (n):", 1, 50, 10)
		with col2:
			p = st.slider("Probability of success (p):", 0.01, 0.99, 0.5, 0.01)

		# Description
		st.markdown("""
        The binomial distribution models the number of successes in a fixed number of independent
        Bernoulli trials, each with the same probability of success.
        """)

		# Properties
		st.markdown("<h4>Properties</h4>", unsafe_allow_html=True)
		col1, col2 = st.columns(2)

		with col1:
			st.markdown("<span class='property-name'>PMF:</span> " + latex('P(X = k) = \\binom{n}{k}p^k(1-p)^{n-k}'),
						unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Mean:</span> " + latex('E[X] = np'), unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Variance:</span> " + latex('Var[X] = np(1-p)'),
						unsafe_allow_html=True)

		with col2:
			st.markdown("<span class='property-name'>MGF:</span> " + latex('M_X(t) = (1-p+pe^t)^n'),
						unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Skewness:</span> " + latex('\\frac{1-2p}{\\sqrt{np(1-p)}}'),
						unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Kurtosis Excess:</span> " + latex('\\frac{1-6p(1-p)}{np(1-p)}'),
						unsafe_allow_html=True)

		# Visualization
		x = np.arange(0, n + 1)
		pmf = stats.binom.pmf(x, n, p)
		cdf = stats.binom.cdf(x, n, p)

		fig = make_subplots(rows=2, cols=1,
							subplot_titles=("Probability Mass Function", "Cumulative Distribution Function"))

		fig.add_trace(
			go.Bar(x=x, y=pmf, marker_color='#1E88E5', name="PMF"),
			row=1, col=1
		)

		fig.add_trace(
			go.Scatter(x=x, y=cdf, mode='lines+markers', marker_color='#FFA000', name="CDF"),
			row=2, col=1
		)

		fig.update_layout(height=600, title_text=f"Binomial Distribution (n={n}, p={p})")
		fig.update_xaxes(title_text="k (number of successes)")
		fig.update_yaxes(title_text="Probability", row=1, col=1)
		fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)

		st.plotly_chart(fig, use_container_width=True)

		# Animation showing the effect of n (sample size)
		st.markdown("<h4>Interactive Animation: Effect of Sample Size n</h4>", unsafe_allow_html=True)

		# Create frames for different values of n
		n_values = list(range(5, 51, 5))

		# Create figure
		fig_n = go.Figure()

		# Add initial trace
		x_initial = np.arange(0, n_values[0] + 1)
		pmf_initial = stats.binom.pmf(x_initial, n_values[0], p)

		fig_n.add_trace(go.Bar(
			x=x_initial,
			y=pmf_initial,
			marker_color='#1E88E5',
			name=f"n={n_values[0]}"
		))

		# Add frames
		frames = []
		for n_val in n_values:
			x_frame = np.arange(0, n_val + 1)
			pmf_frame = stats.binom.pmf(x_frame, n_val, p)

			frames.append(go.Frame(
				data=[go.Bar(
					x=x_frame,
					y=pmf_frame,
					marker_color='#1E88E5'
				)],
				name=f"n={n_val}",
				layout=go.Layout(
					title_text=f"Binomial PMF (n={n_val}, p={p})",
					xaxis=dict(range=[-1, 50]),
					yaxis=dict(range=[0, max(stats.binom.pmf(np.arange(0, 6), 5, p)) * 1.1])
				)
			))

		fig_n.frames = frames

		# Add buttons and sliders
		fig_n.update_layout(
			title="Binomial PMF with varying n (sample size)",
			xaxis=dict(title="Number of successes (k)", range=[-1, 50]),
			yaxis=dict(title="Probability"),
			updatemenus=[{
				'type': 'buttons',
				'showactive': False,
				'buttons': [
					{
						'label': 'Play',
						'method': 'animate',
						'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
					},
					{
						'label': 'Pause',
						'method': 'animate',
						'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
					}
				],
				'direction': 'left',
				'pad': {'r': 10, 't': 10},
				'x': 0.1,
				'xanchor': 'right',
				'y': 0,
				'yanchor': 'top'
			}],
			sliders=[{
				'active': 0,
				'steps': [
					{
						'method': 'animate',
						'args': [[f'n={n_val}'], {
							'frame': {'duration': 300, 'redraw': True},
							'mode': 'immediate',
							'transition': {'duration': 300}
						}],
						'label': f'{n_val}'
					} for n_val in n_values
				],
				'pad': {'t': 50},
				'len': 0.9,
				'x': 0.1,
				'xanchor': 'left',
				'y': 0,
				'yanchor': 'top',
				'currentvalue': {'prefix': 'n = '}
			}]
		)

		st.plotly_chart(fig_n, use_container_width=True)

		# Approximation to Normal
		st.markdown("<h4>Normal Approximation to Binomial</h4>", unsafe_allow_html=True)
		st.markdown("""
        As n increases, the binomial distribution approaches a normal distribution with the same mean and variance.
        This is a result of the Central Limit Theorem.
        """)

		# Visualization of the approximation
		if n >= 20:
			x_norm = np.linspace(0, n, 1000)
			y_norm = stats.norm.pdf(x_norm, n * p, np.sqrt(n * p * (1 - p)))

			x_binomial = np.arange(0, n + 1)
			pmf_binomial = stats.binom.pmf(x_binomial, n, p)

			fig_approx = plt.figure(figsize=(10, 6))
			plt.bar(x_binomial, pmf_binomial, alpha=0.7, color='skyblue', label='Binomial PMF')
			plt.plot(x_norm, y_norm, 'r-', lw=2, label='Normal Approximation')
			plt.xlabel('Number of successes (k)')
			plt.ylabel('Probability')
			plt.title(f'Normal Approximation to Binomial (n={n}, p={p})')
			plt.legend()
			plt.grid(alpha=0.3)

			st.pyplot(fig_approx)

			# Explanation of continuity correction
			st.markdown(
				"<div class='note'>Note: For better approximation with discrete to continuous mapping, a continuity correction can be applied. This means using normal CDF from (k-0.5) to (k+0.5) to approximate binomial probability at k.</div>",
				unsafe_allow_html=True)
		else:
			st.markdown(
				"<div class='note'>Note: The normal approximation becomes more accurate as n increases (generally n ≥ 20) and when p is not too close to 0 or 1. Increase n to see the approximation.</div>",
				unsafe_allow_html=True)

		# Applications
		st.markdown("<h4>Applications</h4>", unsafe_allow_html=True)
		st.markdown("""
        The binomial distribution is used to model:
        - Number of successes in n independent trials
        - Quality control and defect counting
        - A/B testing results
        - Survey responses with binary outcomes
        """)

	elif discrete_dist == "Poisson":
		st.markdown("<h3 class='section-header'>Poisson Distribution</h3>", unsafe_allow_html=True)

		# Parameters
		lambda_param = st.slider("Rate parameter (λ):", 0.1, 20.0, 5.0, 0.1)

		# Description
		st.markdown("""
        The Poisson distribution models the number of events occurring in a fixed interval of time or space, 
        assuming these events occur independently and at a constant average rate.
        """)

		# Properties
		st.markdown("<h4>Properties</h4>", unsafe_allow_html=True)
		col1, col2 = st.columns(2)

		with col1:
			st.markdown(
				"<span class='property-name'>PMF:</span> " + latex('P(X = k) = \\frac{\\lambda^k e^{-\\lambda}}{k!}'),
				unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Mean:</span> " + latex('E[X] = \\lambda'), unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Variance:</span> " + latex('Var[X] = \\lambda'),
						unsafe_allow_html=True)

		with col2:
			st.markdown("<span class='property-name'>MGF:</span> " + latex('M_X(t) = e^{\\lambda(e^t - 1)}'),
						unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Skewness:</span> " + latex('\\frac{1}{\\sqrt{\\lambda}}'),
						unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Kurtosis Excess:</span> " + latex('\\frac{1}{\\lambda}'),
						unsafe_allow_html=True)

		# Visualization
		k_max = max(20, int(lambda_param * 3))
		x = np.arange(0, k_max + 1)
		pmf = stats.poisson.pmf(x, lambda_param)
		cdf = stats.poisson.cdf(x, lambda_param)

		fig = make_subplots(rows=2, cols=1,
							subplot_titles=("Probability Mass Function", "Cumulative Distribution Function"))

		fig.add_trace(
			go.Bar(x=x, y=pmf, marker_color='#7CB342', name="PMF"),
			row=1, col=1
		)

		fig.add_trace(
			go.Scatter(x=x, y=cdf, mode='lines+markers', marker_color='#FB8C00', name="CDF"),
			row=2, col=1
		)

		fig.update_layout(height=600, title_text=f"Poisson Distribution (λ={lambda_param})")
		fig.update_xaxes(title_text="k (number of events)")
		fig.update_yaxes(title_text="Probability", row=1, col=1)
		fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)

		st.plotly_chart(fig, use_container_width=True)

		# Animation showing the effect of lambda
		st.markdown("<h4>Interactive Animation: Effect of Rate Parameter λ</h4>", unsafe_allow_html=True)

		# Create frames for different values of lambda
		lambda_values = np.linspace(0.5, 15, 30)

		# Create figure
		fig_lambda = go.Figure()

		# Add initial trace
		x_initial = np.arange(0, int(lambda_values[0] * 3) + 1)
		pmf_initial = stats.poisson.pmf(x_initial, lambda_values[0])

		fig_lambda.add_trace(go.Bar(
			x=x_initial,
			y=pmf_initial,
			marker_color='#7CB342',
			name=f"λ={lambda_values[0]:.1f}"
		))

		# Add frames
		frames = []
		for lambda_val in lambda_values:
			k_max_frame = int(lambda_val * 3) + 1
			x_frame = np.arange(0, k_max_frame)
			pmf_frame = stats.poisson.pmf(x_frame, lambda_val)

			frames.append(go.Frame(
				data=[go.Bar(
					x=x_frame,
					y=pmf_frame,
					marker_color='#7CB342'
				)],
				name=f"λ={lambda_val:.1f}",
				layout=go.Layout(
					title_text=f"Poisson PMF (λ={lambda_val:.1f})",
					xaxis=dict(range=[-1, 50]),
					yaxis=dict(range=[0, max(stats.poisson.pmf(np.arange(0, 5), 0.5)) * 1.1])
				)
			))

		fig_lambda.frames = frames

		# Add buttons and sliders
		fig_lambda.update_layout(
			title="Poisson PMF with varying λ (rate parameter)",
			xaxis=dict(title="Number of events (k)", range=[-1, 50]),
			yaxis=dict(title="Probability"),
			updatemenus=[{
				'type': 'buttons',
				'showactive': False,
				'buttons': [
					{
						'label': 'Play',
						'method': 'animate',
						'args': [None, {'frame': {'duration': 200, 'redraw': True}, 'fromcurrent': True}]
					},
					{
						'label': 'Pause',
						'method': 'animate',
						'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
					}
				],
				'direction': 'left',
				'pad': {'r': 10, 't': 10},
				'x': 0.1,
				'xanchor': 'right',
				'y': 0,
				'yanchor': 'top'
			}],
			sliders=[{
				'active': 0,
				'steps': [
					{
						'method': 'animate',
						'args': [[f'λ={lambda_val:.1f}'], {
							'frame': {'duration': 200, 'redraw': True},
							'mode': 'immediate',
							'transition': {'duration': 200}
						}],
						'label': f'{lambda_val:.1f}'
					} for lambda_val in lambda_values
				],
				'pad': {'t': 50},
				'len': 0.9,
				'x': 0.1,
				'xanchor': 'left',
				'y': 0,
				'yanchor': 'top',
				'currentvalue': {'prefix': 'λ = '}
			}]
		)

		st.plotly_chart(fig_lambda, use_container_width=True)

		# Poisson as limit of Binomial
		st.markdown("<h4>Poisson as Limit of Binomial</h4>", unsafe_allow_html=True)
		st.markdown("""
        The Poisson distribution can be derived as the limit of the binomial distribution when:
        - The number of trials n approaches infinity
        - The probability of success p approaches zero
        - The product np = λ remains constant

        This makes the Poisson distribution an excellent approximation to the binomial when n is large and p is small.
        """)

		# Interactive demonstration of Poisson approximation to Binomial
		st.markdown("<h4>Poisson Approximation to Binomial</h4>", unsafe_allow_html=True)

		col1, col2 = st.columns(2)
		with col1:
			n_approx = st.slider("Number of trials (n):", 10, 1000, 100)
		with col2:
			p_approx = lambda_param / n_approx
			st.markdown(f"p = λ/n = {p_approx:.6f}")

		# Visualization of the approximation
		x_approx = np.arange(0, min(n_approx, int(lambda_param * 3) + 10))
		binom_pmf = stats.binom.pmf(x_approx, n_approx, p_approx)
		poisson_pmf = stats.poisson.pmf(x_approx, lambda_param)

		fig_approx = plt.figure(figsize=(10, 6))
		plt.bar(x_approx, binom_pmf, alpha=0.5, color='skyblue', label='Binomial PMF')
		plt.plot(x_approx, poisson_pmf, 'r-', lw=2, marker='o', label='Poisson PMF')
		plt.xlabel('Number of events (k)')
		plt.ylabel('Probability')
		plt.title(f'Poisson Approximation to Binomial (n={n_approx}, p={p_approx:.6f}, λ={lambda_param})')
		plt.legend()
		plt.grid(alpha=0.3)

		st.pyplot(fig_approx)

		# Applications
		st.markdown("<h4>Applications</h4>", unsafe_allow_html=True)
		st.markdown("""
        The Poisson distribution is used to model:
        - Number of arrivals in a fixed time period (e.g., customers, calls, emails)
        - Number of defects in a fixed area
        - Rare events in large populations
        - Number of accidents or natural disasters
        - Counts of radioactive decay events
        """)

	elif discrete_dist == "Geometric":
		st.markdown("<h3 class='section-header'>Geometric Distribution</h3>", unsafe_allow_html=True)

		# Parameters
		p = st.slider("Probability of success (p):", 0.01, 0.99, 0.3, 0.01)

		# Two versions of Geometric
		version = st.radio(
			"Select version:",
			["Number of trials until first success (k=1,2,3,...)",
			 "Number of failures before first success (k=0,1,2,...)"]
		)

		first_version = version.startswith("Number of trials")

		# Description
		st.markdown(f"""
        The geometric distribution models the number of {"trials needed to get the first success" if first_version else "failures before the first success"}
        in a sequence of independent Bernoulli trials, each with probability of success p.
        """)

		# Properties
		st.markdown("<h4>Properties</h4>", unsafe_allow_html=True)
		col1, col2 = st.columns(2)

		with col1:
			if first_version:
				st.markdown("<span class='property-name'>PMF:</span> " + latex('P(X = k) = (1-p)^{k-1}p'),
							unsafe_allow_html=True)
				st.markdown("<span class='property-name'>Mean:</span> " + latex('E[X] = \\frac{1}{p}'),
							unsafe_allow_html=True)
				st.markdown("<span class='property-name'>Variance:</span> " + latex('Var[X] = \\frac{1-p}{p^2}'),
							unsafe_allow_html=True)
			else:
				st.markdown("<span class='property-name'>PMF:</span> " + latex('P(X = k) = (1-p)^k p'),
							unsafe_allow_html=True)
				st.markdown("<span class='property-name'>Mean:</span> " + latex('E[X] = \\frac{1-p}{p}'),
							unsafe_allow_html=True)
				st.markdown("<span class='property-name'>Variance:</span> " + latex('Var[X] = \\frac{1-p}{p^2}'),
							unsafe_allow_html=True)

		with col2:
			if first_version:
				st.markdown("<span class='property-name'>MGF:</span> " + latex('M_X(t) = \\frac{pe^t}{1-(1-p)e^t}'),
							unsafe_allow_html=True)
				st.markdown("<span class='property-name'>Skewness:</span> " + latex('\\frac{2-p}{\\sqrt{1-p}}'),
							unsafe_allow_html=True)
				st.markdown("<span class='property-name'>Kurtosis Excess:</span> " + latex('\\frac{6 + p^2}{1-p}'),
							unsafe_allow_html=True)
			else:
				st.markdown("<span class='property-name'>MGF:</span> " + latex('M_X(t) = \\frac{p}{1-(1-p)e^t}'),
							unsafe_allow_html=True)
				st.markdown("<span class='property-name'>Skewness:</span> " + latex('\\frac{2-p}{\\sqrt{1-p}}'),
							unsafe_allow_html=True)
				st.markdown("<span class='property-name'>Kurtosis Excess:</span> " + latex('\\frac{6 + p^2}{1-p}'),
							unsafe_allow_html=True)

		# Memoryless property
		st.markdown(
			"<div class='note'><strong>Memoryless Property:</strong> The geometric distribution is the only discrete memoryless distribution. This means P(X > m+n | X > m) = P(X > n). In other words, if you've already waited m trials without success, the probability of waiting n more trials is the same as the initial probability of waiting n trials.</div>",
			unsafe_allow_html=True)

		# Visualization
		max_k = min(50, int(5 / p))
		if first_version:
			x = np.arange(1, max_k + 1)
			pmf = stats.geom.pmf(x, p)
			cdf = stats.geom.cdf(x, p)
		else:
			x = np.arange(0, max_k + 1)
			pmf = stats.geom.pmf(x + 1, p)
			cdf = stats.geom.cdf(x + 1, p)

		fig = make_subplots(rows=2, cols=1,
							subplot_titles=("Probability Mass Function", "Cumulative Distribution Function"))

		fig.add_trace(
			go.Bar(x=x, y=pmf, marker_color='#26A69A', name="PMF"),
			row=1, col=1
		)

		fig.add_trace(
			go.Scatter(x=x, y=cdf, mode='lines+markers', marker_color='#EF5350', name="CDF"),
			row=2, col=1
		)

		version_text = "trials until first success" if first_version else "failures before first success"
		fig.update_layout(height=600, title_text=f"Geometric Distribution (p={p}, {version_text})")
		fig.update_xaxes(title_text=f"k ({version_text})")
		fig.update_yaxes(title_text="Probability", row=1, col=1)
		fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)

		st.plotly_chart(fig, use_container_width=True)

		# Animation showing the effect of p
		st.markdown("<h4>Interactive Animation: Effect of Probability p</h4>", unsafe_allow_html=True)

		# Create frames for different values of p
		p_values = np.linspace(0.1, 0.9, 17)

		# Create figure
		fig_p = go.Figure()

		# Add initial trace
		max_k_initial = min(50, int(5 / p_values[0]))
		if first_version:
			x_initial = np.arange(1, max_k_initial + 1)
			pmf_initial = stats.geom.pmf(x_initial, p_values[0])
		else:
			x_initial = np.arange(0, max_k_initial + 1)
			pmf_initial = stats.geom.pmf(x_initial + 1, p_values[0])

		fig_p.add_trace(go.Bar(
			x=x_initial,
			y=pmf_initial,
			marker_color='#26A69A',
			name=f"p={p_values[0]:.1f}"
		))

		# Add frames
		frames = []
		for p_val in p_values:
			max_k_frame = min(50, int(5 / p_val))
			if first_version:
				x_frame = np.arange(1, max_k_frame + 1)
				pmf_frame = stats.geom.pmf(x_frame, p_val)
			else:
				x_frame = np.arange(0, max_k_frame + 1)
				pmf_frame = stats.geom.pmf(x_frame + 1, p_val)

			frames.append(go.Frame(
				data=[go.Bar(
					x=x_frame,
					y=pmf_frame,
					marker_color='#26A69A'
				)],
				name=f"p={p_val:.1f}",
				layout=go.Layout(
					title_text=f"Geometric PMF (p={p_val:.1f})",
					xaxis=dict(range=[-1 if not first_version else 0, 50]),
					yaxis=dict(range=[0, 1])
				)
			))

		fig_p.frames = frames

		# Add buttons and sliders
		fig_p.update_layout(
			title=f"Geometric PMF with varying p (probability of success)",
			xaxis=dict(title=f"k ({version_text})", range=[-1 if not first_version else 0, 50]),
			yaxis=dict(title="Probability", range=[0, 1]),
			updatemenus=[{
				'type': 'buttons',
				'showactive': False,
				'buttons': [
					{
						'label': 'Play',
						'method': 'animate',
						'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
					},
					{
						'label': 'Pause',
						'method': 'animate',
						'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
					}
				],
				'direction': 'left',
				'pad': {'r': 10, 't': 10},
				'x': 0.1,
				'xanchor': 'right',
				'y': 0,
				'yanchor': 'top'
			}],
			sliders=[{
				'active': 0,
				'steps': [
					{
						'method': 'animate',
						'args': [[f'p={p_val:.1f}'], {
							'frame': {'duration': 300, 'redraw': True},
							'mode': 'immediate',
							'transition': {'duration': 300}
						}],
						'label': f'{p_val:.1f}'
					} for p_val in p_values
				],
				'pad': {'t': 50},
				'len': 0.9,
				'x': 0.1,
				'xanchor': 'left',
				'y': 0,
				'yanchor': 'top',
				'currentvalue': {'prefix': 'p = '}
			}]
		)

		st.plotly_chart(fig_p, use_container_width=True)

		# Visualization of memoryless property
		st.markdown("<h4>Visualization of Memoryless Property</h4>", unsafe_allow_html=True)

		m = st.slider("Number of trials already elapsed without success (m):", 0, 20, 5)
		n = st.slider("Additional trials to wait (n):", 1, 20, 5)

		if first_version:
			prob_original = 1 - stats.geom.cdf(n - 1, p)
			prob_conditional = 1 - stats.geom.cdf(n - 1, p)
		else:
			prob_original = 1 - stats.geom.cdf(n, p)
			prob_conditional = 1 - stats.geom.cdf(n, p)

		col1, col2 = st.columns(2)

		with col1:
			st.markdown(f"<span class='property-name'>P(X > {n}):</span> {prob_original:.6f}", unsafe_allow_html=True)

		with col2:
			st.markdown(f"<span class='property-name'>P(X > {m + n} | X > {m}):</span> {prob_conditional:.6f}",
						unsafe_allow_html=True)

		st.markdown(
			"<div class='highlight'>The memoryless property shows that regardless of how many unsuccessful trials have already occurred, the probability distribution of additional trials remains the same.</div>",
			unsafe_allow_html=True)

		# Applications
		st.markdown("<h4>Applications</h4>", unsafe_allow_html=True)
		st.markdown("""
        The geometric distribution is used to model:
        - Number of tosses until the first head in a coin toss
        - Number of attempts until first success in repeated trials
        - Number of failures before first detection in quality control
        - Waiting time for a rare event
        """)

	elif discrete_dist == "Negative Binomial":
		st.markdown("<h3 class='section-header'>Negative Binomial Distribution</h3>", unsafe_allow_html=True)

		# Parameters
		col1, col2 = st.columns(2)
		with col1:
			r = st.slider("Number of successes required (r):", 1, 20, 5)
		with col2:
			p = st.slider("Probability of success (p):", 0.01, 0.99, 0.3, 0.01)

		# Two versions of Negative Binomial
		version = st.radio(
			"Select version:",
			["Number of trials needed to get r successes (k=r,r+1,...)",
			 "Number of failures before r successes (k=0,1,2,...)"]
		)

		first_version = version.startswith("Number of trials")

		# Description
		st.markdown(f"""
        The negative binomial distribution models the number of {"trials needed to get r successes" if first_version else "failures before r successes"}
        in a sequence of independent Bernoulli trials, each with probability of success p.
        """)

		# Properties
		st.markdown("<h4>Properties</h4>", unsafe_allow_html=True)
		col1, col2 = st.columns(2)
		with col1:
			if first_version:
				st.markdown(
					"<span class='property-name'>PMF:</span> " + latex('P(X = k) = \\binom{k-1}{r-1}p^r(1-p)^{k-r}'),
					unsafe_allow_html=True)
				st.markdown("<span class='property-name'>Mean:</span> " + latex('E[X] = \\frac{r}{p}'),
							unsafe_allow_html=True)
				st.markdown("<span class='property-name'>Variance:</span> " + latex('Var[X] = \\frac{r(1-p)}{p^2}'),
							unsafe_allow_html=True)
			else:
				st.markdown(
					"<span class='property-name'>PMF:</span> " + latex('P(X = k) = \\binom{k+r-1}{k}p^r(1-p)^k'),
					unsafe_allow_html=True)
				st.markdown("<span class='property-name'>Mean:</span> " + latex('E[X] = \\frac{r(1-p)}{p}'),
							unsafe_allow_html=True)
				st.markdown("<span class='property-name'>Variance:</span> " + latex('Var[X] = \\frac{r(1-p)}{p^2}'),
							unsafe_allow_html=True)

		with col2:
			if first_version:
				st.markdown("<span class='property-name'>MGF:</span> " + latex(
					'M_X(t) = \\left(\\frac{pe^t}{1-(1-p)e^t}\\right)^r'),
							unsafe_allow_html=True)
				st.markdown("<span class='property-name'>Skewness:</span> " + latex('\\frac{2-p}{\\sqrt{r(1-p)}}'),
							unsafe_allow_html=True)
			else:
				st.markdown("<span class='property-name'>MGF:</span> " + latex(
					'M_X(t) = \\left(\\frac{p}{1-(1-p)e^t}\\right)^r'),
							unsafe_allow_html=True)
				st.markdown("<span class='property-name'>Skewness:</span> " + latex('\\frac{2-p}{\\sqrt{r(1-p)}}'),
							unsafe_allow_html=True)

		# Note about special cases
		st.markdown(
			"<div class='note'><strong>Special Cases:</strong> When r = 1, the negative binomial distribution reduces to the geometric distribution. "
			"As r → ∞ and p → 0 such that rp = λ is constant, the negative binomial converges to the Poisson distribution.</div>",
			unsafe_allow_html=True)

		# Visualization
		max_k = min(100, int(r * 5 / p))
		if first_version:
			x = np.arange(r, max_k + 1)
			pmf = stats.nbinom.pmf(x - r, r, p)
			cdf = stats.nbinom.cdf(x - r, r, p)
		else:
			x = np.arange(0, max_k - r + 1)
			pmf = stats.nbinom.pmf(x, r, p)
			cdf = stats.nbinom.cdf(x, r, p)

		fig = make_subplots(rows=2, cols=1,
							subplot_titles=("Probability Mass Function", "Cumulative Distribution Function"))

		fig.add_trace(
			go.Bar(x=x, y=pmf, marker_color='#AB47BC', name="PMF"),
			row=1, col=1
		)

		fig.add_trace(
			go.Scatter(x=x, y=cdf, mode='lines+markers', marker_color='#5C6BC0', name="CDF"),
			row=2, col=1
		)

		version_text = "trials to get r successes" if first_version else "failures before r successes"
		fig.update_layout(height=600, title_text=f"Negative Binomial Distribution (r={r}, p={p}, {version_text})")
		fig.update_xaxes(title_text=f"k ({version_text})")
		fig.update_yaxes(title_text="Probability", row=1, col=1)
		fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)

		st.plotly_chart(fig, use_container_width=True)

		# Animation showing the effect of r (required successes)
		st.markdown("<h4>Interactive Animation: Effect of Required Successes r</h4>", unsafe_allow_html=True)

		# Create frames for different values of r
		r_values = list(range(1, 21))

		# Create figure
		fig_r = go.Figure()

		# Add initial trace
		max_k_initial = min(100, int(r_values[0] * 5 / p))
		if first_version:
			x_initial = np.arange(r_values[0], max_k_initial + 1)
			pmf_initial = stats.nbinom.pmf(x_initial - r_values[0], r_values[0], p)
		else:
			x_initial = np.arange(0, max_k_initial - r_values[0] + 1)
			pmf_initial = stats.nbinom.pmf(x_initial, r_values[0], p)

		fig_r.add_trace(go.Bar(
			x=x_initial,
			y=pmf_initial,
			marker_color='#AB47BC',
			name=f"r={r_values[0]}"
		))

		# Add frames
		frames = []
		for r_val in r_values:
			max_k_frame = min(100, int(r_val * 5 / p))
			if first_version:
				x_frame = np.arange(r_val, max_k_frame + 1)
				pmf_frame = stats.nbinom.pmf(x_frame - r_val, r_val, p)
			else:
				x_frame = np.arange(0, max_k_frame - r_val + 1)
				pmf_frame = stats.nbinom.pmf(x_frame, r_val, p)

			frames.append(go.Frame(
				data=[go.Bar(
					x=x_frame,
					y=pmf_frame,
					marker_color='#AB47BC'
				)],
				name=f"r={r_val}",
				layout=go.Layout(
					title_text=f"Negative Binomial PMF (r={r_val}, p={p})",
					xaxis=dict(range=[-1 if not first_version else 0, 100]),
					yaxis=dict(range=[0, max(pmf_frame) * 1.1])
				)
			))

		fig_r.frames = frames

		# Add buttons and sliders
		fig_r.update_layout(
			title=f"Negative Binomial PMF with varying r (required successes)",
			xaxis=dict(title=f"k ({version_text})", range=[-1 if not first_version else 0, 100]),
			yaxis=dict(title="Probability"),
			updatemenus=[{
				'type': 'buttons',
				'showactive': False,
				'buttons': [
					{
						'label': 'Play',
						'method': 'animate',
						'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
					},
					{
						'label': 'Pause',
						'method': 'animate',
						'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
					}
				],
				'direction': 'left',
				'pad': {'r': 10, 't': 10},
				'x': 0.1,
				'xanchor': 'right',
				'y': 0,
				'yanchor': 'top'
			}],
			sliders=[{
				'active': 0,
				'steps': [
					{
						'method': 'animate',
						'args': [[f'r={r_val}'], {
							'frame': {'duration': 300, 'redraw': True},
							'mode': 'immediate',
							'transition': {'duration': 300}
						}],
						'label': f'{r_val}'
					} for r_val in r_values
				],
				'pad': {'t': 50},
				'len': 0.9,
				'x': 0.1,
				'xanchor': 'left',
				'y': 0,
				'yanchor': 'top',
				'currentvalue': {'prefix': 'r = '}
			}]
		)

		st.plotly_chart(fig_r, use_container_width=True)

		# Sum of Geometric Random Variables
		st.markdown("<h4>Negative Binomial as Sum of Geometric Random Variables</h4>", unsafe_allow_html=True)
		st.markdown("""
        The negative binomial distribution with parameters r and p can be represented as the sum of r independent 
        geometric random variables, each with parameter p.

        If X₁, X₂, ..., Xᵣ are independent geometric random variables (number of trials until first success),
        then Y = X₁ + X₂ + ... + Xᵣ follows a negative binomial distribution with parameters r and p.
        """)

		# Simulation to demonstrate this property
		if st.button("Run Simulation to Demonstrate"):
			n_simulations = 10000
			geo_sums = np.zeros(n_simulations)

			for i in range(n_simulations):
				# Generate r geometric random variables and sum them
				geo_vars = stats.geom.rvs(p, size=r)
				geo_sums[i] = np.sum(geo_vars)

			# Plot histogram of simulated sums vs. theoretical PMF
			fig_sim = plt.figure(figsize=(10, 6))

			# Calculate theoretical PMF
			max_k_sim = int(np.percentile(geo_sums, 99))
			if first_version:
				x_sim = np.arange(r, max_k_sim + 1)
				pmf_sim = stats.nbinom.pmf(x_sim - r, r, p)
			else:
				# Adjust for version difference
				x_sim = np.arange(r, max_k_sim + 1)
				pmf_sim = stats.nbinom.pmf(x_sim - r, r, p)

			# Plot histogram normalized to form a PMF
			plt.hist(geo_sums, bins=range(r, max_k_sim + 2), density=True, alpha=0.6, color='purple',
					 label='Simulated Sum of Geometric RVs')

			# Plot theoretical PMF
			plt.plot(x_sim, pmf_sim, 'ro-', lw=2, label='Theoretical Negative Binomial PMF')

			plt.xlabel('k (Number of trials)')
			plt.ylabel('Probability')
			plt.title(f'Sum of {r} Geometric RVs vs. Negative Binomial Distribution (p={p})')
			plt.legend()
			plt.grid(alpha=0.3)

			st.pyplot(fig_sim)

			st.markdown(
				"<div class='highlight'>The simulation demonstrates that the sum of r independent geometric random variables follows a negative binomial distribution with parameters r and p.</div>",
				unsafe_allow_html=True)

		# Applications
		st.markdown("<h4>Applications</h4>", unsafe_allow_html=True)
		st.markdown("""
        The negative binomial distribution is used to model:
        - Number of trials needed to achieve a specified number of successes
        - Overdispersed count data (when variance exceeds the mean)
        - Waiting time for multiple events to occur
        - Success/failure sequences in sports like baseball innings
        - Number of purchases before a customer buys r specific items
        """)

	elif discrete_dist == "Hypergeometric":
		st.markdown("<h3 class='section-header'>Hypergeometric Distribution</h3>", unsafe_allow_html=True)

		# Parameters
		col1, col2 = st.columns(2)
		with col1:
			N = st.slider("Population size (N):", 10, 100, 50)
			K = st.slider("Number of success states in population (K):", 1, N, min(20, N // 2))
		with col2:
			n = st.slider("Number of draws (n):", 1, N, min(10, N // 2))

		# Description
		st.markdown("""
        The hypergeometric distribution models the number of successes in a sample of size n drawn without 
        replacement from a finite population of size N containing exactly K successes.

        Unlike the binomial distribution, the hypergeometric does not assume independence between draws 
        because sampling is done without replacement.
        """)

		# Properties
		st.markdown("<h4>Properties</h4>", unsafe_allow_html=True)
		col1, col2 = st.columns(2)

		with col1:
			st.markdown("<span class='property-name'>PMF:</span> " + latex(
				'P(X = k) = \\frac{\\binom{K}{k}\\binom{N-K}{n-k}}{\\binom{N}{n}}'),
						unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Mean:</span> " + latex('E[X] = n\\frac{K}{N}'),
						unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Variance:</span> " + latex(
				'Var[X] = n\\frac{K}{N}\\frac{N-K}{N}\\frac{N-n}{N-1}'),
						unsafe_allow_html=True)

		with col2:
			st.markdown(
				"<span class='property-name'>Range:</span> " + latex('\\max(0, n+K-N) \\leq k \\leq \\min(n, K)'),
				unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Skewness:</span> " + latex(
				'\\frac{(N - 2K)\\sqrt{N - 1}(N - 2n)}{\\sqrt{nK(N - K)(N - n)}(N - 2)}'),
						unsafe_allow_html=True)
			st.markdown(
				"<span class='property-name'>Kurtosis Excess:</span> Complex expression depending on N, K, and n",
				unsafe_allow_html=True)

		# Calculate range for k
		k_min = max(0, n + K - N)
		k_max = min(n, K)

		# Visualization
		x = np.arange(k_min, k_max + 1)
		pmf = stats.hypergeom.pmf(x, N, K, n)
		cdf = stats.hypergeom.cdf(x, N, K, n)

		fig = make_subplots(rows=2, cols=1,
							subplot_titles=("Probability Mass Function", "Cumulative Distribution Function"))

		fig.add_trace(
			go.Bar(x=x, y=pmf, marker_color='#FFB74D', name="PMF"),
			row=1, col=1
		)

		fig.add_trace(
			go.Scatter(x=x, y=cdf, mode='lines+markers', marker_color='#4DD0E1', name="CDF"),
			row=2, col=1
		)

		fig.update_layout(height=600, title_text=f"Hypergeometric Distribution (N={N}, K={K}, n={n})")
		fig.update_xaxes(title_text="k (number of successes in sample)")
		fig.update_yaxes(title_text="Probability", row=1, col=1)
		fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)

		st.plotly_chart(fig, use_container_width=True)

		# Visualization of the sampling process
		st.markdown("<h4>Visualization of the Sampling Process</h4>", unsafe_allow_html=True)

		# Create a visual representation of the population and sample
		fig_sampling = plt.figure(figsize=(10, 6))

		# Plot population
		population = np.zeros(N)
		population[:K] = 1  # Success states

		plt.subplot(2, 1, 1)
		colors = ['#4CAF50' if x == 1 else '#F44336' for x in population]
		plt.bar(range(N), np.ones(N), color=colors)
		plt.title(f'Population of Size N={N} with K={K} Success States (Green)')
		plt.xlabel('Population Elements')
		plt.yticks([])

		# Plot a random sample and distribution of successes
		np.random.seed(42)  # For reproducibility
		n_samples = 1000
		success_counts = np.zeros(k_max - k_min + 1)

		for _ in range(n_samples):
			# Draw a sample
			sample_indices = np.random.choice(N, n, replace=False)
			sample = population[sample_indices]
			successes = np.sum(sample)
			success_counts[int(successes) - k_min] += 1

		plt.subplot(2, 1, 2)
		plt.bar(x, success_counts / n_samples, color='#FFB74D', alpha=0.7)
		plt.plot(x, pmf, 'ro-', lw=2, label='Theoretical PMF')
		plt.title(f'Distribution of Successes in {n_samples} Random Samples of Size n={n}')
		plt.xlabel('Number of Successes (k)')
		plt.ylabel('Probability')
		plt.legend()

		plt.tight_layout()
		st.pyplot(fig_sampling)

		# Comparison with Binomial Distribution
		st.markdown("<h4>Comparison with Binomial Distribution</h4>", unsafe_allow_html=True)

		st.markdown("""
        The hypergeometric distribution differs from the binomial distribution in that it models sampling 
        without replacement. When the population size N is large compared to the sample size n, 
        the hypergeometric distribution approaches the binomial distribution with parameters n and p = K/N.
        """)

		# Visualization comparing hypergeometric and binomial
		p_binom = K / N
		x_binom = np.arange(0, n + 1)
		binom_pmf = stats.binom.pmf(x_binom, n, p_binom)

		fig_compare = plt.figure(figsize=(10, 6))

		plt.bar(x, pmf, alpha=0.7, color='#FFB74D', label='Hypergeometric PMF')
		plt.plot(x_binom, binom_pmf, 'bo-', lw=2, label=f'Binomial PMF (n={n}, p={p_binom:.3f})')

		plt.xlabel('Number of Successes (k)')
		plt.ylabel('Probability')
		plt.title(f'Hypergeometric vs. Binomial: N={N}, K={K}, n={n}')
		plt.grid(alpha=0.3)
		plt.legend()

		st.pyplot(fig_compare)

		# Interactive demonstration of convergence to binomial
		st.markdown("<h4>Convergence to Binomial as N Increases</h4>", unsafe_allow_html=True)

		scaling_factor = st.slider("Population Size Multiplier:", 1, 10, 1)

		# Scale N and K while keeping the same proportion and same sample size
		N_scaled = N * scaling_factor
		K_scaled = K * scaling_factor

		# Calculate hypergeometric and binomial PMFs
		x_hyper = np.arange(max(0, n + K_scaled - N_scaled), min(n, K_scaled) + 1)
		pmf_hyper = stats.hypergeom.pmf(x_hyper, N_scaled, K_scaled, n)

		p_binom_scaled = K_scaled / N_scaled
		x_binom_scaled = np.arange(0, n + 1)
		binom_pmf_scaled = stats.binom.pmf(x_binom_scaled, n, p_binom_scaled)

		# Plot comparison
		fig_converge = plt.figure(figsize=(10, 6))

		plt.bar(x_hyper, pmf_hyper, alpha=0.7, color='#FFB74D',
				label=f'Hypergeometric PMF (N={N_scaled}, K={K_scaled})')
		plt.plot(x_binom_scaled, binom_pmf_scaled, 'bo-', lw=2, label=f'Binomial PMF (n={n}, p={p_binom_scaled:.3f})')

		plt.xlabel('Number of Successes (k)')
		plt.ylabel('Probability')
		plt.title(f'Convergence to Binomial as N Increases (Scaling Factor: {scaling_factor}x)')
		plt.grid(alpha=0.3)
		plt.legend()

		st.pyplot(fig_converge)


		# Calculate total variation distance to quantify similarity
		# Define a function to calculate total variation distance
		def tv_distance(p, q):
			common_range = range(max(min(p.keys()), min(q.keys())), min(max(p.keys()), max(q.keys())) + 1)
			return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in common_range)


		# Convert PMFs to dictionaries
		pmf_hyper_dict = {k: p for k, p in zip(x_hyper, pmf_hyper)}
		binom_pmf_dict = {k: p for k, p in zip(x_binom_scaled, binom_pmf_scaled)}

		# Calculate TV distance
		tv_dist = tv_distance(pmf_hyper_dict, binom_pmf_dict)

		st.markdown(
			f"<div class='highlight'>Total Variation Distance between the distributions: {tv_dist:.6f}<br>As this value approaches 0, the distributions become more similar.</div>",
			unsafe_allow_html=True)

		# Applications
		st.markdown("<h4>Applications</h4>", unsafe_allow_html=True)
		st.markdown("""
        The hypergeometric distribution is used to model:
        - Quality control (sampling from a finite batch without replacement)
        - Card drawing problems (e.g., poker hand probability)
        - Auditing (sampling accounts from a population)
        - Election polling before counting all votes
        - Ecological capture-recapture studies
        """)

	elif discrete_dist == "Uniform (Discrete)":
		st.markdown("<h3 class='section-header'>Discrete Uniform Distribution</h3>", unsafe_allow_html=True)

		# Parameters
		col1, col2 = st.columns(2)
		with col1:
			a = st.number_input("Lower bound (a):", value=1, step=1)
		with col2:
			b = st.number_input("Upper bound (b):", value=10, step=1)

		# Ensure a <= b
		if a > b:
			a, b = b, a

		# Description
		st.markdown("""
        The discrete uniform distribution assigns equal probability to each integer in a finite range. 
        It models situations where each outcome in a finite set is equally likely.
        """)

		# Properties
		st.markdown("<h4>Properties</h4>", unsafe_allow_html=True)
		col1, col2 = st.columns(2)

		with col1:
			st.markdown("<span class='property-name'>PMF:</span> " + latex('P(X = k) = \\frac{1}{b - a + 1}'),
						unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Mean:</span> " + latex('E[X] = \\frac{a + b}{2}'),
						unsafe_allow_html=True)
			st.markdown(
				"<span class='property-name'>Variance:</span> " + latex('Var[X] = \\frac{(b - a + 1)^2 - 1}{12}'),
				unsafe_allow_html=True)

		with col2:
			st.markdown("<span class='property-name'>Range:</span> " + latex('k \\in \\{a, a + 1, \\ldots, b\\}'),
						unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Skewness:</span> " + latex('0'),
						unsafe_allow_html=True)
			st.markdown("<span class='property-name'>Kurtosis Excess:</span> " + latex(
				'-\\frac{6(b - a)(b - a + 2)}{5(b - a + 1)^2}'),
						unsafe_allow_html=True)

		# Visualization
		x = np.arange(a, b + 1)
		pmf = np.ones_like(x) / (b - a + 1)
		cdf = np.arange(1, len(x) + 1) / (b - a + 1)

		fig = make_subplots(rows=2, cols=1,
							subplot_titles=("Probability Mass Function", "Cumulative Distribution Function"))

		fig.add_trace(
			go.Bar(x=x, y=pmf, marker_color='#64B5F6', name="PMF"),
			row=1, col=1
		)

		fig.add_trace(
			go.Scatter(x=x, y=cdf, mode='lines+markers', marker_color='#F06292', name="CDF"),
			row=2, col=1
		)

		fig.update_layout(height=600, title_text=f"Discrete Uniform Distribution (a={a}, b={b})")
		fig.update_xaxes(title_text="k")
		fig.update_yaxes(title_text="Probability", row=1, col=1)
		fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)

		st.plotly_chart(fig, use_container_width=True)

		# Simulation of fair dice and other applications
		st.markdown("<h4>Simulation of a Fair Die</h4>", unsafe_allow_html=True)

		if a == 1 and b == 6:
			st.markdown("""
            A perfect example of a discrete uniform distribution is a fair die, where each face (1-6) has an equal probability of 1/6.
            """)

			# Simulate rolling the die
			n_rolls = st.slider("Number of die rolls to simulate:", 10, 10000, 1000)

			# Generate random rolls
			np.random.seed(42)  # For reproducibility
			rolls = np.random.randint(a, b + 1, size=n_rolls)

			# Count frequencies
			unique, counts = np.unique(rolls, return_counts=True)
			frequencies = counts / n_rolls

			# Theoretical probabilities
			theoretical = np.ones_like(unique) / (b - a + 1)

			# Plot comparison
			fig_dice = plt.figure(figsize=(10, 6))

			width = 0.35
			x_pos = np.arange(len(unique))

			plt.bar(x_pos - width / 2, frequencies, width=width, color='#64B5F6', label='Simulated')
			plt.bar(x_pos + width / 2, theoretical, width=width, color='#F06292', label='Theoretical')

			plt.xlabel('Die Face')
			plt.ylabel('Probability')
			plt.title(f'Simulated vs Theoretical Probabilities ({n_rolls} Rolls)')
			plt.xticks(x_pos, unique)
			plt.legend()
			plt.grid(alpha=0.3)

			st.pyplot(fig_dice)

			# Law of large numbers demonstration
			st.markdown("<h4>Law of Large Numbers Demonstration</h4>", unsafe_allow_html=True)

			# Calculate running mean
			cumulative_mean = np.cumsum(rolls) / np.arange(1, n_rolls + 1)

			# Plot convergence to expected value
			fig_lln = plt.figure(figsize=(10, 6))

			plt.plot(np.arange(1, n_rolls + 1), cumulative_mean, color='#3F51B5')
			plt.axhline(y=(a + b) / 2, color='#FF5722', linestyle='--', label=f'Expected Value = {(a + b) / 2}')

			plt.xscale('log')
			plt.xlabel('Number of Rolls (Log Scale)')
			plt.ylabel('Cumulative Mean')
			plt.title('Convergence to Expected Value - Law of Large Numbers')
			plt.legend()
			plt.grid(alpha=0.3)

			st.pyplot(fig_lln)

			st.markdown(
				"<div class='highlight'>The Law of Large Numbers demonstrates that as the number of trials increases, the sample mean converges to the theoretical expected value.</div>",
				unsafe_allow_html=True)
		else:
			# General simulation for any uniform range
			st.markdown(f"""
            Simulating a discrete uniform random variable on the range [{a}, {b}], similar to rolling a {b - a + 1}-sided die
            or drawing randomly from {b - a + 1} equally likely outcomes.
            """)

			# Simulate draws
			n_draws = st.slider("Number of random draws to simulate:", 10, 10000, 1000)

			# Generate random draws
			np.random.seed(42)  # For reproducibility
			draws = np.random.randint(a, b + 1, size=n_draws)

			# Count frequencies
			unique, counts = np.unique(draws, return_counts=True)
			frequencies = counts / n_draws

			# Theoretical probabilities
			theoretical = np.ones_like(unique) / (b - a + 1)

			# Plot comparison
			fig_sim = plt.figure(figsize=(10, 6))

			width = 0.35
			x_pos = np.arange(len(unique))

			plt.bar(x_pos - width / 2, frequencies, width=width, color='#64B5F6', label='Simulated')
			plt.bar(x_pos + width / 2, theoretical, width=width, color='#F06292', label='Theoretical')

			plt.xlabel('Value')
			plt.ylabel('Probability')
			plt.title(f'Simulated vs Theoretical Probabilities ({n_draws} Draws)')
			plt.xticks(x_pos, unique)
			plt.legend()
			plt.grid(alpha=0.3)

			st.pyplot(fig_sim)

		# Applications
		st.markdown("<h4>Applications</h4>", unsafe_allow_html=True)
		st.markdown("""
        The discrete uniform distribution is used to model:
        - Fair dice rolls and random card draws
        - Random number generation in computer science
        - Random selection from a finite set with equal probability
        - Default prior distribution when all outcomes are equally likely
        - Round-robin scheduling algorithms
        """)

		# Entropy and maximum entropy principle
		st.markdown("<h4>Maximum Entropy Principle</h4>", unsafe_allow_html=True)
		st.markdown("""
        The discrete uniform distribution has the maximum entropy among all discrete distributions with a given finite support.

        Entropy is a measure of uncertainty or randomness, defined as:
        """)

		st.markdown(
			f"<div class='formula'>H(X) = -\\sum_{{i}} P(X = i) \\log P(X = i)</div>",
			unsafe_allow_html=True
		)
		# Calculate entropy of the uniform distribution
		entropy_uniform = -np.sum(pmf * np.log(pmf))

		st.markdown(
			f"<div class='highlight'>Entropy of this discrete uniform distribution: {entropy_uniform:.6f} nats (or {entropy_uniform / np.log(2):.6f} bits)</div>",
			unsafe_allow_html=True)

		st.markdown("""
        The maximum entropy principle states that when estimating a probability distribution, 
        you should select the distribution with maximum entropy consistent with your constraints.

        For a given range [a,b], the discrete uniform distribution is the maximum entropy distribution.
        """)

# Continuous Distributions Section
elif selected_section == "Continuous Distributions":
	st.markdown("<h2 class='sub-header'>Continuous Probability Distributions</h2>", unsafe_allow_html=True)

	# Select distribution
	continuous_dist = st.selectbox(
		"Choose a continuous distribution:",
		["Normal (Gaussian)", "Uniform (Continuous)", "Exponential", "Gamma",
		 "Beta", "t-Distribution", "Chi-Square", "F-Distribution", "Lognormal", "Weibull"]
	)

	if continuous_dist == "Normal (Gaussian)":
		st.markdown("<h3 class='section-header'>Normal (Gaussian) Distribution</h3>", unsafe_allow_html=True)

		# Parameters
		col1, col2 = st.columns(2)
		with col1:
			mu = st.slider("Mean (μ):", -10.0, 10.0, 0.0, 0.1)
		with col2:
			sigma = st.slider("Standard Deviation (σ):", 0.1, 10.0, 1.0, 0.1)

		# Description
		st.markdown("""
        The normal distribution is one of the most important probability distributions in statistics. 
        It is a continuous distribution that is symmetric about its mean, with the majority of observations 
        clustered around the central peak, and the probabilities for values further away from the mean 
        taper off equally in both directions.
        """)

		# Properties
		st.markdown("<h4>Properties</h4>", unsafe_allow_html=True)
		col1, col2 = st.columns(2)

		import streamlit as st

		# Assuming col1 and col2 are defined via st.columns()
		col1, col2 = st.columns(2)

		with col1:
			# PDF
			st.markdown(
				"<span class='property-name'>PDF:</span> " + latex(
					r'f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}'),
				unsafe_allow_html=True
			)
			# Mean
			st.markdown(
				"<span class='property-name'>Mean:</span> " + latex(r'E[X] = \mu'),
				unsafe_allow_html=True
			)
			# Variance
			st.markdown(
				"<span class='property-name'>Variance:</span> " + latex(r'Var[X] = \sigma^2'),
				unsafe_allow_html=True
			)

		with col2:
			# MGF
			st.markdown(
				"<span class='property-name'>MGF:</span> " + latex(r'M_X(t) = e^{\mu t + \frac{\sigma^2 t^2}{2}}'),
				unsafe_allow_html=True
			)
			# Skewness
			st.markdown(
				"<span class='property-name'>Skewness:</span> " + latex('0'),
				unsafe_allow_html=True
			)
			# Kurtosis
			st.markdown(
				"<span class='property-name'>Kurtosis:</span> " + latex('3'),
				unsafe_allow_html=True
			)

		# Standard normal and z-scores
		st.markdown(
			"<div class='note'><strong>Standard Normal Distribution:</strong> When μ=0 and σ=1, we have the standard normal distribution (Z). Any normal random variable X can be transformed to the standard normal using the formula Z = (X-μ)/σ. This transformation is the basis for z-scores.</div>",
			unsafe_allow_html=True
		)

		# Visualization
		x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
		pdf = stats.norm.pdf(x, mu, sigma)
		cdf = stats.norm.cdf(x, mu, sigma)

		fig = make_subplots(rows=2, cols=1,
							subplot_titles=("Probability Density Function", "Cumulative Distribution Function"))

		fig.add_trace(
			go.Scatter(x=x, y=pdf, mode='lines', line=dict(color='#1E88E5', width=2), name="PDF"),
			row=1, col=1
		)

		fig.add_trace(
			go.Scatter(x=x, y=cdf, mode='lines', line=dict(color='#FFA000', width=2), name="CDF"),
			row=2, col=1
		)

		fig.update_layout(height=600, title_text=f"Normal Distribution (μ={mu}, σ={sigma})")
		fig.update_xaxes(title_text="x")
		fig.update_yaxes(title_text="Density", row=1, col=1)
		fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)

		st.plotly_chart(fig, use_container_width=True)

		# Empirical Rule (68-95-99.7 rule)
		st.markdown("<h4>Empirical Rule (68-95-99.7 Rule)</h4>", unsafe_allow_html=True)

		# Calculate intervals
		one_sigma = (mu - 1 * sigma, mu + 1 * sigma)
		two_sigma = (mu - 2 * sigma, mu + 2 * sigma)
		three_sigma = (mu - 3 * sigma, mu + 3 * sigma)

		# Calculate probabilities for verification
		p_one_sigma = stats.norm.cdf(one_sigma[1], mu, sigma) - stats.norm.cdf(one_sigma[0], mu, sigma)
		p_two_sigma = stats.norm.cdf(two_sigma[1], mu, sigma) - stats.norm.cdf(two_sigma[0], mu, sigma)
		p_three_sigma = stats.norm.cdf(three_sigma[1], mu, sigma) - stats.norm.cdf(three_sigma[0], mu, sigma)

		# Visualization of the empirical rule
		fig_empirical = go.Figure()

		# Plot PDF
		fig_empirical.add_trace(go.Scatter(
			x=x, y=pdf,
			mode='lines',
			line=dict(color='black', width=2),
			name='PDF'
		))

		# Add colored areas for different sigma ranges
		x_fill = np.linspace(one_sigma[0], one_sigma[1], 500)
		fig_empirical.add_trace(go.Scatter(
			x=x_fill,
			y=stats.norm.pdf(x_fill, mu, sigma),
			fill='tozeroy',
			mode='none',
			name='68% (±1σ)',
			fillcolor='rgba(255, 0, 0, 0.3)'
		))

		x_fill = np.linspace(one_sigma[0], one_sigma[1], 500)
		y_fill = stats.norm.pdf(x_fill, mu, sigma)
		x_fill = np.concatenate([x_fill, [one_sigma[1], two_sigma[1], two_sigma[1], one_sigma[1]]])
		y_fill = np.concatenate([y_fill, [0, 0, stats.norm.pdf(two_sigma[1], mu, sigma), 0]])

		# For ±2σ (need to exclude ±1σ which is already colored)
		x_fill1 = np.linspace(one_sigma[1], two_sigma[1], 200)
		x_fill2 = np.linspace(one_sigma[0], two_sigma[0], 200)
		y_fill1 = stats.norm.pdf(x_fill1, mu, sigma)
		y_fill2 = stats.norm.pdf(x_fill2, mu, sigma)

		fig_empirical.add_trace(go.Scatter(
			x=np.concatenate([x_fill1, [two_sigma[1]], [one_sigma[1]], [one_sigma[1]]]),
			y=np.concatenate([y_fill1, [0], [0], [y_fill1[0]]]),
			fill='toself',
			mode='none',
			name='95% (±2σ) excluding 68%',
			fillcolor='rgba(0, 255, 0, 0.3)'
		))

		fig_empirical.add_trace(go.Scatter(
			x=np.concatenate([[one_sigma[0]], [two_sigma[0]], [two_sigma[0]], x_fill2[::-1]]),
			y=np.concatenate([[y_fill2[0]], [0], [0], y_fill2[::-1]]),
			fill='toself',
			mode='none',
			showlegend=False,
			fillcolor='rgba(0, 255, 0, 0.3)'
		))

		# For ±3σ (need to exclude ±2σ which is already colored)
		x_fill1 = np.linspace(two_sigma[1], three_sigma[1], 200)
		x_fill2 = np.linspace(two_sigma[0], three_sigma[0], 200)
		y_fill1 = stats.norm.pdf(x_fill1, mu, sigma)
		y_fill2 = stats.norm.pdf(x_fill2, mu, sigma)

		fig_empirical.add_trace(go.Scatter(
			x=np.concatenate([x_fill1, [three_sigma[1]], [two_sigma[1]], [two_sigma[1]]]),
			y=np.concatenate([y_fill1, [0], [0], [y_fill1[0]]]),
			fill='toself',
			mode='none',
			name='99.7% (±3σ) excluding 95%',
			fillcolor='rgba(0, 0, 255, 0.3)'
		))

		fig_empirical.add_trace(go.Scatter(
			x=np.concatenate([[two_sigma[0]], [three_sigma[0]], [three_sigma[0]], x_fill2[::-1]]),
			y=np.concatenate([[y_fill2[0]], [0], [0], y_fill2[::-1]]),
			fill='toself',
			mode='none',
			showlegend=False,
			fillcolor='rgba(0, 0, 255, 0.3)'
		))

		# Add annotations for percentages
		fig_empirical.add_annotation(
			x=mu, y=stats.norm.pdf(mu, mu, sigma) * 0.9,
			text=f"68.27%<br>±1σ",
			showarrow=False,
			font=dict(color="red", size=14)
		)

		fig_empirical.add_annotation(
			x=mu + 1.5 * sigma, y=stats.norm.pdf(mu + 1.5 * sigma, mu, sigma) * 0.9,
			text=f"95.45%<br>±2σ",
			showarrow=False,
			font=dict(color="green", size=14)
		)

		fig_empirical.add_annotation(
			x=mu + 2.5 * sigma, y=stats.norm.pdf(mu + 2.5 * sigma, mu, sigma) * 0.9,
			text=f"99.73%<br>±3σ",
			showarrow=False,
			font=dict(color="blue", size=14)
		)

		# Update layout
		fig_empirical.update_layout(
			title="Empirical Rule (68-95-99.7 Rule) for Normal Distribution",
			xaxis_title="x",
			yaxis_title="Density",
			xaxis=dict(
				tickvals=[
					mu - 3 * sigma, mu - 2 * sigma, mu - 1 * sigma,
					mu,
					mu + 1 * sigma, mu + 2 * sigma, mu + 3 * sigma
				],
				ticktext=[
					f"μ-3σ<br>{mu - 3 * sigma:.2f}",
					f"μ-2σ<br>{mu - 2 * sigma:.2f}",
					f"μ-σ<br>{mu - sigma:.2f}",
					f"μ<br>{mu:.2f}",
					f"μ+σ<br>{mu + sigma:.2f}",
					f"μ+2σ<br>{mu + 2 * sigma:.2f}",
					f"μ+3σ<br>{mu + 3 * sigma:.2f}"
				]
			),
			height=500
		)

		st.plotly_chart(fig_empirical, use_container_width=True)

		# Display calculated probabilities
		col1, col2, col3 = st.columns(3)
		with col1:
			st.markdown(
				f"<center><strong>P(μ-σ < X < μ+σ):</strong><br>{p_one_sigma:.6f}<br><i>(Expected: 0.6827)</i></center>",
				unsafe_allow_html=True)
		with col2:
			st.markdown(
				f"<center><strong>P(μ-2σ < X < μ+2σ):</strong><br>{p_two_sigma:.6f}<br><i>(Expected: 0.9545)</i></center>",
				unsafe_allow_html=True)
		with col3:
			st.markdown(
				f"<center><strong>P(μ-3σ < X < μ+3σ):</strong><br>{p_three_sigma:.6f}<br><i>(Expected: 0.9973)</i></center>",
				unsafe_allow_html=True)

		# Probability calculator
		st.markdown("<h4>Normal Probability Calculator</h4>", unsafe_allow_html=True)

		calculation_type = st.radio(
			"Select calculation type:",
			["P(X ≤ x)", "P(X ≥ x)", "P(a ≤ X ≤ b)"]
		)

		if calculation_type == "P(X ≤ x)":
			x_val = st.slider("x value:", float(mu - 4 * sigma), float(mu + 4 * sigma), float(mu), 0.01)
			prob = stats.norm.cdf(x_val, mu, sigma)

			# Visualization
			fig_prob = go.Figure()

			# Plot PDF
			fig_prob.add_trace(go.Scatter(
				x=x, y=pdf,
				mode='lines',
				line=dict(color='black', width=2),
				name='PDF'
			))

			# Add filled area for the probability
			x_fill = np.linspace(min(x), x_val, 500)
			fig_prob.add_trace(go.Scatter(
				x=x_fill,
				y=stats.norm.pdf(x_fill, mu, sigma),
				fill='tozeroy',
				mode='none',
				name=f'P(X ≤ {x_val:.2f})',
				fillcolor='rgba(65, 105, 225, 0.5)'
			))

			# Add vertical line at x_val
			fig_prob.add_shape(
				type="line",
				x0=x_val, y0=0,
				x1=x_val, y1=stats.norm.pdf(x_val, mu, sigma),
				line=dict(color="red", width=2, dash="dash")
			)

			# Add annotation for the probability
			fig_prob.add_annotation(
				x=x_val, y=stats.norm.pdf(x_val, mu, sigma) / 2,
				text=f"P(X ≤ {x_val:.2f}) = {prob:.4f}",
				showarrow=True,
				arrowhead=1
			)

			# Update layout
			fig_prob.update_layout(
				title=f"Probability: P(X ≤ {x_val:.2f})",
				xaxis_title="x",
				yaxis_title="Density",
				height=400
			)

			st.plotly_chart(fig_prob, use_container_width=True)

		elif calculation_type == "P(X ≥ x)":
			x_val = st.slider("x value:", float(mu - 4 * sigma), float(mu + 4 * sigma), float(mu), 0.01)
			prob = 1 - stats.norm.cdf(x_val, mu, sigma)

			# Visualization
			fig_prob = go.Figure()

			# Plot PDF
			fig_prob.add_trace(go.Scatter(
				x=x, y=pdf,
				mode='lines',
				line=dict(color='black', width=2),
				name='PDF'
			))

			# Add filled area for the probability
			x_fill = np.linspace(x_val, max(x), 500)
			fig_prob.add_trace(go.Scatter(
				x=x_fill,
				y=stats.norm.pdf(x_fill, mu, sigma),
				fill='tozeroy',
				mode='none',
				name=f'P(X ≥ {x_val:.2f})',
				fillcolor='rgba(255, 105, 97, 0.5)'
			))

			# Add vertical line at x_val
			fig_prob.add_shape(
				type="line",
				x0=x_val, y0=0,
				x1=x_val, y1=stats.norm.pdf(x_val, mu, sigma),
				line=dict(color="red", width=2, dash="dash")
			)

			# Add annotation for the probability
			fig_prob.add_annotation(
				x=x_val, y=stats.norm.pdf(x_val, mu, sigma) / 2,
				text=f"P(X ≥ {x_val:.2f}) = {prob:.4f}",
				showarrow=True,
				arrowhead=1
			)

			# Update layout
			fig_prob.update_layout(
				title=f"Probability: P(X ≥ {x_val:.2f})",
				xaxis_title="x",
				yaxis_title="Density",
				height=400
			)

			st.plotly_chart(fig_prob, use_container_width=True)

		else:  # P(a ≤ X ≤ b)
			col1, col2 = st.columns(2)
			with col1:
				a_val = st.slider("a value:", float(mu - 4 * sigma), float(mu + 4 * sigma), float(mu - sigma), 0.01)
			with col2:
				b_val = st.slider("b value:", float(mu - 4 * sigma), float(mu + 4 * sigma), float(mu + sigma), 0.01)

			# Ensure a <= b
			if a_val > b_val:
				a_val, b_val = b_val, a_val

			prob = stats.norm.cdf(b_val, mu, sigma) - stats.norm.cdf(a_val, mu, sigma)

			# Visualization
			fig_prob = go.Figure()

			# Plot PDF
			fig_prob.add_trace(go.Scatter(
				x=x, y=pdf,
				mode='lines',
				line=dict(color='black', width=2),
				name='PDF'
			))

			# Add filled area for the probability
			x_fill = np.linspace(a_val, b_val, 500)
			fig_prob.add_trace(go.Scatter(
				x=x_fill,
				y=stats.norm.pdf(x_fill, mu, sigma),
				fill='tozeroy',
				mode='none',
				name=f'P({a_val:.2f} ≤ X ≤ {b_val:.2f})',
				fillcolor='rgba(50, 205, 50, 0.5)'
			))

			# Add vertical lines at a_val and b_val
			fig_prob.add_shape(
				type="line",
				x0=a_val, y0=0,
				x1=a_val, y1=stats.norm.pdf(a_val, mu, sigma),
				line=dict(color="red", width=2, dash="dash")
			)

			fig_prob.add_shape(
				type="line",
				x0=b_val, y0=0,
				x1=b_val, y1=stats.norm.pdf(b_val, mu, sigma),
				line=dict(color="red", width=2, dash="dash")
			)

			# Add annotation for the probability
			fig_prob.add_annotation(
				x=(a_val + b_val) / 2, y=stats.norm.pdf((a_val + b_val) / 2, mu, sigma) / 2,
				text=f"P({a_val:.2f} ≤ X ≤ {b_val:.2f}) = {prob:.4f}",
				showarrow=True,
				arrowhead=1
			)

			# Update layout
			fig_prob.update_layout(
				title=f"Probability: P({a_val:.2f} ≤ X ≤ {b_val:.2f})",
				xaxis_title="x",
				yaxis_title="Density",
				height=400
			)

			st.plotly_chart(fig_prob, use_container_width=True)

		# CLT demonstration
		st.markdown("<h4>Central Limit Theorem Demonstration</h4>", unsafe_allow_html=True)

		st.markdown("""
        The Central Limit Theorem (CLT) states that the sampling distribution of the sample mean approaches 
        a normal distribution as the sample size increases, regardless of the shape of the population distribution.

        Let's demonstrate this by sampling from different distributions and observing how the sample mean distribution becomes normal.
        """)

		dist_type = st.selectbox(
			"Select original distribution to sample from:",
			["Uniform", "Exponential", "Chi-Square (df=2)", "Gamma (k=1, θ=2)", "Binomial (n=1, p=0.3)",
			 "Custom (Bimodal)"]
		)

		sample_size = st.select_slider(
			"Sample size (n):",
			options=[1, 2, 5, 10, 20, 30, 50, 100]
		)

		# Generate samples
		np.random.seed(42)  # For reproducibility
		n_simulations = 5000

		if dist_type == "Uniform":
			# Uniform on [0, 1]
			original_samples = np.random.uniform(0, 1, size=(n_simulations, sample_size))
			theoretical_mean = 0.5
			theoretical_std = 1 / np.sqrt(12) / np.sqrt(sample_size)
			x_range = (0, 1)
			title = "Uniform Distribution (0, 1)"

		elif dist_type == "Exponential":
			# Exponential with lambda=1
			original_samples = np.random.exponential(1, size=(n_simulations, sample_size))
			theoretical_mean = 1
			theoretical_std = 1 / np.sqrt(sample_size)
			x_range = (0, 5)
			title = "Exponential Distribution (λ=1)"

		elif dist_type == "Chi-Square (df=2)":
			# Chi-square with df=2
			original_samples = np.random.chisquare(2, size=(n_simulations, sample_size))
			theoretical_mean = 2
			theoretical_std = np.sqrt(4) / np.sqrt(sample_size)
			x_range = (0, 10)
			title = "Chi-Square Distribution (df=2)"

		elif dist_type == "Gamma (k=1, θ=2)":
			# Gamma with shape=1, scale=2
			original_samples = np.random.gamma(1, 2, size=(n_simulations, sample_size))
			theoretical_mean = 1 * 2  # shape * scale
			theoretical_std = np.sqrt(1 * 2 ** 2) / np.sqrt(sample_size)
			x_range = (0, 10)
			title = "Gamma Distribution (k=1, θ=2)"

		elif dist_type == "Binomial (n=1, p=0.3)":
			# Binomial with n=1, p=0.3 (Bernoulli)
			original_samples = np.random.binomial(1, 0.3, size=(n_simulations, sample_size))
			theoretical_mean = 0.3
			theoretical_std = np.sqrt(0.3 * 0.7) / np.sqrt(sample_size)
			x_range = (0, 1)
			title = "Binomial Distribution (n=1, p=0.3)"

		else:  # Custom (Bimodal)
			# Create a bimodal distribution (mixture of two normals)
			samples1 = np.random.normal(-2, 0.5, size=(n_simulations, sample_size // 2))
			samples2 = np.random.normal(2, 0.5, size=(n_simulations, sample_size - sample_size // 2))
			original_samples = np.column_stack([samples1, samples2])
			theoretical_mean = 0  # (-2 * 0.5 + 2 * 0.5) for perfect symmetry
			theoretical_std = np.sqrt(4.25) / np.sqrt(sample_size)  # Approximate
			x_range = (-4, 4)
			title = "Bimodal Distribution (Mix of N(-2,0.5) and N(2,0.5))"

		# Calculate sample means
		sample_means = np.mean(original_samples, axis=1)

		# Create visualization: original distribution vs distribution of sample means
		fig_clt = make_subplots(
			rows=1, cols=2,
			subplot_titles=["Original Distribution", f"Distribution of Sample Means (n={sample_size})"]
		)

		# Flatten original samples for histogram
		flat_samples = original_samples.flatten()

		# Plot original distribution
		fig_clt.add_trace(
			go.Histogram(
				x=flat_samples,
				histnorm='probability density',
				name="Original",
				marker_color='#4285F4',
				opacity=0.7,
				nbinsx=30
			),
			row=1, col=1
		)

		# Fit KDE to original distribution for smoother visualization
		if len(flat_samples) > 100:  # Only if enough samples
			x_kde = np.linspace(min(flat_samples), max(flat_samples), 1000)
			kde = stats.gaussian_kde(flat_samples)
			fig_clt.add_trace(
				go.Scatter(
					x=x_kde,
					y=kde(x_kde),
					mode='lines',
					line=dict(color='#0F9D58', width=2),
					name="KDE of Original"
				),
				row=1, col=1
			)

		# Plot distribution of sample means
		fig_clt.add_trace(
			go.Histogram(
				x=sample_means,
				histnorm='probability density',
				name="Sample Means",
				marker_color='#DB4437',
				opacity=0.7,
				nbinsx=30
			),
			row=1, col=2
		)

		# Plot normal curve based on CLT
		x_norm = np.linspace(
			min(sample_means),
			max(sample_means),
			1000
		)

		fig_clt.add_trace(
			go.Scatter(
				x=x_norm,
				y=stats.norm.pdf(x_norm, theoretical_mean, theoretical_std),
				mode='lines',
				line=dict(color='#F4B400', width=2),
				name="Normal (CLT)"
			),
			row=1, col=2
		)

		# Update layout
		fig_clt.update_layout(
			title_text=f"Central Limit Theorem Demonstration - {title}",
			height=500,
			barmode='overlay'
		)

		st.plotly_chart(fig_clt, use_container_width=True)

		st.markdown(f"""
        <div class='highlight'>
        <strong>Theoretical Values Based on CLT:</strong><br>
        Mean of Sample Means: {theoretical_mean:.4f}<br>
        Standard Deviation of Sample Means: {theoretical_std:.4f} (= σ/√n)<br>
        Observed Mean of Sample Means: {np.mean(sample_means):.4f}<br>
        Observed Standard Deviation of Sample Means: {np.std(sample_means, ddof=1):.4f}
        </div>
        """, unsafe_allow_html=True)

		# Animation of CLT with increasing sample size
		st.markdown("<h4>Animation: Effect of Sample Size on CLT</h4>", unsafe_allow_html=True)

		if st.button("Generate CLT Animation"):
			# Create animation using Plotly
			sample_sizes = [1, 2, 5, 10, 20, 30, 50, 100]

			# Initialize figure
			fig_anim = go.Figure()

			# Add frames
			frames = []
			for n_size in sample_sizes:
				# Generate samples
				if dist_type == "Uniform":
					samples = np.random.uniform(0, 1, size=(n_simulations, n_size))
					mean = 0.5
					std = 1 / np.sqrt(12) / np.sqrt(n_size)

				elif dist_type == "Exponential":
					samples = np.random.exponential(1, size=(n_simulations, n_size))
					mean = 1
					std = 1 / np.sqrt(n_size)

				elif dist_type == "Chi-Square (df=2)":
					samples = np.random.chisquare(2, size=(n_simulations, n_size))
					mean = 2
					std = np.sqrt(4) / np.sqrt(n_size)

				elif dist_type == "Gamma (k=1, θ=2)":
					samples = np.random.gamma(1, 2, size=(n_simulations, n_size))
					mean = 1 * 2
					std = np.sqrt(1 * 2 ** 2) / np.sqrt(n_size)

				elif dist_type == "Binomial (n=1, p=0.3)":
					samples = np.random.binomial(1, 0.3, size=(n_simulations, n_size))
					mean = 0.3
					std = np.sqrt(0.3 * 0.7) / np.sqrt(n_size)

				else:  # Custom (Bimodal)
					samples1 = np.random.normal(-2, 0.5, size=(n_simulations, n_size // 2))
					samples2 = np.random.normal(2, 0.5, size=(n_simulations, n_size - n_size // 2))
					samples = np.column_stack([samples1, samples2])
					mean = 0
					std = np.sqrt(4.25) / np.sqrt(n_size)

				# Calculate sample means
				means = np.mean(samples, axis=1)

				# Create histogram data
				hist_data = np.histogram(
					means,
					bins=np.linspace(theoretical_mean - 4 * theoretical_std,
									 theoretical_mean + 4 * theoretical_std,
									 40),
					density=True
				)

				# Create normal curve data
				x_norm = np.linspace(
					theoretical_mean - 4 * theoretical_std,
					theoretical_mean + 4 * theoretical_std,
					1000
				)
				y_norm = stats.norm.pdf(x_norm, mean, std)

				# Add frame
				frames.append(go.Frame(
					data=[
						go.Bar(
							x=hist_data[1][:-1],
							y=hist_data[0],
							width=np.diff(hist_data[1]),
							name="Sample Means",
							marker_color='#DB4437',
							opacity=0.7
						),
						go.Scatter(
							x=x_norm,
							y=y_norm,
							mode='lines',
							line=dict(color='#F4B400', width=2),
							name="Normal (CLT)"
						)
					],
					name=f"n={n_size}",
					layout=go.Layout(
						title_text=f"Distribution of Sample Means (n={n_size})",
						xaxis=dict(
							range=[theoretical_mean - 4 * (1 / np.sqrt(12)),
								   theoretical_mean + 4 * (1 / np.sqrt(12))]
						),
						yaxis=dict(
							range=[0, max(max(y_norm), max(hist_data[0])) * 1.1]
						)
					)
				))

			# Initial data
			n_size_init = 1
			if dist_type == "Uniform":
				samples_init = np.random.uniform(0, 1, size=(n_simulations, n_size_init))
				mean_init = 0.5
				std_init = 1 / np.sqrt(12) / np.sqrt(n_size_init)

			elif dist_type == "Exponential":
				samples_init = np.random.exponential(1, size=(n_simulations, n_size_init))
				mean_init = 1
				std_init = 1 / np.sqrt(n_size_init)

			elif dist_type == "Chi-Square (df=2)":
				samples_init = np.random.chisquare(2, size=(n_simulations, n_size_init))
				mean_init = 2
				std_init = np.sqrt(4) / np.sqrt(n_size_init)

			elif dist_type == "Gamma (k=1, θ=2)":
				samples_init = np.random.gamma(1, 2, size=(n_simulations, n_size_init))
				mean_init = 1 * 2
				std_init = np.sqrt(1 * 2 ** 2) / np.sqrt(n_size_init)

			elif dist_type == "Binomial (n=1, p=0.3)":
				samples_init = np.random.binomial(1, 0.3, size=(n_simulations, n_size_init))
				mean_init = 0.3
				std_init = np.sqrt(0.3 * 0.7) / np.sqrt(n_size_init)

			else:  # Custom (Bimodal)
				samples1_init = np.random.normal(-2, 0.5, size=(n_simulations, n_size_init // 2))
				samples2_init = np.random.normal(2, 0.5, size=(n_simulations, n_size_init - n_size_init // 2))
				samples_init = np.column_stack([samples1_init, samples2_init])
				mean_init = 0
				std_init = np.sqrt(4.25) / np.sqrt(n_size_init)

			# Calculate sample means
			means_init = np.mean(samples_init, axis=1)

			# Create histogram data
			hist_data_init = np.histogram(
				means_init,
				bins=np.linspace(theoretical_mean - 4 * theoretical_std,
								 theoretical_mean + 4 * theoretical_std,
								 40),
				density=True
			)

			# Create normal curve data
			x_norm_init = np.linspace(
				theoretical_mean - 4 * theoretical_std,
				theoretical_mean + 4 * theoretical_std,
				1000
			)
			y_norm_init = stats.norm.pdf(x_norm_init, mean_init, std_init)

			# Add initial traces
			fig_anim.add_trace(go.Bar(
				x=hist_data_init[1][:-1],
				y=hist_data_init[0],
				width=np.diff(hist_data_init[1]),
				name="Sample Means",
				marker_color='#DB4437',
				opacity=0.7
			))

			fig_anim.add_trace(go.Scatter(
				x=x_norm_init,
				y=y_norm_init,
				mode='lines',
				line=dict(color='#F4B400', width=2),
				name="Normal (CLT)"
			))

			# Set frames
			fig_anim.frames = frames

			# Add animation controls
			fig_anim.update_layout(
				title=f"Distribution of Sample Means - {title}",
				xaxis=dict(
					range=[theoretical_mean - 4 * (1 / np.sqrt(12)),
						   theoretical_mean + 4 * (1 / np.sqrt(12))]
				),
				yaxis=dict(
					range=[0, max(max(y_norm_init), max(hist_data_init[0])) * 1.1]
				),
				updatemenus=[{
					'type': 'buttons',
					'showactive': False,
					'buttons': [
						{
							'label': 'Play',
							'method': 'animate',
							'args': [None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True}]
						},
						{
							'label': 'Pause',
							'method': 'animate',
							'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
						}
					],
					'direction': 'left',
					'pad': {'r': 10, 't': 10},
					'x': 0.1,
					'xanchor': 'right',
					'y': 0,
					'yanchor': 'top'
				}],
				sliders=[{
					'active': 0,
					'steps': [
						{
							'method': 'animate',
							'args': [[f'n={n_size}'], {
								'frame': {'duration': 300, 'redraw': True},
								'mode': 'immediate',
								'transition': {'duration': 300}
							}],
							'label': str(n_size)
						} for n_size in sample_sizes
					],
					'pad': {'t': 50},
					'currentvalue': {'prefix': 'Sample Size: n = '}
				}]
			)

			# Show animation
			st.plotly_chart(fig_anim, use_container_width=True)

			st.markdown(
				"<div class='highlight'>Observe how the distribution of sample means approaches a normal distribution as the sample size increases, regardless of the shape of the original distribution. This demonstrates the Central Limit Theorem.</div>",
				unsafe_allow_html=True)

		# Applications
		st.markdown("<h4>Applications of the Normal Distribution</h4>", unsafe_allow_html=True)
		st.markdown("""
        The normal distribution is used to model:
        - Measurement errors in physical experiments
        - Heights, weights, and other physical characteristics of populations
        - IQ scores and other standardized test scores
        - Financial returns over long time periods
        - Sampling distributions of means via the Central Limit Theorem
        - Natural phenomena affected by many small, independent random effects
        """)

	elif continuous_dist == "Uniform (Continuous)":
		st.markdown("<h3 class='section-header'>Continuous Uniform Distribution</h3>", unsafe_allow_html=True)

		# Parameters
		col1, col2 = st.columns(2)
		with col1:
			a = st.slider("Lower bound (a):", -10.0, 10.0, 0.0, 0.1)
		with col2:
			b = st.slider("Upper bound (b):", -10.0, 10.0, 1.0, 0.1)

		# Ensure a <= b
		if a > b:
			a, b = b, a

		# Description
		st.markdown("""
        The continuous uniform distribution assigns equal probability to all values in an interval [a, b]. 
        It models situations where any value in a range is equally likely to occur.
        """)

		# Properties
		st.markdown("<h4>Properties</h4>", unsafe_allow_html=True)
		# Assuming col1 and col2 are defined as columns
		col1, col2 = st.columns(2)

		with col1:
			# PDF
			st.markdown(
				"<span class='property-name'>PDF:</span> " + latex(
					'f(x) = \\frac{1}{b-a} \\text{ for } a \\leq x \\leq b'),
				unsafe_allow_html=True
			)
			# CDF
			st.markdown(
				"<span class='property-name'>CDF:</span> " + latex(
					'F(x) = \\begin{cases} 0 & x < a \\\\ \\frac{x-a}{b-a} & a \\leq x \\leq b \\\\ 1 & x > b \\end{cases}'),
				unsafe_allow_html=True
			)
			# Mean
			st.markdown(
				"<span class='property-name'>Mean:</span> " + latex('E[X] = \\frac{a+b}{2}'),
				unsafe_allow_html=True
			)

		with col2:
			# Variance
			st.markdown(
				"<span class='property-name'>Variance:</span> " + latex('Var[X] = \\frac{(b-a)^2}{12}'),
				unsafe_allow_html=True
			)
			# Skewness
			st.markdown(
				"<span class='property-name'>Skewness:</span> " + latex('0'),
				unsafe_allow_html=True
			)
			# Kurtosis
			st.markdown(
				"<span class='property-name'>Kurtosis:</span> " + latex('\\frac{9}{5}'),
				unsafe_allow_html=True
			)
		# Visualization
		x = np.linspace(a - (b - a) * 0.1, b + (b - a) * 0.1, 1000)
		pdf = np.zeros_like(x)
		pdf[(x >= a) & (x <= b)] = 1 / (b - a)

		cdf = np.zeros_like(x)
		cdf[x < a] = 0
		cdf[(x >= a) & (x <= b)] = (x[(x >= a) & (x <= b)] - a) / (b - a)
		cdf[x > b] = 1

		fig = make_subplots(rows=2, cols=1,
							subplot_titles=("Probability Density Function", "Cumulative Distribution Function"))

		fig_prob.add_trace(go.Scatter(
			x=x_fill,
			y=y_fill,
			fill='tozeroy',
			mode='none',
			name=f'P(X ≥ {x_val:.2f})',
			fillcolor='rgba(65, 105, 225, 0.5)'
		))

		# Add vertical line at x_val if within range
		if a <= x_val <= b:
			fig_prob.add_shape(
				type="line",
				x0=x_val, y0=0,
				x1=x_val, y1=1 / (b - a),
				line=dict(color="red", width=2, dash="dash")
			)

		# Add annotation for the probability
		fig_prob.add_annotation(
			x=(max(x_val, a) + b) / 2, y=1 / (b - a) / 2,
			text=f"P(X ≥ {x_val:.2f}) = {prob:.4f}",
			showarrow=True,
			arrowhead=1
		)

		# Update layout
		fig_prob.update_layout(
			title=f"Probability: P(X ≥ {x_val:.2f})",
			xaxis=dict(
				title="x",
				range=[a - (b - a) * 0.1, b + (b - a) * 0.1]
			),
			yaxis=dict(
				title="Density",
				range=[0, max(pdf) * 1.2]
			),
			height=400
		)

		st.plotly_chart(fig_prob, use_container_width=True)

                   calculation_type = st.selectbox("Choose calculation type:", ["P(X ≤ x)", "P(x₁ ≤ X ≤ x₂)"])

# Define sliders using a and b
x_val1 = st.slider("x₁ value:", float(a - (b - a) * 0.5), float(b + (b - a) * 0.5), float(a), 0.01)
x_val2 = st.slider("x₂ value:", float(a - (b - a) * 0.5), float(b + (b - a) * 0.5), float(b), 0.01)

# Example conditional logic
if calculation_type == "P(X ≤ x)":
    st.write(f"Calculating P(X ≤ {x_val1})")
    # Add your probability calculation here
elif calculation_type == "P(x₁ ≤ X ≤ x₂)":
    st.write(f"Calculating P({x_val1} ≤ X ≤ {x_val2})")
		# Calculate probability
		lower_bound = max(a, x_val1)
		upper_bound = min(b, x_val2)

		if upper_bound <= lower_bound:
			prob = 0
		else:
			prob = (upper_bound - lower_bound) / (b - a)

		# Visualization
		fig_prob = go.Figure()

		# Plot PDF
		fig_prob.add_trace(go.Scatter(
			x=x, y=pdf,
			mode='lines',
			line=dict(color='#4CAF50', width=2),
			name='PDF'
		))

		# Add filled area for the probability
		x_fill = x[(x >= lower_bound) & (x <= upper_bound)]
		y_fill = pdf[(x >= lower_bound) & (x <= upper_bound)]

		fig_prob.add_trace(go.Scatter(
			x=x_fill,
			y=y_fill,
			fill='tozeroy',
			mode='none',
			name=f'P({x_val1:.2f} ≤ X ≤ {x_val2:.2f})',
			fillcolor='rgba(65, 105, 225, 0.5)'
		))

		# Add vertical lines at x_val1 and x_val2 if within range
		if a <= x_val1 <= b:
			fig_prob.add_shape(
				type="line",
				x0=x_val1, y0=0,
				x1=x_val1, y1=1 / (b - a),
				line=dict(color="red", width=2, dash="dash")
			)

		if a <= x_val2 <= b:
			fig_prob.add_shape(
				type="line",
				x0=x_val2, y0=0,
				x1=x_val2, y1=1 / (b - a),
				line=dict(color="red", width=2, dash="dash")
			)

		# Add annotation for the probability
		fig_prob.add_annotation(
			x=(lower_bound + upper_bound) / 2, y=1 / (b - a) / 2,
			text=f"P({x_val1:.2f} ≤ X ≤ {x_val2:.2f}) = {prob:.4f}",
			showarrow=True,
			arrowhead=1
		)

		# Update layout
		fig_prob.update_layout(
			title=f"Probability: P({x_val1:.2f} ≤ X ≤ {x_val2:.2f})",
			xaxis=dict(
				title="x",
				range=[a - (b - a) * 0.1, b + (b - a) * 0.1]
			),
			yaxis=dict(
				title="Density",
				range=[0, max(pdf) * 1.2]
			),
			height=400
		)

		st.plotly_chart(fig_prob, use_container_width=True)

	# Normal distribution
	elif selected_distribution == "Normal Distribution":
		st.markdown("""
		        ## Normal Distribution

		        The Normal (or Gaussian) distribution is one of the most important probability distributions 
		        in statistics due to the Central Limit Theorem.

		        ### Formula
		        The probability density function (PDF) of the normal distribution is:

		        $$f(x; \mu, \sigma) = \\frac{1}{\sigma\sqrt{2\pi}} e^{-\\frac{1}{2}(\\frac{x-\mu}{\sigma})^2}$$

		        where:
		        - $\mu$ is the mean (location parameter)
		        - $\sigma$ is the standard deviation (scale parameter)

		        ### Properties
		        - Mean = $\mu$
		        - Median = $\mu$
		        - Mode = $\mu$
		        - Variance = $\sigma^2$
		        - Skewness = 0
		        - Kurtosis = 3
		        - Support: $x \in (-\infty, \infty)$
		        """)

		# Parameters
		col1, col2 = st.columns(2)
		with col1:
			mu = st.slider("Mean (μ):", -10.0, 10.0, 0.0, 0.1)
		with col2:
			sigma = st.slider("Standard Deviation (σ):", 0.1, 5.0, 1.0, 0.1)

		# Generate distribution data
		x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
		pdf = stats.norm.pdf(x, mu, sigma)
		cdf = stats.norm.cdf(x, mu, sigma)

		# Create tabs
		pdf_tab, cdf_tab, calculator_tab = st.tabs(["PDF", "CDF", "Probability Calculator"])

		with pdf_tab:
			# Plot PDF
			fig_pdf = go.Figure()
			fig_pdf.add_trace(go.Scatter(
				x=x, y=pdf,
				mode='lines',
				line=dict(color='#4CAF50', width=3),
				name='PDF'
			))

			# Add vertical lines for mean and standard deviations
			for i in range(-3, 4):
				if i == 0:
					# Mean
					fig_pdf.add_shape(
						type="line",
						x0=mu, y0=0,
						x1=mu, y1=stats.norm.pdf(mu, mu, sigma),
						line=dict(color="red", width=2)
					)
					fig_pdf.add_annotation(
						x=mu, y=stats.norm.pdf(mu, mu, sigma),
						text="μ",
						showarrow=True,
						arrowhead=1
					)
				else:
					# Standard deviations
					sd_x = mu + i * sigma
					fig_pdf.add_shape(
						type="line",
						x0=sd_x, y0=0,
						x1=sd_x, y1=stats.norm.pdf(sd_x, mu, sigma),
						line=dict(color="gray", width=1, dash="dash")
					)
					fig_pdf.add_annotation(
						x=sd_x, y=stats.norm.pdf(sd_x, mu, sigma) / 2,
						text=f"μ{i:+d}σ" if i != 0 else "μ",
						showarrow=False
					)

			# Highlight standard deviation regions with different colors
			colors = ['rgba(173, 216, 230, 0.3)', 'rgba(144, 238, 144, 0.3)', 'rgba(255, 192, 203, 0.3)']
			for i in range(3):
				x_fill = x[(x >= mu - (i + 1) * sigma) & (x <= mu + (i + 1) * sigma)]
				y_fill = pdf[(x >= mu - (i + 1) * sigma) & (x <= mu + (i + 1) * sigma)]

				fig_pdf.add_trace(go.Scatter(
					x=x_fill,
					y=y_fill,
					fill='tozeroy',
					mode='none',
					name=f'μ±{i + 1}σ: {stats.norm.cdf(mu + (i + 1) * sigma, mu, sigma) - stats.norm.cdf(mu - (i + 1) * sigma, mu, sigma):.4f}',
					fillcolor=colors[i]
				))

			fig_pdf.update_layout(
				title="Normal Distribution PDF",
				xaxis_title="x",
				yaxis_title="Probability Density",
				legend_title="Regions",
				height=500
			)

			st.plotly_chart(fig_pdf, use_container_width=True)

			# Add interactive annotations for the 68-95-99.7 rule
			st.markdown("""
		            ### The 68-95-99.7 Rule (Empirical Rule)

		            For the normal distribution:
		            - About 68% of the data falls within 1 standard deviation of the mean
		            - About 95% of the data falls within 2 standard deviations of the mean
		            - About 99.7% of the data falls within 3 standard deviations of the mean
		            """)

			# Create a table with the exact probabilities
			prob_data = {
				"Range": ["μ ± 1σ", "μ ± 2σ", "μ ± 3σ"],
				"Interval": [f"({mu - sigma:.2f}, {mu + sigma:.2f})",
							 f"({mu - 2 * sigma:.2f}, {mu + 2 * sigma:.2f})",
							 f"({mu - 3 * sigma:.2f}, {mu + 3 * sigma:.2f})"],
				"Probability": [
					f"{(stats.norm.cdf(mu + sigma, mu, sigma) - stats.norm.cdf(mu - sigma, mu, sigma)):.4f}",
					f"{(stats.norm.cdf(mu + 2 * sigma, mu, sigma) - stats.norm.cdf(mu - 2 * sigma, mu, sigma)):.4f}",
					f"{(stats.norm.cdf(mu + 3 * sigma, mu, sigma) - stats.norm.cdf(mu - 3 * sigma, mu, sigma)):.4f}"
				]
			}

			df = pd.DataFrame(prob_data)
			st.table(df)

		with cdf_tab:
			# Plot CDF
			fig_cdf = go.Figure()
			fig_cdf.add_trace(go.Scatter(
				x=x, y=cdf,
				mode='lines',
				line=dict(color='#FF5722', width=3),
				name='CDF'
			))

			# Add vertical lines for mean and standard deviations
			for i in range(-3, 4):
				sd_x = mu + i * sigma
				fig_cdf.add_shape(
					type="line",
					x0=sd_x, y0=0,
					x1=sd_x, y1=stats.norm.cdf(sd_x, mu, sigma),
					line=dict(color="gray", width=1, dash="dash")
				)
				fig_cdf.add_annotation(
					x=sd_x, y=stats.norm.cdf(sd_x, mu, sigma) / 2,
					text=f"μ{i:+d}σ" if i != 0 else "μ",
					showarrow=False
				)

			# Add horizontal lines at key probabilities
			probs = [0.025, 0.05, 0.95, 0.975]
			for p in probs:
				x_p = stats.norm.ppf(p, mu, sigma)
				fig_cdf.add_shape(
					type="line",
					x0=x.min(), y0=p,
					x1=x_p, y1=p,
					line=dict(color="blue", width=1, dash="dot")
				)
				fig_cdf.add_shape(
					type="line",
					x0=x_p, y0=0,
					x1=x_p, y1=p,
					line=dict(color="blue", width=1, dash="dot")
				)
				fig_cdf.add_annotation(
					x=x_p, y=p,
					text=f"({x_p:.2f}, {p:.3f})",
					showarrow=True,
					arrowhead=1
				)

			fig_cdf.update_layout(
				title="Normal Distribution CDF",
				xaxis_title="x",
				yaxis_title="Cumulative Probability",
				height=500
			)

			st.plotly_chart(fig_cdf, use_container_width=True)
			with calculator_tab:
				st.markdown("### Probability Calculations for Normal Distribution")
				calculation_type = st.radio(
					"Select calculation type:",
					["P(X ≤ x)", "P(X ≥ x)", "P(x₁ ≤ X ≤ x₂)", "Find x for given probability"]
				)

				if calculation_type == "P(X ≤ x)":
					x_val = st.slider("x value:", float(mu - 4 * sigma), float(mu + 4 * sigma), float(mu), 0.01)

					# Calculate probability
					prob = stats.norm.cdf(x_val, mu, sigma)

					# Visualization
					fig_prob = go.Figure()

					# Plot PDF
					fig_prob.add_trace(go.Scatter(
						x=x, y=pdf,
						mode='lines',
						line=dict(color='#4CAF50', width=2),
						name='PDF'
					))

					# Add filled area for the probability
					x_fill = x[x <= x_val]
					y_fill = pdf[x <= x_val]

					if len(x_fill) > 0:
						fig_prob.add_trace(go.Scatter(
							x=x_fill,
							y=y_fill,
							fill='tozeroy',
							mode='none',
							name=f'P(X ≤ {x_val:.2f})',
							fillcolor='rgba(65, 105, 225, 0.5)'
						))

					# Add vertical line at x_val
					fig_prob.add_shape(
						type="line",
						x0=x_val, y0=0,
						x1=x_val, y1=stats.norm.pdf(x_val, mu, sigma),
						line=dict(color="red", width=2, dash="dash")
					)

					# Add annotation for the probability
					fig_prob.add_annotation(
						x=(mu - 3 * sigma + x_val) / 2,
						y=max(pdf) / 2,
						text=f"P(X ≤ {x_val:.2f}) = {prob:.4f}",
						showarrow=True,
						arrowhead=1
					)

					# Update layout
					fig_prob.update_layout(
						title=f"Probability: P(X ≤ {x_val:.2f})",
						xaxis=dict(
							title="x",
							range=[mu - 4 * sigma, mu + 4 * sigma]
						),
						yaxis=dict(
							title="Density",
							range=[0, max(pdf) * 1.2]
						),
						height=400
					)

					st.plotly_chart(fig_prob, use_container_width=True)

					# Display Z-score
					z_score = (x_val - mu) / sigma
					st.info(f"Z-score: z = (x - μ)/σ = ({x_val:.2f} - {mu:.2f})/{sigma:.2f} = {z_score:.4f}")

				elif calculation_type == "P(X ≥ x)":
					x_val = st.slider("x value:", float(mu - 4 * sigma), float(mu + 4 * sigma), float(mu), 0.01)

					# Calculate probability
					prob = 1 - stats.norm.cdf(x_val, mu, sigma)

					# Visualization
					fig_prob = go.Figure()

					# Plot PDF
					fig_prob.add_trace(go.Scatter(
						x=x, y=pdf,
						mode='lines',
						line=dict(color='#4CAF50', width=2),
						name='PDF'
					))

					# Add filled area for the probability
					x_fill = x[x >= x_val]
					y_fill = pdf[x >= x_val]

					if len(x_fill) > 0:
						fig_prob.add_trace(go.Scatter(
							x=x_fill,
							y=y_fill,
							fill='tozeroy',
							mode='none',
							name=f'P(X ≥ {x_val:.2f})',
							fillcolor='rgba(65, 105, 225, 0.5)'
						))

					# Add vertical line at x_val
					fig_prob.add_shape(
						type="line",
						x0=x_val, y0=0,
						x1=x_val, y1=stats.norm.pdf(x_val, mu, sigma),
						line=dict(color="red", width=2, dash="dash")
					)

					# Add annotation for the probability
					fig_prob.add_annotation(
						x=(x_val + mu + 3 * sigma) / 2,
						y=max(pdf) / 2,
						text=f"P(X ≥ {x_val:.2f}) = {prob:.4f}",
						showarrow=True,
						arrowhead=1
					)

					# Update layout
					fig_prob.update_layout(
						title=f"Probability: P(X ≥ {x_val:.2f})",
						xaxis=dict(
							title="x",
							range=[mu - 4 * sigma, mu + 4 * sigma]
						),
						yaxis=dict(
							title="Density",
							range=[0, max(pdf) * 1.2]
						),
						height=400
					)

					st.plotly_chart(fig_prob, use_container_width=True)

					# Display Z-score
					z_score = (x_val - mu) / sigma
					st.info(f"Z-score: z = (x - μ)/σ = ({x_val:.2f} - {mu:.2f})/{sigma:.2f} = {z_score:.4f}")

				elif calculation_type == "P(x₁ ≤ X ≤ x₂)":
					col1, col2 = st.columns(2)
					with col1:
						x_val1 = st.slider("x₁ value:", float(mu - 4 * sigma), float(mu + 4 * sigma), float(mu - sigma),
										   0.01)
					with col2:
						x_val2 = st.slider("x₂ value:", float(mu - 4 * sigma), float(mu + 4 * sigma), float(mu + sigma),
										   0.01)

					# Ensure x_val1 <= x_val2
					if x_val1 > x_val2:
						x_val1, x_val2 = x_val2, x_val1

					# Calculate probability
					prob = stats.norm.cdf(x_val2, mu, sigma) - stats.norm.cdf(x_val1, mu, sigma)

					# Visualization
					fig_prob = go.Figure()

					# Plot PDF
					fig_prob.add_trace(go.Scatter(
						x=x, y=pdf,
						mode='lines',
						line=dict(color='#4CAF50', width=2),
						name='PDF'
					))

					# Add filled area for the probability
					x_fill = x[(x >= x_val1) & (x <= x_val2)]
					y_fill = pdf[(x >= x_val1) & (x <= x_val2)]

					if len(x_fill) > 0:
						fig_prob.add_trace(go.Scatter(
							x=x_fill,
							y=y_fill,
							fill='tozeroy',
							mode='none',
							name=f'P({x_val1:.2f} ≤ X ≤ {x_val2:.2f})',
							fillcolor='rgba(65, 105, 225, 0.5)'
						))

					# Add vertical lines at x_val1 and x_val2
					fig_prob.add_shape(
						type="line",
						x0=x_val1, y0=0,
						x1=x_val1, y1=stats.norm.pdf(x_val1, mu, sigma),
						line=dict(color="red", width=2, dash="dash")
					)

					fig_prob.add_shape(
						type="line",
						x0=x_val2, y0=0,
						x1=x_val2, y1=stats.norm.pdf(x_val2, mu, sigma),
						line=dict(color="red", width=2, dash="dash")
					)

					# Add annotation for the probability
					fig_prob.add_annotation(
						x=(x_val1 + x_val2) / 2,
						y=max(pdf) / 2,
						text=f"P({x_val1:.2f} ≤ X ≤ {x_val2:.2f}) = {prob:.4f}",
						showarrow=True,
						arrowhead=1
					)

					# Update layout
					fig_prob.update_layout(
						title=f"Probability: P({x_val1:.2f} ≤ X ≤ {x_val2:.2f})",
						xaxis=dict(
							title="x",
							range=[mu - 4 * sigma, mu + 4 * sigma]
						),
						yaxis=dict(
							title="Density",
							range=[0, max(pdf) * 1.2]
						),
						height=400
					)

					st.plotly_chart(fig_prob, use_container_width=True)

					# Display Z-scores
					z_score1 = (x_val1 - mu) / sigma
					z_score2 = (x_val2 - mu) / sigma
					st.info(f"""
			                Z-scores:
			                - z₁ = (x₁ - μ)/σ = ({x_val1:.2f} - {mu:.2f})/{sigma:.2f} = {z_score1:.4f}
			                - z₂ = (x₂ - μ)/σ = ({x_val2:.2f} - {mu:.2f})/{sigma:.2f} = {z_score2:.4f}
			                """)

				elif calculation_type == "Find x for given probability":
					prob_type = st.radio(
						"Select probability type:",
						["P(X ≤ x) = p", "P(X ≥ x) = p"]
					)

					p_val = st.slider("Probability (p):", 0.01, 0.99, 0.95, 0.01)

					if prob_type == "P(X ≤ x) = p":
						x_val = stats.norm.ppf(p_val, mu, sigma)
						title = f"Finding x where P(X ≤ x) = {p_val:.2f}"
					else:  # "P(X ≥ x) = p"
						x_val = stats.norm.ppf(1 - p_val, mu, sigma)
						title = f"Finding x where P(X ≥ x) = {p_val:.2f}"

					# Visualization
					fig_prob = go.Figure()

					# Plot PDF
					fig_prob.add_trace(go.Scatter(
						x=x, y=pdf,
						mode='lines',
						line=dict(color='#4CAF50', width=2),
						name='PDF'
					))

					# Add filled area for the probability
					if prob_type == "P(X ≤ x) = p":
						x_fill = x[x <= x_val]
						y_fill = pdf[x <= x_val]
					else:  # "P(X ≥ x) = p"
						x_fill = x[x >= x_val]
						y_fill = pdf[x >= x_val]

					if len(x_fill) > 0:
						fig_prob.add_trace(go.Scatter(
							x=x_fill,
							y=y_fill,
							fill='tozeroy',
							mode='none',
							name=f'Probability = {p_val:.2f}',
							fillcolor='rgba(65, 105, 225, 0.5)'
						))

					# Add vertical line at x_val
					fig_prob.add_shape(
						type="line",
						x0=x_val, y0=0,
						x1=x_val, y1=stats.norm.pdf(x_val, mu, sigma),
						line=dict(color="red", width=2, dash="dash")
					)

					# Add annotation for the x value
					fig_prob.add_annotation(
						x=x_val,
						y=stats.norm.pdf(x_val, mu, sigma) / 2,
						text=f"x = {x_val:.4f}",
						showarrow=True,
						arrowhead=1
					)

					# Update layout
					fig_prob.update_layout(
						title=title,
						xaxis=dict(
							title="x",
							range=[mu - 4 * sigma, mu + 4 * sigma]
						),
						yaxis=dict(
							title="Density",
							range=[0, max(pdf) * 1.2]
						),
						height=400
					)

					st.plotly_chart(fig_prob, use_container_width=True)

					# Display Z-score
					z_score = (x_val - mu) / sigma
					st.info(f"""
			                Result: x = {x_val:.4f}

			                Z-score: z = {z_score:.4f}

			                This corresponds to the {p_val * 100:.1f}th percentile of the distribution.
			                """)

			# Binomial distribution
	elif selected_distribution == "Binomial Distribution":
			st.markdown("""
			        ## Binomial Distribution

			        The Binomial distribution models the number of successes in a fixed number of independent 
			        Bernoulli trials, each with the same probability of success.

			        ### Formula
			        The probability mass function (PMF) of the binomial distribution is:

			        $$P(X = k) = {n \\choose k} p^k (1-p)^{n-k}$$

			        where:
			        - $n$ is the number of trials
			        - $p$ is the probability of success in a single trial
			        - $k$ is the number of successes
			        - ${n \\choose k} = \\frac{n!}{k!(n-k)!}$ is the binomial coefficient

			        ### Properties
			        - Mean = $np$
			        - Variance = $np(1-p)$
			        - Skewness = $\\frac{1-2p}{\\sqrt{np(1-p)}}$
			        - Support: $k \in \{0, 1, 2, ..., n\}$
			        """)


			# Parameters
			col1, col2 = st.columns(2)
			with col1:
				n = st.slider("Number of trials (n):", 1, 100, 10)
			with col2:
				p = st.slider("Success probability (p):", 0.01, 0.99, 0.5, 0.01)

			# Generate distribution data
			x = np.arange(0, n + 1)
			pmf = stats.binom.pmf(x, n, p)
			cdf = stats.binom.cdf(x, n, p)

			# Calculate mean and variance
			mean = n * p
			variance = n * p * (1 - p)

			# Create tabs
			pmf_tab, cdf_tab, calculator_tab, animation_tab = st.tabs(
				["PMF", "CDF", "Probability Calculator", "Animation"])

			with pmf_tab:
				# Plot PMF
				fig_pmf = go.Figure()
				fig_pmf.add_trace(go.Bar(
					x=x, y=pmf,
					marker_color='#3D9970',
					name='PMF'
				))

				# Add vertical line for mean
				fig_pmf.add_shape(
					type="line",
					x0=mean, y0=0,
					x1=mean, y1=max(pmf),
					line=dict(color="red", width=2, dash="dash")
				)

				fig_pmf.add_annotation(
					x=mean, y=max(pmf) / 2,
					text=f"Mean = {mean:.2f}",
					showarrow=True,
					arrowhead=1
				)

				fig_pmf.update_layout(
					title="Binomial Distribution PMF",
					xaxis_title="Number of Successes (k)",
					yaxis_title="Probability",
					xaxis=dict(tickmode='linear'),
					height=500
				)

				st.plotly_chart(fig_pmf, use_container_width=True)

				# Display distribution properties
				st.markdown(f"""
			            ### Properties of this Binomial Distribution

			            - **Mean (μ)**: np = {n} × {p} = {mean:.4f}
			            - **Variance (σ²)**: np(1-p) = {n} × {p} × {1 - p} = {variance:.4f}
			            - **Standard Deviation (σ)**: √(np(1-p)) = {np.sqrt(variance):.4f}
			            - **Mode**: ⌊(n+1)p⌋ or ⌈(n+1)p-1⌉ = {int(np.floor((n + 1) * p))} or {int(np.ceil((n + 1) * p - 1))}
			            """)

			with cdf_tab:
				# Plot CDF
				fig_cdf = go.Figure()
				fig_cdf.add_trace(go.Scatter(
					x=x, y=cdf,
					mode='lines+markers',
					line=dict(color='#FF5722', width=3),
					marker=dict(size=8),
					name='CDF'
				))

				# Add vertical line for mean
				fig_cdf.add_shape(
					type="line",
					x0=mean, y0=0,
					x1=mean, y1=stats.binom.cdf(int(mean), n, p),
					line=dict(color="red", width=2, dash="dash")
				)

				fig_cdf.update_layout(
					title="Binomial Distribution CDF",
					xaxis_title="Number of Successes (k)",
					yaxis_title="Cumulative Probability",
					xaxis=dict(tickmode='linear'),
					height=500
				)

				st.plotly_chart(fig_cdf, use_container_width=True)

			with calculator_tab:
				st.markdown("### Probability Calculations for Binomial Distribution")
				calculation_type = st.radio(
					"Select calculation type:",
					["P(X = k)", "P(X ≤ k)", "P(X ≥ k)", "P(k₁ ≤ X ≤ k₂)"]
				)

				if calculation_type == "P(X = k)":
					k_val = st.slider("k value:", 0, n, int(mean))

					# Calculate probability
					prob = stats.binom.pmf(k_val, n, p)

					# Visualization
					fig_prob = go.Figure()

					# Plot PMF
					fig_prob.add_trace(go.Bar(
						x=x, y=pmf,
						marker_color=['rgba(61, 153, 112, 0.3)' if i != k_val else 'rgba(61, 153, 112, 1.0)' for i in
									  x],
						name='PMF'
					))

					# Add annotation for the probability
					fig_prob.add_annotation(
						x=k_val, y=pmf[k_val],
						text=f"P(X = {k_val}) = {prob:.4f}",
						showarrow=True,
						arrowhead=1
					)

					# Update layout
					fig_prob.update_layout(
						title=f"Probability: P(X = {k_val})",
						xaxis_title="Number of Successes (k)",
						yaxis_title="Probability",
						xaxis=dict(tickmode='linear'),
						height=400
					)

					st.plotly_chart(fig_prob, use_container_width=True)

					# Display formula calculation
					binom_coef = comb(n, k_val)
					st.info(f"""
			                Calculation:
			                P(X = {k_val}) = ${{n \\choose k}}$ p^k (1-p)^(n-k)
			                            = ${{{n} \\choose {k_val}}}$ × {p:.4f}^{k_val} × {1 - p:.4f}^{n - k_val}
			                            = {binom_coef} × {p ** k_val:.4f} × {(1 - p) ** (n - k_val):.4f}
			                            = {prob:.6f}
			                """)

				elif calculation_type == "P(X ≤ k)":
					k_val = st.slider("k value:", 0, n, int(mean))

					# Calculate probability
					prob = stats.binom.cdf(k_val, n, p)

					# Visualization
					fig_prob = go.Figure()

					# Plot PMF with highlighted area
					fig_prob.add_trace(go.Bar(
						x=x, y=pmf,
						marker_color=['rgba(61, 153, 112, 1.0)' if i <= k_val else 'rgba(61, 153, 112, 0.3)' for i in
									  x],
						name='PMF'
					))

					# Add annotation for the probability
					fig_prob.add_annotation(
						x=k_val / 2, y=max(pmf) / 2,
						text=f"P(X ≤ {k_val}) = {prob:.4f}",
						showarrow=True,
						arrowhead=1
					)

					# Update layout
					fig_prob.update_layout(
						title=f"Probability: P(X ≤ {k_val})",
						xaxis_title="Number of Successes (k)",
						yaxis_title="Probability",
						xaxis=dict(tickmode='linear'),
						height=400
					)

					st.plotly_chart(fig_prob, use_container_width=True)

					# Display formula calculation
					st.info(f"""
			                P(X ≤ {k_val}) = P(X = 0) + P(X = 1) + ... + P(X = {k_val})
			                            = {prob:.6f}
			                """)

				elif calculation_type == "P(X ≥ k)":
					k_val = st.slider("k value:", 0, n, int(mean))

					# Calculate probability
					prob = 1 - stats.binom.cdf(k_val - 1, n, p) if k_val > 0 else 1.0

					# Visualization
					fig_prob = go.Figure()

					# Plot PMF with highlighted area
					fig_prob.add_trace(go.Bar(
						x=x, y=pmf,
						marker_color=['rgba(61, 153, 112, 0.3)' if i < k_val else 'rgba(61, 153, 112, 1.0)' for i in x],
						name='PMF'
					))

					# Add annotation for the probability
					fig_prob.add_annotation(
						x=(k_val + n) / 2, y=max(pmf) / 2,
						text=f"P(X ≥ {k_val}) = {prob:.4f}",
						showarrow=True,
						arrowhead=1
					)

					# Update layout
					fig_prob.update_layout(
						title=f"Probability: P(X ≥ {k_val})",
						xaxis_title="Number of Successes (k)",
						yaxis_title="Probability",
						xaxis=dict(tickmode='linear'),
						height=400
					)

					st.plotly_chart(fig_prob, use_container_width=True)

					# Display formula calculation
					st.info(f"""
			                P(X ≥ {k_val}) = P(X = {k_val}) + P(X = {k_val + 1}) + ... + P(X = {n})
			                            = 1 - P(X ≤ {k_val - 1})
			                            = {prob:.6f}
			                """)

				elif calculation_type == "P(k₁ ≤ X ≤ k₂)":
					col1, col2 = st.columns(2)
					with col1:
						k_val1 = st.slider("k₁ value:", 0, n, max(0, int(mean) - 2))
					with col2:
						k_val2 = st.slider("k₂ value:", 0, n, min(n, int(mean) + 2))

					# Ensure k_val1 <= k_val2
					if k_val1 > k_val2:
						k_val1, k_val2 = k_val2, k_val1

					# Calculate probability
					prob = stats.binom.cdf(k_val2, n, p) - stats.binom.cdf(k_val1 - 1, n,
																		   p) if k_val1 > 0 else stats.binom.cdf(k_val2,
																												 n, p)

					# Visualization
					fig_prob = go.Figure()

					# Plot PMF with highlighted area
					fig_prob.add_trace(go.Bar(
						x=x, y=pmf,
						marker_color=['rgba(61, 153, 112, 1.0)' if k_val1 <= i <= k_val2 else 'rgba(61, 153, 112, 0.3)'
									  for i in x],
						name='PMF'
					))

					# Add annotation for the probability
					fig_prob.add_annotation(
						x=(k_val1 + k_val2) / 2, y=max(pmf) / 2,
						text=f"P({k_val1} ≤ X ≤ {k_val2}) = {prob:.4f}",
						showarrow=True,
						arrowhead=1
					)

					# Update layout
					fig_prob.update_layout(
						title=f"Probability: P({k_val1} ≤ X ≤ {k_val2})",
						xaxis_title="Number of Successes (k)",
						yaxis_title="Probability",
						xaxis=dict(tickmode='linear'),
						height=400
					)

					st.plotly_chart(fig_prob, use_container_width=True)

					# Display formula calculation
					st.info(f"""
			                P({k_val1} ≤ X ≤ {k_val2}) = P(X = {k_val1}) + P(X = {k_val1 + 1}) + ... + P(X = {k_val2})
			                                     = P(X ≤ {k_val2}) - P(X ≤ {k_val1 - 1})
			                                     = {prob:.6f}
			                """)

			with animation_tab:
				st.markdown("### Animated Binomial Distribution")
				st.markdown("""
			            Watch how the Binomial distribution changes as you vary the parameters.
			            This animation helps visualize the effect of sample size and probability on the shape of the distribution.
			            """)

				col1, col2 = st.columns(2)
				with col1:
					animate_param = st.radio("Parameter to animate:",
											 ["Number of trials (n)", "Success probability (p)"])
				with col2:
					animation_speed = st.slider("Animation Speed:", 1, 10, 5)

				if animate_param == "Number of trials (n)":
					max_n = 50
					fixed_p = p

					# Create animation
					fig_anim = go.Figure()

					# Add trace for PMF
					for i in range(1, max_n + 1):
						x_i = np.arange(0, i + 1)
						pmf_i = stats.binom.pmf(x_i, i, fixed_p)

						# Normalize the heights to make them comparable across different n values
						# pmf_i_normalized = pmf_i / max(pmf_i) if max(pmf_i) > 0 else pmf_i

						fig_anim.add_trace(
							go.Bar(
								x=x_i, y=pmf_i,
								marker_color='#3D9970',
								name=f'n = {i}',
								visible=False
							)
						)

					# Set initial visible trace
					fig_anim.data[n - 1].visible = True

					# Define slider steps
					steps = []
					for i in range(1, max_n + 1):
						step = dict(
							method="update",
							args=[{"visible": [False] * len(fig_anim.data)},
								  {"title": f"Binomial Distribution with n = {i}, p = {fixed_p}"}],
							label=str(i)
						)
						step["args"][0]["visible"][i - 1] = True
						steps.append(step)

					# Create slider
					sliders = [dict(
						active=n - 1,
						currentvalue={"prefix": "Number of trials (n): "},
						pad={"t": 50},
						steps=steps
					)]

					# Update layout
					fig_anim.update_layout(
						title=f"Binomial Distribution with n = {n}, p = {fixed_p}",
						xaxis_title="Number of Successes (k)",
						yaxis_title="Probability",
						sliders=sliders,
						height=600
					)

					st.plotly_chart(fig_anim, use_container_width=True)

					# Text explanation
					st.markdown("""
			                #### Effect of increasing the number of trials (n)

			                As the number of trials increases:

			                1. **Shape change**: The distribution becomes more bell-shaped (approaches a normal distribution)
			                2. **Spread**: The variance increases (σ² = np(1-p))
			                3. **Mean shifts**: The mean increases linearly with n (μ = np)
			                4. **Central Limit Theorem**: For large n, the distribution can be approximated by a normal distribution with mean np and variance np(1-p)
			                """)

				else:  # Animate probability p
					fixed_n = n
					p_values = np.round(np.linspace(0.05, 0.95, 19), 2)

					# Create animation
					fig_anim = go.Figure()

					# Add trace for PMF
					for p_i in p_values:
						x_i = np.arange(0, fixed_n + 1)
						pmf_i = stats.binom.pmf(x_i, fixed_n, p_i)
						mean_i = fixed_n * p_i

						fig_anim.add_trace(
							go.Bar(
								x=x_i, y=pmf_i,
								marker_color='#3D9970',
								name=f'p = {p_i:.2f}',
								visible=False
							)
						)

					# Find closest p in p_values to current p
					p_idx = np.abs(p_values - p).argmin()

					# Set initial visible trace
					fig_anim.data[p_idx].visible = True

					# Define slider steps
					steps = []
					for i, p_i in enumerate(p_values):
						step = dict(
							method="update",
							args=[{"visible": [False] * len(fig_anim.data)},
								  {"title": f"Binomial Distribution with n = {fixed_n}, p = {p_i:.2f}"}],
							label=f"{p_i:.2f}"
						)
						step["args"][0]["visible"][i] = True
						steps.append(step)

					# Create slider
					sliders = [dict(
						active=p_idx,
						currentvalue={"prefix": "Success probability (p): "},
						pad={"t": 50},
						steps=steps
					)]

					# Update layout
					fig_anim.update_layout(
						title=f"Binomial Distribution with n = {fixed_n}, p = {p:.2f}",
						xaxis_title="Number of Successes (k)",
						yaxis_title="Probability",
						sliders=sliders,
						height=600
					)

					st.plotly_chart(fig_anim, use_container_width=True)

					# Text explanation
					st.markdown("""
			                #### Effect of changing the success probability (p)

			                As the success probability changes:

			                1. **Symmetry**: The distribution is symmetric only when p = 0.5
			                2. **Skewness**: 
			                   - When p < 0.5: Right-skewed (positive skew)
			                   - When p > 0.5: Left-skewed (negative skew)
			                3. **Mean shifts**: The mean increases linearly with p (μ = np)
			                4. **Variance**: The variance is maximum when p = 0.5 and decreases as p approaches 0 or 1
			                """)

					# Sidebar navigation
					st.sidebar.title("Navigation")
					page = st.sidebar.radio(
						"Select a Topic",
						["Home", "Relationships Between Distributions", "Limit Theorems & Convergence",
						 "Examples & Practice"]
					)


					# Generate an animated matplotlib figure and return as interactive HTML
					def get_animation_html(fig, anim):
						plt.close(fig)
						file = BytesIO()
						anim.save(file, writer='pillow', fps=10)
						file.seek(0)
						data = base64.b64encode(file.read()).decode('utf-8')
						return f'<img src="data:image/gif;base64,{data}" alt="animation">'


					# Home page
					if page == "Home":
						st.markdown("## Welcome to the Interactive Statistics Guide!")
						st.markdown("""
					    This interactive application is designed to help students understand probability distributions, 
					    their relationships, and convergence theorems with dynamic visualizations and examples.

					    ### What you'll find in this app:

					    - **Distributions**: Interactive visualizations of discrete and continuous probability distributions
					    - **Relationships**: Explore how distributions relate to each other
					    - **Convergence**: Animated demonstrations of the Central Limit Theorem, Law of Large Numbers, and more
					    - **Practice**: Interactive examples to test your understanding

					    Use the sidebar to navigate between different topics.
					    """)

						st.image(
							"https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/1200px-Normal_Distribution_PDF.svg.png")

						st.markdown("""
					    <div class="info-box">
					    <strong>For Teachers:</strong> This tool can be used to dynamically demonstrate key statistical concepts in class.
					    <br><br>
					    <strong>For Students:</strong> Experiment with different parameters to build intuition about statistical distributions and theorems.
					    </div>
					    """, unsafe_allow_html=True)

					# Relationships Between Distributions
					elif page == "Relationships Between Distributions":
						st.markdown("<h2 class='sub-header'>Relationships Between Distributions</h2>",
									unsafe_allow_html=True)

						relationship_type = st.selectbox(
							"Select a relationship to explore:",
							["Bernoulli → Binomial → Normal",
							 "Poisson → Normal",
							 "Exponential → Gamma",
							 "Chi-Square → t → F",
							 "Uniform → Normal (CLT)"]
						)

						if relationship_type == "Bernoulli → Binomial → Normal":
							st.markdown("""
					        <div class="explanation">
					        <h3>Bernoulli → Binomial → Normal</h3>
					        <p>This relationship demonstrates how the sum of independent Bernoulli trials forms a Binomial distribution, 
					        and as the number of trials increases, the Binomial distribution approaches a Normal distribution.</p>
					        </div>
					        """, unsafe_allow_html=True)

							p = st.slider("Probability of success (p):", 0.01, 0.99, 0.5, 0.01)

							col1, col2 = st.columns(2)

							with col1:
								st.markdown("<h4>Bernoulli Distribution</h4>", unsafe_allow_html=True)
								fig, ax = plt.subplots(figsize=(8, 4))

								x = np.array([0, 1])
								ax.bar(x, [1 - p, p], width=0.4, alpha=0.7, color=['skyblue', 'navy'])
								ax.set_xticks([0, 1])
								ax.set_xlabel('Outcome')
								ax.set_ylabel('Probability')
								ax.set_title(f'Bernoulli Distribution (p={p})')

								st.pyplot(fig)

								st.markdown(f"""
					            <div class="formula">
					            P(X = x) = p<sup>x</sup>(1-p)<sup>1-x</sup>, x ∈ {{0, 1}}
					            </div>
					            Mean = {p:.4f} | Variance = {p * (1 - p):.4f}
					            """, unsafe_allow_html=True)

							# Animation of Bernoulli to Binomial
							n_values = [1, 2, 3, 5, 10, 20, 30, 50, 100]

							st.markdown("<h4>Transition from Bernoulli to Binomial to Normal</h4>",
										unsafe_allow_html=True)
							n_select = st.select_slider("Number of trials (n):", options=n_values)

							fig, ax = plt.subplots(figsize=(10, 6))

							# Plot Binomial
							x = np.arange(0, n_select + 1)
							binomial_pmf = stats.binom.pmf(x, n_select, p)
							ax.bar(x, binomial_pmf, alpha=0.7, color='skyblue', label='Binomial PMF')

							# Plot Normal approximation
							if n_select >= 10:
								x_normal = np.linspace(0, n_select, 1000)
								mean = n_select * p
								std = np.sqrt(n_select * p * (1 - p))
								normal_pdf = stats.norm.pdf(x_normal, mean, std)
								ax.plot(x_normal, normal_pdf, 'r-', lw=2, label='Normal Approximation')

							ax.set_xlim(-1, n_select + 1)
							ax.set_xlabel('Number of Successes')
							ax.set_ylabel('Probability')
							ax.set_title(f'Binomial Distribution (n={n_select}, p={p})')
							ax.legend()

							st.pyplot(fig)

							st.markdown(f"""
					        <div class="formula">
					        Binomial PMF: P(X = k) = <sup>n!</sup>⁄<sub>k!(n-k)!</sub> p<sup>k</sup>(1-p)<sup>n-k</sup>
					        </div>
					        <p>As n increases, the Binomial distribution approaches a Normal distribution with:</p>
					        <div class="formula">
					        Mean = np = {n_select * p:.4f} <br>
					        Variance = np(1-p) = {n_select * p * (1 - p):.4f}
					        </div>
					        """, unsafe_allow_html=True)

						elif relationship_type == "Poisson → Normal":
							st.markdown("""
					        <div class="explanation">
					        <h3>Poisson → Normal</h3>
					        <p>As the rate parameter λ of a Poisson distribution increases, 
					        the distribution approaches a Normal distribution with mean and variance equal to λ.</p>
					        </div>
					        """, unsafe_allow_html=True)

							lambda_val = st.slider("Rate parameter (λ):", 1.0, 50.0, 10.0, 1.0)

							# Plot Poisson and its Normal approximation
							fig, ax = plt.subplots(figsize=(10, 6))

							# Poisson PMF
							x = np.arange(0, int(lambda_val * 3))
							poisson_pmf = stats.poisson.pmf(x, lambda_val)
							ax.bar(x, poisson_pmf, alpha=0.7, color='skyblue', label='Poisson PMF')

							# Normal approximation
							x_normal = np.linspace(max(0, lambda_val - 4 * np.sqrt(lambda_val)),
												   lambda_val + 4 * np.sqrt(lambda_val), 1000)
							normal_pdf = stats.norm.pdf(x_normal, lambda_val, np.sqrt(lambda_val))
							ax.plot(x_normal, normal_pdf, 'r-', lw=2, label='Normal Approximation')

							ax.set_xlabel('Number of Events')
							ax.set_ylabel('Probability')
							ax.set_title(f'Poisson Distribution (λ={lambda_val}) and Normal Approximation')
							ax.legend()

							st.pyplot(fig)

							st.markdown(f"""
					        <div class="formula">
					        Poisson PMF: P(X = k) = <sup>λ<sup>k</sup>e<sup>-λ</sup></sup>⁄<sub>k!</sub>
					        </div>
					        <p>For large λ, the Poisson distribution approaches a Normal distribution with:</p>
					        <div class="formula">
					        Mean = λ = {lambda_val} <br>
					        Variance = λ = {lambda_val}
					        </div>
					        """, unsafe_allow_html=True)

						elif relationship_type == "Exponential → Gamma":
							st.markdown("""
					        <div class="explanation">
					        <h3>Exponential → Gamma</h3>
					        <p>The sum of independent exponential random variables leads to a Gamma distribution. 
					        The exponential distribution is a special case of the Gamma distribution with shape parameter α = 1.</p>
					        </div>
					        """, unsafe_allow_html=True)

							rate = st.slider("Rate parameter (λ):", 0.1, 5.0, 1.0, 0.1)
							shape = st.slider("Shape parameter (α) for Gamma:", 1, 10, 1, 1)

							col1, col2 = st.columns(2)

							with col1:
								st.markdown("<h4>Exponential Distribution</h4>", unsafe_allow_html=True)
								fig1, ax1 = plt.subplots(figsize=(8, 4))

								x = np.linspace(0, 10 / rate, 1000)
								exp_pdf = stats.expon.pdf(x, scale=1 / rate)
								ax1.plot(x, exp_pdf, 'b-', lw=2)
								ax1.fill_between(x, exp_pdf, alpha=0.3, color='blue')
								ax1.set_xlabel('x')
								ax1.set_ylabel('Probability Density')
								ax1.set_title(f'Exponential Distribution (λ={rate})')

								st.pyplot(fig1)

								st.markdown(f"""
					            <div class="formula">
					            f(x) = λe<sup>-λx</sup>, x ≥ 0
					            </div>
					            Mean = {1 / rate:.4f} | Variance = {1 / (rate ** 2):.4f}
					            """, unsafe_allow_html=True)

							with col2:
								st.markdown("<h4>Gamma Distribution</h4>", unsafe_allow_html=True)
								fig2, ax2 = plt.subplots(figsize=(8, 4))

								x = np.linspace(0, 15 / rate, 1000)
								gamma_pdf = stats.gamma.pdf(x, a=shape, scale=1 / rate)
								ax2.plot(x, gamma_pdf, 'r-', lw=2)
								ax2.fill_between(x, gamma_pdf, alpha=0.3, color='red')
								ax2.set_xlabel('x')
								ax2.set_ylabel('Probability Density')
								ax2.set_title(f'Gamma Distribution (α={shape}, λ={rate})')

								st.pyplot(fig2)

								st.markdown(f"""
					            <div class="formula">
					            f(x) = <sup>λ<sup>α</sup>x<sup>α-1</sup>e<sup>-λx</sup></sup>⁄<sub>Γ(α)</sub>, x ≥ 0
					            </div>
					            Mean = {shape / rate:.4f} | Variance = {shape / (rate ** 2):.4f}
					            """, unsafe_allow_html=True)

							st.markdown("""
					        <div class="info-box">
					        <strong>Relationship:</strong> If X₁, X₂, ..., Xₙ are independent exponential random variables with rate λ, 
					        then their sum Y = X₁ + X₂ + ... + Xₙ follows a Gamma distribution with shape parameter α = n and rate parameter λ.
					        </div>
					        """, unsafe_allow_html=True)

						elif relationship_type == "Chi-Square → t → F":
							st.markdown("""
					        <div class="explanation">
					        <h3>Chi-Square → t → F</h3>
					        <p>These distributions are related through various transformations:</p>
					        <ul>
					            <li>Chi-Square: Sum of squared standard normal random variables</li>
					            <li>t-distribution: Ratio of standard normal to square root of Chi-Square/df</li>
					            <li>F-distribution: Ratio of two Chi-Square distributions, each divided by their degrees of freedom</li>
					        </ul>
					        </div>
					        """, unsafe_allow_html=True)

							col1, col2 = st.columns(2)

							with col1:
								df1 = st.slider("Degrees of freedom for χ² and t:", 1, 30, 5, 1)

							with col2:
								df2 = st.slider("Second degrees of freedom for F:", 1, 30, 10,
												1) if relationship_type == "Chi-Square → t → F" else None

							fig, axes = plt.subplots(3, 1, figsize=(10, 12))

							# Chi-square
							x_chi = np.linspace(0, 20, 1000)
							chi_pdf = stats.chi2.pdf(x_chi, df1)
							axes[0].plot(x_chi, chi_pdf, 'r-', lw=2)
							axes[0].fill_between(x_chi, chi_pdf, alpha=0.3, color='red')
							axes[0].set_xlabel('x')
							axes[0].set_ylabel('Probability Density')
							axes[0].set_title(f'Chi-Square Distribution (df={df1})')

							# t-distribution
							x_t = np.linspace(-4, 4, 1000)
							t_pdf = stats.t.pdf(x_t, df1)
							axes[1].plot(x_t, t_pdf, 'g-', lw=2)
							axes[1].fill_between(x_t, t_pdf, alpha=0.3, color='green')
							axes[1].set_xlabel('x')
							axes[1].set_ylabel('Probability Density')
							axes[1].set_title(f't-Distribution (df={df1})')

							# F-distribution
							x_f = np.linspace(0, 5, 1000)
							f_pdf = stats.f.pdf(x_f, df1, df2)
							axes[2].plot(x_f, f_pdf, 'b-', lw=2)
							axes[2].fill_between(x_f, f_pdf, alpha=0.3, color='blue')
							axes[2].set_xlabel('x')
							axes[2].set_ylabel('Probability Density')
							axes[2].set_title(f'F-Distribution (df1={df1}, df2={df2})')

							plt.tight_layout()
							st.pyplot(fig)

							st.markdown(f"""
					        <div class="formula">
					        <b>Chi-Square:</b> If Z₁, Z₂, ..., Z<sub>n</sub> are independent standard normal variables, 
					        then X = Z₁² + Z₂² + ... + Z<sub>n</sub>² follows a Chi-Square distribution with n degrees of freedom.
					        <br><br>
					        <b>t-distribution:</b> If Z is a standard normal variable and V is a Chi-Square variable with df degrees of freedom, 
					        then T = <sup>Z</sup>⁄<sub>√(V/df)</sub> follows a t-distribution with df degrees of freedom.
					        <br><br>
					        <b>F-distribution:</b> If U and V are independent Chi-Square variables with df₁ and df₂ degrees of freedom, 
					        then F = <sup>(U/df₁)</sup>⁄<sub>(V/df₂)</sub> follows an F-distribution with df₁ and df₂ degrees of freedom.
					        </div>
					        """, unsafe_allow_html=True)

						elif relationship_type == "Uniform → Normal (CLT)":
							st.markdown("""
					        <div class="explanation">
					        <h3>Uniform → Normal via Central Limit Theorem</h3>
					        <p>This demonstrates the Central Limit Theorem: when we add together a large number of independent random variables 
					        (regardless of their distribution), the sum approaches a normal distribution.</p>
					        </div>
					        """, unsafe_allow_html=True)

							n_samples = st.slider("Number of uniform variables to sum:", 1, 100, 30, 1)
							n_points = 10000

							# Generate sums of different numbers of uniform random variables
							uniform_samples = np.random.uniform(0, 1, size=(n_points, n_samples))
							sums = uniform_samples.sum(axis=1)

							# Normalize to have mean 0 and std 1
							if n_samples > 1:
								sums_normalized = (sums - n_samples / 2) / (np.sqrt(n_samples / 12))
							else:
								sums_normalized = sums

							fig, ax = plt.subplots(figsize=(10, 6))

							# Plot histogram
							ax.hist(sums_normalized, bins=50, density=True, alpha=0.7, color='skyblue')

							# Plot normal PDF for comparison
							x = np.linspace(-4, 4, 1000)
							normal_pdf = stats.norm.pdf(x)
							ax.plot(x, normal_pdf, 'r-', lw=2, label='Standard Normal PDF')

							ax.set_xlabel('Normalized Sum')
							ax.set_ylabel('Density')
							ax.set_title(f'Sum of {n_samples} Uniform Variables (Normalized)')
							ax.legend()

							st.pyplot(fig)

							st.markdown(f"""
					        <div class="formula">
					        <b>Central Limit Theorem:</b> If X₁, X₂, ..., X<sub>n</sub> are independent and identically distributed random variables 
					        with mean μ and variance σ², then as n approaches infinity:
					        <br><br>
					        <sup>X₁ + X₂ + ... + X<sub>n</sub> - nμ</sup>⁄<sub>σ√n</sub> → N(0,1)
					        </div>
					        <br>
					        <div class="info-box">
					        For uniform variables on [0,1]:
					        <ul>
					            <li>Mean (μ) = 0.5</li>
					            <li>Variance (σ²) = 1/12</li>
					        </ul>
					        This is why we normalized by subtracting n/2 and dividing by √(n/12).
					        </div>
					        """, unsafe_allow_html=True)

					# Limit Theorems & Convergence
					elif page == "Limit Theorems & Convergence":
						st.markdown("<h2 class='sub-header'>Limit Theorems & Convergence</h2>", unsafe_allow_html=True)

						theorem = st.selectbox(
							"Select a theorem to explore:",
							["Central Limit Theorem (CLT)",
							 "Law of Large Numbers (LLN)",
							 "Convergence Types",
							 "Delta Method"]
						)

						if theorem == "Central Limit Theorem (CLT)":
							st.markdown("""
					        <div class="explanation">
					        <h3>Central Limit Theorem (CLT)</h3>
					        <p>The Central Limit Theorem states that the sampling distribution of the mean of any independent, 
					        random variable will be normal or nearly normal, if the sample size is large enough.</p>
					        </div>
					        """, unsafe_allow_html=True)

							distribution = st.selectbox(
								"Select a distribution to sample from:",
								["Uniform", "Exponential", "Binomial", "Custom (Bimodal)"]
							)

							sample_size = st.slider("Sample size (n):", 1, 100, 30, 1)
							num_samples = st.slider("Number of samples to draw:", 100, 10000, 1000, 100)

							# Set up the distribution
							if distribution == "Uniform":
								a = st.slider("Lower bound (a):", -10.0, 0.0, 0.0, 0.1)
								b = st.slider("Upper bound (b):", 0.1, 10.0, 1.0, 0.1)
								true_mean = (a + b) / 2
								true_var = (b - a) ** 2 / 12


								# Function to generate samples
								def generate_samples():
									return np.random.uniform(a, b, size=(num_samples, sample_size))

							elif distribution == "Exponential":
								rate = st.slider("Rate parameter (λ):", 0.1, 5.0, 1.0, 0.1)
								true_mean = 1 / rate
								true_var = 1 / (rate ** 2)


								# Function to generate samples
								def generate_samples():
									return np.random.exponential(scale=1 / rate, size=(num_samples, sample_size))

							elif distribution == "Binomial":
								n_trials = st.slider("Number of trials:", 1, 50, 10, 1)
								p_success = st.slider("Probability of success:", 0.0, 1.0, 0.5, 0.01)
								true_mean = n_trials * p_success
								true_var = n_trials * p_success * (1 - p_success)


								# Function to generate samples
								def generate_samples():
									return np.random.binomial(n_trials, p_success, size=(num_samples, sample_size))

							elif distribution == "Custom (Bimodal)":
								st.markdown("""
					            This is a bimodal distribution created by mixing two normal distributions.
					            """)
								mix_prop = st.slider("Mixing proportion:", 0.0, 1.0, 0.5, 0.01)
								mean1 = st.slider("Mean of first component:", -10.0, 0.0, -2.0, 0.1)
								mean2 = st.slider("Mean of second component:", 0.0, 10.0, 2.0, 0.1)
								std1 = st.slider("Std dev of first component:", 0.1, 5.0, 1.0, 0.1)
								std2 = st.slider("Std dev of second component:", 0.1, 5.0, 1.0, 0.1)

								true_mean = mix_prop * mean1 + (1 - mix_prop) * mean2
								true_var = mix_prop * (std1 ** 2 + mean1 ** 2) + (1 - mix_prop) * (
											std2 ** 2 + mean2 ** 2) - true_mean ** 2


								# Function to generate samples
								def generate_samples():
									z = np.random.random(size=(num_samples, sample_size)) < mix_prop
									samples = np.zeros((num_samples, sample_size))
									samples[z] = np.random.normal(mean1, std1, size=np.sum(z))
									samples[~z] = np.random.normal(mean2, std2, size=np.sum(~z))
									return samples

							# Generate samples and compute means
							samples = generate_samples()
							sample_means = np.mean(samples, axis=1)

							# Normalize means for CLT comparison
							normalized_means = (sample_means - true_mean) / (np.sqrt(true_var / sample_size))

							# Create visualization
							fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

							# Plot original distribution
							if distribution == "Uniform":
								x = np.linspace(a - 0.1 * (b - a), b + 0.1 * (b - a), 1000)
								if b > a:  # Avoid division by zero
									pdf = np.ones_like(x) / (b - a)
									pdf[(x < a) | (x > b)] = 0
								else:
									pdf = np.zeros_like(x)

								# Also plot histogram of some sample data
								example_data = np.random.uniform(a, b, 10000)
								ax1.hist(example_data, bins=30, density=True, alpha=0.5, color='skyblue')
								ax1.plot(x, pdf, 'r-', lw=2)

							elif distribution == "Exponential":
								x = np.linspace(0, 5 / rate, 1000)
								pdf = rate * np.exp(-rate * x)

								# Also plot histogram of some sample data
								example_data = np.random.exponential(scale=1 / rate, size=10000)
								ax1.hist(example_data, bins=30, density=True, alpha=0.5, color='skyblue')
								ax1.plot(x, pdf, 'r-', lw=2)

							elif distribution == "Binomial":
								x = np.arange(0, n_trials + 1)
								pmf = stats.binom.pmf(x, n_trials, p_success)

								# Also plot histogram of some sample data
								example_data = np.random.binomial(n_trials, p_success, 10000)
								ax1.hist(example_data, bins=np.arange(0, n_trials + 2) - 0.5, density=True, alpha=0.5,
										 color='skyblue')
								ax1.plot(x, pmf, 'ro', markersize=4)
								ax1.vlines(x, 0, pmf, colors='r', lw=2)

							elif distribution == "Custom (Bimodal)":
								x = np.linspace(min(mean1, mean2) - 3 * max(std1, std2),
												max(mean1, mean2) + 3 * max(std1, std2), 1000)
								pdf1 = stats.norm.pdf(x, mean1, std1)
								pdf2 = stats.norm.pdf(x, mean2, std2)
								pdf = mix_prop * pdf1 + (1 - mix_prop) * pdf2

								# Also plot histogram of some sample data
								z = np.random.random(10000) < mix_prop
								example_data = np.zeros(10000)
								example_data[z] = np.random.normal(mean1, std1, size=np.sum(z))
								example_data[~z] = np.random.normal(mean2, std2, size=np.sum(~z))
								ax1.hist(example_data, bins=30, density=True, alpha=0.5, color='skyblue')
								ax1.plot(x, pdf, 'r-', lw=2)

							ax1.set_xlabel('Value')
							ax1.set_ylabel('Density')
							ax1.set_title(f'Original {distribution} Distribution')

							# Plot sampling distribution of the mean
							ax2.hist(normalized_means, bins=30, density=True, alpha=0.5, color='skyblue')

							# Plot standard normal for comparison
							x = np.linspace(-4, 4, 1000)
							normal_pdf = stats.norm.pdf(x)
							ax2.plot(x, normal_pdf, 'r-', lw=2, label='Standard Normal')

							ax2.set_xlabel('Standardized Sample Mean')
							ax2.set_ylabel('Density')
							ax2.set_title(f'Sampling Distribution of Mean (n={sample_size})')
							ax2.legend()

							plt.tight_layout()
							st.pyplot(fig)

							st.markdown(f"""
					        <div class="formula">
					        <b>Central Limit Theorem:</b> 
					        <br>
					        If X₁, X₂, ..., X<sub>n</sub> are independent and identically distributed with mean μ and variance σ², then:
					        <br><br>
					        Z<sub>n</sub> = <sup>X̄<sub>n</sub> - μ</sup>⁄<sub>σ/√n</sub> → N(0,1) as n → ∞
					        <br><br>
					        where X̄<sub>n</sub> is the sample mean: <sup>1</sup>⁄<sub>n</sub>(X₁ + X₂ + ... + X<sub>n</sub>)
					        </div>

					        <div class="info-box">
					        <p>For the {distribution} distribution:</p>
					        <ul>
					            <li>True mean (μ) = {true_mean:.4f}</li>
					            <li>True variance (σ²) = {true_var:.4f}</li>
					            <li>Standard error of the mean (σ/√n) = {np.sqrt(true_var / sample_size):.4f}</li>
					        </ul>
					        <p>As the sample size increases, the sampling distribution of the mean becomes more normal, regardless of the shape of the original distribution.</p>
					        </div>
					        """, unsafe_allow_html=True)

							# Interactive CLT demonstration with increasing sample size
							st.markdown("<h4>Interactive CLT Animation</h4>", unsafe_allow_html=True)

							animate = st.button("Animate CLT with Increasing Sample Size")

							if animate:
								sample_sizes = [1, 2, 3, 5, 10, 20, 30, 50]

								# Create and display figure
								fig, ax = plt.subplots(figsize=(10, 6))


								def update(frame):
									ax.clear()
									n = sample_sizes[frame]

									# Generate samples and compute means
									samples = generate_samples()[:, :n]  # Use only first n columns
									sample_means = np.mean(samples, axis=1)

									# Normalize means for CLT comparison
									normalized_means = (sample_means - true_mean) / (np.sqrt(true_var / n))

									# Plot histogram
									ax.hist(normalized_means, bins=30, density=True, alpha=0.5, color='skyblue')

									# Plot standard normal for comparison
									x = np.linspace(-4, 4, 1000)
									normal_pdf = stats.norm.pdf(x)
									ax.plot(x, normal_pdf, 'r-', lw=2, label='Standard Normal')

									ax.set_xlim(-4, 4)
									ax.set_ylim(0, 0.7)
									ax.set_xlabel('Standardized Sample Mean')
									ax.set_ylabel('Density')
									ax.set_title(f'Sampling Distribution of Mean (n={n})')
									ax.legend()

									return ax,


								anim = FuncAnimation(fig, update, frames=len(sample_sizes), interval=1000, blit=True)

								# Convert animation to HTML
								animation_html = get_animation_html(fig, anim)
								st.markdown(animation_html, unsafe_allow_html=True)

						elif theorem == "Law of Large Numbers (LLN)":
							st.markdown("""
					        <div class="explanation">
					        <h3>Law of Large Numbers (LLN)</h3>
					        <p>The Law of Large Numbers states that as the sample size increases, the sample mean converges to the true population mean.</p>
					        <p>There are two forms of this law:</p>
					        <ul>
					            <li><b>Weak Law:</b> The probability that the sample mean deviates from the true mean by more than any fixed value approaches zero as the sample size increases.</li>
					            <li><b>Strong Law:</b> The sample mean almost surely converges to the expected value as the sample size increases.</li>
					        </ul>
					        </div>
					        """, unsafe_allow_html=True)

							distribution = st.selectbox(
								"Select a distribution to sample from:",
								["Uniform", "Normal", "Exponential", "Bernoulli"]
							)

							if distribution == "Uniform":
								a = st.slider("Lower bound (a):", -10.0, 0.0, 0.0, 0.1)
								b = st.slider("Upper bound (b):", 0.1, 10.0, 1.0, 0.1)
								true_mean = (a + b) / 2


								# Function to generate samples
								def generate_sample(n):
									return np.random.uniform(a, b, size=n)

							elif distribution == "Normal":
								mean = st.slider("Mean (μ):", -10.0, 10.0, 0.0, 0.1)
								std = st.slider("Standard deviation (σ):", 0.1, 10.0, 1.0, 0.1)
								true_mean = mean


								# Function to generate samples
								def generate_sample(n):
									return np.random.normal(mean, std, size=n)

							elif distribution == "Exponential":
								rate = st.slider("Rate parameter (λ):", 0.1, 5.0, 1.0, 0.1)
								true_mean = 1 / rate


								# Function to generate samples
								def generate_sample(n):
									return np.random.exponential(scale=1 / rate, size=n)

							elif distribution == "Bernoulli":
								p = st.slider("Probability of success (p):", 0.0, 1.0, 0.5, 0.01)
								true_mean = p


								# Function to generate samples
								def generate_sample(n):
									return np.random.binomial(1, p, size=n)

							# Generate animation of LLN
							max_n = st.slider("Maximum sample size:", 100, 10000, 1000, 100)
							num_paths = st.slider("Number of sample paths to show:", 1, 10, 5, 1)

							# Points to plot (logarithmically spaced)
							n_points = 100
							plot_indices = np.unique(np.round(np.logspace(0, np.log10(max_n), n_points)).astype(int))
							plot_indices = plot_indices[plot_indices > 0]  # Remove any zeros

							# Generate multiple sample paths
							fig, ax = plt.subplots(figsize=(12, 6))

							for i in range(num_paths):
								# Generate one large sample
								samples = generate_sample(max_n)

								# Calculate running means
								running_means = np.cumsum(samples) / np.arange(1, max_n + 1)

								# Plot the path
								ax.plot(np.arange(1, max_n + 1), running_means, alpha=0.7, lw=1, label=f'Path {i + 1}')

							# Plot the true mean
							ax.axhline(y=true_mean, color='r', linestyle='-', lw=2, label=f'True Mean: {true_mean:.4f}')

							# Add bounds for weak law visualization
							epsilon = 0.5  # Arbitrary choice for demonstration
							ax.axhline(y=true_mean + epsilon, color='k', linestyle='--', alpha=0.5,
									   label=f'μ ± {epsilon}')
							ax.axhline(y=true_mean - epsilon, color='k', linestyle='--', alpha=0.5)

							ax.set_xscale('log')
							ax.set_xlabel('Sample Size (log scale)')
							ax.set_ylabel('Sample Mean')
							ax.set_title(f'Law of Large Numbers: {distribution} Distribution')
							ax.legend()

							ax.grid(True, which="both", ls="--", alpha=0.3)

							st.pyplot(fig)

							st.markdown(f"""
					        <div class="formula">
					        <b>Weak Law of Large Numbers:</b> 
					        <br>
					        For any ε > 0, P(|X̄<sub>n</sub> - μ| > ε) → 0 as n → ∞
					        <br><br>
					        <b>Strong Law of Large Numbers:</b> 
					        <br>
					        P(lim<sub>n→∞</sub> X̄<sub>n</sub> = μ) = 1
					        </div>

					        <div class="info-box">
					        <p>For the {distribution} distribution:</p>
					        <ul>
					            <li>True mean (μ) = {true_mean:.4f}</li>
					            <li>The sample paths demonstrate how individual realizations of the sample mean converge to the true mean as the sample size increases.</li>
					            <li>The dashed lines show bounds μ ± {epsilon}, illustrating the weak law: the probability of being outside these bounds approaches zero.</li>
					        </ul>
					        </div>
					        """, unsafe_allow_html=True)

							# Animated convergence demonstration
							st.markdown("<h4>Animated Demonstration of LLN</h4>", unsafe_allow_html=True)
							animate_lln = st.button("Show LLN Animation")

							if animate_lln:
								# Generate a large sample upfront
								all_samples = generate_sample(max_n)

								# Create animation
								fig, ax = plt.subplots(figsize=(10, 6))


								def update(frame):
									ax.clear()

									# Current sample size
									n = plot_indices[frame]

									# Calculate means for all paths up to current sample size
									current_samples = all_samples[:n]
									current_mean = np.mean(current_samples)

									# Plot histogram of samples
									ax.hist(current_samples, bins=30, density=True, alpha=0.5, color='skyblue')

									# Add vertical line for current mean
									ax.axvline(x=current_mean, color='b', linestyle='-', lw=2,
											   label=f'Current Mean: {current_mean:.4f}')

									# Add vertical line for true mean
									ax.axvline(x=true_mean, color='r', linestyle='-', lw=2,
											   label=f'True Mean: {true_mean:.4f}')

									ax.set_xlabel('Value')
									ax.set_ylabel('Density')
									ax.set_title(f'Sample (n={n}) and Mean Convergence')
									ax.legend()

									# Set consistent x-axis limits based on the distribution
									if distribution == "Uniform":
										ax.set_xlim(a - 0.1 * (b - a), b + 0.1 * (b - a))
									elif distribution == "Normal":
										ax.set_xlim(mean - 3 * std, mean + 3 * std)
									elif distribution == "Exponential":
										ax.set_xlim(0, 5 / rate)
									elif distribution == "Bernoulli":
										ax.set_xlim(-0.5, 1.5)

									return ax,


								anim = FuncAnimation(fig, update, frames=len(plot_indices), interval=200, blit=True)

								# Convert animation to HTML
								animation_html = get_animation_html(fig, anim)
								st.markdown(animation_html, unsafe_allow_html=True)

						elif theorem == "Convergence Types":
							st.markdown("""
					        <div class="explanation">
					        <h3>Types of Convergence in Probability Theory</h3>
					        <p>There are several types of convergence for sequences of random variables, each with different strengths:</p>
					        <ol>
					            <li><b>Almost Sure Convergence</b>: The strongest form, where random variables converge with probability 1</li>
					            <li><b>Convergence in Probability</b>: Weaker than almost sure, but still implies the variables get "close" with high probability</li>
					            <li><b>Convergence in Distribution</b>: The weakest form, where only the distribution functions converge</li>
					            <li><b>Convergence in L<sup>p</sup></b>: Concerns the convergence of moments, particularly mean-square convergence (p=2)</li>
					        </ol>
					        </div>
					        """, unsafe_allow_html=True)

							convergence_type = st.selectbox(
								"Select a type of convergence to visualize:",
								["Almost Sure vs. In Probability",
								 "Convergence in Distribution",
								 "Mean-Square Convergence (L²)"]
							)

							if convergence_type == "Almost Sure vs. In Probability":
								st.markdown("""
					            <div class="explanation">
					            <h4>Almost Sure vs. Convergence in Probability</h4>
					            <p>
					            Let's visualize the difference between almost sure convergence and convergence in probability
					            with a simple example using a sequence of Bernoulli random variables.
					            </p>
					            </div>
					            """, unsafe_allow_html=True)

								# Parameters
								n_points = 100
								n_paths = 10

								# Example 1: Almost Sure Convergence
								# Idea: X_n = 1 with probability 1/n², 0 otherwise
								# Borel-Cantelli ensures almost sure convergence to 0

								# Example 2: Convergence in Probability but not Almost Sure
								# Idea: Y_n = 1 for indices n*k to n*k+k-1, 0 otherwise
								# where k increases gradually

								fig, axes = plt.subplots(2, 1, figsize=(12, 10))

								# Almost Sure Convergence
								for path in range(n_paths):
									x = np.arange(1, n_points + 1)
									y = np.zeros(n_points)

									# Generate random failures with decreasing probability
									for i in range(n_points):
										if np.random.random() < 1 / (i + 1) ** 2:  # Probability 1/n²
											y[i] = 1

									axes[0].plot(x, y, 'o-', markersize=3, alpha=0.7,
												 label=f'Path {path + 1}' if path == 0 else "")

								axes[0].set_xlabel('n')
								axes[0].set_ylabel('X_n')
								axes[0].set_title('Almost Sure Convergence: X_n → 0')
								axes[0].legend()
								axes[0].set_ylim(-0.1, 1.1)
								axes[0].grid(True, alpha=0.3)

								# Convergence in Probability only
								for path in range(n_paths):
									x = np.arange(1, n_points + 1)
									y = np.zeros(n_points)

									# Create oscillating pattern with increasing periods
									block_sizes = np.ceil(np.sqrt(np.arange(1, n_points + 1))).astype(int)
									cum_blocks = np.cumsum(block_sizes)

									current_pos = 0
									current_block = 0

									for i in range(n_points):
										if current_pos < cum_blocks[current_block]:
											# We are in the current block
											y[i] = 1 if current_block % 2 == 0 else 0
											current_pos += 1
										else:
											# Move to next block
											current_block += 1
											if current_block < len(cum_blocks):
												y[i] = 1 if current_block % 2 == 0 else 0
												current_pos += 1
											else:
												# Out of blocks, stay at 0
												y[i] = 0

									# Add some random noise to make paths distinct
									noise_indices = np.random.choice(n_points, size=int(n_points / 10), replace=False)
									y[noise_indices] = 1 - y[noise_indices]

									axes[1].plot(x, y, 'o-', markersize=3, alpha=0.7,
												 label=f'Path {path + 1}' if path == 0 else "")

								axes[1].set_xlabel('n')
								axes[1].set_ylabel('Y_n')
								axes[1].set_title('Convergence in Probability (but not Almost Sure): Y_n → 0')
								axes[1].legend()
								axes[1].set_ylim(-0.1, 1.1)
								axes[1].grid(True, alpha=0.3)

								plt.tight_layout()
								st.pyplot(fig)

								st.markdown("""
					            <div class="formula">
					            <b>Almost Sure Convergence (X<sub>n</sub> →<sup>a.s.</sup> X):</b> 
					            <br>
					            P(lim<sub>n→∞</sub> X<sub>n</sub> = X) = 1
					            <br><br>
					            <b>Convergence in Probability (X<sub>n</sub> →<sup>p</sup> X):</b> 
					            <br>
					            For any ε > 0, lim<sub>n→∞</sub> P(|X<sub>n</sub> - X| > ε) = 0
					            </div>

					            <div class="info-box">
					            <p><b>Explanation:</b></p>
					            <p>In the top panel, we have a sequence of random variables where each X<sub>n</sub> equals 1 with probability 1/n² and 0 otherwise. 
					            By the Borel-Cantelli lemma, since Σ1/n² < ∞, the probability that X<sub>n</sub> = 1 for infinitely many n is zero. 
					            This means almost every sample path will have only finitely many 1's and then stay at 0 forever—demonstrating almost sure convergence to 0.</p>

					            <p>In the bottom panel, we see variables that oscillate with increasingly long periods. 
					            For any fixed probability threshold, if we go far enough in the sequence, the proportion of 1's decreases, 
					            so Y<sub>n</sub> converges to 0 in probability. However, every sample path keeps oscillating between 0 and 1 infinitely often, 
					            so lim<sub>n→∞</sub> Y<sub>n</sub> doesn't exist for any path—meaning there is no almost sure convergence.</p>
					            </div>
					            """, unsafe_allow_html=True)

							elif convergence_type == "Convergence in Distribution":
								st.markdown("""
					            <div class="explanation">
					            <h4>Convergence in Distribution (Weak Convergence)</h4>
					            <p>
					            Convergence in distribution occurs when the cumulative distribution functions (CDFs) of a sequence of random variables 
					            converge to the CDF of another random variable. This is the weakest form of convergence but is often used in the Central Limit Theorem.
					            </p>
					            </div>
					            """, unsafe_allow_html=True)

								# Let's visualize convergence in distribution with a simple example
								dist_example = st.selectbox(
									"Select an example to visualize:",
									["Binomial → Normal", "Sample Maximum → Extreme Value"]
								)

								if dist_example == "Binomial → Normal":
									p = st.slider("Probability of success (p):", 0.01, 0.99, 0.5, 0.01)

									# Sample sizes to demonstrate convergence
									n_values = [1, 2, 5, 10, 20, 50, 100]
									n_select = st.select_slider("Number of trials (n):", options=n_values)

									# Create visualization
									fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

									# Plot PMF/PDF
									x_range = np.linspace(-4, 4, 1000)  # For standardized values

									# Original distribution parameters
									mean = n_select * p
									std = np.sqrt(n_select * p * (1 - p))

									# Plot Binomial PMF
									x_binomial = np.arange(0, n_select + 1)
									binomial_pmf = stats.binom.pmf(x_binomial, n_select, p)

									# Standardized values
									x_binomial_std = (x_binomial - mean) / std

									ax1.bar(x_binomial, binomial_pmf, alpha=0.7, color='skyblue', label='Binomial PMF')
									ax1.set_xlabel('Number of Successes')
									ax1.set_ylabel('Probability')
									ax1.set_title(f'Binomial Distribution (n={n_select}, p={p})')

									# Plot standardized Binomial PMF and Normal PDF
									ax2.bar(x_binomial_std, binomial_pmf, alpha=0.7, color='skyblue',
											label='Standardized Binomial')

									normal_pdf = stats.norm.pdf(x_range)
									ax2.plot(x_range, normal_pdf, 'r-', lw=2, label='Standard Normal')

									ax2.set_xlabel('Standardized Value')
									ax2.set_ylabel('Density')
									ax2.set_title(f'Convergence in Distribution: Binomial → Normal')
									ax2.legend()

									plt.tight_layout()
									st.pyplot(fig)

									# Plot CDFs to demonstrate convergence in distribution
									fig2, ax = plt.subplots(figsize=(10, 6))

									# Standard normal CDF
									normal_cdf = stats.norm.cdf(x_range)
									ax.plot(x_range, normal_cdf, 'r-', lw=2, label='Standard Normal CDF')

									# Plot standardized Binomial CDF
									for n in n_values:
										mean_n = n * p
										std_n = np.sqrt(n * p * (1 - p))

										x_n = np.arange(0, n + 1)
										x_n_std = (x_n - mean_n) / std_n

										binomial_cdf_n = stats.binom.cdf(x_n, n, p)

										# Create a step function for the CDF
										ax.step(x_n_std, binomial_cdf_n, '-', lw=1.5, alpha=0.7,
												label=f'Binomial CDF (n={n})' if n == n_select else "")

									ax.set_xlabel('Standardized Value')
									ax.set_ylabel('Cumulative Probability')
									ax.set_title('Convergence of Standardized Binomial CDF to Normal CDF')
									ax.legend()
									ax.grid(True, alpha=0.3)

									# Zoom in on a section to see convergence clearly
									ax.set_xlim(-3, 3)
									ax.set_ylim(0, 1)

									st.pyplot(fig2)

									st.markdown("""
					                <div class="formula">
					                <b>Convergence in Distribution (X<sub>n</sub> →<sup>d</sup> X):</b> 
					                <br>
					                lim<sub>n→∞</sub> F<sub>X<sub>n</sub></sub>(t) = F<sub>X</sub>(t)
					                <br>
					                for all t where F<sub>X</sub>(t) is continuous. Here, F denotes the cumulative distribution function.
					                </div>

					                <div class="info-box">
					                <p><b>Example: Binomial → Normal</b></p>
					                <p>The standardized binomial distribution converges in distribution to the standard normal distribution as the number of trials increases. 
					                This means that if we take a binomial random variable, subtract its mean, and divide by its standard deviation, 
					                the resulting distribution approaches the standard normal distribution as n increases.</p>

					                <p>This is a special case of the Central Limit Theorem, since a binomial random variable can be viewed as a sum of n independent 
					                Bernoulli random variables.</p>
					                </div>
					                """, unsafe_allow_html=True)

								elif dist_example == "Sample Maximum → Extreme Value":
									dist_type = st.selectbox(
										"Select a distribution to sample from:",
										["Uniform", "Exponential"]
									)

									sample_sizes = [1, 5, 10, 50, 100, 1000]
									n_select = st.select_slider("Sample size (n):", options=sample_sizes)
									n_samples = 10000  # Number of replications

									if dist_type == "Uniform":
										# For Uniform(0,1), the maximum converges to Gumbel after appropriate scaling
										samples = np.random.uniform(0, 1, size=(n_samples, n_select))
										sample_max = np.max(samples, axis=1)

										# Standardization for uniform distribution
										standardized_max = n_select * (1 - sample_max)  # Converges to Exp(1)

										# Theoretical limit
										x_range = np.linspace(0, 10, 1000)
										limit_pdf = stats.expon.pdf(x_range, scale=1)
										limit_cdf = stats.expon.cdf(x_range, scale=1)
										limit_name = "Exponential(1)"

									elif dist_type == "Exponential":
										# For Exponential(1), the maximum converges to Gumbel after appropriate scaling
										samples = np.random.exponential(1, size=(n_samples, n_select))
										sample_max = np.max(samples, axis=1)

										# Standardization for exponential distribution
										standardized_max = sample_max - np.log(n_select)  # Converges to Gumbel

										# Theoretical limit
										x_range = np.linspace(-5, 5, 1000)
										limit_pdf = stats.gumbel_r.pdf(x_range, loc=0, scale=1)
										limit_cdf = stats.gumbel_r.cdf(x_range, loc=0, scale=1)
										limit_name = "Gumbel"

									# Create visualization
									fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

									# Plot histogram of sample maxima
									ax1.hist(sample_max, bins=30, density=True, alpha=0.7, color='skyblue')
									ax1.set_xlabel('Sample Maximum')
									ax1.set_ylabel('Density')
									ax1.set_title(f'Distribution of Maximum from {dist_type}({n_select} samples)')

									# Plot standardized maxima with limit distribution
									ax2.hist(standardized_max, bins=30, density=True, alpha=0.7, color='skyblue',
											 label=f'Standardized Max ({n_select} samples)')
									ax2.plot(x_range, limit_pdf, 'r-', lw=2, label=f'Limit: {limit_name}')
									ax2.set_xlabel('Standardized Maximum')
									ax2.set_ylabel('Density')
									ax2.set_title(f'Convergence to {limit_name} Distribution')
									ax2.legend()

									plt.tight_layout()
									st.pyplot(fig)

									# Show CDF convergence
									fig2, ax = plt.subplots(figsize=(10, 6))

									# Empirical CDF of standardized maxima
									sorted_data = np.sort(standardized_max)
									empirical_cdf = np.arange(1, n_samples + 1) / n_samples
									ax.step(sorted_data, empirical_cdf, 'b-', lw=1.5, alpha=0.7,
											label=f'Empirical CDF (n={n_select})')

									# Limit CDF
									ax.plot(x_range, limit_cdf, 'r-', lw=2, label=f'Limit CDF: {limit_name}')

									ax.set_xlabel('Standardized Maximum')
									ax.set_ylabel('Cumulative Probability')
									ax.set_title(f'Convergence of CDF to {limit_name}')
									ax.legend()
									ax.grid(True, alpha=0.3)

									st.pyplot(fig2)

									st.markdown(f"""
					                <div class="formula">
					                <b>Extreme Value Theory:</b> 
					                <br>
					                If X₁, X₂, ..., X<sub>n</sub> are i.i.d. random variables and M<sub>n</sub> = max(X₁, ..., X<sub>n</sub>), 
					                then under suitable normalization:
					                <br><br>
					                <sup>M<sub>n</sub> - b<sub>n</sub></sup>⁄<sub>a<sub>n</sub></sub> →<sup>d</sup> G
					                <br><br>
					                where G is one of the three extreme value distributions: Gumbel, Fréchet, or Weibull.
					                </div>

					                <div class="info-box">
					                <p><b>Example: Sample Maximum Distribution</b></p>
					                <p>For the {dist_type} distribution, the maximum of n i.i.d. samples converges in distribution to the {limit_name} distribution 
					                after appropriate standardization.</p>

					                <p>This is an example of the Extremal Types Theorem, which is the extreme value analog of the Central Limit Theorem.</p>
					                </div>
					                """, unsafe_allow_html=True)

							elif convergence_type == "Mean-Square Convergence (L²)":
								st.markdown("""
					            <div class="explanation">
					            <h4>Mean-Square Convergence (L² Convergence)</h4>
					            <p>
					            Mean-square convergence occurs when the expected value of the squared difference between a sequence of random variables 
					            and their limit approaches zero. This is particularly important in statistical estimation, as it relates to consistent estimators.
					            </p>
					            </div>
					            """, unsafe_allow_html=True)

								# Example: Sample mean as a consistent estimator in L² sense
								dist_type = st.selectbox(
									"Select a distribution to sample from:",
									["Normal", "Exponential", "Uniform"]
								)

								max_n = st.slider("Maximum sample size:", 10, 1000, 200, 10)

								# Setup parameters for each distribution
								if dist_type == "Normal":
									mean = st.slider("Mean (μ):", -10.0, 10.0, 0.0, 0.1)
									std = st.slider("Standard deviation (σ):", 0.1, 10.0, 1.0, 0.1)


									def generate_sample(n):
										return np.random.normal(mean, std, size=n)


									true_mean = mean
									true_var = std ** 2

								elif dist_type == "Exponential":
									rate = st.slider("Rate parameter (λ):", 0.1, 5.0, 1.0, 0.1)


									def generate_sample(n):
										return np.random.exponential(scale=1 / rate, size=n)


									true_mean = 1 / rate
									true_var = 1 / (rate ** 2)

								elif dist_type == "Uniform":
									a = st.slider("Lower bound (a):", -10.0, 0.0, 0.0, 0.1)
									b = st.slider("Upper bound (b):", 0.1, 10.0, 1.0, 0.1)


									def generate_sample(n):
										return np.random.uniform(a, b, size=n)


									true_mean = (a + b) / 2
									true_var = (b - a) ** 2 / 12

								# Number of Monte Carlo replications
								n_mc = 1000

								# Sample sizes to examine
								sample_sizes = np.unique(np.round(np.logspace(0, np.log10(max_n), 20)).astype(int))
								sample_sizes = sample_sizes[sample_sizes > 0]  # Remove any zeros

								# For each sample size, compute mean squared error of the sample mean
								mse = np.zeros(len(sample_sizes))

								for i, n in enumerate(sample_sizes):
									# Generate n_mc samples of size n
									samples = np.array([np.mean(generate_sample(n)) for _ in range(n_mc)])

									# Compute mean squared error
									mse[i] = np.mean((samples - true_mean) ** 2)

								# Create visualization
								fig, ax = plt.subplots(figsize=(10, 6))

								ax.loglog(sample_sizes, mse, 'bo-', markersize=5, label='Empirical MSE')

								# Also plot the theoretical MSE = var/n
								theoretical_mse = true_var / sample_sizes
								ax.loglog(sample_sizes, theoretical_mse, 'r--', lw=2, label='Theoretical MSE = σ²/n')

								ax.set_xlabel('Sample Size (log scale)')
								ax.set_ylabel('Mean Squared Error (log scale)')
								ax.set_title(f'Mean-Square Convergence of Sample Mean ({dist_type} Distribution)')
								ax.legend()
								ax.grid(True, which="both", ls="--", alpha=0.3)

								st.pyplot(fig)

								# Show the theoretical convergence rate
								fig2, ax2 = plt.subplots(figsize=(10, 6))

								# Plot n * MSE
								ax2.plot(sample_sizes, sample_sizes * mse, 'bo-', markersize=5,
										 label='n × Empirical MSE')
								ax2.axhline(y=true_var, color='r', linestyle='--', lw=2, label='σ² (Theoretical Limit)')

								ax2.set_xlabel('Sample Size')
								ax2.set_ylabel('n × Mean Squared Error')
								ax2.set_title(f'Convergence Rate Visualization: n × MSE → σ²')
								ax2.legend()
								ax2.grid(True, alpha=0.3)

								st.pyplot(fig2)

								st.markdown(f"""
					            <div class="formula">
					            <b>Mean-Square Convergence (X<sub>n</sub> →<sup>L²</sup> X):</b> 
					            <br>
					            lim<sub>n→∞</sub> E[(X<sub>n</sub> - X)²] = 0
					            </div>

					            <div class="info-box">
					            <p><b>Example: Sample Mean Consistency</b></p>
					            <p>For a random sample X₁, X₂, ..., X<sub>n</sub> from a distribution with mean μ and variance σ², 
					            the sample mean X̄<sub>n</sub> converges to μ in mean-square with:</p>

					            <p>E[(X̄<sub>n</sub> - μ)²] = σ²/n</p>

					            <p>This shows that the mean squared error (MSE) of the sample mean decreases at a rate of 1/n, which implies both:</p>
					            <ul>
					                <li>Consistency: MSE → 0 as n → ∞</li>
					                <li>Convergence rate: n × MSE → σ² as n → ∞</li>
					            </ul>

					            <p>For the {dist_type} distribution with variance {true_var:.4f}, the sample mean converges in mean-square to the true mean {true_mean:.4f}.</p>
					            </div>
					            """, unsafe_allow_html=True)

						elif theorem == "Delta Method":
							st.markdown("""
					        <div class="explanation">
					        <h3>The Delta Method</h3>
					        <p>
					        The Delta method is a technique for deriving the asymptotic distribution of a transformation of an asymptotically normal random variable. 
					        It is based on a first-order Taylor series approximation and is widely used in statistical inference.
					        </p>
					        </div>
					        """, unsafe_allow_html=True)

							function = st.selectbox(
								"Select a function to apply the Delta method:",
								["Square (g(x) = x²)", "Exponential (g(x) = e^x)", "Logistic (g(x) = 1/(1+e^(-x)))"]
							)

							# Parameters for normal distribution
							mean = st.slider("Mean of original normal distribution (μ):", -5.0, 5.0, 0.0, 0.1)
							std = st.slider("Standard deviation (σ):", 0.1, 5.0, 1.0, 0.1)

							# Sample size
							n = st.slider("Sample size (n):", 10, 1000, 100, 10)

							# Number of simulations
							n_simulations = 10000

							# Generate samples from original normal distribution
							original_samples = np.random.normal(mean, std / np.sqrt(n), size=n_simulations)

							# Apply transformation
							if function == "Square (g(x) = x²)":
								g = lambda x: x ** 2
								g_prime = lambda x: 2 * x
								g_name = "Square"
								formula = "g(x) = x²"

								# Delta method prediction
								transformed_mean = g(mean)
								transformed_var = (g_prime(mean) * std / np.sqrt(n)) ** 2

							elif function == "Exponential (g(x) = e^x)":
								g = lambda x: np.exp(x)
								g_prime = lambda x: np.exp(x)
								g_name = "Exponential"
								formula = "g(x) = e^x"

								# Delta method prediction
								transformed_mean = g(mean)
								transformed_var = (g_prime(mean) * std / np.sqrt(n)) ** 2

							elif function == "Logistic (g(x) = 1/(1+e^(-x)))":
								g = lambda x: 1 / (1 + np.exp(-x))
								g_prime = lambda x: np.exp(-x) / ((1 + np.exp(-x)) ** 2)
								g_name = "Logistic"
								formula = "g(x) = 1/(1+e^(-x))"

								# Delta method prediction
								transformed_mean = g(mean)
								transformed_var = (g_prime(mean) * std / np.sqrt(n)) ** 2

							# Apply transformation to samples
							transformed_samples = g(original_samples)

							# Create visualization
							fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

							# Plot original distribution
							ax1.hist(original_samples, bins=30, density=True, alpha=0.7, color='skyblue',
									 label=f'Simulated (n={n})')

							# Plot normal PDF using parameters
							x = np.linspace(mean - 4 * std / np.sqrt(n), mean + 4 * std / np.sqrt(n), 1000)
							normal_pdf = stats.norm.pdf(x, mean, std / np.sqrt(n))
							ax1.plot(x, normal_pdf, 'r-', lw=2, label='Normal Distribution')

							ax1.set_xlabel('Value')
							ax1.set_ylabel('Density')
							ax1.set_title('Original Normal Distribution')
							ax1.legend()

							# Plot transformed distribution
							ax2.hist(transformed_samples, bins=30, density=True, alpha=0.7, color='skyblue',
									 label=f'Transformed Samples')

							# Plot normal PDF using delta method parameters
							x_delta = np.linspace(
								transformed_mean - 4 * np.sqrt(transformed_var),
								transformed_mean + 4 * np.sqrt(transformed_var),
								1000
							)
							delta_pdf = stats.norm.pdf(x_delta, transformed_mean, np.sqrt(transformed_var))
							ax2.plot(x_delta, delta_pdf, 'r-', lw=2, label='Delta Method Approximation')

							ax2.set_xlabel('Value')
							ax2.set_ylabel('Density')
							ax2.set_title(f'Transformed Distribution: {g_name} Function')
							ax2.legend()

							plt.tight_layout()
							st.pyplot(fig)

							# QQ plot to assess normality of transformed distribution
							fig2, ax = plt.subplots(figsize=(10, 6))

							# Standardize the transformed samples
							standardized = (transformed_samples - transformed_mean) / np.sqrt(transformed_var)

							# QQ plot
							stats.probplot(standardized, dist="norm", plot=ax)
							ax.set_title(f'QQ Plot: Transformed {g_name} Distribution vs Standard Normal')

							st.pyplot(fig2)

							st.markdown(f"""
					        <div class="formula">
					        <b>Delta Method:</b> 
					        <br>
					        If √n(X<sub>n</sub> - μ) →<sup>d</sup> N(0, σ²), then for a function g with continuous derivative g',
					        <br><br>
					        √n(g(X<sub>n</sub>) - g(μ)) →<sup>d</sup> N(0, [g'(μ)]²σ²)
					        </div>

					        <div class="info-box">
					        <p><b>Applied to {g_name} function: {formula}</b></p>
					        <p>For the normal distribution with mean μ = {mean} and standard deviation σ/√n = {std / np.sqrt(n):.4f}:</p>

					        <p>The Delta method predicts that g(X<sub>n</sub>) is approximately normally distributed with:</p>
					        <ul>
					            <li>Mean: g(μ) = {transformed_mean:.4f}</li>
					            <li>Variance: [g'(μ)]²(σ²/n) = {transformed_var:.6f}</li>
					        </ul>

					        <p>The QQ plot shows how well the transformed distribution matches the normal approximation given by the Delta method.</p>
					        </div>
					        """, unsafe_allow_html=True)

					# Examples & Practice
					elif page == "Examples & Practice":
						st.markdown("<h2 class='sub-header'>Examples & Practice</h2>", unsafe_allow_html=True)

						example_type = st.selectbox(
							"Select an example or practice type:",
							["Distribution Parameter Estimation",
							 "Hypothesis Testing Simulation",
							 "Interactive CLT Explorer",
							 "Distribution Identification Quiz"]
						)

						if example_type == "Distribution Parameter Estimation":
							st.markdown("""
					        <div class="explanation">
					        <h3>Distribution Parameter Estimation</h3>
					        <p>
					        This interactive example demonstrates how to estimate parameters from a probability distribution 
					        using different methods: Method of Moments, Maximum Likelihood Estimation, and Bayesian Estimation.
					        </p>
					        </div>
					        """, unsafe_allow_html=True)

							est_dist = st.selectbox(
								"Select a distribution:",
								["Normal", "Exponential", "Gamma", "Poisson"]
							)

							col1, col2 = st.columns(2)

							with col1:
								if est_dist == "Normal":
									true_mean = st.slider("True mean (μ):", -10.0, 10.0, 0.0, 0.1)
									true_std = st.slider("True standard deviation (σ):", 0.1, 5.0, 1.0, 0.1)


									def generate_sample(n):
										return np.random.normal(true_mean, true_std, size=n)


									param_names = ["Mean (μ)", "Standard Deviation (σ)"]
									true_params = [true_mean, true_std]

								elif est_dist == "Exponential":
									true_rate = st.slider("True rate parameter (λ):", 0.1, 5.0, 1.0, 0.1)


									def generate_sample(n):
										return np.random.exponential(scale=1 / true_rate, size=n)


									param_names = ["Rate (λ)"]
									true_params = [true_rate]

								elif est_dist == "Gamma":
									true_shape = st.slider("True shape parameter (α):", 0.1, 10.0, 2.0, 0.1)
									true_rate = st.slider("True rate parameter (β):", 0.1, 5.0, 1.0, 0.1)


									def generate_sample(n):
										return np.random.gamma(true_shape, scale=1 / true_rate, size=n)


									param_names = ["Shape (α)", "Rate (β)"]
									true_params = [true_shape, true_rate]

								elif est_dist == "Poisson":
									true_lambda = st.slider("True rate parameter (λ):", 0.1, 20.0, 5.0, 0.1)


									def generate_sample(n):
										return np.random.poisson(true_lambda, size=n)


									param_names = ["Rate (λ)"]
									true_params = [true_lambda]

							with col2:
								sample_size = st.slider("Sample size:", 10, 1000, 100, 10)
								est_method = st.selectbox(
									"Estimation method:",
									["Method of Moments", "Maximum Likelihood", "Compare Methods"]
								)

							# Generate a sample
							np.random.seed(42)  # For reproducibility
							sample = generate_sample(sample_size)


							# Define estimation methods
							def mom_estimation(sample, dist):
								if dist == "Normal":
									return [np.mean(sample), np.std(sample, ddof=0)]
								elif dist == "Exponential":
									return [1 / np.mean(sample)]
								elif dist == "Gamma":
									mean = np.mean(sample)
									var = np.var(sample, ddof=0)
									shape = mean ** 2 / var
									rate = mean / var
									return [shape, rate]
								elif dist == "Poisson":
									return [np.mean(sample)]


							def mle_estimation(sample, dist):
								if dist == "Normal":
									return [np.mean(sample), np.std(sample, ddof=0)]  # Same as MoM for normal
								elif dist == "Exponential":
									return [1 / np.mean(sample)]  # Same as MoM for exponential
								elif dist == "Gamma":
									# This is a simplified approach - full MLE for Gamma requires numerical optimization
									mean = np.mean(sample)
									var = np.var(sample, ddof=0)
									shape = mean ** 2 / var
									rate = mean / var
									return [shape, rate]
								elif dist == "Poisson":
									return [np.mean(sample)]  # Same as MoM for Poisson


							# Estimate parameters
							mom_params = mom_estimation(sample, est_dist)
							mle_params = mle_estimation(sample, est_dist)

							# Plot the results
							fig, ax = plt.subplots(figsize=(10, 6))

							# Plot histogram of sample
							ax.hist(sample, bins=30, density=True, alpha=0.5, color='skyblue', label='Sample Data')

							# Plot the true distribution
							x_min, x_max = np.min(sample), np.max(sample)
							padding = 0.2 * (x_max - x_min)
							x = np.linspace(
								max(0, x_min - padding) if est_dist in ["Exponential", "Gamma"] else x_min - padding,
								x_max + padding, 1000)

							if est_dist == "Normal":
								true_pdf = stats.norm.pdf(x, true_mean, true_std)
								mom_pdf = stats.norm.pdf(x, mom_params[0], mom_params[1])
								mle_pdf = stats.norm.pdf(x, mle_params[0], mle_params[1])
							elif est_dist == "Exponential":
								true_pdf = stats.expon.pdf(x, scale=1 / true_rate)
								mom_pdf = stats.expon.pdf(x, scale=1 / mom_params[0])
								mle_pdf = stats.expon.pdf(x, scale=1 / mle_params[0])
							elif est_dist == "Gamma":
								true_pdf = stats.gamma.pdf(x, true_shape, scale=1 / true_rate)
								mom_pdf = stats.gamma.pdf(x, mom_params[0], scale=1 / mom_params[1])
								mle_pdf = stats.gamma.pdf(x, mle_params[0], scale=1 / mle_params[1])
							elif est_dist == "Poisson":
								x_discrete = np.arange(0, int(x_max) + 2)
								true_pdf = stats.poisson.pmf(x_discrete, true_lambda)
								mom_pdf = stats.poisson.pmf(x_discrete, mom_params[0])
								mle_pdf = stats.poisson.pmf(x_discrete, mle_params[0])

								# For discrete distributions, use stem plots
								if est_dist == "Poisson":
									ax.clear()  # Clear previous plot
									ax.hist(sample, bins=np.arange(0, int(x_max) + 2) - 0.5, density=True,
											alpha=0.5, color='skyblue', label='Sample Data')
									ax.stem(x_discrete, true_pdf, linefmt='r-', markerfmt='ro', basefmt=' ',
											label='True Distribution')

									if est_method == "Method of Moments" or est_method == "Compare Methods":
										ax.stem(x_discrete, mom_pdf, linefmt='g-', markerfmt='go', basefmt=' ',
												label='Method of Moments')

									if est_method == "Maximum Likelihood" or est_method == "Compare Methods":
										ax.stem(x_discrete, mle_pdf, linefmt='b-', markerfmt='bo', basefmt=' ',
												label='Maximum Likelihood')
								else:
									ax.plot(x, true_pdf, 'r-', lw=2, label='True Distribution')

									if est_method == "Method of Moments" or est_method == "Compare Methods":
										ax.plot(x, mom_pdf, 'g-', lw=2, label='Method of Moments')

									if est_method == "Maximum Likelihood" or est_method == "Compare Methods":
										ax.plot(x, mle_pdf, 'b-', lw=2, label='Maximum Likelihood')

							ax.set_xlabel('Value')
							ax.set_ylabel('Density')
							ax.set_title(f'{est_dist} Distribution Parameter Estimation (n={sample_size})')
							ax.legend()

							st.pyplot(fig)

							# Display parameter estimates
							st.markdown("<h4>Parameter Estimates</h4>", unsafe_allow_html=True)

							results = []
							for i, param_name in enumerate(param_names):
								param_row = [param_name, true_params[i]]

								if est_method in ["Method of Moments", "Compare Methods"]:
									param_row.append(mom_params[i] if i < len(mom_params) else "")
								else:
									param_row.append("")

								if est_method in ["Maximum Likelihood", "Compare Methods"]:
									param_row.append(mle_params[i] if i < len(mle_params) else "")
								else:
									param_row.append("")

								results.append(param_row)

							columns = ["Parameter", "True Value"]
							if est_method in ["Method of Moments", "Compare Methods"]:
								columns.append("Method of Moments")
							if est_method in ["Maximum Likelihood", "Compare Methods"]:
								columns.append("Maximum Likelihood")

							df = pd.DataFrame(results, columns=columns)
							st.table(df)

							# Add explanation
							st.markdown("""
					        <div class="info-box">
					        <h4>Estimation Methods</h4>
					        <p><b>Method of Moments (MoM):</b> Equates sample moments with theoretical moments to solve for parameters.</p>
					        <p><b>Maximum Likelihood Estimation (MLE):</b> Finds parameter values that maximize the probability of observing the given sample.</p>

					        <p>For some distributions like Normal, Exponential, and Poisson, MoM and MLE give identical estimates. 
					        For others like Gamma, the estimates may differ, especially with small samples.</p>
					        </div>
					        """, unsafe_allow_html=True)

							# Convergence of estimates with increasing sample size
							st.markdown("<h4>Convergence of Estimates with Sample Size</h4>", unsafe_allow_html=True)

							show_convergence = st.button("Show Convergence Plot")

							if show_convergence:
								# Sample sizes to examine
								n_values = np.logspace(1, 3, 20).astype(int)
								n_replications = 100  # For each sample size

								# Arrays to store parameter estimates
								mom_estimates = np.zeros((len(n_values), len(param_names), n_replications))
								mle_estimates = np.zeros((len(n_values), len(param_names), n_replications))

								# Generate estimates for different sample sizes
								for i, n in enumerate(n_values):
									for j in range(n_replications):
										sample_n = generate_sample(n)
										mom_estimates[i, :, j] = mom_estimation(sample_n, est_dist)[:len(param_names)]
										mle_estimates[i, :, j] = mle_estimation(sample_n, est_dist)[:len(param_names)]

								# Calculate mean and variance of estimates
								mom_means = np.mean(mom_estimates, axis=2)
								mom_vars = np.var(mom_estimates, axis=2)
								mle_means = np.mean(mle_estimates, axis=2)
								mle_vars = np.var(mle_estimates, axis=2)

								# Create convergence plots for each parameter
								fig, axes = plt.subplots(len(param_names), 2, figsize=(15, 5 * len(param_names)))

								# If only one parameter, make axes a 2D array
								if len(param_names) == 1:
									axes = np.array([axes])

								for i, param_name in enumerate(param_names):
									# Mean convergence plot
									axes[i, 0].semilogx(n_values, mom_means[:, i], 'g-', lw=2,
														label='Method of Moments')
									axes[i, 0].semilogx(n_values, mle_means[:, i], 'b-', lw=2,
														label='Maximum Likelihood')
									axes[i, 0].axhline(y=true_params[i], color='r', linestyle='-', lw=2,
													   label='True Value')

									axes[i, 0].set_xlabel('Sample Size (log scale)')
									axes[i, 0].set_ylabel(param_name)
									axes[i, 0].set_title(f'Convergence of {param_name} Estimate')
									axes[i, 0].grid(True, alpha=0.3)
									axes[i, 0].legend()

									# Variance convergence plot
									axes[i, 1].loglog(n_values, mom_vars[:, i], 'g-', lw=2, label='Method of Moments')
									axes[i, 1].loglog(n_values, mle_vars[:, i], 'b-', lw=2, label='Maximum Likelihood')

									# Add a 1/n reference line
									ref_line = 1 / n_values
									ref_line = ref_line * mom_vars[0, i] / ref_line[
										0]  # Scale to match starting variance
									axes[i, 1].loglog(n_values, ref_line, 'k--', lw=1, label='1/n Reference')

									axes[i, 1].set_xlabel('Sample Size (log scale)')
									axes[i, 1].set_ylabel(f'Variance of {param_name} Estimate')
									axes[i, 1].set_title(f'Variance Convergence of {param_name} Estimate')
									axes[i, 1].grid(True, which="both", ls="--", alpha=0.3)
									axes[i, 1].legend()

								plt.tight_layout()
								st.pyplot(fig)

								st.markdown("""
					            <div class="info-box">
					            <h4>Convergence Properties</h4>
					            <p>The plots demonstrate two key properties of parameter estimators:</p>
					            <ol>
					                <li><b>Consistency:</b> As sample size increases, the estimators converge to the true parameter values.</li>
					                <li><b>Efficiency:</b> The variance of the estimators decreases at a rate proportional to 1/n.</li>
					            </ol>
					            <p>MLE estimators are asymptotically efficient, meaning they achieve the lowest possible variance as sample size increases.</p>
					            </div>
					            """, unsafe_allow_html=True)

						elif example_type == "Hypothesis Testing Simulation":
							st.markdown("""
					        <div class="explanation">
					        <h3>Hypothesis Testing Simulation</h3>
					        <p>
					        This interactive simulation demonstrates the concepts of Type I error (falsely rejecting a true null hypothesis) 
					        and Type II error (failing to reject a false null hypothesis) in statistical hypothesis testing.
					        </p>
					        </div>
					        """, unsafe_allow_html=True)

							test_type = st.selectbox(
								"Select a hypothesis test:",
								["One-sample t-test", "Two-sample t-test", "Chi-square test"]
							)

							if test_type == "One-sample t-test":
								st.markdown("""
					            <h4>One-sample t-test</h4>
					            <p>Testing whether the mean of a population is equal to a specified value.</p>
					            <p>H₀: μ = μ₀ vs. H₁: μ ≠ μ₀ (two-sided test)</p>
					            """, unsafe_allow_html=True)

								# Parameters
								true_mean = st.slider("True population mean (μ):", -10.0, 10.0, 0.0, 0.1)
								null_mean = st.slider("Null hypothesis value (μ₀):", -10.0, 10.0, 0.0, 0.1)
								pop_std = st.slider("Population standard deviation (σ):", 0.1, 5.0, 1.0, 0.1)
								sample_size = st.slider("Sample size (n):", 5, 100, 30, 5)
								alpha = st.selectbox("Significance level (α):", [0.01, 0.05, 0.1])

								# Critical value
								critical_value = stats.t.ppf(1 - alpha / 2, df=sample_size - 1)

								# Calculate power
								# Standardized effect size
								effect_size = abs(true_mean - null_mean) / (pop_std / np.sqrt(sample_size))
								# Non-centrality parameter
								ncp = effect_size * np.sqrt(sample_size)
								# Power calculation for two-sided test
								power = 1 - (stats.t.cdf(critical_value, df=sample_size - 1, nc=ncp) -
											 stats.t.cdf(-critical_value, df=sample_size - 1, nc=ncp))

								# Visualize the test
								fig, ax = plt.subplots(figsize=(12, 6))

								# X-axis range
								t_range = np.linspace(-5, 5, 1000)

								# Plot the null distribution
								null_pdf = stats.t.pdf(t_range, df=sample_size - 1)
								ax.plot(t_range, null_pdf, 'b-', lw=2, label='Null Distribution (t with df=n-1)')

								# Shade the rejection regions
								reject_region_right = t_range[t_range >= critical_value]
								reject_region_left = t_range[t_range <= -critical_value]
								ax.fill_between(reject_region_right, 0,
												stats.t.pdf(reject_region_right, df=sample_size - 1),
												color='red', alpha=0.3)
								ax.fill_between(reject_region_left, 0,
												stats.t.pdf(reject_region_left, df=sample_size - 1),
												color='red', alpha=0.3)

								# Add vertical lines for critical values
								ax.axvline(x=critical_value, color='r', linestyle='--',
										   label=f'Critical Value: ±{critical_value:.3f}')
								ax.axvline(x=-critical_value, color='r', linestyle='--')

								# If the null hypothesis is false, plot the alternative distribution
								if true_mean != null_mean:
									# Calculate the non-centrality parameter
									ncp = (true_mean - null_mean) / (pop_std / np.sqrt(sample_size))

									# Plot the alternative distribution
									alt_pdf = stats.t.pdf(t_range, df=sample_size - 1, nc=ncp)
									ax.plot(t_range, alt_pdf, 'g-', lw=2,
											label=f'Alternative Distribution (μ={true_mean})')

									# Shade the Type II error region (failing to reject when H0 is false)
									accept_region = t_range[(t_range > -critical_value) & (t_range < critical_value)]
									ax.fill_between(accept_region, 0,
													stats.t.pdf(accept_region, df=sample_size - 1, nc=ncp),
													color='green', alpha=0.3)

								ax.set_xlabel('t-statistic')
								ax.set_ylabel('Probability Density')
								title = "One-sample t-test"
								if true_mean == null_mean:
									title += f" (H₀ is true: μ = {null_mean})"
								else:
									title += f" (H₀ is false: true μ = {true_mean}, μ₀ = {null_mean})"
								ax.set_title(title)
								ax.legend()

								st.pyplot(fig)

								# Display test information
								st.markdown(f"""
					            <div class="info-box">
					            <h4>Test Information</h4>
					            <p><b>Sample size (n):</b> {sample_size}</p>
					            <p><b>Significance level (α):</b> {alpha}</p>
					            <p><b>Critical value:</b> ±{critical_value:.3f}</p>
					            <p><b>Power of the test:</b> {power:.4f}</p>

					            <h4>Interpretation</h4>
					            <p>The t-test compares the sample mean to the null hypothesis value μ₀ = {null_mean}.</p>
					            <p>The test statistic is t = (x̄ - μ₀)/(s/√n), where x̄ is the sample mean and s is the sample standard deviation.</p>
					            <p>We reject H₀ if |t| > {critical_value:.3f}.</p>

					            <p>The red shaded areas represent the probability of Type I error (α = {alpha}), 
					            which occurs when we reject H₀ even though it is true.</p>

					            {f'<p>The green shaded area represents the probability of Type II error (β = {1 - power:.4f}), ' +
								 f'which occurs when we fail to reject H₀ even though it is false.</p>' if true_mean != null_mean else ''}
					            </div>
					            """, unsafe_allow_html=True)

								# Monte Carlo simulation
								run_simulation = st.button("Run Monte Carlo Simulation")

								if run_simulation:
									n_simulations = 10000
									results = np.zeros(n_simulations)

									for i in range(n_simulations):
										# Generate sample from true distribution
										sample = np.random.normal(true_mean, pop_std, size=sample_size)

										# Calculate t-statistic
										t_stat = (np.mean(sample) - null_mean) / (
													np.std(sample, ddof=1) / np.sqrt(sample_size))

										# Record whether we reject the null hypothesis
										results[i] = 1 if abs(t_stat) > critical_value else 0

									empirical_rejection_rate = np.mean(results)

									# Visualize the results
									fig, ax = plt.subplots(figsize=(8, 5))

									# Create bar chart
									labels = ['Reject H₀', 'Fail to Reject H₀']
									counts = [np.sum(results), n_simulations - np.sum(results)]
									percentages = [count / n_simulations * 100 for count in counts]

									ax.bar(labels, percentages, color=['red', 'green'], alpha=0.7)

									ax.set_ylabel('Percentage')
									ax.set_title(f'Monte Carlo Simulation Results ({n_simulations} trials)')

									for i, v in enumerate(percentages):
										ax.text(i, v + 1, f"{v:.1f}%", ha='center')

									st.pyplot(fig)

									# Interpret the results
									if true_mean == null_mean:
										expected_rate = alpha
										error_type = "Type I error"
									else:
										expected_rate = power
										error_type = "Power"

									st.markdown(f"""
					                <div class="info-box">
					                <h4>Simulation Results</h4>
					                <p><b>Number of simulations:</b> {n_simulations}</p>
					                <p><b>Empirical rejection rate:</b> {empirical_rejection_rate:.4f}</p>
					                <p><b>Theoretical {error_type} rate:</b> {expected_rate:.4f}</p>

					                <h4>Interpretation</h4>
					                <p>The simulation shows the proportion of times we would reject H₀ based on random samples from the true distribution.</p>
					                <p>When H₀ is {'true' if true_mean == null_mean else 'false'}, this represents the {'Type I error rate (α)' if true_mean == null_mean else 'Power of the test (1-β)'}.</p>
					                <p>The empirical rate is close to the theoretical value, demonstrating the properties of the t-test.</p>
					                </div>
					                """, unsafe_allow_html=True)

							elif test_type == "Two-sample t-test":
								st.markdown("""
					            <h4>Two-sample t-test</h4>
					            <p>Testing whether the means of two populations are equal.</p>
					            <p>H₀: μ₁ = μ₂ vs. H₁: μ₁ ≠ μ₂ (two-sided test)</p>
					            """, unsafe_allow_html=True)

								# Parameters
								mean1 = st.slider("Mean of population 1 (μ₁):", -10.0, 10.0, 0.0, 0.1)
								mean2 = st.slider("Mean of population 2 (μ₂):", -10.0, 10.0, 0.0, 0.1)
								std1 = st.slider("Standard deviation of population 1 (σ₁):", 0.1, 5.0, 1.0, 0.1)
								std2 = st.slider("Standard deviation of population 2 (σ₂):", 0.1, 5.0, 1.0, 0.1)
								n1 = st.slider("Sample size from population 1 (n₁):", 5, 100, 30, 5)
								n2 = st.slider("Sample size from population 2 (n₂):", 5, 100, 30, 5)
								alpha = st.selectbox("Significance level (α):", [0.01, 0.05, 0.1])

								# Degrees of freedom (Welch's approximation)
								df_numerator = (std1 ** 2 / n1 + std2 ** 2 / n2) ** 2
								df_denominator = (std1 ** 4 / (n1 ** 2 * (n1 - 1)) + std2 ** 4 / (n2 ** 2 * (n2 - 1)))
								df = df_numerator / df_denominator if df_denominator > 0 else n1 + n2 - 2

								# Critical value
								critical_value = stats.t.ppf(1 - alpha / 2, df=df)

								# Calculate power
								# Standardized effect size (Cohen's d)
								effect_size = abs(mean1 - mean2) / np.sqrt((std1 ** 2 + std2 ** 2) / 2)
								# Standard error
								se = np.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)
								# Non-centrality parameter
								ncp = abs(mean1 - mean2) / se
								# Power calculation
								power = 1 - (stats.t.cdf(critical_value, df=df, nc=ncp) -
											 stats.t.cdf(-critical_value, df=df, nc=ncp))

								# Visualize the test
								fig, ax = plt.subplots(figsize=(12, 6))

								# X-axis range
								t_range = np.linspace(-5, 5, 1000)

								# Plot the null distribution
								null_pdf = stats.t.pdf(t_range, df=df)
								ax.plot(t_range, null_pdf, 'b-', lw=2, label=f'Null Distribution (t with df≈{df:.1f})')

								# Shade the rejection regions
								reject_region_right = t_range[t_range >= critical_value]
								reject_region_left = t_range[t_range <= -critical_value]
								ax.fill_between(reject_region_right, 0, stats.t.pdf(reject_region_right, df=df),
												color='red', alpha=0.3)
								ax.fill_between(reject_region_left, 0, stats.t.pdf(reject_region_left, df=df),
												color='red', alpha=0.3)

								# Add vertical lines for critical values
								ax.axvline(x=critical_value, color='r', linestyle='--',
										   label=f'Critical Value: ±{critical_value:.3f}')
								ax.axvline(x=-critical_value, color='r', linestyle='--')

								# If the null hypothesis is false, plot the alternative distribution
								if mean1 != mean2:
									# Calculate the non-centrality parameter
									ncp = (mean1 - mean2) / np.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)

									# Plot the alternative distribution
									alt_pdf = stats.t.pdf(t_range, df=df, nc=ncp)
									ax.plot(t_range, alt_pdf, 'g-', lw=2,
											label=f'Alternative Distribution (μ₁-μ₂={mean1 - mean2})')

									# Shade the Type II error region (failing to reject when H0 is false)
									accept_region = t_range[(t_range > -critical_value) & (t_range < critical_value)]
									ax.fill_between(accept_region, 0, stats.t.pdf(accept_region, df=df, nc=ncp),
													color='green', alpha=0.3)

								ax.set_xlabel('t-statistic')
								ax.set_ylabel('Probability Density')
								title = "Two-sample t-test"
								if mean1 == mean2:
									title += f" (H₀ is true: μ₁ = μ₂ = {mean1})"
								else:
									title += f" (H₀ is false: μ₁ = {mean1}, μ₂ = {mean2})"
								ax.set_title(title)
								ax.legend()

								st.pyplot(fig)

								# Display test information
								st.markdown(f"""
					            <div class="info-box">
					            <h4>Test Information</h4>
					            <p><b>Sample sizes:</b> n₁ = {n1}, n₂ = {n2}</p>
					            <p><b>Standard deviations:</b> σ₁ = {std1}, σ₂ = {std2}</p>
					            <p><b>Significance level (α):</b> {alpha}</p>
					            <p><b>Degrees of freedom:</b> {df:.2f} (Welch approximation)</p>
					            <p><b>Critical value:</b> ±{critical_value:.3f}</p>
					            <p><b>Effect size (Cohen's d):</b> {effect_size:.4f}</p>
					            <p><b>Power of the test:</b> {power:.4f}</p>

					            <h4>Interpretation</h4>
					            <p>The two-sample t-test compares the means of two independent populations.</p>
					            <p>The test statistic is t = (x̄₁ - x̄₂)/√(s₁²/n₁ + s₂²/n₂), where x̄ᵢ is the sample mean and sᵢ is the sample standard deviation for group i.</p>
					            <p>We reject H₀ if |t| > {critical_value:.3f}.</p>

					            <p>The red shaded areas represent the probability of Type I error (α = {alpha}), 
					            which occurs when we reject H₀ even though it is true.</p>

					            {f'<p>The green shaded area represents the probability of Type II error (β = {1 - power:.4f}), ' +
								 f'which occurs when we fail to reject H₀ even though it is false.</p>' if mean1 != mean2 else ''}
					            </div>
					            """, unsafe_allow_html=True)

								# Monte Carlo simulation
								run_simulation = st.button("Run Monte Carlo Simulation")

								if run_simulation:
									n_simulations = 10000
									results = np.zeros(n_simulations)

									for i in range(n_simulations):
										# Generate samples from both populations
										sample1 = np.random.normal(mean1, std1, size=n1)
										sample2 = np.random.normal(mean2, std2, size=n2)

										# Calculate sample means and standard deviations
										mean_1 = np.mean(sample1)
										mean_2 = np.mean(sample2)
										var_1 = np.var(sample1, ddof=1)
										var_2 = np.var(sample2, ddof=1)

										# Calculate t-statistic
										t_stat = (mean_1 - mean_2) / np.sqrt(var_1 / n1 + var_2 / n2)

										# Calculate degrees of freedom (Welch's approximation)
										df_num = (var_1 / n1 + var_2 / n2) ** 2
										df_den = (var_1 ** 2 / (n1 ** 2 * (n1 - 1)) + var_2 ** 2 / (n2 ** 2 * (n2 - 1)))
										df_sim = df_num / df_den if df_den > 0 else n1 + n2 - 2

										# Get critical value for this specific df
										critical_val_sim = stats.t.ppf(1 - alpha / 2, df=df_sim)

										# Record whether we reject the null hypothesis
										results[i] = 1 if abs(t_stat) > critical_val_sim else 0

									empirical_rejection_rate = np.mean(results)

									# Visualize the results
									fig, ax = plt.subplots(figsize=(8, 5))

									# Create bar chart
									labels = ['Reject H₀', 'Fail to Reject H₀']
									counts = [np.sum(results), n_simulations - np.sum(results)]
									percentages = [count / n_simulations * 100 for count in counts]

									ax.bar(labels, percentages, color=['red', 'green'], alpha=0.7)

									ax.set_ylabel('Percentage')
									ax.set_title(f'Monte Carlo Simulation Results ({n_simulations} trials)')

									for i, v in enumerate(percentages):
										ax.text(i, v + 1, f"{v:.1f}%", ha='center')

									st.pyplot(fig)

									# Interpret the results
									if mean1 == mean2:
										expected_rate = alpha
										error_type = "Type I error"
									else:
										expected_rate = power
										error_type = "Power"

									st.markdown(f"""
					                <div class="info-box">
					                <h4>Simulation Results</h4>
					                <p><b>Number of simulations:</b> {n_simulations}</p>
					                <p><b>Empirical rejection rate:</b> {empirical_rejection_rate:.4f}</p>
					                <p><b>Theoretical {error_type} rate:</b> {expected_rate:.4f}</p>

					                <h4>Interpretation</h4>
					                <p>The simulation shows the proportion of times we would reject H₀ based on random samples from the true distributions.</p>
					                <p>When H₀ is {'true' if mean1 == mean2 else 'false'}, this represents the {'Type I error rate (α)' if mean1 == mean2 else 'Power of the test (1-β)'}.</p>
					                <p>The empirical rate is close to the theoretical value, demonstrating the properties of the two-sample t-test.</p>
					                </div>
					                """, unsafe_allow_html=True)

							elif test_type == "Chi-square test":
								st.markdown("""
					            <h4>Chi-square Goodness-of-Fit Test</h4>
					            <p>Testing whether the frequency distribution of observed data matches a theoretical distribution.</p>
					            <p>H₀: The data follows the specified distribution vs. H₁: The data does not follow the specified distribution</p>
					            """, unsafe_allow_html=True)

								# Parameters for a multinomial distribution
								n_categories = st.slider("Number of categories:", 2, 10, 5, 1)

								# Allow user to specify expected proportions
								st.markdown("#### Specify Expected Proportions")

								col1, col2 = st.columns(2)

								with col1:
									expected_props = []
									for i in range(n_categories):
										prop = st.slider(f"Expected proportion for category {i + 1}:", 0.0, 1.0,
														 1.0 / n_categories, 0.01)
										expected_props.append(prop)

									# Normalize to ensure they sum to 1
									expected_props = np.array(expected_props)
									expected_props = expected_props / expected_props.sum()

								with col2:
									# Let user specify true proportions (for simulation)
									st.markdown("#### Specify True Proportions (for simulation)")

									# Option to use expected as true
									use_expected_as_true = st.checkbox("Same as expected (H₀ is true)", value=True)

									if use_expected_as_true:
										true_props = expected_props
									else:
										true_props = []
										for i in range(n_categories):
											prop = st.slider(f"True proportion for category {i + 1}:", 0.0, 1.0,
															 expected_props[i], 0.01,
															 key=f"true_prop_{i}")
											true_props.append(prop)

										# Normalize to ensure they sum to 1
										true_props = np.array(true_props)
										true_props = true_props / true_props.sum()

								# Sample size
								sample_size = st.slider("Sample size (n):", 10, 1000, 100, 10)

								# Significance level
								alpha = st.selectbox("Significance level (α):", [0.01, 0.05, 0.1])

								# Critical value for chi-square test
								df = n_categories - 1  # degrees of freedom
								critical_value = stats.chi2.ppf(1 - alpha, df)

								# Calculate power (for non-central chi-square)
								if not use_expected_as_true:
									# Calculate non-centrality parameter
									ncp = sample_size * np.sum((true_props - expected_props) ** 2 / expected_props)

									# Calculate power
									power = 1 - stats.ncx2.cdf(critical_value, df, ncp)
								else:
									power = alpha  # When H0 is true, power = Type I error rate

								# Visualize the test
								fig, ax = plt.subplots(figsize=(12, 6))

								# X-axis range
								chi2_range = np.linspace(0, stats.chi2.ppf(0.999, df) * 1.5, 1000)

								# Plot the null distribution
								null_pdf = stats.chi2.pdf(chi2_range, df)
								ax.plot(chi2_range, null_pdf, 'b-', lw=2, label=f'Null Distribution (χ² with df={df})')

								# Shade the rejection region
								reject_region = chi2_range[chi2_range >= critical_value]
								ax.fill_between(reject_region, 0, stats.chi2.pdf(reject_region, df),
												color='red', alpha=0.3)

								# Add vertical line for critical value
								ax.axvline(x=critical_value, color='r', linestyle='--',
										   label=f'Critical Value: {critical_value:.3f}')

								# If H0 is false, plot the non-central chi-square distribution
								if not use_expected_as_true:
									# Calculate non-centrality parameter
									ncp = sample_size * np.sum((true_props - expected_props) ** 2 / expected_props)

									# Plot the non-central chi-square distribution
									noncentral_pdf = stats.ncx2.pdf(chi2_range, df, ncp)
									ax.plot(chi2_range, noncentral_pdf, 'g-', lw=2,
											label=f'Alternative Distribution (non-central χ², ncp={ncp:.2f})')

									# Shade the Type II error region
									accept_region = chi2_range[chi2_range < critical_value]
									ax.fill_between(accept_region, 0, stats.ncx2.pdf(accept_region, df, ncp),
													color='green', alpha=0.3)

								ax.set_xlabel('χ² statistic')
								ax.set_ylabel('Probability Density')
								title = "Chi-square Goodness-of-Fit Test"
								if use_expected_as_true:
									title += " (H₀ is true)"
								else:
									title += " (H₀ is false)"
								ax.set_title(title)
								ax.legend()

								st.pyplot(fig)

								# Compare proportions
								fig2, ax2 = plt.subplots(figsize=(10, 6))

								bar_width = 0.35
								x = np.arange(n_categories)

								ax2.bar(x - bar_width / 2, expected_props, bar_width, label='Expected Proportions',
										color='blue', alpha=0.7)
								ax2.bar(x + bar_width / 2, true_props, bar_width, label='True Proportions',
										color='green', alpha=0.7)

								ax2.set_xlabel('Category')
								ax2.set_ylabel('Proportion')
								ax2.set_title('Expected vs. True Proportions')
								ax2.set_xticks(x)
								ax2.set_xticklabels([f'Cat {i + 1}' for i in range(n_categories)])
								ax2.legend()

								st.pyplot(fig2)

								# Display test information
								st.markdown(f"""
					            <div class="info-box">
					            <h4>Test Information</h4>
					            <p><b>Sample size (n):</b> {sample_size}</p>
					            <p><b>Number of categories:</b> {n_categories}</p>
					            <p><b>Degrees of freedom:</b> {df}</p>
					            <p><b>Significance level (α):</b> {alpha}</p>
					            <p><b>Critical value:</b> {critical_value:.3f}</p>
					            <p><b>Power of the test:</b> {power:.4f}</p>

					            <h4>Interpretation</h4>
					            <p>The Chi-square Goodness-of-Fit test compares observed frequencies with expected frequencies under H₀.</p>
					            <p>The test statistic is χ² = Σ(Oᵢ - Eᵢ)²/Eᵢ, where Oᵢ is the observed count and Eᵢ is the expected count for category i.</p>
					            <p>We reject H₀ if χ² > {critical_value:.3f}.</p>

					            <p>The red shaded area represents the probability of Type I error (α = {alpha}), 
					            which occurs when we reject H₀ even though it is true.</p>

					            {f'<p>The green shaded area represents the probability of Type II error (β = {1 - power:.4f}), ' +
								 f'which occurs when we fail to reject H₀ even though it is false.</p>' if not use_expected_as_true else ''}
					            </div>
					            """, unsafe_allow_html=True)

								# Monte Carlo simulation
								run_simulation = st.button("Run Monte Carlo Simulation")

								if run_simulation:
									n_simulations = 10000
									results = np.zeros(n_simulations)

									for i in range(n_simulations):
										# Generate a multinomial sample using true proportions
										observed = np.random.multinomial(sample_size, true_props)

										# Calculate expected counts
										expected = sample_size * expected_props

										# Calculate chi-square statistic
										chi2_stat = np.sum((observed - expected) ** 2 / expected)

										# Record whether we reject the null hypothesis
										results[i] = 1 if chi2_stat > critical_value else 0

									empirical_rejection_rate = np.mean(results)

									# Visualize the results
									fig, ax = plt.subplots(figsize=(8, 5))

									# Create bar chart
									labels = ['Reject H₀', 'Fail to Reject H₀']
									counts = [np.sum(results), n_simulations - np.sum(results)]
									percentages = [count / n_simulations * 100 for count in counts]

									ax.bar(labels, percentages, color=['red', 'green'], alpha=0.7)

									ax.set_ylabel('Percentage')
									ax.set_title(f'Monte Carlo Simulation Results ({n_simulations} trials)')

									for i, v in enumerate(percentages):
										ax.text(i, v + 1, f"{v:.1f}%", ha='center')

									st.pyplot(fig)

									# Interpret the results
									if use_expected_as_true:
										expected_rate = alpha
										error_type = "Type I error"
									else:
										expected_rate = power
										error_type = "Power"

									st.markdown(f"""
					                <div class="info-box">
					                <h4>Simulation Results</h4>
					                <p><b>Number of simulations:</b> {n_simulations}</p>
					                <p><b>Empirical rejection rate:</b> {empirical_rejection_rate:.4f}</p>
					                <p><b>Theoretical {error_type} rate:</b> {expected_rate:.4f}</p>

					                <h4>Interpretation</h4>
					                <p>The simulation shows the proportion of times we would reject H₀ based on random samples from the true distribution.</p>
					                <p>When H₀ is {'true' if use_expected_as_true else 'false'}, this represents the {'Type I error rate (α)' if use_expected_as_true else 'Power of the test (1-β)'}.</p>
					                <p>The empirical rate is close to the theoretical value, demonstrating the properties of the Chi-square test.</p>
					                </div>
					                """, unsafe_allow_html=True)

						elif example_type == "Interactive CLT Explorer":
							st.markdown("""
					        <div class="explanation">
					        <h3>Interactive Central Limit Theorem Explorer</h3>
					        <p>
					        This interactive tool allows you to explore the Central Limit Theorem with different source distributions.
					        You can see how the sampling distribution of the mean approaches a normal distribution as the sample size increases.
					        </p>
					        </div>
					        """, unsafe_allow_html=True)

							# Choose a distribution to sample from
							dist_type = st.selectbox(
								"Select a distribution to sample from:",
								["Uniform", "Exponential", "Gamma", "Beta", "Discrete (Custom)", "Bimodal Mixture"]
							)

							# Set distribution parameters
							if dist_type == "Uniform":
								a = st.slider("Lower bound (a):", -10.0, 0.0, 0.0, 0.1)
								b = st.slider("Upper bound (b):", 0.1, 10.0, 1.0, 0.1)


								def generate_sample(n, sample_size):
									return np.random.uniform(a, b, size=(n, sample_size))


								true_mean = (a + b) / 2
								true_var = (b - a) ** 2 / 12
								dist_name = f"Uniform({a}, {b})"

							elif dist_type == "Exponential":
								rate = st.slider("Rate parameter (λ):", 0.1, 5.0, 1.0, 0.1)


								def generate_sample(n, sample_size):
									return np.random.exponential(scale=1 / rate, size=(n, sample_size))


								true_mean = 1 / rate
								true_var = 1 / (rate ** 2)
								dist_name = f"Exponential(λ={rate})"

							elif dist_type == "Gamma":
								shape = st.slider("Shape parameter (α):", 0.1, 10.0, 2.0, 0.1)
								rate = st.slider("Rate parameter (β):", 0.1, 5.0, 1.0, 0.1)


								def generate_sample(n, sample_size):
									return np.random.gamma(shape, scale=1 / rate, size=(n, sample_size))


								true_mean = shape / rate
								true_var = shape / (rate ** 2)
								dist_name = f"Gamma(α={shape}, β={rate})"

							elif dist_type == "Beta":
								alpha = st.slider("α parameter:", 0.1, 10.0, 2.0, 0.1)
								beta = st.slider("β parameter:", 0.1, 10.0, 5.0, 0.1)


								def generate_sample(n, sample_size):
									return np.random.beta(alpha, beta, size=(n, sample_size))


								true_mean = alpha / (alpha + beta)
								true_var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
								dist_name = f"Beta(α={alpha}, β={beta})"


							elif dist_type == "Discrete (Custom)":

								st.markdown("Define a discrete distribution by specifying values and probabilities:")

								# Define up to 6 values and probabilities

								values = []

								probs = []

								col1, col2 = st.columns(2)

								with col1:

									for i in range(6):
										val = st.number_input(f"Value {i + 1}:", -100.0, 100.0, i, 0.1, key=f"val_{i}")

										values.append(val)

								with col2:

									for i in range(6):
										prob = st.number_input(f"Probability {i + 1}:", 0.0, 1.0, 1 / 6, 0.01,

															   key=f"prob_{i}")

										probs.append(prob)

								# Normalize probabilities

								probs = np.array(probs)

								probs = probs / probs.sum()

								values = np.array(values)


								def generate_sample(n, sample_size):

									return np.random.choice(values, size=(n, sample_size), p=probs)


								true_mean = np.sum(values * probs)

								true_var = np.sum(probs * (values - true_mean) ** 2)

								dist_name = "Custom Discrete Distribution"


							elif dist_type == "Bimodal Mixture":

								st.markdown("Define a mixture of two normal distributions:")

								# Parameters for the first normal distribution

								mu1 = st.slider("Mean of first normal (μ₁):", -10.0, 10.0, -2.0, 0.1)

								sigma1 = st.slider("Standard deviation of first normal (σ₁):", 0.1, 5.0, 1.0, 0.1)

								# Parameters for the second normal distribution

								mu2 = st.slider("Mean of second normal (μ₂):", -10.0, 10.0, 2.0, 0.1)

								sigma2 = st.slider("Standard deviation of second normal (σ₂):", 0.1, 5.0, 1.0, 0.1)

								# Mixing proportion

								p = st.slider("Proportion of first normal (p):", 0.0, 1.0, 0.5, 0.01)


								def generate_sample(n, sample_size):

									# Generate samples from both distributions

									samples = np.zeros((n, sample_size))

									for i in range(n):
										# For each sample, decide which distribution to use for each point

										mask = np.random.random(sample_size) < p

										n1 = np.sum(mask)

										n2 = sample_size - n1

										# Generate from the two distributions

										samples[i, mask] = np.random.normal(mu1, sigma1, n1)

										samples[i, ~mask] = np.random.normal(mu2, sigma2, n2)

									return samples


								true_mean = p * mu1 + (1 - p) * mu2

								true_var = p * (sigma1 ** 2 + mu1 ** 2) + (1 - p) * (
											sigma2 ** 2 + mu2 ** 2) - true_mean ** 2

								dist_name = f"Bimodal Mixture: {p:.1f}×N({mu1}, {sigma1}²) + {1 - p:.1f}×N({mu2}, {sigma2}²)"

							# Now continue with the simulation parameters that would follow these distribution definitions

							sample_sizes = st.multiselect(

								"Select sample sizes to compare:",

								[1, 2, 5, 10, 20, 30, 50, 100],

								default=[1, 5, 30]

							)

							num_samples = st.slider("Number of samples to generate:", 100, 10000, 1000, 100)

							if st.button("Run Simulation"):

								# Create plots

								fig, axs = plt.subplots(len(sample_sizes), 2, figsize=(12, 4 * len(sample_sizes)))

								if len(sample_sizes) == 1:
									axs = np.array([axs])  # Make it 2D for consistent indexing

								for i, sample_size in enumerate(sample_sizes):
									# Generate samples

									samples = generate_sample(num_samples, sample_size)

									sample_means = samples.mean(axis=1)

									# Plot histogram of sample means

									axs[i, 0].hist(sample_means, bins=30, density=True, alpha=0.6)

									# Plot the theoretical normal distribution

									x = np.linspace(min(sample_means), max(sample_means), 1000)

									theoretical_mean = true_mean

									theoretical_std = np.sqrt(true_var / sample_size)

									pdf = norm.pdf(x, theoretical_mean, theoretical_std)

									axs[i, 0].plot(x, pdf, 'r-', lw=2)

									axs[i, 0].set_title(f"Sample Size = {sample_size}")

									axs[i, 0].set_xlabel("Sample Mean")

									axs[i, 0].set_ylabel("Density")

									# QQ plot

									sm.qqplot(sample_means, line='45', ax=axs[i, 1])

									axs[i, 1].set_title(f"QQ Plot for Sample Size = {sample_size}")

								plt.tight_layout()

								st.pyplot(fig)

								# Calculate and display statistics

								st.subheader("Sample Statistics")

								stats_data = []

								for sample_size in sample_sizes:
									samples = generate_sample(num_samples, sample_size)

									sample_means = samples.mean(axis=1)

									mean_of_means = np.mean(sample_means)

									std_of_means = np.std(sample_means, ddof=1)

									theoretical_std = np.sqrt(true_var / sample_size)

									stats_data.append({

										"Sample Size": sample_size,

										"Mean of Sample Means": f"{mean_of_means:.4f}",

										"Std Dev of Sample Means": f"{std_of_means:.4f}",

										"Theoretical Std Dev": f"{theoretical_std:.4f}"

									})

								st.table(pd.DataFrame(stats_data))

								st.markdown(f"""

							    <div class="explanation">

							    <h4>Theoretical Values for {dist_name}</h4>

							    <p>True Mean (μ): {true_mean:.4f}</p>

							    <p>True Variance (σ²): {true_var:.4f}</p>

							    <p>True Standard Deviation (σ): {np.sqrt(true_var):.4f}</p>

							    </div>

							    """, unsafe_allow_html=True)
								# Add a section for showing the convergence of sample means
								if st.checkbox("Show Convergence Animation"):
									st.write(
										"This animation shows how the distribution of sample means converges to a normal distribution as sample size increases.")

									# Create sample sizes for animation
									anim_sample_sizes = np.unique(np.logspace(0, 2.5, 20).astype(int))
									anim_sample_sizes = anim_sample_sizes[
										anim_sample_sizes <= 500]  # Limit to 500 for performance

									# Create a figure for the animation
									fig, ax = plt.subplots(figsize=(10, 6))
									ax.set_xlim(true_mean - 4 * np.sqrt(true_var), true_mean + 4 * np.sqrt(true_var))
									ax.set_ylim(0, 1.5)  # May need adjustment based on distributions

									# Generate a larger set of samples once
									max_size = max(anim_sample_sizes)
									large_samples = generate_sample(num_samples, max_size)

									# Create an empty histogram for initialization
									hist_values, hist_bins = np.histogram([], bins=30, density=True)
									bars = ax.bar(hist_bins[:-1], hist_values, width=np.diff(hist_bins), alpha=0.6,
												  align='edge')

									line, = ax.plot([], [], 'r-', lw=2)
									title = ax.set_title(f"Sample Size = 1")


									# Define animation function
									def update(frame):
										sample_size = anim_sample_sizes[frame]
										# Use the first sample_size elements of each row
										samples = large_samples[:, :sample_size]
										sample_means = samples.mean(axis=1)

										# Update histogram
										hist_values, hist_bins = np.histogram(sample_means, bins=30, density=True)
										for i, bar in enumerate(bars):
											if i < len(hist_values):
												bar.set_height(hist_values[i])
												bar.set_x(hist_bins[i])
												bar.set_width(hist_bins[i + 1] - hist_bins[i])
											else:
												bar.set_height(0)

										# Update theoretical normal curve
										x = np.linspace(min(sample_means), max(sample_means), 1000)
										theoretical_mean = true_mean
										theoretical_std = np.sqrt(true_var / sample_size)
										pdf = norm.pdf(x, theoretical_mean, theoretical_std)
										line.set_data(x, pdf)

										title.set_text(f"Sample Size = {sample_size}")
										return bars + [line, title]


									# Create animation
									anim = FuncAnimation(fig, update, frames=len(anim_sample_sizes), blit=True)

									# Convert animation to HTML5 video
									html = animation.HTML(anim.to_jshtml())
									st.components.v1.html(html.data, height=600)

								# Add exploratory section for comparing theoretical vs empirical results
								if st.checkbox("Explore Theoretical vs Empirical Convergence"):
									st.write(
										"This visualization shows how the standard deviation of sample means approaches the theoretical value as sample size increases.")

									# Define a range of sample sizes
									explore_sample_sizes = np.unique(np.logspace(0, 2.7, 30).astype(int))
									explore_sample_sizes = explore_sample_sizes[explore_sample_sizes <= 500]

									# Calculate empirical standard deviations for each sample size
									empirical_stds = []

									progress_bar = st.progress(0)
									for i, sample_size in enumerate(explore_sample_sizes):
										samples = generate_sample(num_samples, sample_size)
										sample_means = samples.mean(axis=1)
										empirical_stds.append(np.std(sample_means, ddof=1))
										progress_bar.progress((i + 1) / len(explore_sample_sizes))

									# Calculate theoretical standard deviations
									theoretical_stds = [np.sqrt(true_var / size) for size in explore_sample_sizes]

									# Create the plot
									fig, ax = plt.subplots(figsize=(10, 6))
									ax.plot(explore_sample_sizes, empirical_stds, 'bo-', label='Empirical Std Dev')
									ax.plot(explore_sample_sizes, theoretical_stds, 'r-', label='Theoretical Std Dev')
									ax.set_xscale('log')
									ax.set_xlabel('Sample Size (log scale)')
									ax.set_ylabel('Standard Deviation of Sample Means')
									ax.set_title('Convergence of Sample Mean Standard Deviation')
									ax.legend()
									ax.grid(True)

									# Add the formula for the standard error
									formula = r"$\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}$"
									ax.annotate(formula, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=14,
												bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

									st.pyplot(fig)

								# Interactive demonstration of the CLT
								if st.checkbox("Interactive Single Sample Demonstration"):
									st.write("Draw individual samples and observe how their means distribute.")

									col1, col2 = st.columns([1, 2])

									with col1:
										demo_sample_size = st.slider("Sample Size:", 1, 100, 30)

										if st.button("Draw New Sample"):
											# Generate a single new sample and its mean
											new_sample = generate_sample(1, demo_sample_size)[0]
											new_mean = np.mean(new_sample)

											# Store in session state
											if 'demo_samples' not in st.session_state:
												st.session_state.demo_samples = []
												st.session_state.demo_means = []

											st.session_state.demo_samples.append(new_sample)
											st.session_state.demo_means.append(new_mean)

										if st.button("Reset Demonstration"):
											if 'demo_samples' in st.session_state:
												del st.session_state.demo_samples
												del st.session_state.demo_means

									with col2:
										if 'demo_samples' in st.session_state and st.session_state.demo_samples:
											# Create figure with two subplots
											fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

											# Plot the most recent sample
											latest_sample = st.session_state.demo_samples[-1]
											ax1.hist(latest_sample, bins=15, alpha=0.7)
											ax1.axvline(np.mean(latest_sample), color='red', linestyle='dashed',
														linewidth=2)
											ax1.set_title(f"Most Recent Sample (n={demo_sample_size})")
											ax1.set_xlabel("Value")
											ax1.set_ylabel("Count")

											# Plot all sample means
											all_means = st.session_state.demo_means
											ax2.hist(all_means, bins=15, alpha=0.7)
											ax2.axvline(true_mean, color='green', linestyle='dashed', linewidth=2,
														label=f"True Mean = {true_mean:.2f}")

											# Add theoretical normal distribution
											if len(all_means) > 5:  # Only add if we have enough data
												x = np.linspace(min(all_means) - 1, max(all_means) + 1, 1000)
												pdf = norm.pdf(x, true_mean, np.sqrt(true_var / demo_sample_size))
												# Scale the PDF to match histogram height
												pdf_scale = len(all_means) * (
															x[1] - x[0]) * 3  # Adjust the factor as needed
												ax2.plot(x, pdf * pdf_scale, 'r-', linewidth=2,
														 label=f"Theoretical N({true_mean:.2f}, {np.sqrt(true_var / demo_sample_size):.2f})")

											ax2.set_title(f"Distribution of Sample Means (counts: {len(all_means)})")
											ax2.set_xlabel("Sample Mean")
											ax2.set_ylabel("Count")
											ax2.legend()

											plt.tight_layout()
											st.pyplot(fig)

											# Display statistics
											mean_of_means = np.mean(all_means)
											std_of_means = np.std(all_means, ddof=1) if len(all_means) > 1 else np.nan
											theoretical_std = np.sqrt(true_var / demo_sample_size)

											st.markdown(f"""
								            **Statistics from Demonstration:**
								            - Number of samples drawn: {len(all_means)}
								            - Mean of sample means: {mean_of_means:.4f} (True mean: {true_mean:.4f})
								            - Standard deviation of sample means: {std_of_means:.4f} (Theoretical: {theoretical_std:.4f})
								            """)
										else:
											st.write("Press 'Draw New Sample' to begin the demonstration.")

								# Educational explanation
								st.markdown("""
								## About the Central Limit Theorem

								The Central Limit Theorem (CLT) is one of the most important theorems in statistics. It states that when independent random variables are added, their properly normalized sum tends toward a normal distribution even if the original variables themselves are not normally distributed.

								More specifically, if you take sufficiently large samples from a population with mean μ and standard deviation σ and calculate the sample means, the distribution of these sample means will be approximately normally distributed with mean μ and standard deviation σ/√n, where n is the sample size.

								Key points:
								1. The approximmation improves as sample size increases
								2. The theorem applies regardless of the shape of the original distribution
								3. For heavily skewed distributions, larger sample sizes are needed for good approximation

								This simulator allows you to see the CLT in action with different source distributions, helping visualize this fundamental concept in statistics.
								""")
