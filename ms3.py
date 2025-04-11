import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.animation import FuncAnimation
import altair as alt
import time
from PIL import Image
import io
import base64

# Set page configuration
st.set_page_config(
	page_title="Mathematical Statistics for Beginners",
	page_icon="📊",
	layout="wide",
	initial_sidebar_state="expanded"
)

# Custom CSS
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
    .card {
        border-radius: 5px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f5f5f5;
        border-left: 5px solid #1E88E5;
    }
    .formula {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        font-family: "Computer Modern", serif;
        margin: 1rem 0;
    }
    .highlight {
        color: #1E88E5;
        font-weight: bold;
    }
    .note {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 5px solid #ffc107;
    }
    .distribution-title {
        color: #0D47A1;
        font-size: 1.4rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .parameter-box {
        padding: 10px;
        background-color: #e8f5e9;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<h1 class='main-header'>Statistical Distributions Explorer</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <p>This interactive application helps visualize and understand statistical distributions, their properties, 
    and relationships between them. Use the sidebar to navigate through different distributions and concepts.</p>
    <p>Perfect for beginners in probability and mathematical statistics!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
main_options = ["Home", "Discrete Distributions", "Continuous Distributions", "Distribution Relationships",
				"Central Limit Theorem", "About"]
main_selection = st.sidebar.selectbox("Select Section", main_options)


# Function to create animated plots
def create_animation(func, frames, interval=100, repeat=True):
	fig, ax = plt.subplots(figsize=(10, 6))

	def update(frame):
		ax.clear()
		return func(ax, frame)

	anim = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=repeat)

	# Convert animation to HTML5 video
	f = io.BytesIO()
	anim.save(f, writer='pillow', fps=10)
	f.seek(0)

	# Create a base64 string representation
	b64 = base64.b64encode(f.read()).decode('utf-8')

	return f'<img src="data:image/gif;base64,{b64}"/>'


# Discrete distribution functions
def pmf_plot(distribution, params, x_range, name, ax=None):
	if ax is None:
		fig, ax = plt.subplots(figsize=(10, 6))

	x = np.arange(x_range[0], x_range[1] + 1)

	if distribution == stats.binom:
		pmf = distribution.pmf(x, params[0], params[1])
		title = f"{name} PMF (n={params[0]}, p={params[1]})"
	elif distribution == stats.poisson:
		pmf = distribution.pmf(x, params[0])
		title = f"{name} PMF (λ={params[0]})"
	elif distribution == stats.geom:
		pmf = distribution.pmf(x, params[0])
		title = f"{name} PMF (p={params[0]})"
	elif distribution == stats.hypergeom:
		pmf = distribution.pmf(x, params[0], params[1], params[2])
		title = f"{name} PMF (M={params[0]}, n={params[1]}, N={params[2]})"
	elif distribution == stats.nbinom:
		pmf = distribution.pmf(x, params[0], params[1])
		title = f"{name} PMF (n={params[0]}, p={params[1]})"
	else:
		raise ValueError("Distribution not supported")

	ax.bar(x, pmf, alpha=0.7)
	ax.set_xlabel("x")
	ax.set_ylabel("P(X = x)")
	ax.set_title(title)
	ax.grid(alpha=0.3)

	return ax


def cdf_plot(distribution, params, x_range, name, ax=None):
	if ax is None:
		fig, ax = plt.subplots(figsize=(10, 6))

	x = np.arange(x_range[0], x_range[1] + 1)

	if distribution == stats.binom:
		cdf = distribution.cdf(x, params[0], params[1])
		title = f"{name} CDF (n={params[0]}, p={params[1]})"
	elif distribution == stats.poisson:
		cdf = distribution.cdf(x, params[0])
		title = f"{name} CDF (λ={params[0]})"
	elif distribution == stats.geom:
		cdf = distribution.cdf(x, params[0])
		title = f"{name} CDF (p={params[0]})"
	elif distribution == stats.hypergeom:
		cdf = distribution.cdf(x, params[0], params[1], params[2])
		title = f"{name} CDF (M={params[0]}, n={params[1]}, N={params[2]})"
	elif distribution == stats.nbinom:
		cdf = distribution.cdf(x, params[0], params[1])
		title = f"{name} CDF (n={params[0]}, p={params[1]})"
	else:
		raise ValueError("Distribution not supported")

	ax.step(x, cdf, where='post', alpha=0.7)
	ax.set_xlabel("x")
	ax.set_ylabel("P(X ≤ x)")
	ax.set_title(title)
	ax.grid(alpha=0.3)

	return ax


# Continuous distribution functions
def pdf_plot(distribution, params, x_range, name, ax=None):
	if ax is None:
		fig, ax = plt.subplots(figsize=(10, 6))

	x = np.linspace(x_range[0], x_range[1], 1000)

	if distribution == stats.norm:
		pdf = distribution.pdf(x, params[0], params[1])
		title = f"{name} PDF (μ={params[0]}, σ={params[1]})"
	elif distribution == stats.uniform:
		pdf = distribution.pdf(x, params[0], params[1] - params[0])
		title = f"{name} PDF (a={params[0]}, b={params[1]})"
	elif distribution == stats.expon:
		pdf = distribution.pdf(x, scale=1 / params[0])
		title = f"{name} PDF (λ={params[0]})"
	elif distribution == stats.gamma:
		pdf = distribution.pdf(x, params[0], scale=1 / params[1])
		title = f"{name} PDF (α={params[0]}, β={params[1]})"
	elif distribution == stats.beta:
		pdf = distribution.pdf(x, params[0], params[1])
		title = f"{name} PDF (α={params[0]}, β={params[1]})"
	elif distribution == stats.t:
		pdf = distribution.pdf(x, params[0])
		title = f"{name} PDF (df={params[0]})"
	elif distribution == stats.chi2:
		pdf = distribution.pdf(x, params[0])
		title = f"{name} PDF (df={params[0]})"
	elif distribution == stats.f:
		pdf = distribution.pdf(x, params[0], params[1])
		title = f"{name} PDF (dfn={params[0]}, dfd={params[1]})"
	else:
		raise ValueError("Distribution not supported")

	ax.plot(x, pdf, alpha=0.7)
	ax.fill_between(x, pdf, alpha=0.3)
	ax.set_xlabel("x")
	ax.set_ylabel("f(x)")
	ax.set_title(title)
	ax.grid(alpha=0.3)

	return ax


def continuous_cdf_plot(distribution, params, x_range, name, ax=None):
	if ax is None:
		fig, ax = plt.subplots(figsize=(10, 6))

	x = np.linspace(x_range[0], x_range[1], 1000)

	if distribution == stats.norm:
		cdf = distribution.cdf(x, params[0], params[1])
		title = f"{name} CDF (μ={params[0]}, σ={params[1]})"
	elif distribution == stats.uniform:
		cdf = distribution.cdf(x, params[0], params[1] - params[0])
		title = f"{name} CDF (a={params[0]}, b={params[1]})"
	elif distribution == stats.expon:
		cdf = distribution.cdf(x, scale=1 / params[0])
		title = f"{name} CDF (λ={params[0]})"
	elif distribution == stats.gamma:
		cdf = distribution.cdf(x, params[0], scale=1 / params[1])
		title = f"{name} CDF (α={params[0]}, β={params[1]})"
	elif distribution == stats.beta:
		cdf = distribution.cdf(x, params[0], params[1])
		title = f"{name} CDF (α={params[0]}, β={params[1]})"
	elif distribution == stats.t:
		cdf = distribution.cdf(x, params[0])
		title = f"{name} CDF (df={params[0]})"
	elif distribution == stats.chi2:
		cdf = distribution.cdf(x, params[0])
		title = f"{name} CDF (df={params[0]})"
	elif distribution == stats.f:
		cdf = distribution.cdf(x, params[0], params[1])
		title = f"{name} CDF (dfn={params[0]}, dfd={params[1]})"
	else:
		raise ValueError("Distribution not supported")

	ax.plot(x, cdf, alpha=0.7)
	ax.set_xlabel("x")
	ax.set_ylabel("F(x)")
	ax.set_title(title)
	ax.grid(alpha=0.3)

	return ax


# Animated function for parameter changes
def animated_pmf(ax, frame, distribution, params_list, x_range, name):
	pmf_plot(distribution, params_list[frame], x_range, name, ax)
	return ax


def animated_pdf(ax, frame, distribution, params_list, x_range, name):
	pdf_plot(distribution, params_list[frame], x_range, name, ax)
	return ax


def create_distribution_info(name, formula, parameters, properties, examples):
	st.markdown(f"<h3 class='distribution-title'>{name}</h3>", unsafe_allow_html=True)

	with st.expander("Formula and Definition", expanded=True):
		st.markdown(f"<div class='formula'>{formula}</div>", unsafe_allow_html=True)

		st.markdown("<h4>Parameters:</h4>", unsafe_allow_html=True)
		for param, desc in parameters.items():
			st.markdown(f"<div class='parameter-box'><b>{param}</b>: {desc}</div>", unsafe_allow_html=True)

	with st.expander("Properties", expanded=False):
		for prop, value in properties.items():
			st.markdown(f"<b>{prop}:</b> {value}")

	with st.expander("Real-world Examples", expanded=False):
		for example in examples:
			st.markdown(f"• {example}")


# Central Limit Theorem visualization
def clt_demo(sample_size, num_samples, distribution_type):
	if distribution_type == "Uniform":
		dist_func = np.random.uniform
		dist_params = [0, 1]
		dist_mean = 0.5
		dist_std = 1 / np.sqrt(12)
		title = "Uniform Distribution"
	elif distribution_type == "Exponential":
		dist_func = np.random.exponential
		dist_params = [1]
		dist_mean = 1
		dist_std = 1
		title = "Exponential Distribution"
	elif distribution_type == "Bernoulli":
		def bernoulli(size, p=0.5):
			return np.random.binomial(1, p, size)

		dist_func = bernoulli
		dist_params = [0.5]
		dist_mean = 0.5
		dist_std = np.sqrt(0.5 * 0.5)
		title = "Bernoulli Distribution"
	else:  # Normal
		dist_func = np.random.normal
		dist_params = [0, 1]
		dist_mean = 0
		dist_std = 1
		title = "Normal Distribution"

	# Generate samples
	if distribution_type == "Uniform":
		samples = np.array([dist_func(*dist_params, sample_size) for _ in range(num_samples)])
	elif distribution_type == "Exponential":
		samples = np.array([dist_func(*dist_params, sample_size) for _ in range(num_samples)])
	elif distribution_type == "Bernoulli":
		samples = np.array([dist_func(sample_size, *dist_params) for _ in range(num_samples)])
	else:  # Normal
		samples = np.array([dist_func(*dist_params, sample_size) for _ in range(num_samples)])

	# Calculate sample means
	sample_means = np.mean(samples, axis=1)

	# Theoretical distribution of the sample mean
	x = np.linspace(dist_mean - 4 * dist_std / np.sqrt(sample_size),
					dist_mean + 4 * dist_std / np.sqrt(sample_size), 1000)
	pdf = stats.norm.pdf(x, dist_mean, dist_std / np.sqrt(sample_size))

	# Create plots
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

	# Plot original distribution
	if distribution_type == "Uniform":
		x_orig = np.linspace(-0.5, 1.5, 1000)
		y_orig = stats.uniform.pdf(x_orig, *dist_params)
		ax1.plot(x_orig, y_orig)
		ax1.fill_between(x_orig, y_orig, alpha=0.3)
	elif distribution_type == "Exponential":
		x_orig = np.linspace(0, 5, 1000)
		y_orig = stats.expon.pdf(x_orig, scale=dist_params[0])
		ax1.plot(x_orig, y_orig)
		ax1.fill_between(x_orig, y_orig, alpha=0.3)
	elif distribution_type == "Bernoulli":
		ax1.bar([0, 1], [0.5, 0.5], width=0.1)
	else:  # Normal
		x_orig = np.linspace(-4, 4, 1000)
		y_orig = stats.norm.pdf(x_orig, *dist_params)
		ax1.plot(x_orig, y_orig)
		ax1.fill_between(x_orig, y_orig, alpha=0.3)

	ax1.set_title(f"Original {title}")
	ax1.grid(alpha=0.3)

	# Plot sample means distribution
	ax2.hist(sample_means, bins=30, density=True, alpha=0.6)
	ax2.plot(x, pdf, 'r-', lw=2)
	ax2.set_title(f"Distribution of Sample Means (n={sample_size})")
	ax2.grid(alpha=0.3)

	plt.tight_layout()
	return fig


# Home page content
if main_selection == "Home":
	st.markdown("<h2 class='sub-header'>Welcome to Statistical Distributions Explorer</h2>", unsafe_allow_html=True)

	st.markdown("""
    <div class="card">
        <p>This application will help you understand:</p>
        <ul>
            <li>Properties of discrete and continuous probability distributions</li>
            <li>How parameters affect the shape of distributions</li>
            <li>Relationships and connections between different distributions</li>
            <li>Key concepts like the Central Limit Theorem</li>
        </ul>
        <p>Use the sidebar to navigate through different sections.</p>
    </div>
    """, unsafe_allow_html=True)

	st.markdown("<h3 class='sub-header'>Probability Distributions Overview</h3>", unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("""
        <h4>Discrete Distributions</h4>
        <ul>
            <li>Bernoulli</li>
            <li>Binomial</li>
            <li>Geometric</li>
            <li>Negative Binomial</li>
            <li>Poisson</li>
            <li>Hypergeometric</li>
        </ul>
        """, unsafe_allow_html=True)

	with col2:
		st.markdown("""
        <h4>Continuous Distributions</h4>
        <ul>
            <li>Uniform</li>
            <li>Normal</li>
            <li>Exponential</li>
            <li>Gamma</li>
            <li>Beta</li>
            <li>t-distribution</li>
            <li>Chi-square</li>
            <li>F-distribution</li>
        </ul>
        """, unsafe_allow_html=True)

	st.markdown("<h3 class='sub-header'>Featured Concept: Central Limit Theorem</h3>", unsafe_allow_html=True)

	st.markdown("""
    <div class="note">
        <p>The Central Limit Theorem (CLT) is one of the most important concepts in statistics. It states that the sampling 
        distribution of the mean approaches a normal distribution as the sample size gets larger, regardless of the shape 
        of the population distribution.</p>
        <p>Explore this concept in the "Central Limit Theorem" section!</p>
    </div>
    """, unsafe_allow_html=True)

# Discrete Distributions page
elif main_selection == "Discrete Distributions":
	st.markdown("<h2 class='sub-header'>Discrete Probability Distributions</h2>", unsafe_allow_html=True)

	st.markdown("""
    <div class="card">
        <p>Discrete probability distributions describe random variables that can take on only a countable number of values.
        Each possible value has a probability between 0 and 1, and the sum of all probabilities equals 1.</p>
    </div>
    """, unsafe_allow_html=True)

	# Distribution selector
	distribution_options = ["Bernoulli", "Binomial", "Geometric", "Negative Binomial", "Poisson", "Hypergeometric"]
	selected_distribution = st.selectbox("Select a Discrete Distribution", distribution_options)

	# Render the selected distribution
	if selected_distribution == "Bernoulli":
		# Bernoulli distribution
		st.markdown("<h3 class='distribution-title'>Bernoulli Distribution</h3>", unsafe_allow_html=True)

		# Definition and formula
		with st.expander("Formula and Definition", expanded=True):
			st.markdown("""
            <div class="formula">
                P(X = x) = p^x (1-p)^(1-x), x ∈ {0, 1}
            </div>
            <p>The Bernoulli distribution is the discrete probability distribution of a random variable 
            which takes the value 1 with probability p and the value 0 with probability q = 1 - p.</p>
            """, unsafe_allow_html=True)

			st.markdown("<h4>Parameters:</h4>", unsafe_allow_html=True)
			st.markdown("""
            <div class="parameter-box"><b>p</b>: probability of success (0 ≤ p ≤ 1)</div>
            """, unsafe_allow_html=True)

		# Properties
		with st.expander("Properties", expanded=False):
			st.markdown("""
            <b>Mean</b>: p<br>
            <b>Variance</b>: p(1-p)<br>
            <b>Skewness</b>: (1-2p)/√(p(1-p))<br>
            <b>Kurtosis</b>: (1-6p(1-p))/(p(1-p))<br>
            <b>MGF</b>: (1-p) + pe^t
            """, unsafe_allow_html=True)

		# Examples
		with st.expander("Real-world Examples", expanded=False):
			st.markdown("""
            • Coin toss (Heads/Tails)<br>
            • Pass/Fail outcome of an exam<br>
            • Success/Failure of a single product test<br>
            • Presence/Absence of a genetic trait
            """, unsafe_allow_html=True)

		# Interactive visualization
		st.markdown("<h4>Interactive Visualization</h4>", unsafe_allow_html=True)

		p = st.slider("Probability of success (p)", 0.0, 1.0, 0.5, 0.01)

		col1, col2 = st.columns(2)

		with col1:
			fig, ax = plt.subplots(figsize=(8, 5))
			x = np.array([0, 1])
			pmf = np.array([1 - p, p])
			ax.bar(x, pmf, alpha=0.7)
			ax.set_xlabel("x")
			ax.set_ylabel("P(X = x)")
			ax.set_title(f"Bernoulli PMF (p={p})")
			ax.set_xticks([0, 1])
			ax.grid(alpha=0.3)
			st.pyplot(fig)

		with col2:
			fig, ax = plt.subplots(figsize=(8, 5))
			x = np.array([0, 1])
			cdf = np.array([1 - p, 1])
			ax.step([-0.5, 0, 1], [0, 1 - p, 1], where='post', alpha=0.7)
			ax.set_xlabel("x")
			ax.set_ylabel("P(X ≤ x)")
			ax.set_title(f"Bernoulli CDF (p={p})")
			ax.set_xticks([0, 1])
			ax.grid(alpha=0.3)
			st.pyplot(fig)

		# Animation of changing parameter
		st.markdown("<h4>Animated Parameter Change</h4>", unsafe_allow_html=True)

		if st.button("Show Animation of Changing p"):
			frames = 50
			p_values = np.linspace(0.05, 0.95, frames)

			plot_data = []
			for p_val in p_values:
				x = np.array([0, 1])
				pmf = np.array([1 - p_val, p_val])
				plot_data.append((x, pmf, p_val))

			chart_data = pd.DataFrame({
				'frame': np.repeat(range(frames), 2),
				'x': np.tile([0, 1], frames),
				'pmf': np.concatenate([[1 - p_val, p_val] for p_val in p_values]),
				'p_val': np.repeat(p_values, 2)
			})

			# Create animated chart with Altair
			chart = alt.Chart(chart_data).mark_bar().encode(
				x=alt.X('x:O', title='x'),
				y=alt.Y('pmf:Q', title='P(X = x)'),
				color=alt.value('#1f77b4')
			).properties(
				width=600,
				height=400,
				title='Bernoulli PMF with Changing p'
			).facet(
				facet=alt.Facet('p_val:N', title='p value', header=alt.Header(labelFontSize=12))
			)

			# Display animation
			st.altair_chart(chart)

	elif selected_distribution == "Binomial":
		# Binomial distribution
		create_distribution_info(
			name="Binomial Distribution",
			formula=r"P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, k = 0, 1, 2, ..., n",
			parameters={
				"n": "Number of trials (n > 0, integer)",
				"p": "Probability of success on a single trial (0 ≤ p ≤ 1)"
			},
			properties={
				"Mean": "np",
				"Variance": "np(1-p)",
				"Skewness": "(1-2p)/√(np(1-p))",
				"Kurtosis": "(1-6p(1-p))/(np(1-p))",
				"MGF": "(1-p+pe^t)^n"
			},
			examples=[
				"Number of heads in n coin tosses",
				"Number of defective items in a sample of n items",
				"Number of successful free throws out of n attempts",
				"Number of students who pass an exam out of n students"
			]
		)

		# Interactive visualization
		st.markdown("<h4>Interactive Visualization</h4>", unsafe_allow_html=True)

		col1, col2 = st.columns([1, 3])

		with col1:
			n = st.slider("Number of trials (n)", 1, 50, 10, 1)
			p = st.slider("Probability of success (p)", 0.0, 1.0, 0.5, 0.01)

			show_pmf = st.checkbox("Show PMF", value=True)
			show_cdf = st.checkbox("Show CDF", value=True)

			if st.button("Show Animation"):
				st.session_state.show_binomial_animation = True

		with col2:
			if show_pmf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.arange(0, n + 1)
				pmf = stats.binom.pmf(x, n, p)
				ax.bar(x, pmf, alpha=0.7)
				ax.set_xlabel("Number of successes (k)")
				ax.set_ylabel("P(X = k)")
				ax.set_title(f"Binomial PMF (n={n}, p={p})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

			if show_cdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.arange(0, n + 1)
				cdf = stats.binom.cdf(x, n, p)
				ax.step(np.concatenate([[-1], x]), np.concatenate([[0], cdf]), where='post', alpha=0.7)
				ax.set_xlabel("Number of successes (k)")
				ax.set_ylabel("P(X ≤ k)")
				ax.set_title(f"Binomial CDF (n={n}, p={p})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

		# Animation
		if 'show_binomial_animation' in st.session_state and st.session_state.show_binomial_animation:
			st.markdown("<h4>Animated Parameter Changes</h4>", unsafe_allow_html=True)

			animation_type = st.radio("Choose Animation Type", ["Varying p", "Varying n"])

			if animation_type == "Varying p":
				fig = plt.figure(figsize=(10, 6))
				p_values = np.linspace(0.05, 0.95, 10)

				# Create data for animation
				data = []
				for p_val in p_values:
					x = np.arange(0, n + 1)
					y = stats.binom.pmf(x, n, p_val)
					df = pd.DataFrame({'x': x, 'y': y, 'p': [f"p={p_val:.2f}"] * len(x)})
					data.append(df)

				df = pd.concat(data)

				# Create Altair chart with animation
				chart = alt.Chart(df).mark_bar().encode(
					x=alt.X('x:O', title='Number of successes (k)'),
					y=alt.Y('y:Q', title='P(X = k)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title=f'Binomial PMF with n={n} and varying p'
				).facet(
					facet=alt.Facet('p:N', title='Parameter value')
				)

				st.altair_chart(chart)

			else:  # Varying n
				fig = plt.figure(figsize=(10, 6))
				n_values = np.arange(5, 31, 5)

				# Create data for animation
				data = []
				for n_val in n_values:
					x = np.arange(0, n_val + 1)
					y = stats.binom.pmf(x, n_val, p)
					df = pd.DataFrame({'x': x, 'y': y, 'n': [f"n={n_val}"] * len(x)})
					data.append(df)

				df = pd.concat(data)

				# Create Altair chart with animation
				chart = alt.Chart(df).mark_bar().encode(
					x=alt.X('x:O', title='Number of successes (k)'),
					y=alt.Y('y:Q', title='P(X = k)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title=f'Binomial PMF with p={p} and varying n'
				).facet(
					facet=alt.Facet('n:N', title='Parameter value')
				)

				st.altair_chart(chart)

	elif selected_distribution == "Geometric":
		# Geometric distribution
		create_distribution_info(
			name="Geometric Distribution",
			formula=r"P(X = k) = (1-p)^{k-1} p, k = 1, 2, 3, ...",
			parameters={
				"p": "Probability of success on a single trial (0 < p ≤ 1)"
			},
			properties={
				"Mean": "1/p",
				"Variance": "(1-p)/p²",
				"Skewness": "(2-p)/√(1-p)",
				"Kurtosis": "9 + (p²/(1-p))",
				"MGF": "pe^t/(1-(1-p)e^t) for t < -ln(1-p)"
			},
			examples=[
				"Number of coin tosses until the first head appears",
				"Number of attempts until the first success",
				"Number of trials needed to get the first defective item",
				"Number of days until it rains"
			]
		)

		# Interactive visualization
		st.markdown("<h4>Interactive Visualization</h4>", unsafe_allow_html=True)

		col1, col2 = st.columns([1, 3])

		with col1:
			p = st.slider("Probability of success (p)", 0.01, 1.0, 0.3, 0.01)
			max_k = st.slider("Maximum k to display", 1, 30, 15)

			show_pmf = st.checkbox("Show PMF", value=True)
			show_cdf = st.checkbox("Show CDF", value=True)

			if st.button("Show Animation"):
				st.session_state.show_geometric_animation = True

		with col2:
			if show_pmf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.arange(1, max_k + 1)
				pmf = stats.geom.pmf(x, p)
				ax.bar(x, pmf, alpha=0.7)
				ax.set_xlabel("Number of trials (k)")
				ax.set_ylabel("P(X = k)")
				ax.set_title(f"Geometric PMF (p={p})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

			if show_cdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.arange(1, max_k + 1)
				cdf = stats.geom.cdf(x, p)
				ax.step(np.concatenate([[0], x]), np.concatenate([[0], cdf]), where='post', alpha=0.7)
				ax.set_xlabel("Number of trials (k)")
				ax.set_ylabel("P(X ≤ k)")
				ax.set_title(f"Geometric CDF (p={p})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

		# Animation
		if 'show_geometric_animation' in st.session_state and st.session_state.show_geometric_animation:
			st.markdown("<h4>Animated Parameter Changes</h4>", unsafe_allow_html=True)

			fig = plt.figure(figsize=(10, 6))
			p_values = np.linspace(0.1, 0.9, 9)

			# Create data for animation
			data = []
			for p_val in p_values:
				x = np.arange(1, max_k + 1)
				y = stats.geom.pmf(x, p_val)
				df = pd.DataFrame({'x': x, 'y': y, 'p': [f"p={p_val:.1f}"] * len(x)})
				data.append(df)

			df = pd.concat(data)

			# Create Altair chart with animation
			chart = alt.Chart(df).mark_bar().encode(
				x=alt.X('x:O', title='Number of trials (k)'),
				y=alt.Y('y:Q', title='P(X = k)'),
				color=alt.value('#1f77b4')
			).properties(
				width=600,
				height=400,
				title='Geometric PMF with varying p'
			).facet(
				facet=alt.Facet('p:N', title='Parameter value')
			)

			st.altair_chart(chart)

	elif selected_distribution == "Negative Binomial":
		# Negative Binomial distribution
		create_distribution_info(
			name="Negative Binomial Distribution",
			formula=r"P(X = k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}, k = r, r+1, r+2, ...",
			parameters={
				"r": "Number of successes required (r > 0, integer)",
				"p": "Probability of success on a single trial (0 < p ≤ 1)"
			},
			properties={
				"Mean": "r/p",
				"Variance": "r(1-p)/p²",
				"Skewness": "(2-p)/√(r(1-p))",
				"Kurtosis": "6/r + (p²/(r(1-p)))",
				"MGF": "(p/(1-(1-p)e^t))^r for t < -ln(1-p)"
			},
			examples=[
				"Number of coin tosses needed to get r heads",
				"Number of attempts until r successes are achieved",
				"Number of insurance claims until r large claims occur",
				"Number of customers until r sales are made"
			]
		)

		# Interactive visualization
		st.markdown("<h4>Interactive Visualization</h4>", unsafe_allow_html=True)

		col1, col2 = st.columns([1, 3])

		with col1:
			r = st.slider("Number of successes required (r)", 1, 10, 3, 1)
			p = st.slider("Probability of success (p)", 0.01, 1.0, 0.3, 0.01)
			max_k = st.slider("Maximum k to display", r, r + 30, r + 15)

			show_pmf = st.checkbox("Show PMF", value=True)
			show_cdf = st.checkbox("Show CDF", value=True)

			if st.button("Show Animation"):
				st.session_state.show_negbinom_animation = True

		with col2:
			if show_pmf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.arange(r, max_k + 1)
				pmf = stats.nbinom.pmf(x - r, r, p)  # Adjust for scipy's parameterization
				ax.bar(x, pmf, alpha=0.7)
				ax.set_xlabel("Number of trials (k)")
				ax.set_ylabel("P(X = k)")
				ax.set_title(f"Negative Binomial PMF (r={r}, p={p})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

			if show_cdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.arange(r, max_k + 1)
				cdf = stats.nbinom.cdf(x - r, r, p)  # Adjust for scipy's parameterization
				ax.step(np.concatenate([[r - 1], x]), np.concatenate([[0], cdf]), where='post', alpha=0.7)
				ax.set_xlabel("Number of trials (k)")
				ax.set_ylabel("P(X ≤ k)")
				ax.set_title(f"Negative Binomial CDF (r={r}, p={p})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

		# Animation
		if 'show_negbinom_animation' in st.session_state and st.session_state.show_negbinom_animation:
			st.markdown("<h4>Animated Parameter Changes</h4>", unsafe_allow_html=True)

			animation_type = st.radio("Choose Animation Type", ["Varying p", "Varying r"])

			if animation_type == "Varying p":
				fig = plt.figure(figsize=(10, 6))
				p_values = np.linspace(0.1, 0.9, 9)

				# Create data for animation
				data = []
				for p_val in p_values:
					x = np.arange(r, max_k + 1)
					y = stats.nbinom.pmf(x - r, r, p_val)  # Adjust for scipy's parameterization
					df = pd.DataFrame({'x': x, 'y': y, 'p': [f"p={p_val:.1f}"] * len(x)})
					data.append(df)

				df = pd.concat(data)

				# Create Altair chart with animation
				chart = alt.Chart(df).mark_bar().encode(
					x=alt.X('x:O', title='Number of trials (k)'),
					y=alt.Y('y:Q', title='P(X = k)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title=f'Negative Binomial PMF with r={r} and varying p'
				).facet(
					facet=alt.Facet('p:N', title='Parameter value')
				)

				st.altair_chart(chart)

			else:  # Varying r
				fig = plt.figure(figsize=(10, 6))
				r_values = np.arange(1, 6)

				# Create data for animation
				data = []
				for r_val in r_values:
					x = np.arange(r_val, max_k + 1)
					y = stats.nbinom.pmf(x - r_val, r_val, p)  # Adjust for scipy's parameterization
					df = pd.DataFrame({'x': x, 'y': y, 'r': [f"r={r_val}"] * len(x)})
					data.append(df)

				df = pd.concat(data)

				# Create Altair chart with animation
				chart = alt.Chart(df).mark_bar().encode(
					x=alt.X('x:O', title='Number of trials (k)'),
					y=alt.Y('y:Q', title='P(X = k)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title=f'Negative Binomial PMF with p={p} and varying r'
				).facet(
					facet=alt.Facet('r:N', title='Parameter value')
				)

				st.altair_chart(chart)

	elif selected_distribution == "Poisson":
		# Poisson distribution
		create_distribution_info(
			name="Poisson Distribution",
			formula=r"P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, k = 0, 1, 2, ...",
			parameters={
				"λ": "Rate parameter (λ > 0), the average number of events in the given time interval"
			},
			properties={
				"Mean": "λ",
				"Variance": "λ",
				"Skewness": "1/√λ",
				"Kurtosis": "1/λ",
				"MGF": "exp(λ(e^t - 1))"
			},
			examples=[
				"Number of calls received by a call center in an hour",
				"Number of typos on a page",
				"Number of cars arriving at a toll booth in a minute",
				"Number of earthquakes in a region per year"
			]
		)

		# Interactive visualization
		st.markdown("<h4>Interactive Visualization</h4>", unsafe_allow_html=True)

		col1, col2 = st.columns([1, 3])

		with col1:
			lambda_val = st.slider("Rate parameter (λ)", 0.1, 20.0, 5.0, 0.1)
			max_k = st.slider("Maximum k to display", 0, 50, 20)

			show_pmf = st.checkbox("Show PMF", value=True)
			show_cdf = st.checkbox("Show CDF", value=True)

			if st.button("Show Animation"):
				st.session_state.show_poisson_animation = True

		with col2:
			if show_pmf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.arange(0, max_k + 1)
				pmf = stats.poisson.pmf(x, lambda_val)
				ax.bar(x, pmf, alpha=0.7)
				ax.set_xlabel("Number of events (k)")
				ax.set_ylabel("P(X = k)")
				ax.set_title(f"Poisson PMF (λ={lambda_val})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

			if show_cdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.arange(0, max_k + 1)
				cdf = stats.poisson.cdf(x, lambda_val)
				ax.step(np.concatenate([[-1], x]), np.concatenate([[0], cdf]), where='post', alpha=0.7)
				ax.set_xlabel("Number of events (k)")
				ax.set_ylabel("P(X ≤ k)")
				ax.set_title(f"Poisson CDF (λ={lambda_val})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

		# Animation
		if 'show_poisson_animation' in st.session_state and st.session_state.show_poisson_animation:
			st.markdown("<h4>Animated Parameter Changes</h4>", unsafe_allow_html=True)

			fig = plt.figure(figsize=(10, 6))
			lambda_values = np.linspace(0.5, 15, 10)

			# Create data for animation
			data = []
			for lambda_val in lambda_values:
				x = np.arange(0, max_k + 1)
				y = stats.poisson.pmf(x, lambda_val)
				df = pd.DataFrame({'x': x, 'y': y, 'lambda': [f"λ={lambda_val:.1f}"] * len(x)})
				data.append(df)

			df = pd.concat(data)

			# Create Altair chart with animation
			chart = alt.Chart(df).mark_bar().encode(
				x=alt.X('x:O', title='Number of events (k)'),
				y=alt.Y('y:Q', title='P(X = k)'),
				color=alt.value('#1f77b4')
			).properties(
				width=600,
				height=400,
				title='Poisson PMF with varying λ'
			).facet(
				facet=alt.Facet('lambda:N', title='Parameter value')
			)

			st.altair_chart(chart)

	elif selected_distribution == "Hypergeometric":
		# Hypergeometric distribution
		create_distribution_info(
			name="Hypergeometric Distribution",
			formula=r"P(X = k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}, \max(0, n+K-N) \leq k \leq \min(n, K)",
			parameters={
				"N": "Population size (N > 0, integer)",
				"K": "Number of success states in the population (0 ≤ K ≤ N, integer)",
				"n": "Number of draws (0 ≤ n ≤ N, integer)"
			},
			properties={
				"Mean": "n·K/N",
				"Variance": "n·K/N·(1-K/N)·(N-n)/(N-1)",
				"Skewness": "[(N-2K)(N-1)^(1/2)(N-2n)]/[(N-2)·(nK(N-K)(N-n))^(1/2)]",
				"Kurtosis": "Complex expression (depends on all parameters)",
				"MGF": "Complex expression"
			},
			examples=[
				"Drawing cards from a deck without replacement",
				"Quality control sampling from a batch of products",
				"Number of defective items in a sample drawn without replacement",
				"Number of white balls drawn from an urn containing white and black balls"
			]
		)

		# Interactive visualization
		st.markdown("<h4>Interactive Visualization</h4>", unsafe_allow_html=True)

		col1, col2 = st.columns([1, 3])

		with col1:
			N = st.slider("Population size (N)", 1, 100, 50, 1)
			K = st.slider("Number of success states (K)", 0, N, 20, 1)
			n = st.slider("Number of draws (n)", 1, N, 10, 1)

			show_pmf = st.checkbox("Show PMF", value=True)
			show_cdf = st.checkbox("Show CDF", value=True)

			if st.button("Show Animation"):
				st.session_state.show_hypergeom_animation = True

		with col2:
			if show_pmf:
				fig, ax = plt.subplots(figsize=(10, 5))
				k_min = max(0, n + K - N)
				k_max = min(n, K)
				x = np.arange(k_min, k_max + 1)
				pmf = stats.hypergeom.pmf(x, N, K, n)
				ax.bar(x, pmf, alpha=0.7)
				ax.set_xlabel("Number of successes (k)")
				ax.set_ylabel("P(X = k)")
				ax.set_title(f"Hypergeometric PMF (N={N}, K={K}, n={n})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

			if show_cdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				k_min = max(0, n + K - N)
				k_max = min(n, K)
				x = np.arange(k_min, k_max + 1)
				cdf = stats.hypergeom.cdf(x, N, K, n)
				ax.step(np.concatenate([[k_min - 1], x]), np.concatenate([[0], cdf]), where='post', alpha=0.7)
				ax.set_xlabel("Number of successes (k)")
				ax.set_ylabel("P(X ≤ k)")
				ax.set_title(f"Hypergeometric CDF (N={N}, K={K}, n={n})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

		# Animation
		if 'show_hypergeom_animation' in st.session_state and st.session_state.show_hypergeom_animation:
			st.markdown("<h4>Animated Parameter Changes</h4>", unsafe_allow_html=True)

			animation_type = st.radio("Choose Animation Type", ["Varying K", "Varying n"])

			if animation_type == "Varying K":
				fig = plt.figure(figsize=(10, 6))
				K_values = np.linspace(5, N - 5, 10, dtype=int)

				# Create data for animation
				data = []
				for K_val in K_values:
					k_min = max(0, n + K_val - N)
					k_max = min(n, K_val)
					x = np.arange(k_min, k_max + 1)
					y = stats.hypergeom.pmf(x, N, K_val, n)
					df = pd.DataFrame({'x': x, 'y': y, 'K': [f"K={K_val}"] * len(x)})
					data.append(df)

				df = pd.concat(data)

				# Create Altair chart with animation
				chart = alt.Chart(df).mark_bar().encode(
					x=alt.X('x:O', title='Number of successes (k)'),
					y=alt.Y('y:Q', title='P(X = k)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title=f'Hypergeometric PMF with N={N}, n={n}, and varying K'
				).facet(
					facet=alt.Facet('K:N', title='Parameter value')
				)

				st.altair_chart(chart)

			else:  # Varying n
				fig = plt.figure(figsize=(10, 6))
				n_values = np.linspace(5, min(N - 5, 30), 6, dtype=int)

				# Create data for animation
				data = []
				for n_val in n_values:
					k_min = max(0, n_val + K - N)
					k_max = min(n_val, K)
					x = np.arange(k_min, k_max + 1)
					y = stats.hypergeom.pmf(x, N, K, n_val)
					df = pd.DataFrame({'x': x, 'y': y, 'n': [f"n={n_val}"] * len(x)})
					data.append(df)

				df = pd.concat(data)

				# Create Altair chart with animation
				chart = alt.Chart(df).mark_bar().encode(
					x=alt.X('x:O', title='Number of successes (k)'),
					y=alt.Y('y:Q', title='P(X = k)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title=f'Hypergeometric PMF with N={N}, K={K}, and varying n'
				).facet(
					facet=alt.Facet('n:N', title='Parameter value')
				)

				st.altair_chart(chart)

# Continuous Distributions page
elif main_selection == "Continuous Distributions":
	st.markdown("<h2 class='sub-header'>Continuous Probability Distributions</h2>", unsafe_allow_html=True)

	st.markdown("""
    <div class="card">
        <p>Continuous probability distributions describe random variables that can take on an uncountable infinite number of values.
        The probability of the random variable falling within a particular range of values is given by the integral of the variable's 
        probability density function over that range.</p>
    </div>
    """, unsafe_allow_html=True)

	# Distribution selector
	distribution_options = ["Uniform", "Normal", "Exponential", "Gamma", "Beta", "t-distribution", "Chi-square",
							"F-distribution"]
	selected_distribution = st.selectbox("Select a Continuous Distribution", distribution_options)

	# Render the selected distribution
	if selected_distribution == "Uniform":
		# Uniform distribution
		create_distribution_info(
			name="Uniform Distribution",
			formula=r"f(x) = \frac{1}{b - a}, a \leq x \leq b",
			parameters={
				"a": "Lower bound (minimum value)",
				"b": "Upper bound (maximum value), b > a"
			},
			properties={
				"Mean": "(a + b)/2",
				"Variance": "(b - a)²/12",
				"Skewness": "0 (symmetric)",
				"Kurtosis": "-6/5 (platykurtic)",
				"MGF": "(e^(bt) - e^(at))/(t(b-a)), t ≠ 0"
			},
			examples=[
				"Random number generation in a given range",
				"Arrival time within a fixed time window",
				"Position of a randomly broken stick",
				"Error in measurements with a given tolerance"
			]
		)

		# Interactive visualization
		st.markdown("<h4>Interactive Visualization</h4>", unsafe_allow_html=True)

		col1, col2 = st.columns([1, 3])

		with col1:
			a = st.slider("Lower bound (a)", -10.0, 10.0, 0.0, 0.1)
			b = st.slider("Upper bound (b)", a + 0.1, 15.0, a + 5.0, 0.1)

			show_pdf = st.checkbox("Show PDF", value=True)
			show_cdf = st.checkbox("Show CDF", value=True)

			if st.button("Show Animation"):
				st.session_state.show_uniform_animation = True

		with col2:
			if show_pdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.linspace(a - 2, b + 2, 1000)
				pdf = np.where((x >= a) & (x <= b), 1 / (b - a), 0)
				ax.plot(x, pdf, alpha=0.7)
				ax.fill_between(x, pdf, alpha=0.3)
				ax.set_xlabel("x")
				ax.set_ylabel("f(x)")
				ax.set_title(f"Uniform PDF (a={a}, b={b})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

			if show_cdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.linspace(a - 2, b + 2, 1000)
				cdf = np.where(x < a, 0, np.where(x <= b, (x - a) / (b - a), 1))
				ax.plot(x, cdf, alpha=0.7)
				ax.set_xlabel("x")
				ax.set_ylabel("F(x)")
				ax.set_title(f"Uniform CDF (a={a}, b={b})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

		# Animation
		if 'show_uniform_animation' in st.session_state and st.session_state.show_uniform_animation:
			st.markdown("<h4>Animated Parameter Changes</h4>", unsafe_allow_html=True)

			animation_type = st.radio("Choose Animation Type", ["Varying width", "Varying position"])

			if animation_type == "Varying width":
				width_values = np.linspace(1, 10, 10)

				# Create data frames for each width value
				data = []
				x_range = np.linspace(-6, 15, 500)

				for width in width_values:
					local_a = 0
					local_b = local_a + width
					pdf = np.where((x_range >= local_a) & (x_range <= local_b), 1 / width, 0)

					for i, x_val in enumerate(x_range):
						data.append({
							'x': x_val,
							'pdf': pdf[i],
							'width': f"width={width:.1f}"
						})

				df = pd.DataFrame(data)

				# Create chart
				chart = alt.Chart(df).mark_area(opacity=0.6).encode(
					x=alt.X('x:Q', title='x'),
					y=alt.Y('pdf:Q', title='f(x)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title='Uniform PDF with Varying Width (a=0)'
				).facet(
					facet=alt.Facet('width:N', title='Parameter value')
				)

				st.altair_chart(chart)

			else:  # Varying position
				position_values = np.linspace(-5, 5, 11)

				# Create data frames for each position value
				data = []
				x_range = np.linspace(-8, 8, 500)
				width = 3

				for position in position_values:
					local_a = position
					local_b = local_a + width
					pdf = np.where((x_range >= local_a) & (x_range <= local_b), 1 / width, 0)

					for i, x_val in enumerate(x_range):
						data.append({
							'x': x_val,
							'pdf': pdf[i],
							'position': f"a={position:.1f}, b={local_b:.1f}"
						})

				df = pd.DataFrame(data)

				# Create chart
				chart = alt.Chart(df).mark_area(opacity=0.6).encode(
					x=alt.X('x:Q', title='x'),
					y=alt.Y('pdf:Q', title='f(x)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title=f'Uniform PDF with Varying Position (width={width})'
				).facet(
					facet=alt.Facet('position:N', title='Parameter value')
				)

				st.altair_chart(chart)

	elif selected_distribution == "Normal":
		# Normal distribution
		create_distribution_info(
			name="Normal (Gaussian) Distribution",
			formula=r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}, -\infty < x < \infty",
			parameters={
				"μ": "Mean (location parameter), -∞ < μ < ∞",
				"σ": "Standard deviation (scale parameter), σ > 0"
			},
			properties={
				"Mean": "μ",
				"Variance": "σ²",
				"Skewness": "0 (symmetric)",
				"Kurtosis": "0 (mesokurtic)",
				"MGF": "exp(μt + (σ²t²)/2)"
			},
			examples=[
				"Heights of people in a population",
				"Measurement errors",
				"IQ scores",
				"Stock price returns over short intervals"
			]
		)

		# Interactive visualization
		st.markdown("<h4>Interactive Visualization</h4>", unsafe_allow_html=True)

		col1, col2 = st.columns([1, 3])

		with col1:
			mu = st.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1)
			sigma = st.slider("Standard deviation (σ)", 0.1, 5.0, 1.0, 0.1)

			show_pdf = st.checkbox("Show PDF", value=True)
			show_cdf = st.checkbox("Show CDF", value=True)

			if st.button("Show Animation"):
				st.session_state.show_normal_animation = True

		with col2:
			if show_pdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
				pdf = stats.norm.pdf(x, mu, sigma)
				ax.plot(x, pdf, alpha=0.7)
				ax.fill_between(x, pdf, alpha=0.3)

				# Mark the mean and standard deviations
				ax.axvline(x=mu, color='red', linestyle='--', alpha=0.5, label=f"Mean (μ={mu})")
				ax.axvline(x=mu + sigma, color='green', linestyle='--', alpha=0.5, label=f"μ+σ")
				ax.axvline(x=mu - sigma, color='green', linestyle='--', alpha=0.5, label=f"μ-σ")

				ax.set_xlabel("x")
				ax.set_ylabel("f(x)")
				ax.set_title(f"Normal PDF (μ={mu}, σ={sigma})")
				ax.legend()
				ax.grid(alpha=0.3)
				st.pyplot(fig)

			if show_cdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
				cdf = stats.norm.cdf(x, mu, sigma)
				ax.plot(x, cdf, alpha=0.7)
				ax.set_xlabel("x")
				ax.set_ylabel("F(x)")
				ax.set_title(f"Normal CDF (μ={mu}, σ={sigma})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

		# Additional info about standard normal and normal properties
		with st.expander("Standard Normal Distribution", expanded=False):
			st.markdown("""
            <div class="note">
                <p>The standard normal distribution is a special case of the normal distribution where μ = 0 and σ = 1.
                It is often denoted as Z ~ N(0, 1).</p>
                <p>Any normal random variable X ~ N(μ, σ²) can be standardized to a Z-score using the transformation:</p>
                <div class="formula">Z = (X - μ) / σ</div>
                <p>This transformation is useful for probability calculations and hypothesis testing.</p>
            </div>
            """, unsafe_allow_html=True)

			# Show the standard normal distribution
			fig, ax = plt.subplots(figsize=(10, 5))
			x = np.linspace(-4, 4, 1000)
			pdf = stats.norm.pdf(x, 0, 1)
			ax.plot(x, pdf, alpha=0.7)
			ax.fill_between(x, pdf, alpha=0.3)

			# Mark important areas
			ax.fill_between(x, pdf, where=(x >= -1) & (x <= 1), color='green', alpha=0.3)
			ax.fill_between(x, pdf, where=(x >= -2) & (x <= 2), color='yellow', alpha=0.2)
			ax.fill_between(x, pdf, where=(x >= -3) & (x <= 3), color='orange', alpha=0.1)

			ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
			ax.axvline(x=1, color='green', linestyle='--', alpha=0.5)
			ax.axvline(x=-1, color='green', linestyle='--', alpha=0.5)
			ax.axvline(x=2, color='yellow', linestyle='--', alpha=0.5)
			ax.axvline(x=-2, color='yellow', linestyle='--', alpha=0.5)
			ax.axvline(x=3, color='orange', linestyle='--', alpha=0.5)
			ax.axvline(x=-3, color='orange', linestyle='--', alpha=0.5)

			ax.set_xlabel("z")
			ax.set_ylabel("f(z)")
			ax.set_title("Standard Normal Distribution (Z ~ N(0, 1))")
			ax.text(0.1, 0.3, "68.3% within ±1σ", ha='left', fontsize=10)
			ax.text(0.1, 0.2, "95.4% within ±2σ", ha='left', fontsize=10)
			ax.text(0.1, 0.1, "99.7% within ±3σ", ha='left', fontsize=10)
			ax.grid(alpha=0.3)
			st.pyplot(fig)

		# Animation
		if 'show_normal_animation' in st.session_state and st.session_state.show_normal_animation:
			st.markdown("<h4>Animated Parameter Changes</h4>", unsafe_allow_html=True)

			animation_type = st.radio("Choose Animation Type", ["Varying μ", "Varying σ"])

			if animation_type == "Varying μ":
				mu_values = np.linspace(-3, 3, 7)

				# Create data frames for each mu value
				data = []
				x_range = np.linspace(-6, 6, 500)

				for mu_val in mu_values:
					pdf = stats.norm.pdf(x_range, mu_val, sigma)

					for i, x_val in enumerate(x_range):
						data.append({
							'x': x_val,
							'pdf': pdf[i],
							'mu': f"μ={mu_val:.1f}, σ={sigma}"
						})

				df = pd.DataFrame(data)

				# Create chart
				chart = alt.Chart(df).mark_area(opacity=0.6).encode(
					x=alt.X('x:Q', title='x'),
					y=alt.Y('pdf:Q', title='f(x)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title='Normal PDF with Varying Mean'
				).facet(
					facet=alt.Facet('mu:N', title='Parameter value')
				)

				st.altair_chart(chart)

			else:  # Varying sigma
				sigma_values = np.linspace(0.5, 2.5, 5)

				# Create data frames for each sigma value
				data = []
				x_range = np.linspace(-6, 6, 500)

				for sigma_val in sigma_values:
					pdf = stats.norm.pdf(x_range, mu, sigma_val)

					for i, x_val in enumerate(x_range):
						data.append({
							'x': x_val,
							'pdf': pdf[i],
							'sigma': f"μ={mu}, σ={sigma_val:.1f}"
						})

				df = pd.DataFrame(data)

				# Create chart
				chart = alt.Chart(df).mark_area(opacity=0.6).encode(
					x=alt.X('x:Q', title='x'),
					y=alt.Y('pdf:Q', title='f(x)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title='Normal PDF with Varying Standard Deviation'
				).facet(
					facet=alt.Facet('sigma:N', title='Parameter value')
				)

				st.altair_chart(chart)

	elif selected_distribution == "Exponential":
		# Exponential distribution
		create_distribution_info(
			name="Exponential Distribution",
			formula=r"f(x) = \lambda e^{-\lambda x}, x \geq 0",
			parameters={
				"λ": "Rate parameter (λ > 0), the average number of events per unit time"
			},
			properties={
				"Mean": "1/λ",
				"Variance": "1/λ²",
				"Skewness": "2",
				"Kurtosis": "6",
				"MGF": "λ/(λ-t) for t < λ"
			},
			examples=[
				"Time between events in a Poisson process",
				"Time until radioactive decay",
				"Time until device failure",
				"Time between customers arriving at a service center"
			]
		)

		# Interactive visualization
		st.markdown("<h4>Interactive Visualization</h4>", unsafe_allow_html=True)

		col1, col2 = st.columns([1, 3])

		with col1:
			lambda_val = st.slider("Rate parameter (λ)", 0.1, 5.0, 1.0, 0.1)

			show_pdf = st.checkbox("Show PDF", value=True)
			show_cdf = st.checkbox("Show CDF", value=True)

			if st.button("Show Animation"):
				st.session_state.show_exponential_animation = True

		with col2:
			if show_pdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.linspace(0, 5 / lambda_val, 1000)
				pdf = stats.expon.pdf(x, scale=1 / lambda_val)
				ax.plot(x, pdf, alpha=0.7)
				ax.fill_between(x, pdf, alpha=0.3)

				# Mark the mean
				mean = 1 / lambda_val
				ax.axvline(x=mean, color='red', linestyle='--', alpha=0.5, label=f"Mean = 1/λ = {mean:.2f}")

				ax.set_xlabel("x")
				ax.set_ylabel("f(x)")
				ax.set_title(f"Exponential PDF (λ={lambda_val})")
				ax.legend()
				ax.grid(alpha=0.3)
				st.pyplot(fig)

			if show_cdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.linspace(0, 5 / lambda_val, 1000)
				cdf = stats.expon.cdf(x, scale=1 / lambda_val)
				ax.plot(x, cdf, alpha=0.7)
				ax.set_xlabel("x")
				ax.set_ylabel("F(x)")
				ax.set_title(f"Exponential CDF (λ={lambda_val})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

		# Additional info about memoryless property
		with st.expander("Memoryless Property", expanded=False):
			st.markdown("""
            <div class="note">
                <p>The exponential distribution is the only continuous probability distribution that is <b>memoryless</b>. 
                This means that the probability of waiting an additional time t is independent of how long you have already waited.</p>
                <p>Mathematically, for any s, t ≥ 0:</p>
                <div class="formula">P(X > s + t | X > s) = P(X > t)</div>
                <p>This property makes the exponential distribution especially useful for modeling random waiting times for which 
                the probability of waiting an additional period of time is independent of how much time has already elapsed.</p>
            </div>
            """, unsafe_allow_html=True)

		# Animation
		if 'show_exponential_animation' in st.session_state and st.session_state.show_exponential_animation:
			st.markdown("<h4>Animated Parameter Changes</h4>", unsafe_allow_html=True)

			lambda_values = np.linspace(0.5, 2.5, 5)

			# Create data frames for each lambda value
			data = []
			x_range = np.linspace(0, 6, 500)

			for lambda_val in lambda_values:
				pdf = stats.expon.pdf(x_range, scale=1 / lambda_val)

				for i, x_val in enumerate(x_range):
					data.append({
						'x': x_val,
						'pdf': pdf[i],
						'lambda': f"λ={lambda_val:.1f}, mean={1 / lambda_val:.1f}"
					})

			df = pd.DataFrame(data)

			# Create chart
			chart = alt.Chart(df).mark_area(opacity=0.6).encode(
				x=alt.X('x:Q', title='x'),
				y=alt.Y('pdf:Q', title='f(x)'),
				color=alt.value('#1f77b4')
			).properties(
				width=600,
				height=400,
				title='Exponential PDF with Varying λ'
			).facet(
				facet=alt.Facet('lambda:N', title='Parameter value')
			)

			st.altair_chart(chart)

	elif selected_distribution == "Gamma":
		# Gamma distribution
		create_distribution_info(
			name="Gamma Distribution",
			formula=r"f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, x > 0",
			parameters={
				"α": "Shape parameter (α > 0)",
				"β": "Rate parameter (β > 0)"
			},
			properties={
				"Mean": "α/β",
				"Variance": "α/β²",
				"Skewness": "2/√α",
				"Kurtosis": "6/α",
				"MGF": "(1 - t/β)^(-α) for t < β"
			},
			examples=[
				"Waiting time for α events in a Poisson process",
				"Time to failure for devices with multiple components",
				"Amount of rainfall over a time period",
				"Insurance claim sizes"
			]
		)

		# Interactive visualization
		st.markdown("<h4>Interactive Visualization</h4>", unsafe_allow_html=True)

		col1, col2 = st.columns([1, 3])

		with col1:
			alpha = st.slider("Shape parameter (α)", 0.1, 10.0, 2.0, 0.1)
			beta = st.slider("Rate parameter (β)", 0.1, 5.0, 1.0, 0.1)

			show_pdf = st.checkbox("Show PDF", value=True)
			show_cdf = st.checkbox("Show CDF", value=True)

			if st.button("Show Animation"):
				st.session_state.show_gamma_animation = True

		with col2:
			if show_pdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				if alpha < 1:
					x = np.linspace(0.001, 10 / beta, 1000)  # Avoid x=0 for α<1
				else:
					x = np.linspace(0, 10 / beta, 1000)
				pdf = stats.gamma.pdf(x, alpha, scale=1 / beta)
				ax.plot(x, pdf, alpha=0.7)
				ax.fill_between(x, pdf, alpha=0.3)

				# Mark the mean
				mean = alpha / beta
				ax.axvline(x=mean, color='red', linestyle='--', alpha=0.5, label=f"Mean = α/β = {mean:.2f}")

				ax.set_xlabel("x")
				ax.set_ylabel("f(x)")
				ax.set_title(f"Gamma PDF (α={alpha}, β={beta})")
				ax.legend()
				ax.grid(alpha=0.3)
				st.pyplot(fig)

			if show_cdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.linspace(0, 10 / beta, 1000)
				cdf = stats.gamma.cdf(x, alpha, scale=1 / beta)
				ax.plot(x, cdf, alpha=0.7)
				ax.set_xlabel("x")
				ax.set_ylabel("F(x)")
				ax.set_title(f"Gamma CDF (α={alpha}, β={beta})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

		# Additional info about gamma and exponential relationship
		with st.expander("Relationship with Other Distributions", expanded=False):
			st.markdown("""
            <div class="note">
                <p>The Gamma distribution has important relationships with several other distributions:</p>
                <ul>
                    <li><b>Exponential Distribution</b>: When α = 1, the Gamma distribution is equivalent to the Exponential distribution with rate parameter β.</li>
                    <li><b>Chi-square Distribution</b>: A Chi-square distribution with k degrees of freedom is a special case of the Gamma distribution with α = k/2 and β = 1/2.</li>
                    <li><b>Erlang Distribution</b>: When α is a positive integer, the Gamma distribution is known as the Erlang distribution, which describes the waiting time until the α-th event in a Poisson process.</li>
                </ul>
                <p>If X₁, X₂, ..., Xₙ are independent, identically distributed Exponential random variables with rate parameter β, then their sum follows a Gamma distribution with shape parameter α = n and rate parameter β.</p>
            </div>
            """, unsafe_allow_html=True)

		# Animation
		if 'show_gamma_animation' in st.session_state and st.session_state.show_gamma_animation:
			st.markdown("<h4>Animated Parameter Changes</h4>", unsafe_allow_html=True)

			animation_type = st.radio("Choose Animation Type", ["Varying α (shape)", "Varying β (rate)"])

			if animation_type == "Varying α (shape)":
				alpha_values = np.linspace(0.5, 5, 5)

				# Create data frames for each alpha value
				data = []

				for alpha_val in alpha_values:
					if alpha_val < 1:
						x_range = np.linspace(0.001, 10 / beta, 500)  # Avoid x=0 for α<1
					else:
						x_range = np.linspace(0, 10 / beta, 500)

					pdf = stats.gamma.pdf(x_range, alpha_val, scale=1 / beta)

					for i, x_val in enumerate(x_range):
						data.append({
							'x': x_val,
							'pdf': pdf[i],
							'alpha': f"α={alpha_val:.1f}, β={beta}"
						})

				df = pd.DataFrame(data)

				# Create chart
				chart = alt.Chart(df).mark_area(opacity=0.6).encode(
					x=alt.X('x:Q', title='x'),
					y=alt.Y('pdf:Q', title='f(x)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title='Gamma PDF with Varying Shape Parameter (α)'
				).facet(
					facet=alt.Facet('alpha:N', title='Parameter value')
				)

				st.altair_chart(chart)

			else:  # Varying beta
				beta_values = np.linspace(0.5, 2.5, 5)

				# Create data frames for each beta value
				data = []

				for beta_val in beta_values:
					if alpha < 1:
						x_range = np.linspace(0.001, 10 / beta_val, 500)  # Avoid x=0 for α<1
					else:
						x_range = np.linspace(0, 10 / beta_val, 500)

					pdf = stats.gamma.pdf(x_range, alpha, scale=1 / beta_val)

					for i, x_val in enumerate(x_range):
						data.append({
							'x': x_val,
							'pdf': pdf[i],
							'beta': f"α={alpha}, β={beta_val:.1f}"
						})

				df = pd.DataFrame(data)

				# Create chart
				chart = alt.Chart(df).mark_area(opacity=0.6).encode(
					x=alt.X('x:Q', title='x'),
					y=alt.Y('pdf:Q', title='f(x)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title='Gamma PDF with Varying Rate Parameter (β)'
				).facet(
					facet=alt.Facet('beta:N', title='Parameter value')
				)

				st.altair_chart(chart)

	elif selected_distribution == "Beta":
		# Beta distribution
		create_distribution_info(
			name="Beta Distribution",
			formula=r"f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, 0 \leq x \leq 1",
			parameters={
				"α": "Shape parameter (α > 0)",
				"β": "Shape parameter (β > 0)"
			},
			properties={
				"Mean": "α/(α+β)",
				"Variance": "αβ/((α+β)²(α+β+1))",
				"Skewness": "2(β-α)√(α+β+1)/((α+β+2)√(αβ))",
				"Kurtosis": "Complex expression (depends on both parameters)",
				"MGF": "No simple closed form"
			},
			examples=[
				"Probability of success given past successes and failures",
				"Modeling random proportions or percentages",
				"Distributions of order statistics",
				"Uncertainty about a probability in Bayesian statistics"
			]
		)

		# Interactive visualization
		st.markdown("<h4>Interactive Visualization</h4>", unsafe_allow_html=True)

		col1, col2 = st.columns([1, 3])

		with col1:
			alpha = st.slider("Shape parameter (α)", 0.1, 10.0, 2.0, 0.1)
			beta = st.slider("Shape parameter (β)", 0.1, 10.0, 2.0, 0.1)

			show_pdf = st.checkbox("Show PDF", value=True)
			show_cdf = st.checkbox("Show CDF", value=True)

			if st.button("Show Animation"):
				st.session_state.show_beta_animation = True

		with col2:
			if show_pdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.linspace(0, 1, 1000)
				pdf = stats.beta.pdf(x, alpha, beta)
				ax.plot(x, pdf, alpha=0.7)
				ax.fill_between(x, pdf, alpha=0.3)

				# Mark the mean
				mean = alpha / (alpha + beta)
				ax.axvline(x=mean, color='red', linestyle='--', alpha=0.5, label=f"Mean = α/(α+β) = {mean:.2f}")

				ax.set_xlabel("x")
				ax.set_ylabel("f(x)")
				ax.set_title(f"Beta PDF (α={alpha}, β={beta})")
				ax.legend()
				ax.grid(alpha=0.3)
				st.pyplot(fig)

			if show_cdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.linspace(0, 1, 1000)
				cdf = stats.beta.cdf(x, alpha, beta)
				ax.plot(x, cdf, alpha=0.7)
				ax.set_xlabel("x")
				ax.set_ylabel("F(x)")
				ax.set_title(f"Beta CDF (α={alpha}, β={beta})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

		# Additional info about beta distribution
		with st.expander("Special Cases and Applications", expanded=False):
			st.markdown("""
            <div class="note">
                <p>The Beta distribution is extremely flexible and can take on many different shapes depending on its parameters:</p>
                <ul>
                    <li><b>α = β = 1</b>: Uniform distribution over [0, 1]</li>
                    <li><b>α < 1, β < 1</b>: U-shaped (bimodal at 0 and 1)</li>
                    <li><b>α > 1, β > 1</b>: Unimodal</li>
                    <li><b>α = β > 1</b>: Symmetric around 0.5</li>
                    <li><b>α < 1, β ≥ 1</b>: J-shaped, decreasing</li>
                    <li><b>α ≥ 1, β < 1</b>: J-shaped, increasing</li>
                </ul>
                <p>The Beta distribution is commonly used in Bayesian statistics as a conjugate prior for the parameter of a Bernoulli, 
                binomial, or geometric distribution. This makes it particularly useful for modeling evolving probabilities 
                as more data is observed.</p>
            </div>
            """, unsafe_allow_html=True)

		# Animation
		if 'show_beta_animation' in st.session_state and st.session_state.show_beta_animation:
			st.markdown("<h4>Animated Parameter Changes</h4>", unsafe_allow_html=True)

			animation_type = st.radio("Choose Animation Type", ["Varying α", "Varying β", "Varying both (α=β)"])

			if animation_type == "Varying α":
				alpha_values = np.linspace(0.5, 5, 10)

				# Create data frames for each alpha value
				data = []
				x_range = np.linspace(0, 1, 200)

				for alpha_val in alpha_values:
					pdf = stats.beta.pdf(x_range, alpha_val, beta)

					for i, x_val in enumerate(x_range):
						data.append({
							'x': x_val,
							'pdf': pdf[i],
							'alpha': f"α={alpha_val:.1f}, β={beta}"
						})

				df = pd.DataFrame(data)

				# Create chart
				chart = alt.Chart(df).mark_area(opacity=0.6).encode(
					x=alt.X('x:Q', title='x'),
					y=alt.Y('pdf:Q', title='f(x)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title=f'Beta PDF with Varying α (fixed β={beta})'
				).facet(
					facet=alt.Facet('alpha:N', title='Parameter value')
				)

				st.altair_chart(chart)

			elif animation_type == "Varying β":
				beta_values = np.linspace(0.5, 5, 10)

				# Create data frames for each beta value
				data = []
				x_range = np.linspace(0, 1, 200)

				for beta_val in beta_values:
					pdf = stats.beta.pdf(x_range, alpha, beta_val)

					for i, x_val in enumerate(x_range):
						data.append({
							'x': x_val,
							'pdf': pdf[i],
							'beta': f"α={alpha}, β={beta_val:.1f}"
						})

				df = pd.DataFrame(data)

				# Create chart
				chart = alt.Chart(df).mark_area(opacity=0.6).encode(
					x=alt.X('x:Q', title='x'),
					y=alt.Y('pdf:Q', title='f(x)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title=f'Beta PDF with Varying β (fixed α={alpha})'
				).facet(
					facet=alt.Facet('beta:N', title='Parameter value')
				)

				st.altair_chart(chart)

			else:  # Varying both (α=β)
				ab_values = np.linspace(0.5, 5, 10)

				# Create data frames for each alpha=beta value
				data = []
				x_range = np.linspace(0, 1, 200)

				for ab_val in ab_values:
					pdf = stats.beta.pdf(x_range, ab_val, ab_val)

					for i, x_val in enumerate(x_range):
						data.append({
							'x': x_val,
							'pdf': pdf[i],
							'ab': f"α=β={ab_val:.1f}"
						})

				df = pd.DataFrame(data)

				# Create chart
				chart = alt.Chart(df).mark_area(opacity=0.6).encode(
					x=alt.X('x:Q', title='x'),
					y=alt.Y('pdf:Q', title='f(x)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title='Beta PDF with Varying α=β (Symmetric Cases)'
				).facet(
					facet=alt.Facet('ab:N', title='Parameter value')
				)

				st.altair_chart(chart)

	elif selected_distribution == "t-distribution":
		# t-distribution
		create_distribution_info(
			name="Student's t-Distribution",
			formula=r"f(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\Gamma(\frac{\nu}{2})}\left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}, -\infty < x < \infty",
			parameters={
				"ν": "Degrees of freedom (ν > 0)"
			},
			properties={
				"Mean": "0 for ν > 1, otherwise undefined",
				"Variance": "ν/(ν-2) for ν > 2, ∞ for 1 < ν ≤ 2, otherwise undefined",
				"Skewness": "0 for ν > 3, otherwise undefined",
				"Kurtosis": "6/(ν-4) for ν > 4, otherwise undefined",
				"MGF": "Does not exist in closed form"
			},
			examples=[
				"Statistical inference when the sample size is small",
				"Estimating the mean of a normally distributed population",
				"Constructing confidence intervals with small samples",
				"Hypothesis testing with the t-test"
			]
		)

		# Interactive visualization
		st.markdown("<h4>Interactive Visualization</h4>", unsafe_allow_html=True)

		col1, col2 = st.columns([1, 3])

		with col1:
			df = st.slider("Degrees of freedom (ν)", 1, 30, 5, 1)

			show_pdf = st.checkbox("Show PDF", value=True)
			show_cdf = st.checkbox("Show CDF", value=True)
			show_comparison = st.checkbox("Compare with Normal", value=True)

			if st.button("Show Animation"):
				st.session_state.show_t_animation = True

		with col2:
			if show_pdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.linspace(-5, 5, 1000)
				t_pdf = stats.t.pdf(x, df)
				ax.plot(x, t_pdf, alpha=0.7, label=f"t-distribution (ν={df})")
				ax.fill_between(x, t_pdf, alpha=0.3)

				if show_comparison:
					# Add normal distribution for comparison
					norm_pdf = stats.norm.pdf(x, 0, 1)
					ax.plot(x, norm_pdf, 'r--', alpha=0.7, label="Standard Normal")

				ax.set_xlabel("x")
				ax.set_ylabel("f(x)")
				ax.set_title(f"Student's t-Distribution PDF (ν={df})")
				ax.legend()
				ax.grid(alpha=0.3)
				st.pyplot(fig)

			if show_cdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.linspace(-5, 5, 1000)
				t_cdf = stats.t.cdf(x, df)
				ax.plot(x, t_cdf, alpha=0.7, label=f"t-distribution (ν={df})")

				if show_comparison:
					# Add normal distribution for comparison
					norm_cdf = stats.norm.cdf(x, 0, 1)
					ax.plot(x, norm_cdf, 'r--', alpha=0.7, label="Standard Normal")

				ax.set_xlabel("x")
				ax.set_ylabel("F(x)")
				ax.set_title(f"Student's t-Distribution CDF (ν={df})")
				ax.legend()
				ax.grid(alpha=0.3)
				st.pyplot(fig)

		# Additional info about t-distribution
		with st.expander("Relationship with Normal Distribution", expanded=False):
			st.markdown("""
            <div class="note">
                <p>The t-distribution approaches the standard normal distribution as the degrees of freedom (ν) increase:</p>
                <ul>
                    <li>For small ν, the t-distribution has heavier tails than the normal distribution, reflecting the additional uncertainty due to estimating the variance from a small sample.</li>
                    <li>As ν increases, the t-distribution converges to the standard normal distribution.</li>
                    <li>When ν > 30, the t-distribution is very close to the standard normal distribution for most practical purposes.</li>
                </ul>
                <p>The t-distribution is derived by taking the ratio of a standard normal random variable to the square root of a chi-square random variable divided by its degrees of freedom.</p>
                <p>If Z ~ N(0, 1) and V ~ χ²(ν) are independent, then:</p>
                <div class="formula">T = \frac{Z}{\sqrt{V/\nu}} ~ t(ν)</div>
            </div>
            """, unsafe_allow_html=True)

		# Animation
		if 'show_t_animation' in st.session_state and st.session_state.show_t_animation:
			st.markdown("<h4>Animated Parameter Changes</h4>", unsafe_allow_html=True)

			df_values = [1, 2, 3, 5, 10, 30]

			# Create data frames for each degrees of freedom value
			data = []
			x_range = np.linspace(-5, 5, 500)

			# Add normal distribution for comparison
			norm_pdf = stats.norm.pdf(x_range, 0, 1)
			for x_val in x_range:
				data.append({
					'x': x_val,
					'pdf': norm_pdf[x_range == x_val][0],
					'df': 'Normal'
				})

			for df_val in df_values:
				t_pdf = stats.t.pdf(x_range, df_val)

				for i, x_val in enumerate(x_range):
					data.append({
						'x': x_val,
						'pdf': t_pdf[i],
						'df': f"t (ν={df_val})"
					})

			df_data = pd.DataFrame(data)

			# Create chart
			t_chart = alt.Chart(df_data).mark_line().encode(
				x=alt.X('x:Q', title='x'),
				y=alt.Y('pdf:Q', title='f(x)'),
				color=alt.Color('df:N', title='Distribution'),
				strokeDash=alt.condition(
					alt.datum.df == 'Normal',
					alt.value([5, 5]),
					alt.value([0])
				)
			).properties(
				width=600,
				height=400,
				title="Comparison of t-distributions with Normal Distribution"
			)

			st.altair_chart(t_chart)

	elif selected_distribution == "Chi-square":
		# Chi-square distribution
		create_distribution_info(
			name="Chi-square Distribution",
			formula=r"f(x) = \frac{1}{2^{k/2}\Gamma(k/2)} x^{k/2-1} e^{-x/2}, x > 0",
			parameters={
				"k": "Degrees of freedom (k > 0, integer)"
			},
			properties={
				"Mean": "k",
				"Variance": "2k",
				"Skewness": "√(8/k)",
				"Kurtosis": "12/k",
				"MGF": "(1-2t)^(-k/2) for t < 1/2"
			},
			examples=[
				"Sum of squares of standard normal random variables",
				"Goodness-of-fit tests",
				"Testing independence in contingency tables",
				"Confidence intervals for variance in normal distributions"
			]
		)

		# Interactive visualization
		st.markdown("<h4>Interactive Visualization</h4>", unsafe_allow_html=True)

		col1, col2 = st.columns([1, 3])

		with col1:
			k = st.slider("Degrees of freedom (k)", 1, 30, 5, 1)

			show_pdf = st.checkbox("Show PDF", value=True)
			show_cdf = st.checkbox("Show CDF", value=True)

			if st.button("Show Animation"):
				st.session_state.show_chi2_animation = True

		with col2:
			if show_pdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.linspace(0.01, 3 * k, 1000)  # Avoid x=0 for k=1, 2
				pdf = stats.chi2.pdf(x, k)
				ax.plot(x, pdf, alpha=0.7)
				ax.fill_between(x, pdf, alpha=0.3)

				# Mark the mean
				ax.axvline(x=k, color='red', linestyle='--', alpha=0.5, label=f"Mean = k = {k}")

				ax.set_xlabel("x")
				ax.set_ylabel("f(x)")
				ax.set_title(f"Chi-square PDF (k={k})")
				ax.legend()
				ax.grid(alpha=0.3)
				st.pyplot(fig)

			if show_cdf:
				fig, ax = plt.subplots(figsize=(10, 5))
				x = np.linspace(0.01, 3 * k, 1000)
				cdf = stats.chi2.cdf(x, k)
				ax.plot(x, cdf, alpha=0.7)
				ax.set_xlabel("x")
				ax.set_ylabel("F(x)")
				ax.set_title(f"Chi-square CDF (k={k})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

		# Additional info about chi-square distribution
		with st.expander("Relationship with Other Distributions", expanded=False):
			st.markdown("""
            <div class="note">
                <p>The Chi-square distribution has important connections to several other distributions:</p>
                <ul>
                    <li><b>Normal Distribution</b>: If Z₁, Z₂, ..., Zₖ are independent standard normal random variables, then the sum of their squares, ∑Z²ᵢ, follows a chi-square distribution with k degrees of freedom.</li>
                    <li><b>Gamma Distribution</b>: A Chi-square distribution with k degrees of freedom is equivalent to a Gamma distribution with shape parameter α = k/2 and scale parameter θ = 2 (or rate parameter β = 1/2).</li>
                    <li><b>t-Distribution</b>: If Z ~ N(0, 1) and V ~ χ²(k) are independent, then Z/√(V/k) follows a t-distribution with k degrees of freedom.</li>
                    <li><b>F-Distribution</b>: If U ~ χ²(d₁) and V ~ χ²(d₂) are independent, then (U/d₁)/(V/d₂) follows an F-distribution with d₁ and d₂ degrees of freedom.</li>
                </ul>
                <p>The chi-square distribution is widely used in hypothesis testing, particularly for:</p>
                <ul>
                    <li>Testing goodness-of-fit of observed data to theoretical distributions</li>
                    <li>Testing independence between categorical variables</li>
                    <li>Testing homogeneity of proportions</li>
                    <li>Constructing confidence intervals for the variance of a normal distribution</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

		# Animation
		if 'show_chi2_animation' in st.session_state and st.session_state.show_chi2_animation:
			st.markdown("<h4>Animated Parameter Changes</h4>", unsafe_allow_html=True)

			k_values = [1, 2, 3, 5, 10, 20]

			# Create data frames for each degrees of freedom value
			data = []
			max_x = max(k_values) * 3
			x_range = np.linspace(0.01, max_x, 500)  # Avoid x=0 for k=1, 2

			for k_val in k_values:
				pdf = stats.chi2.pdf(x_range, k_val)

				for i, x_val in enumerate(x_range):
					data.append({
						'x': x_val,
						'pdf': pdf[i],
						'k': f"k={k_val}"
					})

			df = pd.DataFrame(data)

			# Create chart
			chart = alt.Chart(df).mark_area(opacity=0.6).encode(
				x=alt.X('x:Q', title='x'),
				y=alt.Y('pdf:Q', title='f(x)'),
				color=alt.value('#1f77b4')
			).properties(
				width=600,
				height=400,
				title='Chi-square PDF with Varying Degrees of Freedom (k)'
			).facet(
				facet=alt.Facet('k:N', title='Parameter value')
			)

			st.altair_chart(chart)

	elif selected_distribution == "F-distribution":
		# F-distribution
		create_distribution_info(
			name="F-Distribution",
			formula=r"f(x) = \frac{\sqrt{\frac{(d_1 x)^{d_1} \cdot d_2^{d_2}}{(d_1 x + d_2)^{d_1+d_2}}}}{\mathrm{B}\left(\frac{d_1}{2}, \frac{d_2}{2}\right) \cdot x}, x > 0",
			parameters={
				"d₁": "Numerator degrees of freedom (d₁ > 0, integer)",
				"d₂": "Denominator degrees of freedom (d₂ > 0, integer)"
			},
			properties={
				"Mean": "d₂/(d₂-2) for d₂ > 2, otherwise undefined",
				"Variance": "2d₂²(d₁+d₂-2)/(d₁(d₂-2)²(d₂-4)) for d₂ > 4, otherwise undefined",
				"Skewness": "Complex expression (defined for d₂ > 6)",
				"Kurtosis": "Complex expression (defined for d₂ > 8)",
				"MGF": "Does not exist in closed form"
			},
			examples=[
				"Analysis of variance (ANOVA)",
				"Testing equality of variances (F-test)",
				"Regression analysis",
				"Testing nested statistical models"
			]
		)

		# Interactive visualization
		st.markdown("<h4>Interactive Visualization</h4>", unsafe_allow_html=True)

		col1, col2 = st.columns([1, 3])

		with col1:
			d1 = st.slider("Numerator degrees of freedom (d₁)", 1, 30, 5, 1)
			d2 = st.slider("Denominator degrees of freedom (d₂)", 1, 30, 10, 1)

			show_pdf = st.checkbox("Show PDF", value=True)
			show_cdf = st.checkbox("Show CDF", value=True)

			if st.button("Show Animation"):
				st.session_state.show_f_animation = True

		with col2:
			if show_pdf:
				fig, ax = plt.subplots(figsize=(10, 5))

				# Calculate the range to show (depends on degrees of freedom)
				max_x = 5 if d1 > 2 else 10
				x = np.linspace(0.01, max_x, 1000)

				pdf = stats.f.pdf(x, d1, d2)
				ax.plot(x, pdf, alpha=0.7)
				ax.fill_between(x, pdf, alpha=0.3)

				# Mark the mean if it exists
				if d2 > 2:
					mean = d2 / (d2 - 2)
					if mean <= max_x:
						ax.axvline(x=mean, color='red', linestyle='--', alpha=0.5, label=f"Mean = {mean:.2f}")
						ax.legend()

				ax.set_xlabel("x")
				ax.set_ylabel("f(x)")
				ax.set_title(f"F-Distribution PDF (d₁={d1}, d₂={d2})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

			if show_cdf:
				fig, ax = plt.subplots(figsize=(10, 5))

				# Calculate the range to show (depends on degrees of freedom)
				max_x = 5 if d1 > 2 else 10
				x = np.linspace(0.01, max_x, 1000)

				cdf = stats.f.cdf(x, d1, d2)
				ax.plot(x, cdf, alpha=0.7)
				ax.set_xlabel("x")
				ax.set_ylabel("F(x)")
				ax.set_title(f"F-Distribution CDF (d₁={d1}, d₂={d2})")
				ax.grid(alpha=0.3)
				st.pyplot(fig)

		# Additional info about F-distribution
		with st.expander("Applications and Properties", expanded=False):
			st.markdown("""
            <div class="note">
                <p>The F-distribution has several important applications in statistical analysis:</p>
                <ul>
                    <li><b>Analysis of Variance (ANOVA)</b>: Used to test if the means of multiple groups are equal.</li>
                    <li><b>F-test for Equality of Variances</b>: Tests if two populations have the same variance.</li>
                    <li><b>Multiple Regression Analysis</b>: Tests the significance of a regression model or compares nested models.</li>
                </ul>
                <p>The F-distribution is related to the ratio of chi-square distributions:</p>
                <p>If U ~ χ²(d₁) and V ~ χ²(d₂) are independent, then:</p>
                <div class="formula">F = \frac{U/d_1}{V/d_2} ~ F(d_1, d_2)</div>
                <p>Special cases and properties:</p>
                <ul>
                    <li>F(1, d₂) is related to the square of a t-distribution with d₂ degrees of freedom.</li>
                    <li>If X ~ F(d₁, d₂), then 1/X ~ F(d₂, d₁) (reciprocal property).</li>
                    <li>As d₂ → ∞ with fixed d₁, the distribution approaches a scaled chi-square distribution.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

		# Animation
		if 'show_f_animation' in st.session_state and st.session_state.show_f_animation:
			st.markdown("<h4>Animated Parameter Changes</h4>", unsafe_allow_html=True)

			animation_type = st.radio("Choose Animation Type", ["Varying d₁", "Varying d₂"])

			if animation_type == "Varying d₁":
				d1_values = [1, 2, 5, 10, 20]

				# Create data frames for each d1 value
				data = []
				max_x = 5
				x_range = np.linspace(0.01, max_x, 500)

				for d1_val in d1_values:
					pdf = stats.f.pdf(x_range, d1_val, d2)

					for i, x_val in enumerate(x_range):
						data.append({
							'x': x_val,
							'pdf': pdf[i],
							'd1': f"d₁={d1_val}, d₂={d2}"
						})

				df = pd.DataFrame(data)

				# Create chart
				chart = alt.Chart(df).mark_area(opacity=0.6).encode(
					x=alt.X('x:Q', title='x'),
					y=alt.Y('pdf:Q', title='f(x)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title='F-Distribution PDF with Varying Numerator Degrees of Freedom (d₁)'
				).facet(
					facet=alt.Facet('d1:N', title='Parameter value')
				)

				st.altair_chart(chart)

			else:  # Varying d2
				d2_values = [2, 5, 10, 20, 30]

				# Create data frames for each d2 value
				data = []
				max_x = 5
				x_range = np.linspace(0.01, max_x, 500)

				for d2_val in d2_values:
					pdf = stats.f.pdf(x_range, d1, d2_val)

					for i, x_val in enumerate(x_range):
						data.append({
							'x': x_val,
							'pdf': pdf[i],
							'd2': f"d₁={d1}, d₂={d2_val}"
						})

				df = pd.DataFrame(data)

				# Create chart
				chart = alt.Chart(df).mark_area(opacity=0.6).encode(
					x=alt.X('x:Q', title='x'),
					y=alt.Y('pdf:Q', title='f(x)'),
					color=alt.value('#1f77b4')
				).properties(
					width=600,
					height=400,
					title='F-Distribution PDF with Varying Denominator Degrees of Freedom (d₂)'
				).facet(
					facet=alt.Facet('d2:N', title='Parameter value')
				)

				st.altair_chart(chart)

