import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Master Thesis Dashboard",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]


def find_images(folder):
    folder = ROOT / folder
    if not folder.exists():
        return []

    images = [
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    images = [
        p for p in images
        if ".ipynb_checkpoints" not in str(p)
    ]

    return sorted(images)


def find_activation_minmax_plots():
    images = find_images("outputs/activation_analysis")

    keep_models = ["llama7b", "mistral7b", "olmo1b", "olmo7b", "phi3"]

    selected = []
    for p in images:
        name = p.name.lower()
        parent = p.parent.name.lower()

        is_minmax = (
            "input_output_max" in name
            or "input_output" in name
            or "max" in name
        )

        is_redistribution = "redistribution" in name or "redistribution" in parent

        model_match = any(model in str(p).lower() for model in keep_models)

        if is_minmax and model_match and not is_redistribution:
            selected.append(p)

    final = {}
    for p in selected:
        text = str(p).lower()
        if "llama7b" in text and "llama7b" not in final:
            final["llama7b"] = p
        elif "mistral7b" in text and "mistral7b" not in final:
            final["mistral7b"] = p
        elif "olmo1b" in text and "olmo1b" not in final:
            final["olmo1b"] = p
        elif "olmo7b" in text and "olmo7b" not in final:
            final["olmo7b"] = p
        elif "phi3" in text and "phi3" not in final:
            final["phi3"] = p

    return [final[k] for k in ["olmo1b", "olmo7b", "llama7b", "mistral7b", "phi3"] if k in final]


def show_specific_image_comparison(images, title, columns=2):
    st.subheader(title)

    if not images:
        st.warning("No matching plots found.")
        return

    st.caption(f"{len(images)} plots found")

    selected_images = st.multiselect(
        "Select models to compare",
        options=images,
        default=images,
        format_func=lambda p: p.parent.name.replace("_", "-")
    )

    if not selected_images:
        st.info("Select at least one plot.")
        return

    cols = st.columns(columns)

    for i, img_path in enumerate(selected_images):
        with cols[i % columns]:
            st.image(
                Image.open(img_path),
                caption=str(img_path.relative_to(ROOT)),
                use_container_width=True
            )


def show_image_gallery(folder, title, filter_text=None, exclude_text=None, columns=2):
    st.subheader(title)

    images = find_images(folder)

    if filter_text:
        images = [
            p for p in images
            if filter_text.lower() in p.name.lower()
        ]

    if exclude_text:
        images = [
            p for p in images
            if exclude_text.lower() not in str(p).lower()
        ]

    if not images:
        st.warning(f"No images found in: {folder}")
        return

    st.caption(f"{len(images)} plots found")

    selected_images = st.multiselect(
        "Select plots to compare",
        options=images,
        default=images,
        format_func=lambda p: str(p.relative_to(ROOT))
    )

    if not selected_images:
        st.info("Select at least one plot.")
        return

    cols = st.columns(columns)

    for i, img_path in enumerate(selected_images):
        with cols[i % columns]:
            st.image(
                Image.open(img_path),
                caption=str(img_path.relative_to(ROOT)),
                use_container_width=True
            )


def show_table(data, title):
    st.subheader(title)
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    return df


st.title("Master Thesis Dashboard")
st.markdown("## Superweights and Quantization in Large Language Models")
st.write(
    "Interactive overview of the central plots, tables, and empirical results used in the thesis."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Chapter 3: Superweights",
        "Chapter 5: Quantization",
        "Automatic Plot Browser"
    ]
)


if page == "Overview":

    st.header("Overview")

    st.markdown("""
    This dashboard accompanies the Master's thesis

    **"An Empirical Study of Superweights and Quantization in Large Language Models"**

    and provides an interactive overview of the main experiments and findings.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Models", "5")
    col2.metric("Experiments", "20+")
    col3.metric("Benchmarks", "10+")
    col4.metric("Main Topics", "2")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Research Goal")

        st.markdown("""
Investigate whether a small number of **superweights** dominate the behaviour of
large language models and study how this property interacts with modern
quantization techniques.

The experiments cover five open-source language models ranging from
1B to 7B parameters.
""")

    with col2:

        st.subheader("Main Findings")

        st.markdown("""
- Superweights cause large activation spikes.
- Removing only a few superweights leads to major performance degradation.
- Much of the lost functionality can be recovered through redistribution training.
- Protecting superweights substantially improves low-bit quantization.
""")

    st.divider()

    st.subheader("Dashboard Structure")

    st.markdown("""
### Chapter 3 — Superweight Analysis

Explore how superweights emerge, how they affect downstream performance,
and how their functionality can be redistributed across the model.

Included experiments:

- Activation analysis
- Top superweight candidates
- Ablation studies
- Category-level effects
- Scaling experiments
- Redistribution analysis

---

### Chapter 5 — Quantization

Investigate the interaction between superweights and model quantization.

Included experiments:

- Activation quantization
- Weight quantization
- Quantization across model sizes
- FLORES multilingual evaluation
- Super-W8 scaling
- Protected Superweights (SW-AWQ)
""")

    st.info(
        "Use the navigation panel on the left to explore each experiment individually. "
        "Every section contains the corresponding figures and tables used in the thesis."
    )


elif page == "Chapter 3: Superweights":
    st.header("Chapter 3: Superweight Analysis")

    section = st.sidebar.radio(
        "Superweight Section",
        [
            "Activation Analysis",
            "Top Superweight Candidates",
            "Task-Level Ablation",
            "HellaSwag Ablation",
            "Category-Level Effects",
            "Scaling",
            "Redistribution"
        ]
    )

    if section == "Activation Analysis":
        st.markdown("""
        This section compares the original maximum input and output activations of the
        MLP down-projection across layers for the five evaluated models. These plots
        are used to identify activation spikes that indicate potential superweight locations.
        """)

        activation_plots = find_activation_minmax_plots()

        show_specific_image_comparison(
            activation_plots,
            "Input vs Output Max Activation per Layer",
            columns=2
        )

    elif section == "Top Superweight Candidates":
        st.markdown("""
        This section summarizes the top global superweight candidates identified
        through activation-spike analysis in OLMo-1B.
        """)

        data = {
            "Layer": [15, 1, 3, 2, 4, 6, 12, 5, 0, 0],
            "Row": [1764, 1764, 1764, 1764, 1764, 1764, 1764, 1764, 623, 623],
            "Col": [6840, 1710, 1902, 8041, 556, 4472, 353, 3977, 2271, 2660],
            "Input Max": [289.50, 437.00, 64.19, 69.81, 73.38, 26.64, 15.33, 10.88, 6.61, 4.73],
            "Output Max": [432.50, 281.00, 65.13, 42.66, 36.78, 9.09, 7.05, 3.07, 3.76, 4.39],
            "Score": [125208.75, 122797.00, 4180.21, 2977.94, 2698.82, 242.06, 108.02, 33.43, 24.82, 20.73]
        }

        df = show_table(data, "Top-10 global superweight candidates")
        st.bar_chart(df.set_index("Col")["Score"])

    elif section == "Task-Level Ablation":
        st.markdown("""
        This section shows task-level performance drops in OLMo-1B under top-k
        superweight ablation.
        """)

        task_plot = ROOT / "outputs/plots/olmo_topk_task_heatmap_rownorm.png"

        st.subheader("Task-level performance drop under top-k superweight ablation")
        
        if task_plot.exists():
            st.image(
                Image.open(task_plot),
                caption="outputs/plots/olmo_topk_task_heatmap_rownorm.png",
                use_container_width=True
            )
        else:
            st.warning("Plot not found: outputs/plots/olmo_topk_task_heatmap_rownorm.png")

    elif section == "HellaSwag Ablation":
        st.markdown("""
        This section compares original model performance, random ablation, and
        targeted superweight ablation on HellaSwag.
        """)

        data = {
            "Model": ["LLaMA-7B", "Mistral-7B", "OLMo-1B", "OLMo-7B", "Phi-3-mini"],
            "Baseline": [0.740, 0.806, 0.694, 0.788, 0.764],
            "Random Ablation": [0.740, 0.804, 0.694, 0.788, 0.762],
            "Superweight Ablation": [0.360, 0.268, 0.298, 0.350, 0.294],
            "# Superweights": [1, 1, 2, 4, 6],
        }

        df = show_table(data, "Impact of superweight ablation on HellaSwag")
        st.bar_chart(
            df.set_index("Model")[["Baseline", "Random Ablation", "Superweight Ablation"]]
        )

    elif section == "Category-Level Effects":
        st.markdown("""
        This section shows how superweight removal changes the model's output
        probability distribution across token categories.
        """)

        show_image_gallery(
            "results/category_analysis",
            "Category-level shift in token probabilities",
            columns=2
        )

    elif section == "Scaling":
        st.markdown("""
        This section shows task-specific performance changes under scaling of a
        single identified superweight.
        """)

        st.subheader("Task-specific performance change under scaling of a single superweight")

        plot_path = ROOT / "outputs/plots/olmo_sw1_scaling_delta_heatmap.png"
        
        if plot_path.exists():
            st.image(
                Image.open(plot_path),
                caption="Performance change under SW1 scaling (relative to the baseline model)",
                use_container_width=True,
            )
        else:
            st.warning("Plot not found: outputs/plots/olmo_sw1_scaling_delta_heatmap.png")

    elif section == "Redistribution":

        st.markdown("""
        This section summarizes the superweight redistribution experiments.
        Superweights were repeatedly disabled during training to encourage the
        model to distribute the associated functionality across other parameters.
        """)
    
        data = {
            "Model": ["OLMo-1B", "OLMo-7B", "LLaMA-7B", "Mistral-7B", "Phi-3-mini"],
            "Original": [0.694, 0.788, 0.740, 0.806, 0.764],
            "Ablation": [0.298, 0.350, 0.360, 0.268, 0.294],
            "Redistribution": [0.572, 0.746, 0.708, 0.626, 0.674],
            "Recovery (%)": [69.2, 95.7, 91.9, 67.3, 81.0],
        }
    
        df = show_table(data, "Effect of superweight redistribution")
        st.bar_chart(df.set_index("Model")[["Original", "Ablation", "Redistribution"]])
    
        st.divider()
    
        st.markdown("""
        For each model, three complementary activation plots are shown:
        activation delta, activation concentration delta, and redistribution input-output maxima.
        """)
    
        models = {
            "OLMo-1B": "olmo1b",
            "OLMo-7B": "olmo7b",
            "LLaMA-7B": "llama7b",
            "Mistral-7B": "mistral7b",
            "Phi-3-mini": "phi3",
        }
    
        model = st.selectbox("Select model", list(models.keys()))
        model_key = models[model]
        folder = ROOT / "outputs" / "activation_analysis" / model_key
    
        def find_plot(folder, required_terms, excluded_terms=None):
            excluded_terms = excluded_terms or []
            if not folder.exists():
                return None
    
            candidates = []
            for p in folder.rglob("*.png"):
                name = p.name.lower()
                if all(term.lower() in name for term in required_terms):
                    if not any(term.lower() in name for term in excluded_terms):
                        candidates.append(p)
    
            return sorted(candidates)[0] if candidates else None
    
        activation_delta = find_plot(
            folder,
            required_terms=["activation_delta"],
            excluded_terms=["concentration"]
        )
    
        concentration_delta = find_plot(
            folder,
            required_terms=["activation_concentration_delta"]
        )
    
        redistribution_input_output = find_plot(
            folder,
            required_terms=["redistribution", "input", "output", "max"]
        )
    
        plots = [
            ("Activation Delta", activation_delta),
            ("Activation Concentration Delta", concentration_delta),
            ("Redistribution Input vs Output Max", redistribution_input_output),
        ]
    
        cols = st.columns(3)
    
        for col, (title, path) in zip(cols, plots):
            with col:
                st.subheader(title)
    
                if path is not None and path.exists():
                    st.image(
                        Image.open(path),
                        caption=str(path.relative_to(ROOT)),
                        use_container_width=True
                    )
                else:
                    st.warning(f"Missing plot for {model}: {title}")
        

elif page == "Chapter 5: Quantization":
    st.header("Chapter 5: Quantization")

    section = st.sidebar.radio(
        "Quantization Section",
        [
            "Activation Bit-Width",
            "Task-Level Quantization",
            "Language / FLORES",
            "Model Size",
            "Quantization Methods",
            "Superweight Scaling W8",
            "Protected Superweights AWQ"
        ]
    )

    if section == "Activation Bit-Width":
        st.markdown("""
        This section shows how activation bit-width affects perplexity.
        Lower perplexity is better.
        """)

        data = {
            "Bit-Width": ["W16A8", "W16A8", "W16A8", "W16A4", "W16A4"],
            "Method": ["FP16", "Naive", "Superweight", "Naive", "Superweight"],
            "WikiText-2": [8.51, 9.04, 8.51, 142.88, 8.85],
            "C4": [6.96, 7.12, 6.96, 89.45, 7.19],
        }

        df = show_table(data, "Table 5.1: LLaMA-7B activation quantization perplexity")
        st.bar_chart(df.set_index("Method")[["WikiText-2", "C4"]])

    elif section == "Task-Level Quantization":
        st.markdown("""
        This section compares W8A8 quantization methods across task categories.
        Higher is better for accuracy tasks, lower is better for perplexity tasks.
        """)

        data = {
            "Category": [
                "Commonsense Reasoning",
                "Natural Language Understanding",
                "Coreference Reasoning",
                "Cross-lingual Reasoning",
                "Language Modeling",
                "Language Modeling",
            ],
            "Task": ["HellaSwag", "BoolQ", "WinoGrande", "XCOPA", "WikiText-2", "C4"],
            "FP16": [0.424, 0.662, 0.592, 0.790, 14.28, 13.56],
            "Naive": [0.418, 0.604, 0.558, 0.770, 17.22, 15.58],
            "Super": [0.420, 0.620, 0.564, 0.770, 17.20, 15.58],
        }

        df = show_table(data, "Table 5.2: W8A8 quantization by task category")

        acc_df = df[~df["Task"].isin(["WikiText-2", "C4"])]
        ppl_df = df[df["Task"].isin(["WikiText-2", "C4"])]

        st.subheader("Accuracy tasks")
        st.bar_chart(acc_df.set_index("Task")[["FP16", "Naive", "Super"]])

        st.subheader("Perplexity tasks")
        st.bar_chart(ppl_df.set_index("Task")[["FP16", "Naive", "Super"]])

    elif section == "Language / FLORES":
        st.markdown("""
        This section shows FLORES multilingual translation perplexity under W16A4
        activation quantization. Lower is better.
        """)

        data = {
            "Language Pair": ["DE → EN", "EN → DE", "EN → ES", "EN → FR", "ES → EN", "FR → EN"],
            "FP16": [3.08, 3.58, 4.95, 2.39, 3.90, 2.80],
            "Naive": [5.34, 7.16, 9.03, 3.74, 6.51, 4.29],
            "Super": [3.21, 3.82, 5.21, 2.48, 4.00, 2.92],
        }

        df = show_table(data, "Table 5.3: FLORES translation perplexity")
        st.bar_chart(df.set_index("Language Pair")[["FP16", "Naive", "Super"]])

    elif section == "Model Size":
        st.markdown("""
        This section compares activation quantization robustness across model sizes.
        Lower perplexity is better.
        """)

        data = {
            "Model": ["OLMo-1B", "OLMo-1B", "OLMo-7B", "OLMo-7B"],
            "Dataset": ["WikiText-2", "C4", "WikiText-2", "C4"],
            "FP16": [17.819, 13.485, 13.298, 10.265],
            "Naive W16A8": [18.606, 13.807, 25.946, 14.682],
            "Super W16A8": [17.821, 13.572, 13.589, 10.351],
            "Naive Δ (%)": [4.42, 2.39, 95.11, 43.03],
            "Super Δ (%)": [0.01, 0.65, 2.19, 0.84],
        }

        df = show_table(data, "Table 5.4: W16A8 activation quantization across model sizes")

        chart_df = df.copy()
        chart_df["Model-Dataset"] = chart_df["Model"] + " / " + chart_df["Dataset"]

        st.bar_chart(chart_df.set_index("Model-Dataset")[["FP16", "Naive W16A8", "Super W16A8"]])

    elif section == "Quantization Methods":
        st.markdown("""
        This section compares different 4-bit quantization methods on OLMo-1B.
        Higher is better for all metrics.
        """)

        data = {
            "Task": ["BoolQ", "HellaSwag", "WinoGrande", "XCOPA-en"],
            "FP16": [0.6620, 0.5160, 0.5920, 0.7900],
            "Naive W4": [0.3940, 0.2520, 0.5180, 0.5600],
            "Super W4": [0.4960, 0.3220, 0.5180, 0.6600],
            "GPTQ": [0.6300, 0.4200, 0.5580, 0.7800],
            "AWQ": [0.5460, 0.4160, 0.5300, 0.7800],
        }

        df = show_table(data, "Table 5.5: Quantization method comparison")
        st.bar_chart(df.set_index("Task")[["FP16", "Naive W4", "Super W4", "GPTQ", "AWQ"]])

    elif section == "Superweight Scaling W8":
        st.markdown("""
        This section shows the effect of superweight scaling under Super-W8 quantization.
        Values are accuracy differences in percentage points relative to α = 1.0.
        """)

        data = {
            "α": [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0],
            "BoolQ": [-3.8, -1.6, -0.6, 0.0, -0.6, -1.8, -3.0, -2.2, -0.6, -3.4],
            "PIQA": [0.2, 0.2, 0.4, 0.0, 0.8, 0.8, 0.8, 0.2, -0.2, -1.2],
            "HellaSwag": [-0.6, 0.4, 0.0, 0.0, 0.2, 0.0, 0.2, 0.2, 0.8, -1.6],
            "WinoGrande": [1.2, 0.6, 0.2, 0.0, 1.2, 1.4, 1.4, 1.4, -0.4, -1.2],
            "ARC-E": [-1.0, 0.2, 0.0, 0.0, -0.4, -0.4, -0.8, -1.4, -2.0, -3.4],
            "ARC-C": [-0.7, -0.3, 0.7, 0.0, -0.3, -0.3, -0.3, 0.0, 0.0, -0.3],
        }

        df = show_table(data, "Table 5.6: Super-W8 scaling results")
        st.line_chart(df.set_index("α"))

    elif section == "Protected Superweights AWQ":
        st.markdown("""
        This section summarizes the best SW-AWQ hyperparameter configurations.
        Higher accuracy is better.
        """)

        data = {
            "Task": ["BoolQ", "HellaSwag", "PIQA", "SciQ"],
            "Baseline": [0.544, 0.416, 0.744, 0.936],
            "Best α0": [2.0, 1.5, 1.5, 1.0],
            "Best λ": [1.0, 0.75, 0.75, 0.0],
            "Best Acc.": [0.582, 0.422, 0.748, 0.936],
            "Δ": [0.038, 0.006, 0.004, 0.000],
        }

        df = show_table(data, "Table 5.7: Best SW-AWQ configurations")
        st.bar_chart(df.set_index("Task")[["Baseline", "Best Acc."]])


elif page == "Automatic Plot Browser":
    st.header("Automatic Plot Browser")

    st.markdown("""
    This page automatically detects image files in selected repository folders.
    It is useful for checking whether newly generated plots are already visible
    in the dashboard.
    """)

    root_choice = st.selectbox(
        "Select root folder",
        [
            "outputs",
            "results",
            "results/superweights",
            "results/quantization",
            "outputs/plots",
            "outputs/activation_analysis",
            "results/category_analysis"
        ]
    )

    show_image_gallery(
        root_choice,
        f"All detected plots in {root_choice}",
        columns=2
    )