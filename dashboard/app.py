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
        st.subheader(
            "Incremental Performance Changes under Top-k Super Weight Ablation"
        )

        st.markdown("""
        The heatmap reports the performance decrease relative to the immediately
        preceding ablation setting. Positive values indicate an additional
        performance loss, whereas negative values indicate an improvement.
        """)

        task_plot = (
            ROOT
            / "outputs"
            / "plots"
            / "olmo_topk_heatmap_incremental_no_title.png"
        )
        
        if task_plot.exists():
            st.image(
                Image.open(task_plot),
                caption=(
                    "Incremental task-level performance changes in OLMo-1B "
                    "under top-k super weight ablation."
                ),
                use_container_width=True
            )
        else:
            st.warning(
                "Plot not found: "
                "outputs/plots/olmo_topk_heatmap_incremental_no_title.png"
            )

    elif section == "HellaSwag Ablation":
        st.subheader("Impact of Super Weight Ablation on HellaSwag")

        st.markdown("""
        This section compares the original models with random ablation and
        targeted super weight ablation on HellaSwag. Results are reported as
        mean ± standard deviation over five evaluation seeds.
        """)

        ablation_table = pd.DataFrame(
            {
                "Model": [
                    "LLaMA-7B",
                    "Mistral-7B",
                    "OLMo-1B",
                    "OLMo-7B",
                    "Phi-3-mini",
                ],
                "Baseline": [
                    "0.7208 ± 0.01",
                    "0.8004 ± 0.02",
                    "0.6584 ± 0.01",
                    "0.7744 ± 0.01",
                    "0.7504 ± 0.01",
                ],
                "Random Ablation": [
                    "0.7208 ± 0.01",
                    "0.8000 ± 0.01",
                    "0.6580 ± 0.01",
                    "0.7744 ± 0.01",
                    "0.7504 ± 0.01",
                ],
                "Super Weight Ablation": [
                    "0.3452 ± 0.02",
                    "0.2652 ± 0.03",
                    "0.2804 ± 0.02",
                    "0.3288 ± 0.01",
                    "0.2976 ± 0.02",
                ],
                "# Super Weights": [1, 1, 2, 4, 6],
            }
        )

        st.dataframe(
            ablation_table,
            hide_index=True,
            use_container_width=True,
        )

        plot_data = pd.DataFrame(
            {
                "Model": [
                    "LLaMA-7B",
                    "Mistral-7B",
                    "OLMo-1B",
                    "OLMo-7B",
                    "Phi-3-mini",
                ],
                "Baseline": [0.7208, 0.8004, 0.6584, 0.7744, 0.7504],
                "Random Ablation": [0.7208, 0.8000, 0.6580, 0.7744, 0.7504],
                "Super Weight Ablation": [
                    0.3452,
                    0.2652,
                    0.2804,
                    0.3288,
                    0.2976,
                ],
            }
        )

        st.bar_chart(
            plot_data.set_index("Model"),
            use_container_width=True,
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
        st.subheader("Effect of Super Weight Redistribution")

        st.markdown("""
        This section evaluates whether performance lost through super weight
        ablation can be recovered by redistributing the associated functionality
        across other model parameters. Results are reported as mean ± standard
        deviation over five evaluation seeds.
        """)

        redistribution_table = pd.DataFrame(
            {
                "Model": [
                    "OLMo-1B",
                    "OLMo-7B",
                    "LLaMA-7B",
                    "Mistral-7B",
                    "Phi-3 Mini",
                ],
                "Original": [
                    "0.6492 ± 0.01",
                    "0.7572 ± 0.01",
                    "0.7228 ± 0.02",
                    "0.7928 ± 0.01",
                    "0.7520 ± 0.02",
                ],
                "Ablation": [
                    "0.2644 ± 0.02",
                    "0.3272 ± 0.03",
                    "0.3592 ± 0.01",
                    "0.2504 ± 0.02",
                    "0.3000 ± 0.03",
                ],
                "Redistribution": [
                    "0.5408 ± 0.03",
                    "0.7008 ± 0.03",
                    "0.6916 ± 0.02",
                    "0.5936 ± 0.03",
                    "0.6792 ± 0.01",
                ],
                "Recovery (%)": [
                    71.8,
                    86.9,
                    91.4,
                    63.3,
                    83.9,
                ],
            }
        )

        st.dataframe(
            redistribution_table,
            hide_index=True,
            use_container_width=True,
        )

        redistribution_plot = pd.DataFrame(
            {
                "Model": [
                    "OLMo-1B",
                    "OLMo-7B",
                    "LLaMA-7B",
                    "Mistral-7B",
                    "Phi-3 Mini",
                ],
                "Original": [
                    0.6492,
                    0.7572,
                    0.7228,
                    0.7928,
                    0.7520,
                ],
                "Ablation": [
                    0.2644,
                    0.3272,
                    0.3592,
                    0.2504,
                    0.3000,
                ],
                "Redistribution": [
                    0.5408,
                    0.7008,
                    0.6916,
                    0.5936,
                    0.6792,
                ],
            }
        )

        st.bar_chart(
            redistribution_plot.set_index("Model"),
            use_container_width=True,
            stack=False,
        )

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
            "Proposed Extensions",
        ]
    )

    if section == "Activation Bit-Width":
        st.subheader(
            "Table 5.1: LLaMA-7B Activation Quantization Perplexity"
        )

        st.markdown("""
        This section compares LLaMA-7B perplexity under different activation
        bit-widths and quantization strategies. Results are reported as
        mean ± standard deviation over five evaluation seeds; lower values
        indicate better performance.
        """)

        activation_table = pd.DataFrame(
            {
                "Bit-Width": [
                    "W16A8",
                    "W16A8",
                    "W16A8",
                    "W16A4",
                    "W16A4",
                ],
                "Method": [
                    "FP16",
                    "Naive",
                    "Super Weight",
                    "Naive",
                    "Super Weight",
                ],
                "WikiText-2": [
                    "8.69 ± 0.43",
                    "9.29 ± 0.48",
                    "8.70 ± 0.42",
                    "145.96 ± 8.46",
                    "9.04 ± 0.46",
                ],
                "C4": [
                    "7.51 ± 0.42",
                    "7.69 ± 0.42",
                    "7.52 ± 0.42",
                    "90.68 ± 7.19",
                    "7.76 ± 0.42",
                ],
            }
        )

        st.dataframe(
            activation_table,
            hide_index=True,
            use_container_width=True,
        )

        activation_plot = pd.DataFrame(
            {
                "Configuration": [
                    "W16A8 – FP16",
                    "W16A8 – Naive",
                    "W16A8 – Super Weight",
                    "W16A4 – Naive",
                    "W16A4 – Super Weight",
                ],
                "WikiText-2": [
                    8.69,
                    9.29,
                    8.70,
                    145.96,
                    9.04,
                ],
                "C4": [
                    7.51,
                    7.69,
                    7.52,
                    90.68,
                    7.76,
                ],
            }
        )

        st.subheader("Perplexity by Quantization Configuration")

        st.bar_chart(
            activation_plot.set_index("Configuration"),
            use_container_width=True,
            stack=False,
        )

    elif section == "Task-Level Quantization":
        st.subheader("Table 5.2: W8A8 Quantization by Task Category")

        st.markdown("""
        This section compares W8A8 quantization methods on OLMo-1B across
        different task categories. Results are reported as mean ± standard
        deviation over five evaluation seeds. Higher values indicate better
        performance for accuracy-based tasks, whereas lower values indicate
        better performance for language-modeling benchmarks.
        """)

        task_table = pd.DataFrame(
            {
                "Category": [
                    "Commonsense Reasoning",
                    "Natural Language Understanding",
                    "Coreference Reasoning",
                    "Cross-lingual Reasoning",
                    "Language Modeling",
                    "Language Modeling",
                ],
                "Task": [
                    "HellaSwag",
                    "BoolQ",
                    "WinoGrande",
                    "XCOPA",
                    "WikiText-2",
                    "C4",
                ],
                "FP16": [
                    "0.6396 ± 0.01",
                    "0.6152 ± 0.01",
                    "0.5768 ± 0.02",
                    "0.7900 ± 0.00",
                    "14.60 ± 0.64",
                    "14.03 ± 0.54",
                ],
                "Naive": [
                    "0.6164 ± 0.01",
                    "0.6288 ± 0.03",
                    "0.5640 ± 0.03",
                    "0.7700 ± 0.00",
                    "15.20 ± 0.69",
                    "14.36 ± 0.54",
                ],
                "Super": [
                    "0.6188 ± 0.01",
                    "0.6188 ± 0.02",
                    "0.5612 ± 0.03",
                    "0.7700 ± 0.00",
                    "14.62 ± 0.65",
                    "14.12 ± 0.54",
                ],
            }
        )

        st.dataframe(
            task_table,
            hide_index=True,
            use_container_width=True,
        )

        accuracy_plot = pd.DataFrame(
            {
                "Task": [
                    "HellaSwag",
                    "BoolQ",
                    "WinoGrande",
                    "XCOPA",
                ],
                "FP16": [
                    0.6396,
                    0.6152,
                    0.5768,
                    0.7900,
                ],
                "Naive": [
                    0.6164,
                    0.6288,
                    0.5640,
                    0.7700,
                ],
                "Super": [
                    0.6188,
                    0.6188,
                    0.5612,
                    0.7700,
                ],
            }
        )

        perplexity_plot = pd.DataFrame(
            {
                "Task": [
                    "WikiText-2",
                    "C4",
                ],
                "FP16": [
                    14.60,
                    14.03,
                ],
                "Naive": [
                    15.20,
                    14.36,
                ],
                "Super": [
                    14.62,
                    14.12,
                ],
            }
        )

        st.subheader("Accuracy Tasks")

        st.bar_chart(
            accuracy_plot.set_index("Task"),
            use_container_width=True,
            stack=False,
        )

        st.subheader("Language Modeling Tasks")

        st.bar_chart(
            perplexity_plot.set_index("Task"),
            use_container_width=True,
            stack=False,
        )

    elif section == "Language / FLORES":
        st.subheader("Table 5.3: FLORES Translation Perplexity")

        st.markdown("""
        This section compares multilingual translation perplexity for OLMo-1B
        under W16A4 activation quantization. Results are reported as mean ±
        standard deviation over five evaluation seeds. Lower values indicate
        better performance.
        """)

        flores_table = pd.DataFrame(
            {
                "Language Pair": [
                    "DE → EN",
                    "EN → DE",
                    "EN → ES",
                    "EN → FR",
                    "ES → EN",
                    "FR → EN",
                ],
                "FP16": [
                    "3.36 ± 0.14",
                    "3.65 ± 0.14",
                    "4.47 ± 0.10",
                    "2.19 ± 0.11",
                    "3.78 ± 0.26",
                    "2.55 ± 0.14",
                ],
                "Naive": [
                    "5.69 ± 0.44",
                    "7.02 ± 0.24",
                    "7.89 ± 0.28",
                    "3.35 ± 0.18",
                    "6.20 ± 0.61",
                    "3.95 ± 0.34",
                ],
                "Super": [
                    "3.49 ± 0.18",
                    "3.93 ± 0.11",
                    "4.65 ± 0.09",
                    "2.27 ± 0.11",
                    "3.87 ± 0.29",
                    "2.65 ± 0.15",
                ],
            }
        )

        st.dataframe(
            flores_table,
            hide_index=True,
            use_container_width=True,
        )

        flores_plot = pd.DataFrame(
            {
                "Language Pair": [
                    "DE → EN",
                    "EN → DE",
                    "EN → ES",
                    "EN → FR",
                    "ES → EN",
                    "FR → EN",
                ],
                "FP16": [
                    3.36,
                    3.65,
                    4.47,
                    2.19,
                    3.78,
                    2.55,
                ],
                "Naive": [
                    5.69,
                    7.02,
                    7.89,
                    3.35,
                    6.20,
                    3.95,
                ],
                "Super": [
                    3.49,
                    3.93,
                    4.65,
                    2.27,
                    3.87,
                    2.65,
                ],
            }
        )

        st.subheader("Perplexity by Language Pair")

        st.bar_chart(
            flores_plot.set_index("Language Pair"),
            use_container_width=True,
            stack=False,
        )

    elif section == "Model Size":
        st.subheader(
            "Table 5.4: W16A8 Activation Quantization Across Model Sizes"
        )

        st.markdown("""
        This section compares activation quantization robustness across model
        sizes. Results are reported as mean ± standard deviation over five
        evaluation seeds. Relative changes are calculated against FP16.
        Lower perplexity is better.
        """)

        model_size_table = pd.DataFrame(
            {
                "Model": [
                    "OLMo-1B",
                    "OLMo-1B",
                    "OLMo-7B",
                    "OLMo-7B",
                ],
                "Dataset": [
                    "WikiText-2",
                    "C4",
                    "WikiText-2",
                    "C4",
                ],
                "FP16": [
                    "14.60 ± 0.65",
                    "14.03 ± 0.55",
                    "10.87 ± 0.59",
                    "10.82 ± 0.48",
                ],
                "Naive W16A8": [
                    "15.20 ± 0.69",
                    "14.36 ± 0.54",
                    "20.81 ± 0.76",
                    "15.02 ± 0.59",
                ],
                "Super W16A8": [
                    "14.62 ± 0.65",
                    "14.12 ± 0.54",
                    "11.14 ± 0.70",
                    "10.92 ± 0.49",
                ],
                "Naive Δ (%)": [
                    4.12,
                    2.32,
                    91.54,
                    38.75,
                ],
                "Super Δ (%)": [
                    0.13,
                    0.65,
                    2.53,
                    0.90,
                ],
            }
        )

        st.dataframe(
            model_size_table,
            hide_index=True,
            use_container_width=True,
        )

        model_size_plot = pd.DataFrame(
            {
                "Configuration": [
                    "OLMo-1B – WikiText-2",
                    "OLMo-1B – C4",
                    "OLMo-7B – WikiText-2",
                    "OLMo-7B – C4",
                ],
                "FP16": [
                    14.60,
                    14.03,
                    10.87,
                    10.82,
                ],
                "Naive W16A8": [
                    15.20,
                    14.36,
                    20.81,
                    15.02,
                ],
                "Super W16A8": [
                    14.62,
                    14.12,
                    11.14,
                    10.92,
                ],
            }
        )

        st.subheader("Perplexity by Model and Dataset")

        st.bar_chart(
            model_size_plot.set_index("Configuration"),
            use_container_width=True,
            stack=False,
        )

    elif section == "Quantization Methods":
            st.subheader("Table 5.5: Quantization Method Comparison")

            st.markdown("""
            This section compares different 4-bit weight-quantization methods on
            OLMo-1B across selected zero-shot benchmarks. Results are reported as
            mean ± standard deviation over five evaluation seeds. Higher values
            indicate better performance for all benchmarks.
            """)

            method_table = pd.DataFrame(
                {
                    "Task": [
                        "BoolQ",
                        "HellaSwag",
                        "WinoGrande",
                        "XCOPA-en",
                    ],
                    "FP16": [
                        "0.6152 ± 0.01",
                        "0.6396 ± 0.01",
                        "0.5768 ± 0.02",
                        "0.7900 ± 0.00",
                    ],
                    "Naive W4": [
                        "0.4072 ± 0.03",
                        "0.2536 ± 0.01",
                        "0.5440 ± 0.02",
                        "0.5600 ± 0.00",
                    ],
                    "Super W4": [
                        "0.5196 ± 0.03",
                        "0.2992 ± 0.02",
                        "0.5440 ± 0.02",
                        "0.6600 ± 0.00",
                    ],
                    "GPTQ": [
                        "0.4948 ± 0.02",
                        "0.6336 ± 0.01",
                        "0.5568 ± 0.02",
                        "0.7400 ± 0.00",
                    ],
                    "AWQ": [
                        "0.6084 ± 0.02",
                        "0.6220 ± 0.01",
                        "0.5536 ± 0.03",
                        "0.7800 ± 0.00",
                    ],
                }
            )

            st.dataframe(
                method_table,
                hide_index=True,
                use_container_width=True,
            )

            method_plot = pd.DataFrame(
                {
                    "Task": [
                        "BoolQ",
                        "HellaSwag",
                        "WinoGrande",
                        "XCOPA-en",
                    ],
                    "FP16": [
                        0.6152,
                        0.6396,
                        0.5768,
                        0.7900,
                    ],
                    "Naive W4": [
                        0.4072,
                        0.2536,
                        0.5440,
                        0.5600,
                    ],
                    "Super W4": [
                        0.5196,
                        0.2992,
                        0.5440,
                        0.6600,
                    ],
                    "GPTQ": [
                        0.4948,
                        0.6336,
                        0.5568,
                        0.7400,
                    ],
                    "AWQ": [
                        0.6084,
                        0.6220,
                        0.5536,
                        0.7800,
                    ],
                }
            )

            st.subheader("Performance by Quantization Method")

            st.bar_chart(
                method_plot.set_index("Task"),
                use_container_width=True,
                stack=False,
            )


    elif section == "Proposed Extensions":
        st.header("Proposed Extensions to Super Weight-Aware Quantization")

        st.markdown("""
        Two extensions are evaluated to investigate whether explicitly modifying
        or protecting super weights can further improve quantized model performance.
        The first scales restored super weights, whereas the second integrates
        explicit super weight protection into AWQ.
        """)

        st.subheader("Table 5.6: Super Weight Scaling")

        st.markdown("""
        Moderate scaling can improve performance under Super-W8 quantization,
        although the optimal factor varies across tasks. Large deviations from
        the original scale generally reduce performance.
        """)

        scaling_table = pd.DataFrame(
            {
                "α": [
                    0.5,
                    0.8,
                    0.9,
                    1.0,
                    1.1,
                    1.2,
                    1.3,
                    1.5,
                    2.0,
                    3.0,
                ],
                "BoolQ": [
                    "-3.8",
                    "-1.6",
                    "-0.6",
                    "0.0",
                    "-0.6",
                    "-1.8",
                    "-3.0",
                    "-2.2",
                    "-0.6",
                    "-3.4",
                ],
                "PIQA": [
                    "+0.2",
                    "+0.2",
                    "+0.4",
                    "0.0",
                    "+0.8",
                    "+0.8",
                    "+0.8",
                    "+0.2",
                    "-0.2",
                    "-1.2",
                ],
                "HellaSwag": [
                    "-0.6",
                    "+0.4",
                    "0.0",
                    "0.0",
                    "+0.2",
                    "0.0",
                    "+0.2",
                    "+0.2",
                    "+0.8",
                    "-1.6",
                ],
                "WinoGrande": [
                    "+1.2",
                    "+0.6",
                    "+0.2",
                    "0.0",
                    "+1.2",
                    "+1.4",
                    "+1.4",
                    "+1.4",
                    "-0.4",
                    "-1.2",
                ],
                "ARC-E": [
                    "-1.0",
                    "+0.2",
                    "0.0",
                    "0.0",
                    "-0.4",
                    "-0.4",
                    "-0.8",
                    "-1.4",
                    "-2.0",
                    "-3.4",
                ],
                "ARC-C": [
                    "-0.7",
                    "-0.3",
                    "+0.7",
                    "0.0",
                    "-0.3",
                    "-0.3",
                    "-0.3",
                    "0.0",
                    "0.0",
                    "-0.3",
                ],
            }
        )

        st.dataframe(
            scaling_table,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Accuracy differences in percentage points relative to α = 1.0 "
            "for Super-W8 quantization on OLMo-1B."
        )

        st.subheader("Table 5.7: Protecting Critical Weights in AWQ")

        st.markdown("""
        Explicit restoration and scaling of super weights after AWQ improves
        performance on BoolQ, HellaSwag, and PIQA. SciQ does not benefit from
        additional scaling in this experiment.
        """)

        sw_awq_table = pd.DataFrame(
            {
                "Task": [
                    "BoolQ",
                    "HellaSwag",
                    "PIQA",
                    "SciQ",
                ],
                "Baseline": [
                    "0.544",
                    "0.416",
                    "0.744",
                    "0.936",
                ],
                "Best α₀": [
                    "2.0",
                    "1.5",
                    "1.5",
                    "1.0",
                ],
                "Best λ": [
                    "1.0",
                    "0.75",
                    "0.75",
                    "0.0",
                ],
                "Best Accuracy": [
                    "0.582",
                    "0.422",
                    "0.748",
                    "0.936",
                ],
                "Δ": [
                    "+0.038",
                    "+0.006",
                    "+0.004",
                    "+0.000",
                ],
            }
        )

        st.dataframe(
            sw_awq_table,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Best SW-AWQ configurations compared with the corresponding "
            "AWQ baseline on OLMo-1B."
        )


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
