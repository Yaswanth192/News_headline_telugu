"""
app.py
======
Telugu News Headline Studio — Complete Streamlit Application (Phase 6).

Production web interface for local Telugu news headline generation,
relevance evaluation, model comparison, research analytics, and architecture details.

Hardware Constraints:
- NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM).
- Sequential memory-safe loading & unloading.
- Strict local execution.
"""

import html
import json
import os
import pathlib
import sys
import time
from typing import Dict, Any, List

import pandas as pd
import streamlit as st

# Configure UTF-8 for Telugu character display
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.manager import get_model_manager, MODEL_REGISTRY, CLASSIFIER_REGISTRY

# Page configuration
st.set_page_config(
    page_title="Telugu News Headline Studio",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished typography and badges
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .telugu-text {
        font-family: 'Mandali', 'Suranna', 'Gautami', 'Nirmala UI', sans-serif;
        font-size: 1.15rem;
        line-height: 1.8;
    }
    .headline-card {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 1.2rem;
        border-radius: 6px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .badge-hrel {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-mrel {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-lrel {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .metric-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Preset Telugu sample articles from actual validation corpus
PRESET_SAMPLES = [
    {
        "title": "శేఖర్ కమ్ముల తండ్రి శేషయ్య కన్నుమూత (సినీ వార్తలు)",
        "article": "ప్రముఖ దర్శకులు శేఖర్ కమ్ముల ఇంట విషాదం నెలకొంది . ఆయన తండ్రి కమ్ముల శేషయ్య ( 89 ) కన్నుమూశారు . ఆయన గత కొద్దికాలంగా వృద్ధాప్య సమస్యలతో బాధపడుతున్నారు . ఈ క్రమంలో హైదరాబాద్ లోని ఓ ప్రైవేట్ ఆస్పత్రిలో చికిత్స పొందుతూ శనివారం ఉదయం 6 గంటలకు తుదిశ్వాస విడిచారు . శేషయ్య మృతి పట్ల సినీ , రాజకీయ ప్రముఖులు సంతాపం వ్యక్తం చేస్తున్నారు . శేఖర్ కమ్ముల కుటుంబ సభ్యులకు ప్రగాఢ సానుభూతిని తెలియజేస్తున్నారు . శేఖర్ కమ్ముల స్వగ్రామం నల్గొండ జిల్లాలోని మిర్యాలగూడ సమీపంలో ఉండగా , తండ్రి భౌతికకాయాన్ని స్వగ్రామానికి తరలించనున్నట్లు కుటుంబసభ్యులు తెలిపారు . సాయంత్రం అంత్యక్రియలు నిర్వహించనున్నారు .",
        "reference_headline": "శేఖర్ కమ్ముల తండ్రి శేషయ్య కన్నుమూత",
    },
    {
        "title": "ఘోర రోడ్డు ప్రమాదం : ఐదుగురు దుర్మరణం (ప్రమాదాలు)",
        "article": "గుజరాత్‌లోని ఆనంద్ జిల్లాలో ఘోర రోడ్డు ప్రమాదం సంభవించింది . తారాపూర్ వద్ద కారు , లారీ ఒకదానికొకటి ఢీకొన్నాయి . ఈ ప్రమాదంలో ఐదుగురు అక్కడికక్కడే మృతి చెందగా , మరో ముగ్గురు తీవ్రంగా గాయపడ్డారు . సమాచారం అందుకున్న పోలీసులు వెంటనే ఘటనా స్థలానికి చేరుకుని సహాయక చర్యలు చేపట్టారు . క్షతగాత్రులను సమీపంలోని ఆస్పత్రికి తరలించారు . మృతుల వివరాలు ఇంకా తెలియరాలేదు . కేసు నమోదు చేసుకున్న పోలీసులు దర్యాప్తు చేస్తున్నారు . పోలీసులు కేసు నమోదు చేసుకుని దర్యాప్తు ప్రారంభించారు .",
        "reference_headline": "ఘార రోడ్డు ప్రమాదం : ఐదుగురు దుర్మరణం",
    },
    {
        "title": "విశాఖకు ముఖ్యమంత్రి జగన్ (రాజకీయ వార్తలు)",
        "article": "ఆంధ్రప్రదేశ్ ముఖ్యమంత్రి వైఎస్ జగన్ మోహన్ రెడ్డి నేడు విశాఖపట్నంలో పర్యటించనున్నారు . ఈ పర్యటనలో భాగంగా ఆయన పలు అభివృద్ధి పనులకు శంకుస్థాపనలు , ప్రారంభోత్సవాలు చేయనున్నారు . ముఖ్యంగా విశాఖ నగరంలో ఐటీ హబ్ , నూతన రహదారుల నిర్మాణ పనులకు శ్రీకారం చుట్టనున్నారు . అనంతరం స్థానిక నాయకులతో సమావేశమై సమీక్ష నిర్వహించనున్నారు . ముఖ్యమంత్రి పర్యటన సందర్భంగా విశాఖ నగరంలో భారీ భద్రతా ఏర్పాట్లు చేశారు .",
        "reference_headline": "విశాఖకు ముఖ్యమంత్రి వైఎస్ జగన్",
    },
]


def render_sidebar():
    """Render sidebar with navigation and live GPU monitor."""
    st.sidebar.image("https://img.icons8.com/fluency/96/news.png", width=64)
    st.sidebar.title("Telugu News Studio")
    st.sidebar.caption("Fine-Tuned NLP on Local RTX 3050 (4 GB VRAM)")

    page = st.sidebar.radio(
        "Navigation",
        [
            "1. Headline Generation",
            "2. Relevance Analysis",
            "3. Sequential Comparison",
            "4. Research Dashboard",
            "5. About & Architecture",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Hardware & GPU Status")
    mgr = get_model_manager()
    gpu_info = mgr.get_gpu_memory_info()

    if gpu_info["available"]:
        st.sidebar.success(f"GPU: {gpu_info['device_name']}")
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Allocated", f"{gpu_info['allocated_mib']} MiB")
        col2.metric("Reserved", f"{gpu_info['reserved_mib']} MiB")
        vram_pct = int((gpu_info["reserved_mib"] / max(gpu_info["total_mib"], 1.0)) * 100)
        st.sidebar.progress(min(max(vram_pct, 0), 100))
        st.sidebar.caption(f"Physical Limit: {gpu_info['total_mib']} MiB (~4 GB)")
    else:
        st.sidebar.info("Running in CPU Mode")

    if st.sidebar.button("🧹 Clear GPU VRAM Cache", help="Reclaims allocated GPU memory and empties cache."):
        mgr.unload_active_generator()
        st.sidebar.success("VRAM cache cleared!")

    return page


def render_page_generation():
    """Page 1: Telugu Headline Generation with interactive controls."""
    st.markdown('<div class="main-title">Telugu Headline Generation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Generate concise, high-relevance Telugu news headlines from raw news stories using fine-tuned seq2seq models.</div>',
        unsafe_allow_html=True,
    )

    col_ctrl, col_input = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Model Configuration")
        model_options = {k: v["name"] for k, v in MODEL_REGISTRY.items()}
        selected_model_key = st.selectbox(
            "Select Checkpoint",
            options=list(model_options.keys()),
            format_func=lambda k: model_options[k],
            index=0,
            help="Choose between baseline mT5-small, IndicBART, or low-resource variants.",
        )
        meta = MODEL_REGISTRY[selected_model_key]
        st.info(f"**Architecture:** `{meta['architecture']}`\n\n{meta['description']}")

        st.subheader("Decoding Parameters")
        num_beams = st.slider("Beam Search Width (num_beams)", min_value=1, max_value=8, value=4, step=1)
        max_length = st.slider("Max Headline Length (tokens)", min_value=16, max_value=128, value=64, step=8)
        do_sample = st.checkbox("Enable Sampling (Creativity)", value=False)
        temperature = 1.0
        if do_sample:
            temperature = st.slider("Temperature", min_value=0.2, max_value=2.0, value=1.0, step=0.1)

    with col_input:
        st.subheader("Input Telugu Article")
        preset_idx = st.selectbox(
            "Load Sample Article (or enter custom text below):",
            options=["Custom Input"] + [s["title"] for s in PRESET_SAMPLES],
            index=1,
        )

        initial_text = ""
        initial_ref = ""
        if preset_idx != "Custom Input":
            for s in PRESET_SAMPLES:
                if s["title"] == preset_idx:
                    initial_text = s["article"]
                    initial_ref = s["reference_headline"]
                    break

        article_input = st.text_area(
            "Telugu News Story (వార్తా వ్యాసం):",
            value=initial_text,
            height=200,
            placeholder="ఇక్కడ తెలుగు వార్తా వ్యాసాన్ని నమోదు చేయండి...",
        )

        col_w, col_c = st.columns(2)
        words = len(article_input.split())
        chars = len(article_input)
        col_w.caption(f"Word Count: **{words}** words")
        col_c.caption(f"Character Count: **{chars}** characters")

        if initial_ref:
            st.caption(f"**Ground Truth Reference:** `{initial_ref}`")

        generate_btn = st.button("🚀 Generate Telugu Headline", type="primary", use_container_width=True)

    if generate_btn:
        if not article_input.strip():
            st.error("Please enter a valid Telugu news article.")
            return
        if len(article_input.strip()) < 20:
            st.warning("The input article is very short. Generating a headline might yield lower relevance.")

        mgr = get_model_manager()
        with st.spinner(f"Generating headline with {meta['name']} on {mgr.device}..."):
            try:
                res = mgr.generate(
                    model_key=selected_model_key,
                    article_text=article_input,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                
                st.markdown("---")
                st.subheader("Generated Headline")

                st.markdown(
                    f"""
                    <div class="headline-card">
                        <div class="telugu-text" style="font-size: 1.4rem; font-weight: 700; color: #0F172A;">
                            {html.escape(res['headline'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Metrics row
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric("Latency", f"{res['latency_seconds']:.2f} s")
                m_c2.metric("Length", f"{res['generated_words']} words")
                m_c3.metric("Device", res["device"].upper())
                m_c4.metric("Model", res["model_key"])

                # Copy & Download Action
                d_c1, d_c2 = st.columns([1, 1])
                d_c1.text_input("Copy Output:", value=res["headline"], key="copy_headline")
                export_json = json.dumps({
                    "model": res["model_name"],
                    "headline": res["headline"],
                    "latency_seconds": res["latency_seconds"],
                    "article": article_input,
                }, ensure_ascii=False, indent=2)
                d_c2.download_button(
                    "💾 Download Prediction JSON",
                    data=export_json.encode("utf-8"),
                    file_name="telugu_headline_prediction.json",
                    mime="application/json",
                )

            except Exception as e:
                st.error(f"Error during headline generation: {str(e)}")


def render_page_relevance():
    """Page 2: Telugu Headline Relevance Analysis (TeClass mBERT / TF-IDF)."""
    st.markdown('<div class="main-title">Headline Relevance Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Audit whether a candidate headline directly reflects the factual main event (FME) or is secondary/sensationalized.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        > [!NOTE]
        > **TeClass Taxonomy Definitions:**
        > - **HREL (Highly Related):** The headline directly captures the article's core Factual Main Event (**FME**).
        > - **MREL (Moderately Related):** The headline focuses on Secondary Events (**FSE**), Logical Conclusions (**STC**), or Minor Takeaways (**WKC**).
        > - **LREL (Least Related):** The headline is Sensationalized (**SEN**), Clickbait (**CBT**), Misleading (**MLC**), or Unsupported (**USO**).
        """
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Classifier Settings")
        clf_type = st.radio(
            "Select Classifier Architecture",
            options=["bert", "tfidf"],
            format_func=lambda x: "Multilingual BERT (mBERT, Phase 2 Final)" if x == "bert" else "TF-IDF + Logistic Regression (Baseline)",
            help="mBERT evaluates full cross-attention between article and headline.",
        )
        preset_idx = st.selectbox(
            "Load Sample Article & Candidate Headline:",
            options=["Custom Input"] + [s["title"] for s in PRESET_SAMPLES],
            index=1,
        )

        init_art = ""
        init_hl = ""
        if preset_idx != "Custom Input":
            for s in PRESET_SAMPLES:
                if s["title"] == preset_idx:
                    init_art = s["article"]
                    init_hl = s["reference_headline"]
                    break

        art_in = st.text_area("Article Text (వార్తా కథనం):", value=init_art, height=180)

    with col2:
        st.subheader("Candidate Headline")
        hl_in = st.text_input("Candidate Headline (శీర్షిక):", value=init_hl)
        st.caption("You can test ground-truth headlines, model-generated headlines, or deliberately misleading headlines.")

        classify_btn = st.button("🔍 Evaluate Relevance", type="primary", use_container_width=True)

    if classify_btn:
        if not art_in.strip() or not hl_in.strip():
            st.error("Please provide both an article and a headline to evaluate.")
            return

        mgr = get_model_manager()
        with st.spinner(f"Evaluating relevance with {clf_type.upper()} classifier..."):
            try:
                res = mgr.classify_relevance(article=art_in, headline=hl_in, model_type=clf_type)
                
                st.markdown("---")
                st.subheader("Classification Results")

                pred_class = res["predicted_class"]
                probs = res["probabilities"]

                badge_class = {
                    "HREL": "badge-hrel",
                    "MREL": "badge-mrel",
                    "LREL": "badge-lrel",
                }.get(pred_class, "badge-mrel")

                st.markdown(
                    f"""
                    <div style="margin-bottom: 1rem;">
                        <span style="font-size: 1.1rem; font-weight: 600; margin-right: 10px;">Predicted Category:</span>
                        <span class="{badge_class}">{pred_class}</span>
                    </div>
                    <p><em>{html.escape(res['explanation'])}</em></p>
                    """,
                    unsafe_allow_html=True,
                )

                # Probabilities breakdown
                p_c1, p_c2, p_c3 = st.columns(3)
                p_c1.metric("HREL Confidence", f"{probs.get('HREL', 0.0) * 100:.1f} %")
                p_c2.metric("MREL Confidence", f"{probs.get('MREL', 0.0) * 100:.1f} %")
                p_c3.metric("LREL Confidence", f"{probs.get('LREL', 0.0) * 100:.1f} %")

                # Progress bar visualization
                prob_df = pd.DataFrame({
                    "Category": ["HREL", "MREL", "LREL"],
                    "Probability": [probs.get("HREL", 0.0), probs.get("MREL", 0.0), probs.get("LREL", 0.0)]
                })
                st.bar_chart(prob_df.set_index("Category"))

                # Disclaimer
                st.caption(
                    "⚠️ **Disclaimer:** TeClass annotations reflect structural news alignment. "
                    "HREL indicates high overlap with the main factual event, but does not substitute for human editorial fact-checking."
                )

            except Exception as e:
                st.error(f"Error during relevance classification: {str(e)}")


def render_page_comparison():
    """Page 3: Sequential Model Comparison with safe memory unloading."""
    st.markdown('<div class="main-title">Sequential Model Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Compare two models on the exact same article while guaranteeing safe sequential execution within 4 GB VRAM.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        > [!IMPORTANT]
        > **VRAM Safety Guarantee:** To prevent out-of-memory errors on 4 GB GPUs, Model A is loaded, executed, and completely unloaded from VRAM before Model B is initialized.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        model_a_key = st.selectbox(
            "Select Model A",
            options=list(MODEL_REGISTRY.keys()),
            format_func=lambda k: MODEL_REGISTRY[k]["name"],
            index=0,  # mt5_small_final
        )
    with col2:
        model_b_key = st.selectbox(
            "Select Model B",
            options=list(MODEL_REGISTRY.keys()),
            format_func=lambda k: MODEL_REGISTRY[k]["name"],
            index=2,  # indicbart_final
        )

    preset_idx = st.selectbox(
        "Choose Comparison Article:",
        options=[s["title"] for s in PRESET_SAMPLES],
        index=0,
    )
    article_text = ""
    ref_headline = ""
    for s in PRESET_SAMPLES:
        if s["title"] == preset_idx:
            article_text = s["article"]
            ref_headline = s["reference_headline"]
            break

    custom_text = st.text_area("Article Content:", value=article_text, height=140)
    compare_btn = st.button("⚖️ Run Sequential Head-to-Head Comparison", type="primary", use_container_width=True)

    if compare_btn:
        if not custom_text.strip():
            st.error("Please provide valid article text.")
            return

        mgr = get_model_manager()
        
        # Step 1: Run Model A
        progress_bar = st.progress(10, text=f"Step 1/4: Loading and running {MODEL_REGISTRY[model_a_key]['name']}...")
        try:
            res_a = mgr.generate(model_key=model_a_key, article_text=custom_text)
            progress_bar.progress(40, text=f"Step 2/4: Unloading {MODEL_REGISTRY[model_a_key]['name']} to free VRAM...")
            mgr.unload_active_generator()

            # Step 2: Run Model B
            progress_bar.progress(60, text=f"Step 3/4: Loading and running {MODEL_REGISTRY[model_b_key]['name']}...")
            res_b = mgr.generate(model_key=model_b_key, article_text=custom_text)
            mgr.unload_active_generator()

            # Step 3: Run Classifier on both outputs
            progress_bar.progress(85, text="Step 4/4: Evaluating relevance of generated headlines...")
            rel_a = mgr.classify_relevance(article=custom_text, headline=res_a["headline"], model_type="bert")
            rel_b = mgr.classify_relevance(article=custom_text, headline=res_b["headline"], model_type="bert")
            progress_bar.progress(100, text="Comparison complete!")

            st.markdown("---")
            st.subheader("Comparison Results")
            if ref_headline:
                st.info(f"**Ground Truth Reference:** `{ref_headline}`")

            c_a, c_b = st.columns(2)

            with c_a:
                st.markdown(f"### {res_a['model_name']}")
                st.markdown(
                    f"""
                    <div class="headline-card">
                        <div class="telugu-text" style="font-weight: 600; font-size: 1.25rem;">
                            {html.escape(res_a['headline'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write(f"⏱️ **Latency:** `{res_a['latency_seconds']:.2f} s`")
                st.write(f"📏 **Length:** `{res_a['generated_words']} words`")
                st.write(f"🎯 **Predicted Relevance:** **{rel_a['predicted_class']}**")
                st.caption(f"HREL Confidence: {rel_a['probabilities']['HREL']*100:.1f}%")

            with c_b:
                st.markdown(f"### {res_b['model_name']}")
                st.markdown(
                    f"""
                    <div class="headline-card">
                        <div class="telugu-text" style="font-weight: 600; font-size: 1.25rem;">
                            {html.escape(res_b['headline'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write(f"⏱️ **Latency:** `{res_b['latency_seconds']:.2f} s`")
                st.write(f"📏 **Length:** `{res_b['generated_words']} words`")
                st.write(f"🎯 **Predicted Relevance:** **{rel_b['predicted_class']}**")
                st.caption(f"HREL Confidence: {rel_b['probabilities']['HREL']*100:.1f}%")

        except Exception as e:
            st.error(f"Error during comparison: {str(e)}")


def render_page_dashboard():
    """Page 4: Research Results & Experiment Dashboard."""
    st.markdown('<div class="main-title">Research Results & Experiment Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Authentic metrics, learning curves, and resource scaling from local training on RTX 3050.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["📊 Phase 3 Model Comparison", "📈 Low-Resource Scaling (Phase 4)", "🚫 Gemma 2 Feasibility Audit"])

    with tab1:
        st.subheader("Controlled Comparison: mT5-small vs IndicBART ($N=50$ Validation Samples)")
        comp_path = PROJECT_ROOT / "outputs" / "metrics" / "model_comparison_phase3.json"
        if comp_path.exists():
            with open(comp_path, "r", encoding="utf-8") as f:
                comp_data = json.load(f)
            
            models = comp_data.get("models", {})
            rows = []
            for m_name, m_stats in models.items():
                rows.append({
                    "Model": m_name,
                    "ROUGE-1": m_stats["rouge"]["rouge1"],
                    "ROUGE-2": m_stats["rouge"]["rouge2"],
                    "ROUGE-L": m_stats["rouge"]["rougeL"],
                    "chrF++": m_stats["chrf"],
                    "Avg Latency (s)": m_stats["avg_latency_s"],
                    "Peak VRAM (MiB)": m_stats["peak_vram_mib"],
                    "HREL Count": m_stats["relevance_distribution"]["HREL"],
                    "MREL Count": m_stats["relevance_distribution"]["MREL"],
                    "LREL Count": m_stats["relevance_distribution"]["LREL"],
                })
            df_comp = pd.DataFrame(rows)
            st.dataframe(df_comp, use_container_width=True)

            st.markdown(
                """
                **Key Scientific Observations:**
                - **mT5-small** achieved superior lexical overlap (**ROUGE-1: 0.1025**, **ROUGE-2: 0.0283**) compared to IndicBART on this Telugu domain.
                - **IndicBART** demonstrated strong character-level fidelity (**chrF: 14.02** vs 12.78) but suffers from high repetition and latency (2.62s vs 0.27s).
                - Both models maintained >50% high-relevance (**HREL**) predictions according to the TeClass classifier.
                """
            )
        else:
            st.warning("Comparison metrics file not found.")

    with tab2:
        st.subheader("Low-Resource Data Scaling (5% vs 25% vs 100%)")
        st.caption("All models trained deterministically for 75 steps under identical batch and learning configurations.")

        low_path = PROJECT_ROOT / "outputs" / "metrics" / "low_resource_results.json"
        if low_path.exists():
            with open(low_path, "r", encoding="utf-8") as f:
                low_data = json.load(f)

            summary_rows = []
            for tier, data in low_data.items():
                summary_rows.append({
                    "Subset Tier": f"{data['proportion_pct']}% Data",
                    "ROUGE-1": data["rouge"]["rouge1"],
                    "ROUGE-2": data["rouge"]["rouge2"],
                    "ROUGE-L": data["rouge"]["rougeL"],
                    "chrF++": data["chrf"],
                    "Train Time (s)": data["training_time_s"],
                    "Peak VRAM (MiB)": data["peak_vram_mib"],
                    "HREL Count": data["relevance_distribution"]["HREL"],
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            st.warning(
                "📌 **Integrity Note:** These preliminary metrics were evaluated on $N=15$ test samples. "
                "The 100% data tier achieved the highest overall ROUGE-1 (0.2230) and chrF++ (29.80)."
            )

        st.subheader("Learning Curves & Artifacts")
        p_c1, p_c2 = st.columns(2)
        plot_rouge = PROJECT_ROOT / "outputs" / "plots" / "learning_curve_rouge.png"
        plot_chrf = PROJECT_ROOT / "outputs" / "plots" / "learning_curve_chrf.png"

        if plot_rouge.exists():
            p_c1.image(str(plot_rouge), caption="ROUGE Score Trajectory across Data Tiers")
        if plot_chrf.exists():
            p_c2.image(str(plot_chrf), caption="chrF Score Trajectory across Data Tiers")

    with tab3:
        st.subheader("Gemma 2 2B Feasibility Audit (Phase 5)")
        gemma_path = PROJECT_ROOT / "outputs" / "metrics" / "gemma2_feasibility_report.json"
        if gemma_path.exists():
            with open(gemma_path, "r", encoding="utf-8") as f:
                gemma_data = json.load(f)

            st.error(f"Overall Status: **{gemma_data.get('overall_status', 'BLOCKED')}**")
            st.markdown("### Specific Hardware & Environmental Blockers")
            for b in gemma_data.get("blockers", []):
                st.markdown(f"- ❌ **{html.escape(b)}**")

            st.markdown("### Memory Budget Analysis")
            mb = gemma_data.get("stages", {}).get("stage3_memory_budget", {})
            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric("Parameter Count", mb.get("parameter_count", "2.61 B"))
            col_b2.metric("Full FP16 Weights", f"{mb.get('full_fp16_weights_gb', 0)} GB")
            col_b3.metric("Physical VRAM Limit", f"{mb.get('dedicated_vram_limit_gb', 0)} GB")

            st.info(
                "**Conclusion:** Gemma 2 2B cannot be fine-tuned or loaded in FP16 on 4 GB VRAM. "
                "Furthermore, 4-bit QLoRA is blocked due to Hugging Face gated access requirements and lack of native Windows `bitsandbytes` CUDA binaries."
            )


def render_page_about():
    """Page 5: System Overview, Architecture, and Limitations."""
    st.markdown('<div class="main-title">About Telugu News Headline Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">System design, technical limitations, and research ethics.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 1. Research Objective
        The objective of this project is to build an end-to-end, locally reproducible Telugu news headline generation
        and relevance evaluation pipeline. The project is tailored to low-resource Indian language settings and operates
        entirely within the constraints of an **NVIDIA RTX 3050 Laptop GPU (4 GB VRAM)**.

        ---

        ### 2. Architecture & Pipeline
        - **Generation:**
          - **mT5-small:** Multilingual Text-to-Text Transfer Transformer (~300M parameters). Fine-tuned on the Telugu news corpus.
          - **IndicBART:** Sequence-to-sequence model pretrained on Indic languages (~244M parameters) using `<2te>` language tokens.
        - **Classification:**
          - **mBERT Classifier:** Cross-attention relevance classifier fine-tuned on the TeClass dataset to evaluate whether candidate headlines reflect the **Factual Main Event (FME)**.
          - **TF-IDF Baseline:** Character and word n-gram linear model for fast baseline inference.

        ---

        ### 3. Hardware Constraints & VRAM Management
        - **Physical VRAM:** Strictly limited to 4,095.5 MiB.
        - **Memory Optimization:**
          - **Single Active Model:** At any time, at most one transformer is loaded in GPU memory.
          - **Garbage Collection & Cache Purge:** When switching models, the previous model is garbage collected and `torch.cuda.empty_cache()` is called.
          - **Mixed Precision (FP16):** Used during training to prevent Out-Of-Memory (OOM) errors.

        ---

        ### 4. Known Limitations & Research Caveats
        1. **Repetition in Generation:** IndicBART displays phrase repetition without aggressive `no_repeat_ngram_size` constraints.
        2. **TeClass Classifier Sensitivity:** The mBERT classifier achieves a Macro-F1 of ~0.51. HREL predictions reflect structural alignment with the lead event, but should not be treated as a guarantee of complete factual accuracy.
        3. **Low-Resource Evaluation Size:** Preliminary low-resource metric points were calculated over $N=15$ test samples due to local compute budgets.
        4. **Gemma 2 Status:** Documented and verified as **BLOCKED** on this hardware setup.
        """
    )


def main():
    page = render_sidebar()
    if page == "1. Headline Generation":
        render_page_generation()
    elif page == "2. Relevance Analysis":
        render_page_relevance()
    elif page == "3. Sequential Comparison":
        render_page_comparison()
    elif page == "4. Research Dashboard":
        render_page_dashboard()
    elif page == "5. About & Architecture":
        render_page_about()


if __name__ == "__main__":
    main()
