import streamlit as st
import threading
from agents import EvoKgAgent
import logging
import base64

st.set_page_config(
    page_title="EvoKG Assistant",
    layout="centered",
    initial_sidebar_state="expanded",
)


def get_img_as_base64(file_path: str):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.warning(f"Could not load image {file_path}: {str(e)}")
        return None


def clear_chat():
    if "agents" in st.session_state:
        for agent in st.session_state.agents.values():
            agent["agent"].clear_chat_history()
    st.rerun()


def set_clear_chat_timer():
    threading.Timer(3600, clear_chat).start()  # 3600 seconds = 1 hour


def initialize_session_state():
    if "logger" not in st.session_state:
        st.session_state.logger = logging.getLogger(__name__)
        st.session_state.logger.handlers = []
        st.session_state.logger.setLevel(logging.INFO)
        st.session_state.logger.addHandler(logging.StreamHandler())

    st.session_state.setdefault("user_api_key", "")
    st.session_state.setdefault(
        "original_api_key", st.secrets["OPENAI_API_KEY"]
    )  # Store the original API key
    st.session_state.setdefault("show_function_calls", False)
    st.session_state.setdefault("ui_disabled", False)
    st.session_state.setdefault("lock_widgets", False)

    greeting = """
        # Welcome to EvoKG Chatbot

        #### I'm the EvoKG Assistant, and I’m here to help you explore and understand the EvoKG knowledge graph.

        ## Sample Questions You Can Ask
        To get started, try asking questions like:

        * Get details about the disease Stomach Neoplasms.
        * How many nodes are connected to Stomach Neoplasms in EvoKG?
        * Predict new drug-disease links for a specific drug.
        * Predict new drug-aging phenotype links for the same drug.

        These examples highlight how EvoKG can answer specific queries and assist in predictive biological analysis.

        #### Feel free to ask questions, and I’ll do my best to assist you!
        ---
                """

    if "agents" not in st.session_state:
        st.session_state.agents = {
            "EvoKG Assistant": {
                "agent": EvoKgAgent(
                    "EvoKG Assistant",
                    model="gpt-4o-mini",
                    openai_api_key=get_current_api_key_for_agent_use(),
                    auto_summarize_buffer_tokens=10000,
                ),
                "greeting": greeting,
                "avatar": "ℹ️",
                "user_avatar": "👤",
            }
        }

    if "clear_chat_timer_set" not in st.session_state:
        set_clear_chat_timer()
        st.session_state.clear_chat_timer_set = True

    st.session_state.setdefault(
        "current_agent_name", list(st.session_state.agents.keys())[0]
    )

    for agent in st.session_state.agents.values():
        if "conversation_started" not in agent:
            agent["conversation_started"] = False
        if "messages" not in agent:
            agent["messages"] = []

    st.session_state.setdefault("current_page", "intro")

    # Add navigation menu style
    st.session_state.setdefault(
        "nav_style",
        """
        <style>
        .nav-link {
            padding: 8px 16px;
            text-decoration: none;
            border-radius: 4px;
            margin-bottom: 4px;
            display: inline-block;
            width: 100%;
            color: #444;
        }
        .nav-link:hover {
            background-color: #f0f2f6;
        }
        .nav-link.active {
            background-color: #e6e9ef;
            font-weight: bold;
        }
        </style>
    """,
    )


def initialize_page():
    try:
        # Apply background globally for dark theme
        page_bg_img = """
        <style>
        [data-testid="stAppViewContainer"] {
            background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                        url("https://www.nayuki.io/res/animated-floating-graph-nodes/floating-graph-nodes.png");
            background-size: cover;
            background-position: center;
        }

        [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0);
        }
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not set background: {str(e)}")


def get_current_api_key_for_agent_use():
    """Get the current API key, either the user's or the default."""
    key = (
        st.session_state.user_api_key
        if st.session_state.user_api_key
        else st.session_state.original_api_key
    )
    if key is None:
        key = "placeholder"
    return key


def update_agents_api_key():
    """Update all agents with the new API key."""
    for agent in st.session_state.agents.values():
        agent["agent"].set_api_key(get_current_api_key_for_agent_use())


def has_valid_api_key():
    """Check if a valid API key is available."""
    return bool(st.session_state.user_api_key) or bool(
        st.session_state.original_api_key
    )


def render_message(message):
    """Render a chat message in Streamlit."""
    current_agent_avatar = st.session_state.agents[
        st.session_state.current_agent_name
    ].get("avatar", None)
    current_user_avatar = st.session_state.agents[
        st.session_state.current_agent_name
    ].get("user_avatar", None)

    if message.role == "user":
        with st.chat_message("user", avatar=current_user_avatar):
            st.write(message.content)

    elif message.role == "system":
        with st.chat_message("assistant", avatar="ℹ️"):
            st.write(message.content)

    elif message.role == "assistant" and not message.is_function_call:
        with st.chat_message("assistant", avatar=current_agent_avatar):
            st.write(message.content)

    if st.session_state.show_function_calls:
        if message.is_function_call:
            with st.chat_message("assistant", avatar="🛠️"):
                st.text(f"{message.func_name}(params = {message.func_arguments})")
        elif message.role == "function":
            with st.chat_message("assistant", avatar="✔️"):
                st.text(message.content)

    current_action = "*Thinking...*"

    if message.is_function_call:
        current_action = f"*Checking source ({message.func_name})...*"
    elif message.role == "function":
        current_action = f"*Evaluating result ({message.func_name})...*"

    return current_action


def handle_chat_input():
    """Handle user chat input in the Streamlit chat interface."""
    if prompt := st.chat_input(
        disabled=st.session_state.lock_widgets, on_submit=lock_ui
    ):
        agent = st.session_state.agents[st.session_state.current_agent_name]

        # Continue with conversation
        if not agent.get("conversation_started", False):
            messages = agent["agent"].chat(prompt, yield_prompt_message=True)
            agent["conversation_started"] = True
        else:
            messages = agent["agent"].chat(prompt, yield_prompt_message=True)

        st.session_state.current_action = "*Thinking...*"
        while True:
            try:
                with st.spinner(st.session_state.current_action):
                    message = next(messages)
                    agent["messages"].append(message)
                    st.session_state.current_action = render_message(message)

                    # Log message info
                    session_id = st.runtime.scriptrunner.add_script_run_ctx().streamlit_script_run_ctx.session_id
                    info = {
                        "session_id": session_id,
                        "message": message.model_dump(),
                        "agent": st.session_state.current_agent_name,
                    }
                    st.session_state.logger.info(info)
            except StopIteration:
                break

        st.session_state.lock_widgets = False
        st.rerun()


def clear_chat_current_agent():
    """Clear chat for the current agent."""
    current_agent = st.session_state.agents[st.session_state.current_agent_name]
    current_agent["conversation_started"] = False
    current_agent["agent"].clear_history()
    st.session_state.agents[st.session_state.current_agent_name]["messages"] = []


def lock_ui():
    """Lock the UI when user submits input."""
    st.session_state.lock_widgets = True


def show_intro_page():
    """
    Show the Introduction page with content
    """
    st.markdown(
        """
        <div style="display: flex; align-items: center;">
            <!-- Main heading text -->
            <h1 style="margin-right: 10px;">Welcome to EvoKG Chatbot</h1>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # Custom hover CSS for sections
    st.markdown(
        """
        <style>
        .hover-section {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            padding: 1rem;
            border-radius: 4px;
        }
        .hover-section:hover {
            transform: translateY(-15px) scale(1.02);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- INTRO SECTIONS ---
    st.markdown(
        """
        <div class="hover-section">
          <h2>What is EvoKG?</h2>
          <p>
            EvoKG is a groundbreaking Evolutionary Knowledge Graph that brings together
            biological insights across six species, organized in evolutionary order:
          </p>
          <ul>
            <li>Y: <em>Saccharomyces cerevisiae</em> (Yeast)</li>
            <li>C: <em>Caenorhabditis elegans</em> (Nematode)</li>
            <li>D: <em>Drosophila melanogaster</em> (Fruit Fly)</li>
            <li>Z: <em>Danio rerio</em> (Zebrafish)</li>
            <li>M: <em>Mus musculus</em> (Mouse)</li>
            <li>H: <em>Homo sapiens</em> (Human)</li>
          </ul>
          <p>
            This comprehensive knowledge graph serves as a biological encyclopedia
            with a special focus on metabolites, genes, and proteins, their relationships,
            and their roles in health, disease, and aging. EvoKG includes 907,746 entities,
            connected by 31 distinct relationship types, forming a network of 175,678,450 links.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hover-section">
            <h2>Key Features of EvoKG</h2>
            <ul>
              <li><strong>Cross-Species Insights:</strong> Explore data from six species to understand evolutionary relationships and biological functions.</li>
              <li><strong>Metabolite Encyclopedia:</strong> Access detailed information about metabolites, genes, and proteins, and their alternative functions across species.</li>
              <li><strong>Aging-Related Data:</strong> Delve into connections between aging and its associated metabolites, genes, and proteins.</li>
              <li><strong>Link Prediction:</strong> Uncover new potential relationships (e.g., between drugs and diseases or aging phenotypes) using EvoKG's predictive capabilities.</li>
              <li><strong>Biological Complexity:</strong> Navigate through diverse node types, including:
                <ul>
                  <li>GENES, PROTEINS, CHEMICALS, METABOLITES, PHENOTYPES, DISEASES</li>
                  <li>AGING_PHENOTYPES, HALLMARKS, TISSUES</li>
                  <li>ANTI-AGING INTERVENTIONS, EPIGENETIC_MODIFICATIONS</li>
                </ul>
              </li>
              <li><strong>Relationship Diversity:</strong> Investigate interactions with 31 edge types, such as gene-disease, protein-tissue, chemical-aging phenotype, and more.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="hover-section">
            <h2>What Can EvoKG Do for You?</h2>
            <p>
              EvoKG is designed to assist researchers, biologists, and data scientists in exploring,
              analyzing, and generating new insights into complex biological systems. You can use
              this chatbot to:
            </p>
            <ul>
              <li>Access encyclopedic details about specific diseases, metabolites, or genes.</li>
              <li>Discover new alternative functions for biological entities like metabolites, genes, and proteins.</li>
              <li>Gain insights into aging and its associated biochemical pathways.</li>
              <li>Predict new links between existing entities in the knowledge graph, helping you hypothesize novel biological interactions.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="hover-section">
            <h2>Let's Explore!</h2>
            <p>
              With EvoKG, you're not just asking questions—you're unlocking the mysteries of evolution, aging,
              and biological complexity across species. Start exploring today and see how EvoKG can advance
              your research and understanding!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Centered button with custom styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Exploring with EvoKG Chatbot →", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()


def show_chat_page():
    """Show the chatbot page."""
    st.header(st.session_state.current_agent_name)

    current_agent_avatar = st.session_state.agents[
        st.session_state.current_agent_name
    ].get("avatar", None)
    with st.chat_message("assistant", avatar=current_agent_avatar):
        st.write(
            st.session_state.agents[st.session_state.current_agent_name]["greeting"]
        )

    for message in st.session_state.agents[st.session_state.current_agent_name][
        "messages"
    ]:
        render_message(message)

    if has_valid_api_key():
        handle_chat_input()
    else:
        st.chat_input(placeholder="Enter an API key to begin chatting.", disabled=True)


def show_tutorial_page():
    """Show the tutorial page."""
    st.title("EVOKG Tutorials")
    st.markdown(
        """
        ## Getting Started
        Learn how to effectively use the EvoKG Assistant with these tutorials:

        ### Basic Queries
        1. Searching for genes
        2. Finding disease relationships
        3. Exploring protein interactions

        ### Advanced Features
        1. Complex relationship queries
        2. Subgraph analysis
        3. Entity predictions
        """
    )


def show_contact_page():
    """Show the contact page."""
    st.title("Contact Us")
    st.markdown(
        """
        ## Get in Touch

        For questions, feedback, or support:
        - Email: gaurav.ahuja@iiitd.ac.in
        - GitHub: [EvoKG Repository](https://github.com/zakmii/Evo-KG-Chatbot/tree/main)
        - Twitter: [@EvoKG](https://twitter.com/evokg)

        Developer:
        - Ankit Singh : https://github.com/zakmii
        - Arushi Sharma : https://github.com/AruShar

        ### Report Issues
        If you encounter any problems, please report them on our GitHub repository.
        """
    )


def main():
    """Main Streamlit UI."""
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
                <img src="data:image/png;base64,{}" alt="Logo" style="height: 150px;">
                <h1 style="margin: 0; font-size: 44px;">EvoKG</h1>
            </div>
        """.format(get_img_as_base64("logo.png")),
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Navigation menu
        pages = {
            "intro": "Introduction",
            "chat": "Chatbot",
            "tutorial": "Tutorial",
            "contact": "Contact Us",
        }

        # Custom CSS for navigation
        st.markdown(st.session_state.nav_style, unsafe_allow_html=True)

        # Navigation buttons with active state
        for page_id, page_name in pages.items():
            if st.button(
                page_name,
                key=f"nav_{page_id}",
                help=f"Go to {page_name} page",
                use_container_width=True,
            ):
                st.session_state.current_page = page_id
                st.rerun()

        st.markdown("---")

        # Only show agent controls on chat page
        if st.session_state.current_page == "chat":
            st.button(
                label="Clear chat for current assistant",
                on_click=clear_chat_current_agent,
                disabled=st.session_state.lock_widgets,
            )

            st.markdown("---")
            # If you need an API Key input here, uncomment:
            user_key = st.text_input(
                "Set API Key",
                value=st.session_state.user_api_key,
                max_chars=51,
                type="password",
                help="Enter your OpenAI API key here to override the default provided by the app.",
                disabled=st.session_state.lock_widgets,
            )
            if user_key != st.session_state.user_api_key and len(user_key) == 51:
                st.session_state.user_api_key = user_key
                update_agents_api_key()
                st.write(f"Using API key: `{user_key[:3]}...{user_key[-3:]}`")

    # Display current page
    if st.session_state.current_page == "intro":
        show_intro_page()
    elif st.session_state.current_page == "chat":
        show_chat_page()
    elif st.session_state.current_page == "tutorial":
        show_tutorial_page()
    elif st.session_state.current_page == "contact":
        show_contact_page()


if __name__ == "__main__":
    initialize_page()
    initialize_session_state()
    main()
