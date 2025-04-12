import kani_utils.kani_streamlit_server as ks
import os
import dotenv
from kani.engines.openai import OpenAIEngine
from agents import EvoKgAgent
import pathlib
import logging

dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path for local assets to avoid path issues
current_dir = pathlib.Path(__file__).parent.absolute()
logo_path = str(current_dir / "logo.png")
bg_image_path = str(current_dir / "floating-graph-nodes.png")

# Check if files exist
if not pathlib.Path(logo_path).exists():
    logger.warning(f"Logo file not found at {logo_path}")
    logo_path = None

if not pathlib.Path(bg_image_path).exists():
    logger.warning(f"Background image not found at {bg_image_path}, using default URL")
    # Use remote URL as fallback
    bg_image_path = "https://www.nayuki.io/res/animated-floating-graph-nodes/floating-graph-nodes.png"
else:
    logger.info(f"Using local background image: {bg_image_path}")
    # For local files in Streamlit, they need to be loaded differently
    # We'll use a data URL approach for local images if needed
    # This remains a URL string, so it will be handled correctly

# initialize the application and set some page settings
# parameters here are passed to streamlit.set_page_config,
# see more at https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config
# this function MUST be run first
ks.initialize_app_config(
    show_function_calls=False,
    page_title="EvoLLM",
    app_title="EvoLLM",
    logo_path=logo_path,
    background_image=bg_image_path,
    page_icon="🧬",  # can also be a URL
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/zakmii/Evo-KG-Chatbot",
        "Report a Bug": "https://github.com/zakmii/Evo-KG-Chatbot/issues",
        "About": "EvoLLM is built on GPT-4o-mini, Streamlit, zhudotexe/kani, hourfu/redlines, and oneilsh/kani-utils.",
    },
)

# define an engine to use (see Kani documentation for more info)
engine = OpenAIEngine(os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
# mistralEngine = HuggingEngine(
#     model_id="mistralai/Mistral-7B-Instruct-v0.3", token=os.environ["MISTRAL_TOKEN"]
# )


# We also have to define a function that returns a dictionary of agents to serve
# Agents are keyed by their name, which is what the user will see in the UI
def get_agents():
    return {
        "EvoLLM (4o-mini)": EvoKgAgent(
            engine
        ),  # prompt_tokens_cost = 0.005, completion_tokens_cost = 0.015),
        # "EvoLLM (Mistral)": EvoKgAgent(mistralEngine),
    }


# tell the app to use that function to create agents when needed
ks.set_app_agents(get_agents)


########################
##### 3 - Serve App
########################

ks.serve_app()
