from agent_smith_ai.utility_agent import UtilityAgent

import textwrap
import os
from typing import Any, Dict



## A UtilityAgent can call API endpoints and local methods
class EvoKgAgent(UtilityAgent):

    def __init__(self, name, model = "gpt-4o-mini", openai_api_key = None, auto_summarize_buffer_tokens = 500):
        
        ## define a system message
        system_message = textwrap.dedent(f"""
            You are the Evo-KG Assistant, an AI-powered chatbot that can answer questions about data from the Evo-KG knowledge graph. 
            You can retrieve information about genes and proteins from the Evo-KG knowledge graph.
            Given a gene or protein ID, you can retrieve information about the gene or protein, including its name, description.
            """).strip()
        
        super().__init__(name,                                             # Name of the agent
                         system_message,                                   # Openai system message
                         model = model,                     # Openai model name
                         openai_api_key = openai_api_key,    # API key; will default to OPENAI_API_KEY env variable
                         auto_summarize_buffer_tokens = auto_summarize_buffer_tokens,               # Summarize and clear the history when fewer than this many tokens remains in the context window. Checked prior to each message sent to the model.
                         summarize_quietly = False,                        # If True, do not alert the user when a summarization occurs
                         max_tokens = None,                                # maximum number of tokens this agent can bank (default: None, no limit)
                         token_refill_rate = 50000.0 / 3600.0)             # number of tokens to add to the bank per second

        ## register some API endpoints (inherited from UtilityAgent)
        ## the openapi.json spec must be available at the spec_url:
        ##    callable endpoints must have a "description" and "operationId"
        ##    params can be in body or query, but must be fully specified
        self.register_api("monarch", 
                          spec_url = "https://neo4j-fastapi.vercel.app/openapi.json", 
                          base_url = "https://neo4j-fastapi.vercel.app",
                          callable_endpoints = ['get_gene',
                                                'get_protein'])