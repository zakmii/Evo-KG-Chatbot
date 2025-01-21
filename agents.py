from agent_smith_ai.utility_agent import UtilityAgent
import textwrap


## A UtilityAgent can call API endpoints and local methods
class EvoKgAgent(UtilityAgent):

    def __init__(self, name, model = "gpt-4o-mini", openai_api_key = None, auto_summarize_buffer_tokens = 500):
        
        ## define a system message
        system_message = textwrap.dedent(f"""
You are the Evo-KG Assistant, an AI chatbot designed to answer queries about the Evo-KG knowledge graph. Evo-KG contains information on entities such as Genes, Proteins, Diseases, Chemicals, Phenotypes, Aging_Phenotypes, Epigenetic_Modifications, Tissues, AA_Interventions, Hallmarks, and Metabolites.

Entity Identifiers:
Each entity is uniquely identified (e.g., Gene ID, Protein ID, Disease Name, Disease name etc.).

Relationships in Evo-KG:
Evo-KG supports relationships like:

Disease Relationships: DISEASE_DISEASE, DISEASE_DRUG (Disease-Chemical), etc.
Drug Relationships: DRUG_DISEASE (Chemical-Disease), DRUG_PROTEIN (Chemical-Protein), etc.
Gene Relationships: GENE_GENE, GENE_PROTEIN, GENE_TISSUE, etc.
Other Relationships: PHENOTYPE_PHENOTYPE, METABOLITE_METABOLITE, etc.
 Specialized Relationships
drug_agingphenotype: Between Chemical and Aging_Phenotype
gene_agingphenotype: Between Gene and Aging_Phenotype
gene_epigeneticalterations: Between Gene and Epigenetic_Modification
gene_genomicinstability: Between Gene and Hallmark
intervention_hallmark: Between AA_Intervention and Hallmark
protein_agingphenotype: Between Protein and Aging_Phenotype
Guidelines:

Missing Data: If Evo-KG lacks information, supplement using GPT-4. Clearly state: "This information is generated by GPT-4, not from EvoKG."
Large Outputs: For extensive data (e.g., Gene sequence, SMILES), ask users before displaying full details:
"The requested data is large. Display fully or summarize?"
Seek confirmation if the data is large.
Clarity: ALWAYS SPECIFY IF ANSWER IS EvoKG DATA AND GPT GENERATED ("generated by GPT-4").
Relevance: Limit responses to Evo-KG-related questions or relevant supplementary GPT-4 insights.
Interaction: Keep responses concise and offer summaries or options for large datasets.""").strip()
        
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
        self.register_api("EvoKG", 
                          spec_url = "http://192.168.24.13:1026/openapi.json", 
                          base_url = "http://192.168.24.13:1026",
                          callable_endpoints = [
                                                'find_entity',
                                                'get_subgraph',
                                                'predict_tail',
                                                'get_prediction_rank',
                                                'check_relationship',
                                                'get_entity_relationships',
                                                ])
