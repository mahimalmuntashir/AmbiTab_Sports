# Model identity and runtime

Exact repository IDs are supplied for Qwen2.5-7B-Instruct, google/gemma-4-E4B-it, and Mistral-7B-Instruct-v0.2. Gemma 4 E4B has 4.5B effective parameters and 8B with embeddings according to its official model card; it is distinct from Gemma 3n. Historical commit SHAs remain not_recorded and must not be invented or replaced by current HEAD. The pinned analysis dependencies describe the repaired CPU analysis, not a historical GPU lock.

Optional model_inference.py supports explicit revision arguments for new exploratory runs. It retains historical-style prompt handling and is not exercised by saved-response reproduction. New experiments need their own recorded revisions and settings.
