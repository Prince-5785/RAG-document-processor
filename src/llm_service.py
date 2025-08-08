import json
import logging
import re
import asyncio
import random
from typing import Dict, Any, List, Optional, Callable

try:
    from groq import Groq, APIError, AsyncGroq, RateLimitError
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq SDK not available. Install with 'pip install groq'")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with 'pip install transformers torch'")

from .utils import Timer, extract_json_from_text


class LLMService:
    """
    Service for LLM-based tasks with key rotation, exponential backoff, and a fallback chain.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config.get('llm', {})
        self.logger = logging.getLogger(__name__)

        api_config = self.llm_config.get('api', {})
        self.model_sequence = api_config.get('model_sequence', [])
        
        self.groq_api_keys = api_config.get('groq_api_keys', [])
        self.key_index = 0
        if not self.groq_api_keys:
            self.logger.warning("No Groq API keys found in configuration. API calls will fail.")

        retry_config = api_config.get('retries', {})
        self.max_attempts = retry_config.get('max_attempts', 5)
        self.initial_delay = retry_config.get('initial_delay', 1.0)
        self.max_delay = retry_config.get('max_delay', 10.0)
        
        local_model_config = self.llm_config.get('local_model', {})
        self.local_model_name = local_model_config.get('model_name')
        self.device = local_model_config.get('device', 'cpu')

        gen_params = self.llm_config.get('generation_params', {})
        self.temperature = gen_params.get('temperature', 0.1)
        self.top_p = gen_params.get('top_p', 0.9)
        self.max_tokens_parsing = gen_params.get('max_new_tokens_parsing', 250)
        self.max_tokens_decision = gen_params.get('max_new_tokens_decision', 500)

        self.local_model = None
        self.local_tokenizer = None
        self.local_pipeline = None
        
        self.load_local_model()

    def load_local_model(self) -> None:
        """Loads the local language model for the final LLM fallback."""
        if not TRANSFORMERS_AVAILABLE or not self.local_model_name:
            self.logger.info("No local model configured or transformers not installed. Skipping load.")
            return
        if self.local_model is not None:
            return
        
        self.logger.info(f"Loading local fallback LLM model: {self.local_model_name}")
        try:
            with Timer(f"Loading local LLM model {self.local_model_name}"):
                self.local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)
                if self.local_tokenizer.pad_token is None:
                    self.local_tokenizer.pad_token = self.local_tokenizer.eos_token
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    self.local_model_name, device_map='auto' if self.device == 'cuda' else None
                )
                self.local_pipeline = pipeline(
                    "text-generation",
                    model=self.local_model,
                    tokenizer=self.local_tokenizer,
                    device=0 if self.device == 'cuda' and torch.cuda.is_available() else -1
                )
                self.logger.info(f"Local model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load local LLM model: {e}")
            self.local_pipeline = None

    async def parse_query(self, query: str, preferred_model: Optional[str] = None) -> Dict[str, Any]:
        """Asynchronously parses a user query using the sequential fallback chain."""
        self.logger.info(f"Starting query parsing for: '{query[:50]}...'")
        prompt = self._create_query_parsing_prompt(query)
        parsed_data = await self._execute_task_with_fallback(
            prompt=prompt,
            parse_function=self._extract_json_from_response,
            max_new_tokens=self.max_tokens_parsing,
            preferred_model=preferred_model
        )
        if parsed_data:
            return parsed_data
        self.logger.warning("All LLM attempts failed. Using final rule-based parser.")
        return self._fallback_query_parsing(query)

    async def make_decision(self, query_data: Dict[str, Any], retrieved_contexts: List[Dict[str, Any]], preferred_model: Optional[str] = None) -> Dict[str, Any]:
        """Asynchronously makes an insurance decision using the sequential fallback chain."""
        self.logger.info("Starting decision making process.")
        prompt = self._create_decision_prompt(query_data, retrieved_contexts)
        decision_data = await self._execute_task_with_fallback(
            prompt=prompt,
            parse_function=self._parse_decision_response,
            max_new_tokens=self.max_tokens_decision,
            preferred_model=preferred_model
        )
        if decision_data:
            return decision_data
        self.logger.warning("All LLM attempts failed. Using final rule-based decision maker.")
        return self._fallback_decision_making(query_data, retrieved_contexts)

    async def answer_question(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """Asynchronously generates a direct answer to a question using provided context."""
        self.logger.info(f"Generating answer for question: '{question[:50]}...'")
        prompt = self._create_qa_prompt(question, contexts)

        answer_data = await self._execute_task_with_fallback(
            prompt=prompt,
            parse_function=lambda text: {"answer": text.strip()},
            max_new_tokens=300,
            # preferred_model="llama-3.1-8b-instant"
        )

        if answer_data and answer_data.get("answer"):
            return answer_data["answer"]
        
        self.logger.warning("All LLM attempts failed to generate an answer.")
        return "The information could not be retrieved from the provided documents."

    async def _execute_task_with_fallback(self, prompt: str, parse_function: Callable, max_new_tokens: int, preferred_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Core logic to attempt a task with a sequence of models."""
        model_run_sequence = list(self.model_sequence)
        if preferred_model and preferred_model in model_run_sequence:
            model_run_sequence.remove(preferred_model)
            model_run_sequence.insert(0, preferred_model)

        for model_id in model_run_sequence:
            response_text = await self._generate_response_from_api(model_id, prompt, max_new_tokens)
            if response_text:
                parsed_result = parse_function(response_text)
                if parsed_result:
                    self.logger.info(f"Success with API model: {model_id}")
                    if isinstance(parsed_result, dict):
                        parsed_result['model_used'] = model_id
                    return parsed_result
                else:
                    self.logger.warning(f"Failed to parse response from {model_id}.")
            self.logger.warning(f"Attempt with {model_id} failed. Trying next model.")

        if self.local_pipeline:
            loop = asyncio.get_running_loop()
            local_response = await loop.run_in_executor(
                None, self._generate_response_from_local, prompt, max_new_tokens
            )
            if local_response:
                parsed_result = parse_function(local_response)
                if parsed_result:
                    self.logger.info(f"Success with local model: {self.local_model_name}")
                    if isinstance(parsed_result, dict):
                        parsed_result['model_used'] = self.local_model_name
                    return parsed_result
        return None

    def _get_next_api_key(self) -> Optional[str]:
        """Rotates through the list of API keys."""
        if not self.groq_api_keys:
            return None
        key = self.groq_api_keys[self.key_index]
        self.key_index = (self.key_index + 1) % len(self.groq_api_keys)
        return key

    async def _generate_response_from_api(self, model_id: str, prompt: str, max_new_tokens: int) -> Optional[str]:
        """Asynchronously generates a response using Groq API with key rotation and exponential backoff."""
        for attempt in range(self.max_attempts):
            api_key = self._get_next_api_key()
            if not api_key:
                self.logger.error("No API keys available to make a request.")
                return None
            
            try:
                async_client = AsyncGroq(api_key=api_key)
                chat_completion = await async_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_id,
                    temperature=self.temperature,
                    max_tokens=max_new_tokens,
                    top_p=self.top_p,
                )
                return chat_completion.choices[0].message.content
            except (RateLimitError, APIError) as e:
                self.logger.warning(f"API Error (attempt {attempt + 1}/{self.max_attempts}) with key index {self.key_index}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}", exc_info=True)

            if attempt < self.max_attempts - 1:
                delay = min(self.initial_delay * (2 ** attempt), self.max_delay)
                jitter = delay * random.uniform(0.1, 0.5)
                wait_time = delay + jitter
                self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
        
        self.logger.error(f"API request failed after {self.max_attempts} attempts for model {model_id}.")
        return None

    def _generate_response_from_local(self, prompt: str, max_new_tokens: int) -> Optional[str]:
        """Generates a response using the local Hugging Face pipeline (synchronous)."""
        if not self.local_pipeline: return None
        outputs = self.local_pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.local_tokenizer.eos_token_id,
            return_full_text=False
        )
        if outputs and len(outputs) > 0:
            return outputs[0]['generated_text'].strip()
        return None

    def _create_query_parsing_prompt(self, query: str) -> str:
        """Creates a prompt for the query parsing task."""
        return f"""You are an expert at extracting structured information from insurance queries. Analyze the user's query and output a single, valid JSON object. Do not add any extra text or explanations.

Query: "{query}"

JSON format:
{{
  "age": <integer or null>, "gender": "<male/female/null>", "procedure": "<medical procedure or null>",
  "location": "<city/location or null>", "policy_duration_months": <integer or null>, "policy_type": "<policy type or null>"
}}

JSON:
"""
    
    def _create_decision_prompt(self, query_data: Dict[str, Any], contexts: List[Dict[str, Any]]) -> str:
        """Creates a prompt for the decision-making task using multi-step reasoning."""
        context_text = ""
        for i, context in enumerate(contexts[:5], 1):
            context_text += f"Clause {i}: {context.get('text', '')}\n\n"
        
        return f"""You are a senior insurance claims processor. Your task is to meticulously evaluate an insurance claim based on the provided information and policy clauses. You must follow a strict reasoning process.

### REASONING STEPS:
1.  **Analyze Policy Requirements:** First, carefully read the 'Relevant Policy Clauses' and identify the key conditions for approval.
2.  **Assess User Information:** Review the 'Query Information' provided. Note which details are present and which are missing.
3.  **Compare and Identify Gaps:** Compare the user's information against the policy requirements. Explicitly state any critical information that is missing.
4.  **Formulate a Conclusion:** Based on your comparison, make a final decision.
    - **APPROVE:** Only if you have all necessary information and the claim clearly meets all policy requirements.
    - **REJECT:** If the claim clearly violates a policy rule OR if critical information is missing.
5.  **Write Justification:** Your justification must be based on your reasoning. If rejecting due to missing information, you MUST state exactly what information is needed.

---
### TASK:

**Query Information:**
{json.dumps(query_data, indent=2)}

**Relevant Policy Clauses:**
{context_text}

---
### FINAL OUTPUT:
Provide your response strictly in the following format.

Decision: [APPROVED/REJECTED]
Payout: ₹[integer amount]
Justification: [Your clear, step-by-step justification.]

Response:"""

    def _create_qa_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """Creates a prompt for the question answering task."""
        if not contexts:
            context_text = "No relevant context found."
        else:
            context_text = "\n\n---\n\n".join([ctx.get('text', '') for ctx in contexts])

        return f"""You are an expert assistant specialized in reading and understanding insurance policy documents. Your task is to answer the user's question based *only* on the provided context.

**CONTEXT FROM THE POLICY DOCUMENT:**
{context_text}

**USER'S QUESTION:**
{question}

**INSTRUCTIONS:**
- Answer concisely and accurately.
- If the answer cannot be found in the context, you MUST state: "The answer to this question could not be found in the provided document."
- Do not use any prior knowledge.

**ANSWER:**
"""

    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extracts a JSON object from a model's response text."""
        return extract_json_from_text(response)
    
    def _parse_decision_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parses the structured decision from a response."""
        try:
            decision_match = re.search(r'Decision:\s*(APPROVED|REJECTED)', response, re.IGNORECASE)
            payout_match = re.search(r'Payout:\s*₹?\s*(\d[\d,]*\d|\d)', response)
            justification_match = re.search(r'Justification:\s*(.+)', response, re.DOTALL)
            
            if not decision_match or not payout_match or not justification_match:
                return None

            decision = decision_match.group(1).upper()
            payout = int(payout_match.group(1).replace(',', ''))
            justification = justification_match.group(1).strip()
            
            return {
                'decision': decision, 'payout': payout,
                'justification': justification, 'raw_response': response
            }
        except Exception as e:
            self.logger.error(f"Error parsing decision response: {e}")
            return None
    
    def _fallback_query_parsing(self, query: str) -> Dict[str, Any]:
        """Final fallback using rule-based regular expressions."""
        self.logger.info("Executing rule-based query parsing.")
        query_lower = query.lower()
        age_match = re.search(r'(\d+)[-\s]*year', query_lower)
        age = int(age_match.group(1)) if age_match else None
        gender = 'male' if 'male' in query_lower else 'female' if 'female' in query_lower else None
        procedures = ['surgery', 'treatment', 'operation', 'procedure', 'therapy']
        procedure = next((proc for proc in procedures if proc in query_lower), None)
        cities = ['mumbai', 'delhi', 'bangalore', 'pune', 'chennai', 'kolkata', 'hyderabad']
        location = next((city.title() for city in cities if city in query_lower), None)
        month_match = re.search(r'(\d+)[-\s]*month', query_lower)
        duration = int(month_match.group(1)) if month_match else 12 if 'annual' in query_lower or 'year' in query_lower else None
        
        return {
            'age': age, 'gender': gender, 'procedure': procedure, 'location': location,
            'policy_duration_months': duration, 'model_used': 'rule-based-fallback',
            'policy_type': 'health' if any(term in query_lower for term in ['health', 'medical', 'surgery']) else None
        }

    def _fallback_decision_making(self, query_data: Dict[str, Any], contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Final fallback using simple business rules."""
        self.logger.info("Executing rule-based decision making.")
        decision = "APPROVED"
        payout = 50000
        justification = f"Claim approved based on policy terms. Age: {query_data.get('age', 'N/A')}, Procedure: {query_data.get('procedure', 'N/A')}"
        age = query_data.get('age')
        if age and age > 65:
            decision = "REJECTED"
            payout = 0
            justification = "Age exceeds policy limit of 65 years."
        
        return {
            'decision': decision, 'payout': payout, 'justification': justification,
            'raw_response': f"Fallback decision: {decision}", 'model_used': 'rule-based-fallback'
        }