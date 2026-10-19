import json
import logging
import re
import asyncio
import random
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator

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
    Service for LLM-based insurance parsing, decision making, QA, and token streaming
    with key rotation, exponential backoff, and local fallback.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config.get('llm', {})
        self.logger = logging.getLogger(__name__)

        api_config = self.llm_config.get('api', {})
        self.model_sequence = api_config.get('model_sequence', [
            'qwen/qwen3.8-27b',
            'openai/gpt-oss-120b',
            'openai/gpt-oss-20b',
            'allam-2-7b'
        ])
        
        self.groq_api_keys = api_config.get('groq_api_keys', [])
        self.key_index = 0
        if not self.groq_api_keys:
            self.logger.warning("No Groq API keys found in configuration. API calls will use rule-based fallback.")

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
        """Asynchronously parses a user query into structured attributes."""
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

    async def make_decision(
        self,
        query_data: Dict[str, Any],
        retrieved_contexts: List[Dict[str, Any]],
        preferred_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Asynchronously makes an insurance claim decision with structured justification."""
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
        """Asynchronously generates a direct answer grounded in provided policy context."""
        self.logger.info(f"Generating answer for question: '{question[:50]}...'")
        prompt = self._create_qa_prompt(question, contexts)

        answer_data = await self._execute_task_with_fallback(
            prompt=prompt,
            parse_function=lambda text: {"answer": text.strip()},
            max_new_tokens=350,
        )

        if answer_data and answer_data.get("answer"):
            return answer_data["answer"]
        
        self.logger.warning("All LLM attempts failed to generate an answer. Falling back to rule-based answer.")
        return self._rule_based_answer(question, contexts)

    async def stream_answer_question(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        preferred_model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Streams token-by-token answer via Groq API.
        Falls back to yielding a complete string if streaming fails.
        """
        if not contexts:
            yield "The answer to this question could not be found in the provided document."
            return

        prompt = self._create_qa_prompt(question, contexts)
        model_id = preferred_model or (self.model_sequence[0] if self.model_sequence else "qwen/qwen3.8-27b")
        
        stream_success = False
        if GROQ_AVAILABLE and self.groq_api_keys:
            for attempt in range(self.max_attempts):
                api_key = self._get_next_api_key()
                if not api_key:
                    break
                try:
                    client = AsyncGroq(api_key=api_key)
                    completion = await client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_id,
                        temperature=self.temperature,
                        max_tokens=350,
                        top_p=self.top_p,
                        stream=True
                    )
                    stream_success = True
                    async for chunk in completion:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                    return
                except Exception as e:
                    self.logger.warning(f"Streaming error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_attempts - 1:
                        await asyncio.sleep(self.initial_delay)

        if not stream_success:
            # Fallback non-streaming answer yielded word-by-word
            fallback_ans = await self.answer_question(question, contexts)
            words = fallback_ans.split(' ')
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                await asyncio.sleep(0.01)

    async def _execute_task_with_fallback(
        self,
        prompt: str,
        parse_function: Callable,
        max_new_tokens: int,
        preferred_model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
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
        if not GROQ_AVAILABLE or not self.groq_api_keys:
            return None

        for attempt in range(self.max_attempts):
            api_key = self._get_next_api_key()
            if not api_key:
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
            page_info = f" (Page {context.get('metadata', {}).get('page', 1)})" if context.get('metadata') else ""
            context_text += f"Clause {i}{page_info}: {context.get('text', '')}\n\n"
        
        return f"""You are a senior insurance claims processor. Your task is to meticulously evaluate an insurance claim based on the provided information and policy clauses. You must follow a strict reasoning process.

### REASONING STEPS:
1.  **Analyze Policy Requirements:** First, carefully read the 'Relevant Policy Clauses' and identify key conditions for approval.
2.  **Assess User Information:** Review the 'Query Information' provided.
3.  **Compare and Identify Gaps:** Check whether waiting periods, age limits, or exclusions apply.
4.  **Formulate Conclusion:**
    - **APPROVE:** Only if you have all necessary information and the claim clearly meets policy requirements.
    - **REJECT:** If the claim clearly violates a policy rule or exclusion.
5.  **Write Justification:** Provide the exact reason, referencing specific clause numbers.

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
            pieces = []
            for ctx in contexts:
                page_str = f" [Page {ctx.get('metadata', {}).get('page', 1)}]" if ctx.get('metadata') else ""
                pieces.append(f"{ctx.get('text', '')}{page_str}")
            context_text = "\n\n---\n\n".join(pieces)

        return f"""You are an expert assistant specialized in reading and understanding insurance policy documents. Your task is to answer the user's question based *only* on the provided context.

**CONTEXT FROM THE POLICY DOCUMENT:**
{context_text}

**USER'S QUESTION:**
{question}

**INSTRUCTIONS:**
- Answer concisely and accurately based solely on the provided clauses.
- If the answer cannot be found in the context, you MUST state: "The answer to this question could not be found in the provided document."
- Do not make assumptions or fabricate terms.

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

    def _rule_based_answer(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """Fallback extractive answer when LLM is offline."""
        if not contexts:
            return "The answer to this question could not be found in the provided document."
        top_clause = contexts[0].get('text', '')
        # Pick the most relevant sentence from top clause
        sentences = [s.strip() for s in re.split(r'[.\n]+', top_clause) if len(s.strip()) > 10]
        q_words = set(re.findall(r'\b\w+\b', question.lower()))
        best_sentence = top_clause[:200]
        max_overlap = -1
        for sent in sentences:
            s_words = set(re.findall(r'\b\w+\b', sent.lower()))
            overlap = len(q_words.intersection(s_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = sent
        return f"{best_sentence}."

    def _fallback_query_parsing(self, query: str) -> Dict[str, Any]:
        """Final fallback using rule-based regular expressions."""
        self.logger.info("Executing rule-based query parsing.")
        query_lower = query.lower()
        age_match = re.search(r'(\d+)[-\s]*year', query_lower)
        age = int(age_match.group(1)) if age_match else None
        gender = 'male' if 'male' in query_lower else 'female' if 'female' in query_lower else None
        procedures = ['surgery', 'treatment', 'operation', 'procedure', 'therapy', 'hospitalization', 'maternity', 'dental']
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
        age = query_data.get('age')
        procedure = query_data.get('procedure', '')
        
        if age and age > 65:
            return {
                'decision': 'REJECTED', 'payout': 0,
                'justification': "Age exceeds maximum entry age limit of 65 years.",
                'raw_response': "Fallback decision: REJECTED", 'model_used': 'rule-based-fallback'
            }
        
        if procedure and any(ex in procedure.lower() for ex in ['cosmetic', 'dental']):
            return {
                'decision': 'REJECTED', 'payout': 0,
                'justification': f"Procedure '{procedure}' falls under policy permanent exclusions.",
                'raw_response': "Fallback decision: REJECTED", 'model_used': 'rule-based-fallback'
            }

        return {
            'decision': 'APPROVED', 'payout': 50000,
            'justification': f"Claim tentatively approved under general hospitalization terms. Age: {age or 'N/A'}, Procedure: {procedure or 'medical treatment'}.",
            'raw_response': "Fallback decision: APPROVED", 'model_used': 'rule-based-fallback'
        }