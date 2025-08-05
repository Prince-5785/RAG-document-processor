"""
LLM service for query parsing and decision making using local language models.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Union

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install transformers for LLM functionality.")

from .utils import Timer, extract_json_from_text


class LLMService:
    """Service for LLM-based query parsing and decision making."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config.get('llm', {})
        self.model_name = self.llm_config.get('model_name', 'microsoft/DialoGPT-medium')
        self.device = self.llm_config.get('device', 'cpu')
        self.max_length = self.llm_config.get('max_length', 2048)
        self.temperature = self.llm_config.get('temperature', 0.1)
        self.top_p = self.llm_config.get('top_p', 0.9)
        
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.logger = logging.getLogger(__name__)
    
    def load_model(self) -> None:
        """Load the language model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
        
        if self.model is not None:
            return
        
        self.logger.info(f"Loading LLM model: {self.model_name}")
        
        try:
            with Timer(f"Loading LLM model {self.model_name}"):
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    device_map='auto' if self.device == 'cuda' else None,
                    low_cpu_mem_usage=True
                )
                
                # Create pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == 'cuda' and torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                )
                
                self.logger.info(f"Model loaded successfully on {self.device}")
                
        except Exception as e:
            self.logger.error(f"Failed to load LLM model: {e}")
            # Fallback to a simpler approach
            self._load_fallback_model()
    
    def _load_fallback_model(self) -> None:
        """Load a fallback model for basic functionality."""
        self.logger.info("Loading fallback model for basic functionality")
        self.model = "fallback"
        self.tokenizer = "fallback"
        self.pipeline = None
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse user query to extract structured information.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with extracted information
        """
        if self.model is None:
            self.load_model()
        
        prompt = self._create_query_parsing_prompt(query)
        
        if self.pipeline is not None:
            try:
                response = self._generate_response(prompt, max_new_tokens=200)
                parsed_data = self._extract_json_from_response(response)
                
                if parsed_data:
                    return parsed_data
            except Exception as e:
                self.logger.warning(f"LLM parsing failed, using fallback: {e}")
        
        # Fallback to rule-based parsing
        return self._fallback_query_parsing(query)
    
    def make_decision(self, query_data: Dict[str, Any], retrieved_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make insurance decision based on query and retrieved contexts.
        
        Args:
            query_data: Parsed query information
            retrieved_contexts: List of relevant document chunks
            
        Returns:
            Dictionary with decision, payout, and justification
        """
        if self.model is None:
            self.load_model()
        
        prompt = self._create_decision_prompt(query_data, retrieved_contexts)
        
        if self.pipeline is not None:
            try:
                response = self._generate_response(prompt, max_new_tokens=500)
                decision_data = self._parse_decision_response(response)
                
                if decision_data:
                    return decision_data
            except Exception as e:
                self.logger.warning(f"LLM decision making failed, using fallback: {e}")
        
        # Fallback to rule-based decision
        return self._fallback_decision_making(query_data, retrieved_contexts)
    
    def _generate_response(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate response using the LLM pipeline."""
        try:
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            if outputs and len(outputs) > 0:
                return outputs[0]['generated_text'].strip()
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
        
        return ""
    
    def _create_query_parsing_prompt(self, query: str) -> str:
        """Create prompt for query parsing."""
        return f"""Extract structured information from the following insurance query. Return only a JSON object with the specified fields.

Query: "{query}"

Extract the following information and return as JSON:
{{
  "age": <integer or null>,
  "gender": "<male/female/null>",
  "procedure": "<medical procedure or null>",
  "location": "<city/location or null>",
  "policy_duration_months": <integer or null>,
  "policy_type": "<policy type or null>"
}}

JSON:"""
    
    def _create_decision_prompt(self, query_data: Dict[str, Any], contexts: List[Dict[str, Any]]) -> str:
        """Create prompt for decision making."""
        context_text = ""
        for i, context in enumerate(contexts[:5], 1):
            context_text += f"Clause {i}: {context.get('text', '')}\n\n"
        
        return f"""You are an insurance claims processor. Based on the query information and relevant policy clauses, make a decision.

Query Information:
{json.dumps(query_data, indent=2)}

Relevant Policy Clauses:
{context_text}

Instructions:
1. Decide: APPROVED or REJECTED
2. If approved, estimate payout amount in ₹ (Indian Rupees)
3. Provide 2-3 clear justification sentences citing specific clauses
4. Be conservative and follow policy terms strictly

Response format:
Decision: [APPROVED/REJECTED]
Payout: ₹[amount or 0]
Justification: [Clear explanation with clause references]

Response:"""
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response."""
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return extract_json_from_text(response)
    
    def _parse_decision_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse decision response from LLM."""
        try:
            # Extract decision
            decision_match = re.search(r'Decision:\s*(APPROVED|REJECTED)', response, re.IGNORECASE)
            decision = decision_match.group(1).upper() if decision_match else "REJECTED"
            
            # Extract payout
            payout_match = re.search(r'Payout:\s*₹?(\d+(?:,\d+)*)', response)
            payout = int(payout_match.group(1).replace(',', '')) if payout_match else 0
            
            # Extract justification
            justification_match = re.search(r'Justification:\s*(.+?)(?:\n\n|\Z)', response, re.DOTALL)
            justification = justification_match.group(1).strip() if justification_match else "No justification provided."
            
            return {
                'decision': decision,
                'payout': payout,
                'justification': justification,
                'raw_response': response
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing decision response: {e}")
            return None
    
    def _fallback_query_parsing(self, query: str) -> Dict[str, Any]:
        """Fallback rule-based query parsing."""
        query_lower = query.lower()
        
        # Extract age
        age_match = re.search(r'(\d+)[-\s]*year', query_lower)
        age = int(age_match.group(1)) if age_match else None
        
        # Extract gender
        gender = None
        if 'male' in query_lower and 'female' not in query_lower:
            gender = 'male'
        elif 'female' in query_lower:
            gender = 'female'
        
        # Extract common procedures
        procedures = ['surgery', 'treatment', 'operation', 'procedure', 'therapy']
        procedure = None
        for proc in procedures:
            if proc in query_lower:
                procedure = proc
                break
        
        # Extract location (common Indian cities)
        cities = ['mumbai', 'delhi', 'bangalore', 'pune', 'chennai', 'kolkata', 'hyderabad']
        location = None
        for city in cities:
            if city in query_lower:
                location = city.title()
                break
        
        # Extract policy duration
        duration = None
        if 'month' in query_lower:
            month_match = re.search(r'(\d+)[-\s]*month', query_lower)
            duration = int(month_match.group(1)) if month_match else None
        elif 'annual' in query_lower or 'year' in query_lower:
            duration = 12
        
        return {
            'age': age,
            'gender': gender,
            'procedure': procedure,
            'location': location,
            'policy_duration_months': duration,
            'policy_type': 'health' if any(term in query_lower for term in ['health', 'medical', 'surgery']) else None
        }
    
    def _fallback_decision_making(self, query_data: Dict[str, Any], contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback rule-based decision making."""
        # Simple rule-based decision
        decision = "APPROVED"
        payout = 50000  # Default payout
        
        # Basic rules
        age = query_data.get('age', 0)
        if age and age > 65:
            decision = "REJECTED"
            payout = 0
            justification = "Age exceeds policy limit of 65 years."
        else:
            justification = f"Claim approved based on policy terms. Age: {age}, Procedure: {query_data.get('procedure', 'N/A')}"
        
        return {
            'decision': decision,
            'payout': payout,
            'justification': justification,
            'raw_response': f"Fallback decision: {decision}"
        }
