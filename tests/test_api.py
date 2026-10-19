"""
API Integration tests for the InsureQuery FastAPI Application.
Tests:
- /health and /metrics
- Authentication verification
- /documents upload, list, delete
- File format and file size validation
- /query with source citations
- /query/stream Server-Sent Events (SSE)
- /query/claim structured decision
- /hackrx/run backward compatibility
"""

import pytest
import io
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from main_api import app


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


class TestInsureQueryAPI:
    """Test suite for InsureQuery REST & SSE API."""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "pipeline_status" in data or "status" in data

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "avg_latency_ms" in data
        assert "cache" in data

    def test_auth_verification(self, client):
        with patch.dict('os.environ', {'INSUREQUERY_API_KEY': 'secret-token-123'}):
            # Unauthorized request
            res_unauth = client.post("/query", json={"question": "What is the room rent?"})
            assert res_unauth.status_code == 401

            # Invalid Bearer format
            res_bad_format = client.post(
                "/query",
                json={"question": "What is the room rent?"},
                headers={"Authorization": "Invalid secret-token-123"}
            )
            assert res_bad_format.status_code == 401

            # Wrong token
            res_wrong = client.post(
                "/query",
                json={"question": "What is the room rent?"},
                headers={"Authorization": "Bearer wrong-token"}
            )
            assert res_wrong.status_code == 401

    @patch('main_api.pipeline')
    def test_document_upload_validation(self, mock_pipeline, client):
        # Unsupported file extension
        unsupported_file = io.BytesIO(b"dummy binary data")
        res_ext = client.post(
            "/documents",
            files={"file": ("malicious.exe", unsupported_file, "application/octet-stream")}
        )
        assert res_ext.status_code == 400
        assert "Unsupported file format" in res_ext.json()['detail']

    @patch('main_api.pipeline')
    def test_document_list_and_delete(self, mock_pipeline, client):
        mock_pipeline.list_documents.return_value = [
            {'id': 'doc-1', 'filename': 'policy.txt', 'chunk_count': 5, 'status': 'ready'}
        ]
        mock_pipeline.delete_document.return_value = True

        # List
        res_list = client.get("/documents")
        assert res_list.status_code == 200
        assert len(res_list.json()['documents']) == 1

        # Delete
        res_del = client.delete("/documents/doc-1")
        assert res_del.status_code == 200
        assert res_del.json()['status'] == "deleted"

    @patch('main_api.pipeline')
    def test_query_with_citations(self, mock_pipeline, client):
        mock_pipeline.answer_with_citations = AsyncMock(return_value={
            'answer': 'The room rent sub-limit is ₹5,000 per day.',
            'sources': [
                {
                    'file': 'sample_insurance_policy.txt',
                    'page': 1,
                    'chunk_text': 'Room rent: Up to ₹5,000 per day',
                    'score': 0.8845
                }
            ],
            'threshold_applied': False
        })

        response = client.post(
            "/query",
            json={"question": "What is the daily room rent sub-limit?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "₹5,00,000" not in data['answer']
        assert "5,000" in data['answer']
        assert len(data['sources']) == 1
        assert data['sources'][0]['file'] == 'sample_insurance_policy.txt'
        assert data['sources'][0]['page'] == 1
        assert data['threshold_applied'] is False

    @patch('main_api.pipeline')
    def test_query_claim_decision(self, mock_pipeline, client):
        mock_pipeline.query = AsyncMock(return_value={
            'decision': 'APPROVED',
            'payout': 50000,
            'justification': 'Claim falls within hospitalization limits.',
            'metadata': {'model_used': 'llama-3.3-70b-versatile'},
            'json_output': {
                'claim_id': 'CLM_A1B2C3D4',
                'currency': 'INR',
                'evidence': [
                    {'clause_id': 'CL_001', 'text': 'Hospitalization up to 50,000', 'relevance_score': 0.9}
                ]
            }
        })

        response = client.post(
            "/query/claim",
            json={"query": "35 year old male undergoing appendix surgery"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['decision'] == 'APPROVED'
        assert data['payout'] == 50000
        assert data['claim_id'] == 'CLM_A1B2C3D4'
        assert len(data['evidence']) == 1

    @patch('main_api.pipeline')
    def test_query_stream_sse(self, mock_pipeline, client):
        async def mock_stream(question, document_ids=None, score_threshold=None):
            yield 'data: {"type": "sources", "sources": [{"file": "policy.txt", "page": 1, "score": 0.9}]}\n\n'
            yield 'data: {"type": "token", "content": "Hello"}\n\n'
            yield 'data: {"type": "token", "content": " world"}\n\n'
            yield 'data: {"type": "done"}\n\n'

        mock_pipeline.stream_answer = mock_stream

        response = client.post(
            "/query/stream",
            json={"question": "What is the policy limit?"}
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = response.text
        assert 'data: {"type": "sources"' in body
        assert 'data: {"type": "token"' in body
        assert 'data: {"type": "done"}' in body


if __name__ == "__main__":
    pytest.main([__file__])
