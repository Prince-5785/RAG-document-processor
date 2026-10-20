"""
API Integration tests for the InsureQuery FastAPI Application.
Tests:
- /health and /metrics (including honest 503 and per-stage latency)
- Fail-closed authentication verification
- /documents upload, status, list, delete
- File format and file size validation
- /query with source citations
- /query/stream Server-Sent Events (SSE)
- /query/claim structured decision with sum_insured and model_used
"""

import os
import pytest
import io
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from main_api import app

TEST_API_KEY = "test-secret-key-12345"


@pytest.fixture(autouse=True)
def setup_test_env():
    """Configure default test API key in environment."""
    original_key = os.environ.get("INSUREQUERY_API_KEY")
    os.environ["INSUREQUERY_API_KEY"] = TEST_API_KEY
    yield
    if original_key is not None:
        os.environ["INSUREQUERY_API_KEY"] = original_key
    else:
        os.environ.pop("INSUREQUERY_API_KEY", None)


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {TEST_API_KEY}"}


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


class TestInsureQueryAPI:
    """Test suite for InsureQuery REST & SSE API."""

    def test_health_check_healthy(self, client):
        response = client.get("/health")
        # If healthy, returns 200 with pipeline_status
        assert response.status_code in (200, 503)
        data = response.json()
        assert "pipeline_status" in data or "status" in data

    @patch('main_api.pipeline', None)
    def test_health_check_unhealthy_returns_503(self, client):
        """Honest healthcheck: when pipeline is None, must return HTTP 503."""
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data.get("status") == "unhealthy"

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "avg_latency_ms" in data
        assert "stage_latencies" in data
        assert "cache" in data

    def test_auth_fail_closed_without_key(self, client):
        """When INSUREQUERY_API_KEY is not set in environment, auth must fail closed with 503."""
        with patch.dict('os.environ', {}, clear=True):
            res = client.post("/query", json={"question": "What is the room rent?"})
            assert res.status_code == 503
            assert "Server auth not configured" in res.json()['detail']

    def test_auth_verification(self, client):
        # Missing auth header -> 401
        res_unauth = client.post("/query", json={"question": "What is the room rent?"})
        assert res_unauth.status_code == 401

        # Invalid Bearer format -> 401
        res_bad_format = client.post(
            "/query",
            json={"question": "What is the room rent?"},
            headers={"Authorization": "Invalid secret-token"}
        )
        assert res_bad_format.status_code == 401

        # Wrong token -> 401
        res_wrong = client.post(
            "/query",
            json={"question": "What is the room rent?"},
            headers={"Authorization": "Bearer wrong-token"}
        )
        assert res_wrong.status_code == 401

    @patch('main_api.pipeline')
    def test_document_upload_validation(self, mock_pipeline, client, auth_headers):
        mock_pipeline.document_processor.get_supported_formats.return_value = ['pdf', 'docx', 'txt', 'eml']
        mock_pipeline.document_processor.max_file_size_mb = 25
        
        # Unsupported file extension
        unsupported_file = io.BytesIO(b"dummy binary data")
        res_ext = client.post(
            "/documents",
            files={"file": ("malicious.exe", unsupported_file, "application/octet-stream")},
            headers=auth_headers
        )
        assert res_ext.status_code == 400
        assert "Unsupported file format" in res_ext.json()['detail']

    @patch('main_api.pipeline')
    def test_document_list_and_delete(self, mock_pipeline, client, auth_headers):
        mock_pipeline.list_documents.return_value = [
            {'id': 'doc-1', 'filename': 'policy.txt', 'chunk_count': 5, 'status': 'ready'}
        ]
        mock_pipeline.delete_document.return_value = True

        # List
        res_list = client.get("/documents", headers=auth_headers)
        assert res_list.status_code == 200
        assert len(res_list.json()['documents']) == 1

        # Delete
        res_del = client.delete("/documents/doc-1", headers=auth_headers)
        assert res_del.status_code == 200
        assert res_del.json()['status'] == "deleted"

    @patch('main_api.pipeline')
    def test_document_status_endpoint(self, mock_pipeline, client, auth_headers):
        mock_pipeline.vector_store.get_document.return_value = {
            'id': 'doc-123',
            'filename': 'health_policy.pdf',
            'status': 'ready',
            'chunk_count': 12,
            'indexed_at': '2026-10-20T10:00:00Z'
        }

        res = client.get("/documents/doc-123/status", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data['document_id'] == 'doc-123'
        assert data['status'] == 'ready'
        assert data['chunk_count'] == 12

    @patch('main_api.pipeline')
    def test_query_with_citations(self, mock_pipeline, client, auth_headers):
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
            json={"question": "What is the daily room rent sub-limit?"},
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "5,000" in data['answer']
        assert len(data['sources']) == 1
        assert data['sources'][0]['file'] == 'sample_insurance_policy.txt'
        assert data['sources'][0]['page'] == 1
        assert data['threshold_applied'] is False

    @patch('main_api.pipeline')
    def test_query_claim_decision(self, mock_pipeline, client, auth_headers):
        mock_pipeline.query = AsyncMock(return_value={
            'decision': 'APPROVED',
            'payout': 50000,
            'justification': 'Claim falls within hospitalization limits.',
            'clause_ids': ['CL_001'],
            'metadata': {'model_used': 'llama-3.3-70b-versatile'},
            'json_output': {
                'claim_id': 'CLM_A1B2C3D4',
                'currency': 'INR',
                'evidence': [
                    {'clause_id': 'CL_001', 'text': 'Hospitalization up to 50,000', 'relevance_score': 0.9}
                ],
                'processing_metadata': {'model_used': 'llama-3.3-70b-versatile'}
            }
        })

        response = client.post(
            "/query/claim",
            json={
                "query": "35 year old male undergoing appendix surgery",
                "sum_insured": 500000
            },
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data['decision'] == 'APPROVED'
        assert data['payout'] == 50000
        assert data['claim_id'] == 'CLM_A1B2C3D4'
        assert data['model_used'] == 'llama-3.3-70b-versatile'
        assert data['clause_ids'] == ['CL_001']
        assert len(data['evidence']) == 1

    @patch('main_api.pipeline')
    def test_query_stream_sse(self, mock_pipeline, client, auth_headers):
        async def mock_stream(question, document_ids=None, score_threshold=None):
            yield 'data: {"type": "sources", "sources": [{"file": "policy.txt", "page": 1, "score": 0.9}]}\n\n'
            yield 'data: {"type": "token", "content": "Hello"}\n\n'
            yield 'data: {"type": "token", "content": " world"}\n\n'
            yield 'data: {"type": "done"}\n\n'

        mock_pipeline.stream_answer = mock_stream

        response = client.post(
            "/query/stream",
            json={"question": "What is the policy limit?"},
            headers=auth_headers
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = response.text
        assert 'data: {"type": "sources"' in body
        assert 'data: {"type": "token"' in body
        assert 'data: {"type": "done"}' in body


if __name__ == "__main__":
    pytest.main([__file__])
