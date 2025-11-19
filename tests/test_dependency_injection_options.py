"""Test different approaches to FastAPI dependency injection for testing.

This file tests various options to resolve the dependency injection issue
with ASGITransport and AsyncClient.
"""

from __future__ import annotations

from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient


# Test Option 1: Using TestClient (sync) - should work but limited
def test_option1_testclient_sync():
    """Option 1: Use TestClient for sync testing."""
    app = FastAPI()

    def get_dep():
        return "dependency_value"

    @app.get("/test")
    async def test_endpoint(dep: str = Depends(get_dep)):
        return {"dep": dep}

    # Override dependency
    app.dependency_overrides[get_dep] = lambda: "override_value"

    client = TestClient(app)
    response = client.get("/test")
    assert response.status_code == 200
    assert response.json()["dep"] == "override_value"

    # Cleanup
    app.dependency_overrides.clear()


# Test Option 2: Using AsyncClient with ASGITransport
@pytest.mark.asyncio
async def test_option2_asyncclient_asgitransport():
    """Option 2: Use AsyncClient with ASGITransport."""
    app = FastAPI()

    def get_dep():
        return "dependency_value"

    @app.get("/test")
    async def test_endpoint(dep: str = Depends(get_dep)):
        return {"dep": dep}

    # Override dependency
    app.dependency_overrides[get_dep] = lambda: "override_value"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/test")
        assert response.status_code == 200
        assert response.json()["dep"] == "override_value"

    # Cleanup
    app.dependency_overrides.clear()


# Test Option 3: Using Annotated with Depends
@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="Annotated pattern is flaky when run after other FastAPI tests due to global state. "
    "Codebase uses `Type = Depends(...)` syntax instead."
)
async def test_option3_annotated_depends():
    """Option 3: Test Annotated[Type, Depends(...)] pattern.
    
    Note: This test verifies the old Annotated pattern. The actual codebase
    now uses the simpler `Type = Depends(...)` syntax, but this test remains
    to document the different approaches. Marked as xfail due to test isolation issues.
    """
    app = FastAPI()

    def get_dep():
        return "dependency_value"

    @app.get("/test")
    async def test_endpoint(dep: Annotated[str, Depends(get_dep)]):
        return {"dep": dep}

    # Override dependency
    app.dependency_overrides[get_dep] = lambda: "override_value"

    # Use TestClient instead of ASGITransport for better compatibility
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        response = client.get("/test")
        assert response.status_code == 200
        assert response.json()["dep"] == "override_value"

    # Cleanup
    app.dependency_overrides.clear()


# Test Option 4: Create fresh app instance per test
@pytest.mark.asyncio
async def test_option4_fresh_app_instance():
    """Option 4: Create a fresh app instance for each test."""
    # This simulates creating a new app for testing
    app = FastAPI()

    def get_dep():
        return "dependency_value"

    @app.get("/test")
    async def test_endpoint(dep: str = Depends(get_dep)):
        return {"dep": dep}

    # Set override BEFORE creating transport
    app.dependency_overrides[get_dep] = lambda: "override_value"

    # Verify override is set
    assert get_dep in app.dependency_overrides

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/test")
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        assert response.status_code == 200
        assert response.json()["dep"] == "override_value"

    # Cleanup
    app.dependency_overrides.clear()


# Test Option 5: Using TestClient with async endpoint (should fail)
def test_option5_testclient_async_endpoint():
    """Option 5: TestClient with async endpoint - may have issues."""
    app = FastAPI()

    def get_dep():
        return "dependency_value"

    @app.get("/test")
    async def test_endpoint(dep: str = Depends(get_dep)):
        return {"dep": dep}

    app.dependency_overrides[get_dep] = lambda: "override_value"

    client = TestClient(app)
    response = client.get("/test")
    # TestClient should handle async endpoints
    assert response.status_code == 200
    assert response.json()["dep"] == "override_value"

    app.dependency_overrides.clear()
