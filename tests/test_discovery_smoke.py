import json
import os
import unittest
from unittest import mock

import server
from pydantic import ValidationError
from starlette.testclient import TestClient


class DiscoverySmokeTests(unittest.TestCase):
    def test_bootstrap_tool_catalog_matches_contracts_resource(self) -> None:
        bootstrap_tools = {tool.name for tool in server.bootstrap().tools}
        contract_tools = set(json.loads(server.get_contracts_resource())["tools"])
        self.assertSetEqual(bootstrap_tools, contract_tools)

    def test_bootstrap_resources_include_connect_kit(self) -> None:
        resource_uris = {resource.uri for resource in server.bootstrap().resources}
        self.assertIn("auralmind://connect-kit", resource_uris)

    def test_server_info_includes_version_and_supported_transports(self) -> None:
        payload = json.loads(server.get_server_info())
        self.assertEqual(payload["name"], server.SERVER_NAME)
        self.assertEqual(payload["version"], server.VERSION)
        self.assertEqual(payload["transport"], server.capabilities().transport)
        self.assertSequenceEqual(payload["supported_transports"], list(server.SUPPORTED_TRANSPORTS))

    def test_transport_aliases_normalize_to_supported_values(self) -> None:
        with mock.patch.dict(os.environ, {server.ACTIVE_TRANSPORT_ENV: "http"}, clear=False):
            self.assertEqual(server._active_transport(), "streamable-http")
        with mock.patch.dict(os.environ, {server.ACTIVE_TRANSPORT_ENV: "streamable_http"}, clear=False):
            self.assertEqual(server._active_transport(), "streamable-http")

    def test_transport_fallback_defaults_to_streamable_http(self) -> None:
        with mock.patch.dict(os.environ, {server.ACTIVE_TRANSPORT_ENV: ""}, clear=False):
            self.assertEqual(server._active_transport(), "streamable-http")

    def test_http_bind_defaults_and_run_kwargs(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                server.ACTIVE_TRANSPORT_ENV: "streamable-http",
                server.HTTP_HOST_ENV: "",
                server.HTTP_PORT_ENV: "",
                server.HTTP_PATH_ENV: "",
            },
            clear=False,
        ):
            self.assertEqual(server._http_host(), server.DEFAULT_HTTP_HOST)
            self.assertEqual(server._http_port(), server.DEFAULT_HTTP_PORT)
            self.assertEqual(server._http_path(), server.DEFAULT_HTTP_PATH)
            self.assertEqual(
                server._run_kwargs_for_active_transport(),
                {
                    "transport": "streamable-http",
                    "host": server.DEFAULT_HTTP_HOST,
                    "port": server.DEFAULT_HTTP_PORT,
                    "path": server.DEFAULT_HTTP_PATH,
                    "json_response": True,
                },
            )

    def test_http_health_and_root_routes_are_available(self) -> None:
        client = TestClient(server.app)
        root_response = client.get("/")
        self.assertEqual(root_response.status_code, 200)
        self.assertEqual(root_response.json()["mcp_path"], server._http_path())

        health_response = client.get("/health")
        self.assertEqual(health_response.status_code, 200)
        payload = health_response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["name"], server.SERVER_NAME)

    def test_bootstrap_examples_reference_known_tools(self) -> None:
        packet = server.bootstrap()
        tool_names = {tool.name for tool in packet.tools}
        for example in packet.example_calls.values():
            params = example.get("params", {})
            name = params.get("name")
            self.assertIn(name, tool_names)

    def test_upload_model_rejects_missing_or_conflicting_payloads(self) -> None:
        with self.assertRaises(ValidationError):
            server.UploadIn(filename="song.wav")
        with self.assertRaises(ValidationError):
            server.UploadIn(filename="song.wav", payload_b64="abc", hex_payload="00ff")

    def test_system_prompt_resource_is_not_empty(self) -> None:
        prompt = server.get_system_prompt()
        self.assertTrue(prompt.strip())
        self.assertIn("AuralMind", prompt)

    def test_workflow_resource_matches_connect_packet(self) -> None:
        workflow_payload = json.loads(self._run_async(server.get_workflow_resource()))
        packet = server.get_connect_packet()
        self.assertEqual(workflow_payload["workflow"], packet.workflow_steps)
        self.assertEqual(
            workflow_payload["recommended_first_path"],
            packet.recommended_first_path,
        )

    @staticmethod
    def _run_async(awaitable):
        try:
            import asyncio

            return asyncio.run(awaitable)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(awaitable)
            finally:
                loop.close()


if __name__ == "__main__":
    unittest.main()
