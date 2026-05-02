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
        self.assertIn("auralmind://control-surface", resource_uris)
        self.assertIn("auralmind://premium-trap-workflow", resource_uris)
        self.assertIn("config://maintainer-guide", resource_uris)

    def test_bootstrap_prompts_include_premium_trap_session(self) -> None:
        prompt_names = {prompt.name for prompt in server.bootstrap().prompts}
        self.assertIn("premium_trap_mastering_session", prompt_names)

    def test_fastmcp_instructions_guide_clients_to_contracts_and_async_jobs(self) -> None:
        self.assertIn("auralmind://contracts", server.mcp.instructions)
        self.assertIn("auralmind://premium-trap-workflow", server.mcp.instructions)
        self.assertIn("run_master_job", server.mcp.instructions)

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
        self.assertIn("plan_mastering_strategy", tool_names)

    def test_upload_model_rejects_missing_or_conflicting_payloads(self) -> None:
        with self.assertRaises(ValidationError):
            server.UploadIn(filename="song.wav")
        with self.assertRaises(ValidationError):
            server.UploadIn(filename="song.wav", payload_b64="abc", hex_payload="00ff")

    def test_system_prompt_resource_is_not_empty(self) -> None:
        prompt = server.get_system_prompt()
        self.assertTrue(prompt.strip())
        self.assertIn("AuralMind", prompt)

    def test_premium_trap_workflow_resource_is_available(self) -> None:
        guide = server.get_premium_trap_workflow_resource()
        self.assertIn("Premium Trap Workflow", guide)
        self.assertIn("run_master_job", guide)
        self.assertIn("Quality Gates", guide)

    def test_premium_trap_prompt_points_to_guidance_and_async_lifecycle(self) -> None:
        text = self._run_async(server.premium_trap_mastering_session_prompt("song.wav"))
        self.assertIn("auralmind://premium-trap-workflow", text)
        self.assertIn("run_master_job", text)
        self.assertIn("job_status", text)

    def test_workflow_resource_matches_connect_packet(self) -> None:
        workflow_payload = json.loads(self._run_async(server.get_workflow_resource()))
        packet = server.get_connect_packet()
        self.assertEqual(workflow_payload["workflow"], packet.workflow_steps)
        self.assertEqual(
            workflow_payload["recommended_first_path"],
            packet.recommended_first_path,
        )

    def test_contracts_include_control_profile_and_strategy_models(self) -> None:
        payload = json.loads(server.get_contracts_resource())
        self.assertIn("plan_mastering_strategy", payload["tools"])
        self.assertIn("MasteringControlProfile", payload["models"])
        self.assertIn("StrategyPlanIn", payload["models"])
        self.assertIn("StrategyPlanOut", payload["models"])

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
