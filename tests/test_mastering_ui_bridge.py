from __future__ import annotations

import asyncio
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import run_ui
from mastering_ui_bridge import MasteringUIBridge


class RunUiLauncherTests(unittest.TestCase):
    def test_main_defaults_to_connected_ui_server(self) -> None:
        with mock.patch.object(run_ui, "start_ui_server") as start_ui_server:
            result = run_ui.main([])

        self.assertEqual(result, 0)
        start_ui_server.assert_called_once()

    def test_main_routes_export_mode_to_async_export(self) -> None:
        with mock.patch.object(run_ui, "start_ui_server") as start_ui_server:
            with mock.patch.object(run_ui, "export_pending_masters", new=mock.AsyncMock()) as export_pending:
                def fake_asyncio_run(coro: object) -> None:
                    if hasattr(coro, "close"):
                        coro.close()

                with mock.patch.object(run_ui.asyncio, "run", side_effect=fake_asyncio_run) as async_run:
                    result = run_ui.main(["--export"])

        self.assertEqual(result, 0)
        start_ui_server.assert_not_called()
        async_run.assert_called_once()
        export_pending.assert_called_once()

    def test_export_pending_masters_uses_live_ui_sessions(self) -> None:
        session = types.SimpleNamespace(
            job_id="job_123",
            status="done",
            output_file=None,
            error=None,
        )

        def refresh_session(target: object) -> object:
            target.output_file = "Album_Ignorance_is_bliss/masters/TestSong_TestPreset_Master.wav"
            return target

        fake_ui = types.SimpleNamespace(
            mastering_sessions={"sess_123": session},
            _refresh_session=mock.Mock(side_effect=refresh_session),
        )

        with mock.patch.object(run_ui, "_load_mastering_ui", return_value=fake_ui):
            with mock.patch("builtins.print") as print_mock:
                asyncio.run(run_ui.export_pending_masters())

        fake_ui._refresh_session.assert_called_once_with(session)
        printed = " ".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn("sess_123", printed)
        self.assertIn("TestSong_TestPreset_Master.wav", printed)


class MasteringUIBridgeTests(unittest.TestCase):
    def test_build_output_path_sanitizes_song_and_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge = MasteringUIBridge(export_root=Path(tmpdir))
            output_path = bridge._build_output_path(
                song_name="Face Time / 2",
                preset="Hi Fi / Streaming",
                filename="master.wav",
            )

        self.assertEqual(output_path.parent, Path(tmpdir))
        self.assertTrue(output_path.name.endswith("_Master.wav"))
        self.assertIn("Face_Time", output_path.name)
        self.assertIn("Hi_Fi", output_path.name)
        self.assertNotIn(" ", output_path.name)
        self.assertNotIn("/", output_path.name)


if __name__ == "__main__":
    unittest.main()
