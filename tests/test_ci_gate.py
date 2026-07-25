import json

from ci.run_gate import run_gate


def test_run_gate_creates_output(tmp_path):
    output = tmp_path / "result.json"
    result = run_gate("abc123", output)
    assert output.exists()
    assert result["commit_sha"] == "abc123"
    assert result["overall"] in ("pass", "warning", "critical")


def test_run_gate_output_format(tmp_path):
    output = tmp_path / "result.json"
    run_gate("abc123", output)
    data = json.loads(output.read_text())
    assert "commit_sha" in data
    assert "overall" in data
    assert "details" in data
    assert isinstance(data["details"], list)
