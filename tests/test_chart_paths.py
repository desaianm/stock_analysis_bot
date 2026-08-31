from pathlib import Path

from stockbot.tools.data import ChartingTool


def test_chart_filename_is_slugged_and_contained_for_agent_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    output = ChartingTool().run("../../escape", [1, 2])

    plots = (tmp_path / "plots").resolve()
    assert Path(output.file_path).resolve().is_relative_to(plots)
    assert Path(output.file_path).name == "escape_chart.png"


def test_chart_filename_preserves_normal_metric_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    output = ChartingTool().run("Annual Revenue", [1, 2])

    assert Path(output.file_path).name == "annual_revenue_chart.png"
