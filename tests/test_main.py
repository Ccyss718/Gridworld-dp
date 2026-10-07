import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import main as main_module


def test_main_runs_complete_pipeline(tmp_path, capsys, monkeypatch):
    fake_main_path = tmp_path / "main.py"
    monkeypatch.setattr(main_module, "__file__", str(fake_main_path))
    monkeypatch.setattr(main_module.plt, "show", lambda: None)

    main_module.main()

    captured = capsys.readouterr().out
    output_path = tmp_path / "outputs" / "value_heatmap.png"

    assert "随机策略状态价值" in captured
    assert "最优状态价值" in captured
    assert "最优策略" in captured
    assert "X" in captured
    assert "G" in captured
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close("all")
