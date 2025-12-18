from pathlib import Path

from agent import Agent


def test_fetch_url_nonexistent_domain_does_not_crash(tmp_path, monkeypatch):
    """
    Regression:
    - A fetch_url call to a non-existent domain should hard-fail
    - Agent should replan and still return a final answer
    - Episodic memory should be persisted (episodes.jsonl)
    """

    test_mem_dir = tmp_path / "memory"
    test_mem_dir.mkdir(parents=True, exist_ok=True)

    # Monkeypatch config values at runtime (must stay Path objects)
    import config
    monkeypatch.setattr(config, "MEMORY_DIR", test_mem_dir)
    monkeypatch.setattr(config, "EPISODES_PATH", test_mem_dir / "episodes.jsonl")
    monkeypatch.setattr(config, "CHROMA_DIR", test_mem_dir / "chroma")

    # Run
    agent = Agent(max_replans=2)
    user_input = "Fetch the contents of https://this-domain-does-not-exist-12345.com and explain what happened."
    answer = agent.run(user_input)

    # Assertions: agent returns *something* user-facing
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0

    # It should explain failure (DNS / name resolution / couldn't fetch)
    lowered = answer.lower()
    assert ("failed" in lowered) or ("could not" in lowered) or ("unable" in lowered)

    # Memory should be written
    episodes_path = test_mem_dir / "episodes.jsonl"
    assert episodes_path.exists()
    content = episodes_path.read_text(encoding="utf-8").strip()
    assert content != ""
    assert "this-domain-does-not-exist-12345.com" in content