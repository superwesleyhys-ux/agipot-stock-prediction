"""One command surface for local checks and CI; optional services stay explicit."""
from pathlib import Path
import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "tests", "properties", "types", "distribution"]


@nox.session(python=["3.11", "3.12", "3.13"], reuse_venv=True)
def tests(session):
    session.install(".[dev,harness,data]")
    session.run("python", "-m", "pytest", "-q", *session.posargs)


@nox.session(reuse_venv=True)
def coverage(session):
    session.install(".[dev,harness,data]")
    session.run("agipot-harness", "run", "coverage")


@nox.session(reuse_venv=True)
def lint(session):
    session.install("ruff>=0.11,<1")
    session.run("ruff", "check", "src", "tests", "scripts", "noxfile.py")


@nox.session(reuse_venv=True)
def types(session):
    session.install(".[harness]")
    session.run("mypy")


@nox.session(reuse_venv=True)
def properties(session):
    session.install(".[harness]")
    session.run("pytest", "tests/test_formula_properties.py", "tests/test_streaming_properties.py", "-q")


@nox.session(reuse_venv=True, default=False)
def mutation(session):
    session.install(".[harness]")
    session.run("mutmut", "run", *(session.posargs or ["agipot_stock_prediction.research.cost_model.*"]))
    session.run("mutmut", "results")


@nox.session(reuse_venv=True)
def distribution(session):
    session.install(".[dev,harness]")
    session.run("python", "-m", "build")
    session.run("python", "scripts/check_distribution.py")
    wheels = sorted(Path("dist").glob("agipot_stock_prediction-*.whl"), key=lambda p: p.stat().st_mtime)
    session.install("--force-reinstall", str(wheels[-1]))
    test_directory = str(Path("tests").resolve())
    # Run the installed wheel and its tests from outside the checkout.
    with session.chdir(session.create_tmp()):
        session.run("python", "-c", "import agipot_stock_prediction as p; assert 'site-packages' in p.__file__; print(p.__version__)")
        session.run("agipot-predict", "--demo", "--output", "demo.json", env={"PYTHONPATH": ""})
        session.run("agipot-harness", "catalog", env={"PYTHONPATH": ""})
        session.run("agipot-harness", "demo", env={"PYTHONPATH": ""})
        session.run("pytest", test_directory, "-q", env={"PYTHONPATH": ""})


@nox.session(reuse_venv=True, default=False)
def audit(session):
    session.install(".[harness]")
    session.run("bandit", "-r", "src", "-c", "config/harness/bandit.yaml")
    session.run("pip-audit", "--skip-editable")


@nox.session(python=False, default=False)
def secrets(session):
    session.run("gitleaks", "git", "--redact", "--no-banner", ".", external=True)


@nox.session(reuse_venv=True, default=False)
def research(session):
    session.install(".[harness,integrations]")
    session.run("agipot-harness", "demo")
    session.run("pytest", "tests/test_integrations.py", "-q")


@nox.session(reuse_venv=True, default=False)
def benchmark(session):
    session.install(".[harness]")
    session.run("pytest", "benchmarks", "--benchmark-json=reports/benchmark.json", "-q")


@nox.session(reuse_venv=True, default=False)
def memory(session):
    session.install(".", "memray")
    session.run("python", "-m", "memray", "run", "-o", "reports/memory.bin", "examples/demo.py")


@nox.session(reuse_venv=True, default=False)
def regression(session):
    session.install(".[harness]")
    session.run("pytest", "tests/test_output_regression.py", "-q")
