from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
DAG_PATH = REPO_ROOT / "dags" / "mlops_pipeline.py"


class FakeDag:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeOperator:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __rshift__(self, other):
        return other


class FakeMount:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def fake_airflow_modules() -> dict[str, types.ModuleType]:
    airflow = types.ModuleType("airflow")
    airflow.DAG = FakeDag

    operators = types.ModuleType("airflow.operators")
    bash = types.ModuleType("airflow.operators.bash")
    bash.BashOperator = FakeOperator

    providers = types.ModuleType("airflow.providers")
    provider_docker = types.ModuleType("airflow.providers.docker")
    provider_operators = types.ModuleType("airflow.providers.docker.operators")
    provider_operator_docker = types.ModuleType("airflow.providers.docker.operators.docker")
    provider_operator_docker.DockerOperator = FakeOperator

    docker = types.ModuleType("docker")
    docker_types = types.ModuleType("docker.types")
    docker_types.Mount = FakeMount

    return {
        "airflow": airflow,
        "airflow.operators": operators,
        "airflow.operators.bash": bash,
        "airflow.providers": providers,
        "airflow.providers.docker": provider_docker,
        "airflow.providers.docker.operators": provider_operators,
        "airflow.providers.docker.operators.docker": provider_operator_docker,
        "docker": docker,
        "docker.types": docker_types,
    }


class AirflowDagContractTests(unittest.TestCase):
    def test_uses_host_path_for_runner_mount_and_airflow_path_for_build(self):
        host_repo_dir = "/host/workspaces/mlops/repos/mlops-examples"
        spec = importlib.util.spec_from_file_location("mlops_pipeline_contract_test", DAG_PATH)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)

        with (
            patch.dict(sys.modules, fake_airflow_modules()),
            patch.dict(
                os.environ,
                {"MLOPS_EXAMPLES_REPO_HOST_DIR": host_repo_dir},
                clear=False,
            ),
        ):
            spec.loader.exec_module(module)

        self.assertEqual(module.COMMON_MOUNTS[0].kwargs["source"], host_repo_dir)
        self.assertEqual(module.COMMON_MOUNTS[0].kwargs["target"], "/work")

        build_command = module.setup_environment.kwargs["bash_command"]
        self.assertIn(str(REPO_ROOT), build_command)
        self.assertIn(str(REPO_ROOT / "Dockerfile.runner"), build_command)
        self.assertNotIn("/opt/mlops-examples", build_command)


if __name__ == "__main__":
    unittest.main()
