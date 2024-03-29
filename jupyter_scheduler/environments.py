import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Dict, List

from jupyter_scheduler.models import RuntimeEnvironment


class EnvironmentManager(ABC):
    @abstractmethod
    def list_environments(self) -> List[RuntimeEnvironment]:
        pass

    @abstractmethod
    def manage_environments_command(self) -> str:
        pass

    @abstractmethod
    def output_formats_mapping(self) -> Dict[str, str]:
        """Dictionary of all output formats with human readable names
        supported by this environment manager. This should include all
        supported output formats, not just by individual environments.
        """
        pass


class CondaEnvironmentManager(EnvironmentManager):
    """Provides a list of Conda environments. If Conda is not
    installed or activated, it defaults to providing exclusively
    the Python executable that JupyterLab is currently running in.
    """

    def list_environments(self) -> List[RuntimeEnvironment]:
        environments = []

        try:
            envs = subprocess.check_output(["conda", "env", "list", "--json"])
            envs = json.loads(envs).get("envs", [])
        except subprocess.CalledProcessError as e:
            envs = []
        except FileNotFoundError as e:
            envs = []

        current_python_root = sys.prefix
        if not envs or current_python_root not in envs:
            envs = [sys.executable]

        for env in envs:
            name = os.path.basename(env)
            environments.append(
                RuntimeEnvironment(
                    name=name,
                    label=name,
                    description=f"Environment: {name}",
                    file_extensions=["ipynb"],
                    output_formats=["ipynb", "html"],
                    metadata={"path": env},
                )
            )

        return environments

    def manage_environments_command(self) -> str:
        return ""

    def output_formats_mapping(self) -> Dict[str, str]:
        return {"ipynb": "Notebook", "html": "HTML"}


class StaticEnvironmentManager(EnvironmentManager):
    """Provides a static list of environments, for demo purpose only"""

    def list_environments(self) -> List[RuntimeEnvironment]:
        name = "jupyterlab-env"
        path = os.path.join(os.environ["HOME"], name)
        return [
            RuntimeEnvironment(
                name=name,
                label=name,
                description=f"Virtual environment: {name}",
                file_extensions=["ipynb"],
                metadata={"path": path},
                output_formats=["ipynb", "html"],
            )
        ]

    def manage_environments_command(self) -> str:
        return ""

    def output_formats_mapping(self) -> Dict[str, str]:
        return {"ipynb": "Notebook", "html": "HTML"}


class EnvironmentRetrievalError(Exception):
    pass
