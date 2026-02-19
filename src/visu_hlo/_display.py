"""Display utilities for HLO and DOT graphs."""

import os
import platform
import subprocess
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import graphviz
from jaxlib.xla_client import _xla as xla  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from IPython.core.interactiveshell import InteractiveShell


class GraphvizNotFoundError(Exception):
    """Raised when Graphviz is not installed or not found in PATH."""


_system = platform.system()
if _system == 'Linux':
    DISPLAY_PROGRAM = 'xdg-open'
elif _system == 'Darwin':
    DISPLAY_PROGRAM = 'open'
elif _system == 'Windows':
    DISPLAY_PROGRAM = 'start'
else:
    raise RuntimeError(f'Unsupported platform: {_system}')


class DotGraphViewer:
    """Viewer for DOT graph representations.

    Attributes:
        dot_graph: DOT graph string.
    """

    def __init__(self, dot_graph: str) -> None:
        self.dot_graph = dot_graph

    def show(self) -> None:
        """Display the DOT graph as SVG.

        In a Jupyter notebook, displays inline. Otherwise, opens the system's default SVG viewer.
        """
        if self._in_notebook():
            self._show_in_notebook()
        else:
            self._show_in_program()

    def _show_in_notebook(self) -> None:
        from IPython.display import SVG, display

        svg_graph = self._as_svg()
        display(SVG(svg_graph))

    def _show_in_program(self) -> None:
        tmp_path = None
        try:
            with NamedTemporaryFile(suffix='.svg', delete=False) as file:
                tmp_path = file.name
                self.write_svg(tmp_path)
                subprocess.run([DISPLAY_PROGRAM, tmp_path], check=False)
        except Exception:
            # Clean up temporary file on error
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def write_dot(self, path: str | Path) -> None:
        """Write the Dot Graph as file.

        Args:
            path: Path to the Dot file.
        """
        Path(path).write_text(self.dot_graph)

    def write_svg(self, path: str | Path) -> None:
        """Write the Dot Graph as an SVG file.

        Args:
            path: Path to the SVG file.
        """
        svg_graph = self._as_svg()
        Path(path).write_text(svg_graph)

    def _as_svg(self) -> str:
        try:
            result: str = graphviz.pipe_string('dot', 'svg', self.dot_graph, encoding='utf-8')
            return result
        except graphviz.ExecutableNotFound as e:
            raise GraphvizNotFoundError(
                'Graphviz is not installed or not found in PATH. '
                'Please install Graphviz: https://graphviz.org/download/'
            ) from e

    @staticmethod
    def _in_notebook() -> bool:
        """Return True if running inside a Jupyter notebook."""
        try:
            from IPython import get_ipython
        except ImportError:
            return False

        shell: InteractiveShell | None = get_ipython()
        if shell is None:
            return False
        if 'IPKernelApp' not in shell.config:
            return False
        return True


class HLOViewer:
    """Viewer for HLO representations.

    Attributes:
        hlo: HLO text representation.
    """

    def __init__(self, hlo: str) -> None:
        self.hlo = hlo

    def show(self) -> None:
        """Display the HLO graph as SVG.

        In a Jupyter notebook, displays inline. Otherwise, opens the system's default SVG viewer.
        """
        self._dot_graph_viewer.show()

    def write_dot(self, path: str | Path) -> None:
        """Write the Dot Graph as file.

        Args:
            path: Path to the Dot file.
        """
        self._dot_graph_viewer.write_dot(path)

    def write_svg(self, path: str | Path) -> None:
        """Write the Dot Graph as an SVG file.

        Args:
            path: Path to the SVG file.
        """
        self._dot_graph_viewer.write_svg(path)

    @cached_property
    def _dot_graph_viewer(self) -> DotGraphViewer:
        hlo_module = xla.hlo_module_from_text(self.hlo)
        dot_graph = xla.hlo_module_to_dot_graph(hlo_module)
        return DotGraphViewer(dot_graph)
