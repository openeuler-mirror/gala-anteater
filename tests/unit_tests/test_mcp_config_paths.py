#!/usr/bin/python3
# ******************************************************************************
# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# gala-anteater is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan
# PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************************/
"""Guard against drift of the configuration paths of the MCP server.

The server used to read /etc/gala-anteater-mcp/config/prompts.json while
setup.py installed suggestion_generation_prompts.json, so the prompt template
could never be loaded. The runtime paths now live in one module
(common/paths.py) which is derived from the same file names that setup.py
installs. This test keeps the three sides, the installing setup.py, the
reading code and the shipped template content, tied together.

Everything is read from the source tree, either through ast or as plain
text, so the test runs without installing any dependency.
"""

import ast
import importlib.util
import json
import os
import unittest
from glob import glob
from os.path import basename, dirname, isfile, join, realpath

REPO_ROOT = realpath(join(dirname(__file__), "..", ".."))
MCP_ROOT = join(REPO_ROOT, "anteater_mcp")
MCP_SETUP = join(MCP_ROOT, "setup.py")
MCP_CONFIG_DIR = join(MCP_ROOT, "config")
MCP_PACKAGE_DIR = join(MCP_ROOT, "container_disruption_detection_mcp")
PATHS_MODULE = join(MCP_PACKAGE_DIR, "common", "paths.py")


def _load_paths(config_dir=None):
    """Load common/paths.py standalone, it only depends on os"""
    previous = os.environ.get("ANTEATER_MCP_CONFIG_DIR")
    if config_dir is None:
        os.environ.pop("ANTEATER_MCP_CONFIG_DIR", None)
    else:
        os.environ["ANTEATER_MCP_CONFIG_DIR"] = config_dir
    try:
        spec = importlib.util.spec_from_file_location(
            "anteater_mcp_config_paths", PATHS_MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("ANTEATER_MCP_CONFIG_DIR", None)
        else:
            os.environ["ANTEATER_MCP_CONFIG_DIR"] = previous


def _normalize(path):
    """Compare the POSIX install destination of setup.py on any platform"""
    return path.replace("\\", "/").rstrip("/")


def _data_files(path):
    """[(destination directory, installed file name)] of a setup.py"""
    with open(path, encoding="utf-8") as file:
        tree = ast.parse(file.read())
    installed = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", "") != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg != "data_files":
                continue
            for entry in getattr(keyword.value, "elts", []):
                if not isinstance(entry, ast.Tuple):
                    continue
                dest = entry.elts[0]
                if not isinstance(dest, ast.Constant):
                    continue
                for pattern in entry.elts[1:]:
                    for name in _globbed_names(pattern, path):
                        installed.append((dest.value, name))
    return installed


def _globbed_names(node, setup_path):
    """File names matched by one glob("...") call of a setup.py"""
    if not isinstance(node, ast.Call):
        return []
    if getattr(node.func, "id", "") != "glob":
        return []
    pattern = node.args[0].value
    # setup.py is run from its own directory, so its patterns are relative
    # to it; the repository root keeps the result independent of the cwd
    root = dirname(realpath(setup_path))
    matched = glob(join(root, pattern))
    return sorted(basename(name) for name in matched)


class TestMcpConfigPaths(unittest.TestCase):
    """The paths read at runtime must be the files installed on the host"""

    @classmethod
    def setUpClass(cls):
        cls.paths = _load_paths()
        cls.installed = _data_files(MCP_SETUP)
        cls.config_names = {
            cls.paths.MAIN_CONFIG_FILE,
            cls.paths.JOB_CONFIG_FILE,
            cls.paths.PROMPT_CONFIG_FILE,
        }
        cls.config_dest = _normalize(cls.paths.DEFAULT_MCP_CONFIG_DIR)
        cls.installed_config = sorted(
            name for dest, name in cls.installed
            if _normalize(dest) == cls.config_dest
        )

    def test_config_files_should_be_installed_in_the_runtime_dir(self):
        installed = {
            name: _normalize(dest) for dest, name in self.installed
            if name in self.config_names
        }
        for name in sorted(self.config_names):
            self.assertIn(
                name, installed,
                "setup.py does not install %s at all" % name)
            self.assertEqual(
                installed[name], self.config_dest,
                "anteater_mcp/config/%s goes to %s, the runtime reads %s"
                % (name, installed[name], self.config_dest))

    def test_every_config_file_should_be_installed(self):
        missing = sorted(self.config_names - set(self.installed_config))
        self.assertEqual(
            [], missing,
            "setup.py installs %s but not %s" % (
                self.installed_config, missing))

    def test_prompt_template_should_be_the_one_setup_py_installs(self):
        # the bug this guard locks out: the code asked for prompts.json
        self.assertNotEqual("prompts.json", self.paths.PROMPT_CONFIG_FILE)
        self.assertIn(
            self.paths.PROMPT_CONFIG_FILE, self.installed_config)
        self.assertIn(
            basename(self.paths.prompt_config_path()), self.installed_config)

    def test_config_files_should_exist_in_the_source_tree(self):
        for name in (self.paths.MAIN_CONFIG_FILE, self.paths.JOB_CONFIG_FILE,
                     self.paths.PROMPT_CONFIG_FILE):
            self.assertTrue(
                isfile(join(MCP_CONFIG_DIR, name)),
                "anteater_mcp/config/%s is missing" % name)

    def test_config_dir_should_be_overridable(self):
        temp_dir = join(MCP_ROOT, "config")
        overridden = _load_paths(config_dir=temp_dir)
        self.assertEqual(
            join(temp_dir, self.paths.PROMPT_CONFIG_FILE),
            overridden.prompt_config_path())
        self.assertEqual(
            join(temp_dir, self.paths.JOB_CONFIG_FILE),
            overridden.job_config_path())
        self.assertEqual(
            join(temp_dir, self.paths.MAIN_CONFIG_FILE),
            overridden.main_config_path())
        self.assertEqual(
            self.paths.DEFAULT_MCP_CONFIG_DIR,
            _load_paths().DEFAULT_MCP_CONFIG_DIR)

    def test_runtime_code_should_not_hardcode_a_path(self):
        hardcoded = []
        for path, _, names in os.walk(MCP_PACKAGE_DIR):
            if "__pycache__" in path:
                continue
            for name in sorted(names):
                if not name.endswith(".py") or name == "paths.py":
                    continue
                with open(join(path, name), encoding="utf-8") as file:
                    if "/etc" in file.read():
                        hardcoded.append(name)
        self.assertEqual(
            [], hardcoded,
            "read the path from common/paths.py instead of hardcoding it")

    def test_prompt_template_should_fit_the_loader(self):
        path = join(MCP_CONFIG_DIR, self.paths.PROMPT_CONFIG_FILE)
        with open(path, encoding="utf-8") as file:
            prompts = json.load(file)
        self.assertIn("user_prompt", prompts)
        self.assertIn("format_examples", prompts)
        for placeholder in ("{detection_report}", "{analysis_report}"):
            self.assertIn(
                placeholder, prompts["user_prompt"],
                "the loader formats user_prompt with %s" % placeholder)
        # naive_recovery_suggestion_llm validates every item against
        # RecoverySuggestion, so the examples have to stay in that shape
        self.assertTrue(prompts["format_examples"])
        for example in prompts["format_examples"]:
            self.assertEqual(
                {"suggestion", "evidence", "example"}, set(example))


if __name__ == "__main__":
    unittest.main()
