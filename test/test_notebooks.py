# coding=utf-8
"""
These tests try to execute all Jupyter/IPython notebooks found in ./notebooks and convert them to
separate HTML pages.

In preparation to HTML conversion and to allow portability between Python 2 and 3, all notebooks'
metadata is patched to match the current environment.
"""
import codecs
import json
import os
import os.path
import pprint
import subprocess as sp
import sys

import nose.tools

BASE_PATH = os.path.abspath(os.path.join(__file__, '..', '..'))
IPYNB_PATH = os.path.join(BASE_PATH, 'notebooks')
HTML_PATH = os.path.join(IPYNB_PATH, 'html')
TEMP_PATH = os.path.join(IPYNB_PATH, 'temp')

IS_PY2 = sys.version_info[0] == 2
IS_PY3 = sys.version_info[0] == 3


def get_notebooks():
    """
    Walks through <pyMG>/notebooks and lists all '.ipynb' files.

    Jupyter's checkpoint directories ('.ipynb_checkpoints') and temporary directories are skipped.

    Returns:
        list of tuples of str: Each tuple contains the absolute path to the file and the file name.
    """
    _notebooks = []

    for root, _, files in os.walk(os.path.join(BASE_PATH, 'notebooks')):
        if root.endswith('.ipynb_checkpoints'):
            # skip IPython checkpoints
            continue
        if root == TEMP_PATH:
            # skip previously converted notebooks
            continue

        for f in files:
            if f.endswith('ipynb'):
                _notebooks.append((root, f))

    return _notebooks


def convert_notebook_py2_py3(ipynb_dir, ipynb_file):
    """
    Patches given IPython notebook's metadata to match current environment

    Args:
        ipynb_dir: absolute path to the notebook
        ipynb_file: the file name of the notebook

    Returns:
        str: absolute path to the converted notebook file
    """
    ipynb_path = os.path.join(ipynb_dir, ipynb_file)
    nose.tools.assert_true(os.path.isfile(ipynb_path))

    with codecs.open(ipynb_path, mode='r', encoding='utf-8') as fh:
        json_doc = json.load(fh)

    json_doc['metadata']['kernelspec']['display_name'] = u"Python %d" % sys.version_info[0]
    json_doc['metadata']['kernelspec']['name'] = u"python%d" % sys.version_info[0]
    json_doc['metadata']['language_info']['pygments_lexer'] = u"ipython%d" % sys.version_info[0]
    json_doc['metadata']['language_info']['codemirror_mode']['version'] = sys.version_info[0]
    json_doc['metadata']['language_info']['version'] = \
        u"%s.%s.%s" % (sys.version_info[0], sys.version_info[1], sys.version_info[2])

    pprint.pprint(json_doc['metadata'])

    if not os.path.isdir(TEMP_PATH):
        os.mkdir(TEMP_PATH)

    converted_path = os.path.join(TEMP_PATH, ipynb_file)
    with codecs.open(converted_path, mode='w', encoding='utf-8') as fh:
        json.dump(json_doc, fh, indent=1, ensure_ascii=False)
    nose.tools.assert_true(os.path.isfile(converted_path))

    return converted_path


def convert_notebook_to_html(ipynb_dir, ipynb_file):
    """
    Calls jupyter-nbconvert to execute and convert given notebook to HTML

    Args:
        ipynb_dir: absolute path to the notebook
        ipynb_file: the file name of the notebook
    """
    html_path = os.path.join(HTML_PATH, ipynb_file).replace('ipynb', 'html')

    ipynb_path = convert_notebook_py2_py3(ipynb_dir, ipynb_file)
    nose.tools.assert_true(os.path.isfile(ipynb_path))

    return_code = sp.check_call(['jupyter-nbconvert', '--to', 'html', '--execute', ipynb_path,
                                 '--output', html_path], stdout=sp.PIPE, stderr=sp.STDOUT)

    nose.tools.assert_equal(return_code, 0)


def test_notebooks():
    """
    Test generator for notebook execution and conversion
    """
    if not os.path.isdir(HTML_PATH):
        os.mkdir(HTML_PATH)

    for root_nb in get_notebooks():
        ipynb_dir, ipynb_file = root_nb

        yield convert_notebook_to_html, ipynb_dir, ipynb_file
