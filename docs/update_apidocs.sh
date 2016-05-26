#!/usr/bin/env bash

# Assuring we are running in the project's root
[[ -d "${PWD}/docs" && "./docs/update_apidocs.sh" == "$0" ]] ||
    {
        echo "ERROR: You must be in the project root."
        exit 1
    }

SPHINX_APIDOC="`which sphinx-apidoc`"
[[ -x "$SPHINX_APIDOC" ]] ||
    {
        echo "ERROR: sphinx-apidoc not found."
        exit 1
    }

echo "removing existing .rst files ..."
rm ${PWD}/docs/source/pymg/*.rst
rm ${PWD}/docs/source/project/*.rst

echo ""
echo "generating new .rst files ..."
${SPHINX_APIDOC} -o docs/source/pymg -e pymg --force
${SPHINX_APIDOC} -o docs/source/project -e project --force
