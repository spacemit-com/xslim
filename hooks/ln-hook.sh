#!/bin/sh
# soft link the git hooks script to .git/hooks

ln -s $PWD/hooks/pre-commit $PWD/.git/hooks
