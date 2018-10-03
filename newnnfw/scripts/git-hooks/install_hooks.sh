#!/usr/bin/env bash

REPO_PATH=$(git rev-parse --show-toplevel)
REPO_HOOKS_PATH=scripts/git-hooks
GIT_HOOKS_PATH=$REPO_PATH/.git/hooks
REPO_PATH_REL=../.. # Relative path from REPO_HOOKS_PATH

# Create symbolic links to hooks dir

# NOTE `ln -s` does not overwrite if the file exists.
ln -s $REPO_PATH_REL/$REPO_HOOKS_PATH/pre-push $GIT_HOOKS_PATH/pre-push
