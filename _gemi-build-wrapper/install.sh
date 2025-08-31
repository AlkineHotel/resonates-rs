#!/bin/bash
#
# Installation script for the Rust build wrapper's bash completion.

# --- SCRIPT SETUP ---
# Exit immediately if a command exits with a non-zero status.
set -e

# --- PATHS AND VARIABLES ---
# Get the directory of the current script to reliably locate the completion file.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPLETION_SCRIPT_PATH="${SCRIPT_DIR}/build-completion.sh"

# --- MAIN LOGIC ---
echo "Bash Completion Installer for the Rust Build Wrapper"
echo "----------------------------------------------------"

# --- DETECT SHELL PROFILE ---
# Determine the user's shell profile file.
PROFILE_FILE=""
if [ -n "$BASH_VERSION" ]; then
    if [ -f "$HOME/.bashrc" ]; then
        PROFILE_FILE="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        PROFILE_FILE="$HOME/.bash_profile"
    fi
elif [ -n "$ZSH_VERSION" ]; then
    if [ -f "$HOME/.zshrc" ]; then
        PROFILE_FILE="$HOME/.zshrc"
    fi
fi

if [ -z "$PROFILE_FILE" ]; then
    echo "Error: Could not find a supported shell profile file (~/.bashrc, ~/.bash_profile, or ~/.zshrc)." >&2
    echo "Please add the following line to your shell's startup file manually:" >&2
    echo "  source \"$COMPLETION_SCRIPT_PATH\"" >&2
    exit 1
fi

# --- USER CONFIRMATION ---
echo "This script will add a line to your shell profile file to enable bash completion."
echo "The following file will be modified: $PROFILE_FILE"
echo "The following line will be added:"
echo "  source \"$COMPLETION_SCRIPT_PATH\""
echo ""
read -p "Do you want to proceed? (y/n) " -n 1 -r
echo # Move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled by user." >&2
    exit 1
fi

# --- INSTALLATION ---
# Check if the completion is already installed to avoid duplicate entries.
if grep -Fxq "source \"$COMPLETION_SCRIPT_PATH\"" "$PROFILE_FILE"; then
    echo "Completion is already installed in $PROFILE_FILE. No changes were made."
else
    # Add the source line to the profile file.
    echo -e "\n# Added by the Rust build wrapper installer for bash completion" >> "$PROFILE_FILE"
    echo "source \"$COMPLETION_SCRIPT_PATH\"" >> "$PROFILE_FILE"
    echo ""
    echo "SUCCESS: Bash completion has been installed."
    echo "Please restart your shell or run the following command for the changes to take effect:"
    echo "  source \"$PROFILE_FILE\""
fi

exit 0
