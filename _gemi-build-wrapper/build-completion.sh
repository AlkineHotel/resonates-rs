#!/bin/bash
#
# Bash completion for the Rust Cross-Compilation Build Wrapper

_resonates_build_completion() {
    local cur prev_word commands targets

    # COMP_WORDS: an array of words in the current command line.
    # COMP_CWORD: the index of the word being completed.
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev_word="${COMP_WORDS[COMP_CWORD-1]}"

    commands="list list-installed build help"

    # If we are completing the first argument (the command itself)
    if [ "$COMP_CWORD" -eq 1 ]; then
        COMPREPLY=( $(compgen -W "${commands}" -- "${cur}") )
        return 0
    fi

    # If the previous word was 'build', we should suggest targets.
    # This is the main completion logic for build targets.
    if [ "$prev_word" == "build" ]; then
        # Get all available targets from rustup, and extract the first column (the target triple).
        # This is done once and stored in a variable to avoid multiple calls.
        all_targets=$(rustup target list | awk ''{print $1}'')

        # Get all words in the current command line starting from the third word (after 'build.sh build').
        # These are the targets that have already been typed.
        local existing_targets
        existing_targets=$(printf "%s\n" "${COMP_WORDS[@]:2}")

        # Find targets that are available but not yet typed in the command line.
        # `comm -23` shows lines unique to the first file (all_targets).
        local filtered_targets
        filtered_targets=$(comm -23 <(echo "$all_targets" | sort) <(echo "$existing_targets" | sort))

        # Generate completions from the filtered list of targets.
        COMPREPLY=( $(compgen -W "${filtered_targets}" -- "${cur}") )
        return 0
    fi
}

# Register the completion function for the build.sh script.
# This allows completion to work whether you call it as 'build.sh' or './build.sh'
complete -F _resonates_build_completion build.sh
complete -F _resonates_build_completion ./build.sh
