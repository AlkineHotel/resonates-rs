#!/bin/bash
#
# Rust Cross-Compilation Build Wrapper
#
# This script simplifies the process of building a Rust crate for multiple
# target architectures. It can list available and installed targets, and
# build for one or more targets interactively.
#
# Note for Windows users: This is a bash script and requires a bash-compatible
# shell to run, such as Git Bash, or the Windows Subsystem for Linux (WSL).

# --- HELP FUNCTION ---
# Shows how to use the script
usage() {
    echo "Rust Cross-Compilation Build Wrapper"
    echo "------------------------------------"
    echo "Usage: ./build.sh [command]"
    echo ""
    echo "Commands:"
    echo "  list              List all available Rust compilation targets."
    echo "  list-installed    List all currently installed Rust targets."
    echo "  build <target...> Build the crate for one or more specified targets."
    echo "                    (e.g., x86_64-pc-windows-msvc aarch64-apple-darwin)"
    echo "  help              Show this help message."
    echo ""
    echo "Bash Completion:"
    echo "  To permanently install bash completion, run the installer script:"
    echo "  ./_gemi-build-wrapper/install.sh"
    echo "  This will guide you through adding the completion to your shell startup file."
    echo ""
    echo "Prerequisites:"
    echo " - rustup and cargo must be installed and in your PATH."
    echo " - For some targets, you may need to install the appropriate C linker."
    echo ""
    echo "Example:"
    echo "  # Build for 64-bit Linux and Windows"
    echo "  ./build.sh build x86_64-unknown-linux-gnu x86_64-pc-windows-msvc"
}

# --- TARGET LISTING FUNCTIONS ---
list_available_targets() {
    echo "Querying for available Rust targets..."
    rustup target list
}

list_installed_targets() {
    echo "Querying for installed Rust targets..."
    rustup target list --installed
}

# --- BUILD FUNCTION ---
# Builds the crate for the given target triples
build_targets() {
    if [ $# -eq 0 ]; then
        echo "Error: No build targets specified for the 'build' command." >&2
        echo "Please provide at least one target triple." >&2
        echo "Example: ./build.sh build x86_64-unknown-linux-musl" >&2
        exit 1
    fi

    echo "Starting build process for: $@"
    
    for target in "$@"; do
        echo ""
        echo "========================================"
        echo "Processing target: $target"
        echo "========================================"

        # Check if the target is installed
        if ! rustup target list --installed | grep -q "^$target"; then
            echo "Target '$target' is not installed."
            # Ask the user if they want to install it
            read -p "Would you like to install it now? (y/n) " -n 1 -r
            echo # Move to a new line
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "Installing target '$target' with rustup..."
                rustup target add "$target"
                if [ $? -ne 0 ]; then
                    echo "Error: Failed to install target '$target'. Skipping." >&2
                    continue # Move to the next target
                fi
                echo "Target '$target' installed successfully."
            else
                echo "Skipping build for target '$target'."
                continue # Move to the next target
            fi
        else
            echo "Target '$target' is already installed."
        fi

        # Build the project for the target
        echo "Running: cargo build --release --target "$target""
        cargo build --release --target "$target"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "SUCCESS: Build for target '$target' completed."
            echo "Binary is available at: ./target/$target/release/"
        else
            echo ""
            echo "ERROR: Build for target '$target' failed." >&2
            echo "This may be due to a missing linker or other system dependencies." >&2
            echo "Please check the output above for more details." >&2
        fi
    done

    echo ""
    echo "========================================"
    echo "Build process finished."
    echo "========================================"
}

# --- MAIN SCRIPT LOGIC ---
# Parse command-line arguments

# No arguments given, show usage.
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

case "$1" in
    list)
        list_available_targets
        ;;
    list-installed)
        list_installed_targets
        ;;
    build)
        shift # Remove 'build' from the arguments list
        build_targets "$@"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Error: Unknown command '$1'." >&2
        usage
        exit 1
        ;;
esac
