#!/bin/bash
# build_all.sh - Script di build per regression library

set -e  # Exit on error
set -o pipefail  # Capture pipe errors

echo "=========================================="
echo "BUILD REGRESSION LIBRARY + PYTHON BINDINGS"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directory progetto
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo -e "${BLUE}Project directory:${NC} $PROJECT_ROOT"

# Check requirements
echo -e "\n${BLUE}1. CHECKING REQUIREMENTS...${NC}"

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}✗ $1 not found${NC}"
        return 1
    else
        echo -e "${GREEN}✓ $1 found${NC} ($($1 --version | head -n1))"
        return 0
    fi
}

# Essential commands
check_command cmake || exit 1
check_command make || exit 1
check_command g++ || check_command clang++ || { echo -e "${RED}✗ No C++ compiler found${NC}"; exit 1; }

# Python and pybind11 check (optional)
if python3 -c "import pybind11" 2>/dev/null; then
    echo -e "${GREEN}✓ pybind11 found in Python${NC}"
    PYTHON_BINDINGS="ON"
else
    echo -e "${YELLOW}⚠ pybind11 not found in Python${NC}"
    PYTHON_BINDINGS="OFF"
fi

# Clean previous builds
echo -e "\n${BLUE}2. CLEANING...${NC}"
read -p "Do you want to clean previous builds? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing build directories..."
    rm -rf "$PROJECT_ROOT/build" "$PROJECT_ROOT/output" "$PROJECT_ROOT/lib"
fi

# Create directories
mkdir -p "$PROJECT_ROOT/build"
mkdir -p "$PROJECT_ROOT/output"

# Configure CMake
echo -e "\n${BLUE}3. CONFIGURING CMAKE...${NC}"
cd "$PROJECT_ROOT/build"

# Detect number of CPU cores
if [[ "$OSTYPE" == "darwin"* ]]; then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=$(nproc)
fi

# Build options
BUILD_OPTS=(
    "-DCMAKE_BUILD_TYPE=Release"
    "-DBUILD_PYTHON_BINDINGS=${PYTHON_BINDINGS}"
    "-DBUILD_TESTS=ON"
    "-DBUILD_EXAMPLES=ON"
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
)

# Configure
echo -e "${YELLOW}CMake options:${NC} ${BUILD_OPTS[*]}"
cmake .. "${BUILD_OPTS[@]}"

# Compile
echo -e "\n${BLUE}4. COMPILING...${NC}"
echo -e "${YELLOW}Using $NPROC parallel jobs${NC}"
make -j$NPROC VERBOSE=1

# Run tests
echo -e "\n${BLUE}5. RUNNING TESTS...${NC}"
if [[ -f "$PROJECT_ROOT/build/CTestTestfile.cmake" ]]; then
    cd "$PROJECT_ROOT/build"
    if ctest --output-on-failure; then
        echo -e "${GREEN}✓ All tests passed${NC}"
    else
        echo -e "${RED}✗ Some tests failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠ No tests found${NC}"
fi

# Check generated files
echo -e "\n${BLUE}6. CHECKING GENERATED FILES...${NC}"

print_files() {
    local dir="$1"
    local pattern="$2"
    local description="$3"
    
    if [[ -d "$dir" ]]; then
        echo -e "\n${YELLOW}$description:${NC}"
        find "$dir" -name "$pattern" -type f | while read -r file; do
            size=$(ls -lh "$file" | awk '{print $5}')
            echo -e "  ${GREEN}✓${NC} $(basename "$file") ($size)"
        done
    fi
}

# Check libraries
print_files "$PROJECT_ROOT/build" "*.a" "Static libraries (.a)"
print_files "$PROJECT_ROOT/build" "*.so" "Shared libraries (.so)"

# Check Python module if built
if [[ "$PYTHON_BINDINGS" == "ON" ]]; then
    echo -e "\n${YELLOW}Python module check:${NC}"
    PY_MODULE=$(find "$PROJECT_ROOT/build" -name "regression*.so" -type f | head -1)
    if [[ -n "$PY_MODULE" ]]; then
        echo -e "  ${GREEN}✓ Python module found:${NC} $(basename "$PY_MODULE")"
        
        # Try to import
        echo -e "  ${YELLOW}Testing import...${NC}"
        if python3 -c "import sys; sys.path.insert(0, '$(dirname "$PY_MODULE")'); import regression; print('    Import successful: ' + str(regression.__version__))" 2>/dev/null; then
            echo -e "  ${GREEN}✓ Python module imports correctly${NC}"
        else
            echo -e "  ${RED}✗ Python module import failed${NC}"
        fi
    else
        echo -e "  ${RED}✗ No Python module found${NC}"
    fi
fi

# Check executables
echo -e "\n${YELLOW}Executables:${NC}"
find "$PROJECT_ROOT/build" -type f -executable ! -name "*.so" ! -name "*.dylib" ! -name "*.dll" | while read -r exe; do
    echo -e "  ${GREEN}✓${NC} $(basename "$exe")"
done

# Generate compile_commands.json symlink for tools
echo -e "\n${BLUE}7. SETTING UP DEVELOPMENT TOOLS...${NC}"
if [[ -f "$PROJECT_ROOT/build/compile_commands.json" ]]; then
    ln -sf "$PROJECT_ROOT/build/compile_commands.json" "$PROJECT_ROOT/compile_commands.json" 2>/dev/null || true
    echo -e "${GREEN}✓ Created compile_commands.json symlink${NC}"
fi

# Summary
echo -e "\n${BLUE}==========================================${NC}"
echo -e "${GREEN}BUILD COMPLETED SUCCESSFULLY!${NC}"
echo -e "${BLUE}==========================================${NC}"
echo -e "\n${YELLOW}Output directories:${NC}"
echo -e "  Build artifacts: ${PROJECT_ROOT}/build/"
echo -e "  Libraries: ${PROJECT_ROOT}/output/"
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  Run examples: cd ${PROJECT_ROOT}/build/examples && ./example_usage"
echo -e "  Test Python: python3 -c \"import sys; sys.path.insert(0, '${PROJECT_ROOT}/build'); import regression\""

if [[ "$PYTHON_BINDINGS" == "ON" ]]; then
    echo -e "\n${YELLOW}Python installation (optional):${NC}"
    echo "  # Install to user site-packages"
    echo "  pip install --user ${PROJECT_ROOT}/build/pybinding/"
fi

echo -e "\n${BLUE}==========================================${NC}"