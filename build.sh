#!/bin/bash
# build.sh - Script ottimizzato per Modern CMake + Pyenv

set -e  
set -o pipefail 

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' 

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}   BUILD REGRESSION LIBRARY (PYENV)       ${NC}"
echo -e "${BLUE}==========================================${NC}"

# 1. RILEVAMENTO AMBIENTE (PYENV)
PYTHON_EXE=$(which python)
PYTHON_VER=$($PYTHON_EXE --version)

echo -e "${YELLOW}Using Python:${NC} $PYTHON_EXE ($PYTHON_VER)"

# 2. ARGOMENTI
CLEAN_BUILD="OFF"
for arg in "$@"; do
    [[ "$arg" == "--clean" ]] && CLEAN_BUILD="ON"
done

# 3. PULIZIA
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$CLEAN_BUILD" == "ON" ]]; then
    echo -e "${YELLOW}Pulizia directory build...${NC}"
    rm -rf "$PROJECT_ROOT/build"
fi

mkdir -p "$PROJECT_ROOT/build"
cd "$PROJECT_ROOT/build"

# 4. CONFIGURAZIONE CMAKE
# Passiamo esplicitamente l'eseguibile di pyenv a CMake
echo -e "\n${BLUE}CONFIGURAZIONE CMAKE...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$PYTHON_EXE" \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_TESTS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# 5. COMPILAZIONE
echo -e "\n${BLUE}COMPILAZIONE IN CORSO...${NC}"
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
make -j$NPROC

# 6. VERIFICA RISULTATI
echo -e "\n${BLUE}CHECK GENERATED FILES...${NC}"

# Controlla librerie core
if ls lib/libregression* 1> /dev/null 2>&1; then
    echo -e "${GREEN}✓ Librerie generate in build/lib:${NC}"
    ls -h lib/libregression*
else
    echo -e "${RED}✗ Errore: Librerie core non trovate!${NC}"
fi

# Controlla modulo python
echo -e "\n${YELLOW}Esecuzione test import Python...${NC}"
$PYTHON_EXE "$PROJECT_ROOT/tests/test_python_import.py"

#!/bin/bash
# build.sh - Script per build automatica

set -e

echo "🧱 ML Library Build Script"
echo "=========================="

# Configurazione
BUILD_DIR="build"
INSTALL_DIR="${INSTALL_DIR:-${HOME}/.local}"
PYTHON_TEST=true
CLEAN_BUILD=false

# Parse argomenti
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --no-python)
            PYTHON_TEST=false
            shift
            ;;
        --install-dir=*)
            INSTALL_DIR="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--clean] [--no-python] [--install-dir=/path]"
            exit 1
            ;;
    esac
done

# Clean build se richiesto
if [ "$CLEAN_BUILD" = true ] && [ -d "$BUILD_DIR" ]; then
    echo "🧹 Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Crea directory build
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configura CMake
echo "🔧 Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_TESTS=ON \
    -DVERBOSE_OUTPUT=ON \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

# Build
echo "🏗️  Building..."
make -j$(nproc)

# Test C++
echo "🧪 Running C++ tests..."
if ctest --output-on-failure; then
    echo "✅ C++ tests passed"
else
    echo "❌ C++ tests failed"
    exit 1
fi

# Test Python
if [ "$PYTHON_TEST" = true ]; then
    echo "🐍 Testing Python module..."
    if make test_python_module; then
        echo "✅ Python module test passed"
    else
        echo "❌ Python module test failed"
        exit 1
    fi
fi

# Install
echo "📦 Installing to ${INSTALL_DIR}..."
make install

echo ""
echo "🎉 Build completed successfully!"
echo ""
echo "Quick test:"
echo "  python3 -c \""
echo "  import sys"
echo "  sys.path.insert(0, '${INSTALL_DIR}/lib/python*/site-packages')"
echo "  import regression_module as ml"
echo "  print('ML Library:', ml.__version__)"
echo "  \""
echo ""
echo "Library installed in: ${INSTALL_DIR}"# PY_MOD=$(find . -name "regression_module*.so")
# if [[ -n "$PY_MOD" ]]; then
#     echo -e "${GREEN}✓ Modulo Python generato:${NC} $PY_MOD"
#     # Test rapido di import
#     echo -n "Test import modulo... "
#     if $PYTHON_EXE -c "import sys; sys.path.insert(0, '$(dirname "$PY_MOD")'); import regression_module; print('OK')" 2>/dev/null; then
#         echo -e "${GREEN}SUCCESS${NC}"
#     else
#         echo -e "${RED}FAILED${NC} (Verifica i binding in C++)"
#     fi
# fi

echo -e "\n${BLUE}==========================================${NC}"
echo -e "${GREEN}BUILD COMPLETATA!${NC}"
echo -e "Esegui i test con: ${YELLOW}cd build && ctest${NC}"
echo -e "${BLUE}==========================================${NC}"